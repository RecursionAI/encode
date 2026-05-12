"""relay() and relay_async() — the chat/completions + responses wrapper.

Single entry point that:

- talks to /v1/chat/completions or /v1/responses (auto-selected from inputs)
- runs an automatic tool-call loop with Python callables
- supports response_format (Pydantic model or dict) for structured outputs
- exposes intercept callbacks via both ``relay(...).intercept(cb)`` and
  ``on_intercept=`` kwarg, with explicit ``event.stop()`` for early termination
- supports streaming, including across the full tool-call loop
"""

from __future__ import annotations

import inspect
import itertools
import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from . import _http, errors, tools
from ._schema import pydantic_to_response_format
from ._streaming import (
    StreamEvent,
    accumulate_chat_tool_calls,
    aiter_chat_completions,
    aiter_responses,
    finalize_chat_tool_calls,
    iter_chat_completions,
    iter_responses,
)
from .events import Event, EventType
from .executor import LocalToolExecutor, ToolExecutor
from .messages import Message, Messages, to_chat_messages, to_responses_input
from .responses import AssistantTurn, RelayResponse, ToolCallRecord, Usage
from .session import Session

InterceptCallback = Callable[["InterceptEvent"], Any]
AsyncInterceptCallback = Callable[["InterceptEvent"], Any]


@dataclass
class InterceptEvent:
    """Snapshot + mutation point between iterations of the tool loop.

    ``messages`` is a *live* :class:`Messages` view of the conversation as it
    stands now. Mutations applied inside the intercept callback (via the
    helpers below, or by editing ``messages`` directly) are picked up by the
    harness and become the input to the next iteration.

    When the harness was started with ``session=``, mutations also emit a
    ``context.modify`` event into the session log so the durable record
    captures what the intercept decided.

    For the ``responses`` endpoint, mutating ``messages`` causes the harness
    to reproject the next iteration's ``input`` items from the chat-style
    history; this loses any non-message typed items the model may have
    emitted (e.g. reasoning items). For most context-engineering use cases
    this is fine; if you need to preserve typed items, leave ``messages``
    alone and use ``stop()`` instead.
    """

    iteration: int
    endpoint: Literal["chat", "responses"]
    assistant_turn: AssistantTurn
    tool_calls: list[ToolCallRecord]
    raw_response: dict[str, Any]
    will_continue: bool
    messages: Messages = field(default_factory=Messages)
    _session: Session | None = field(default=None, repr=False)
    _initial_snapshot: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _stopped: bool = field(default=False, repr=False)

    @property
    def session(self) -> Session | None:
        """The Session attached to this run, or ``None`` if running stateless.

        Read-only handle to the durable event log; mutate via verbs like
        ``register_tool`` rather than poking the events list directly.
        """
        return self._session

    def __post_init__(self) -> None:
        # Capture starting state for mutation detection. Use deep-enough copies
        # so user mutations don't bleed back into the snapshot.
        self._initial_snapshot = [dict(m) for m in self.messages]

    # ------------------------- backward-compat view -------------------------

    @property
    def messages_so_far(self) -> list[dict[str, Any]]:
        """Read-only list view of the current conversation.

        Kept for backward compatibility. Prefer ``messages`` (mutable) or the
        explicit helpers below.
        """
        return list(self.messages)

    # ------------------------- mutation helpers -------------------------

    def append(self, message: Message | dict[str, Any]) -> None:
        """Append a single message to the end of the conversation."""
        self.messages.append(message)

    def insert(self, idx: int, message: Message | dict[str, Any]) -> None:
        """Insert a message at the given index."""
        d = message.model_dump(exclude_none=True) if isinstance(message, Message) else dict(message)
        self.messages._items.insert(idx, d)

    def replace(self, messages: Sequence[Message | dict[str, Any]]) -> None:
        """Replace the entire conversation with ``messages``."""
        self.messages.clear()
        self.messages.extend(messages)

    def edit_last_tool_result(
        self, fn: Callable[[Any], Any]
    ) -> None:
        """Apply ``fn`` to the ``content`` of the most recent tool message.

        Use to trim noisy tool output, redact, or otherwise transform what
        the model will see on the next iteration.
        """
        for i in range(len(self.messages) - 1, -1, -1):
            m = self.messages._items[i]
            if m.get("role") == "tool":
                m["content"] = fn(m.get("content"))
                return

    def compact(
        self,
        fn: Callable[[list[dict[str, Any]]], Sequence[Message | dict[str, Any]]],
    ) -> None:
        """Apply ``fn(messages) -> messages``; replace the conversation with the result.

        Use for summarization, trimming, or any whole-history transformation.
        """
        new = list(fn(list(self.messages)))
        self.messages.clear()
        self.messages.extend(new)

    # ------------------------- introspection -------------------------

    @property
    def mutated(self) -> bool:
        """True if the conversation was changed during the intercept callback."""
        return list(self.messages) != self._initial_snapshot

    def stop(self) -> None:
        """Mark the loop for termination after the current iteration completes."""
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def register_tool(self, tool: Any) -> bool:
        """Add a tool to the run's :class:`Session` for the next iteration.

        Idempotent: same-name registrations are no-ops. For the new tool to
        appear in the next iteration's request, the harness was started with
        ``tools=session.tools`` (or any list that shares identity with the
        session's tool list) — otherwise the tool is recorded in the durable
        log but the model won't see it on the next turn.

        Raises ``RuntimeError`` if the run has no session attached, since
        tool registrations are an audit-log event.

        Returns ``True`` if newly registered, ``False`` if a same-name tool
        already exists on the session.
        """
        if self._session is None:
            raise RuntimeError(
                "register_tool requires session=... on the relay() call"
            )
        return self._session.register_tool(tool, by="intercept")


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

_PASSTHROUGH_CHAT = (
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "user",
    "tool_choice",
)

_PASSTHROUGH_RESPONSES = (
    "temperature",
    "top_p",
    "max_output_tokens",
    "max_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "user",
    "tool_choice",
    "instructions",
)


def _resolve_endpoint(
    endpoint: Literal["chat", "responses", "auto"],
    *,
    messages: Any,
    input: Any,
    instructions: Any,
) -> Literal["chat", "responses"]:
    if endpoint != "auto":
        return endpoint
    if input is not None or instructions is not None:
        return "responses"
    if messages is not None:
        return "chat"
    raise ValueError("relay() requires `messages=` (chat) or `input=`/`instructions=` (responses)")


def _build_response_format(
    response_format: type[BaseModel] | dict[str, Any] | None,
    endpoint: Literal["chat", "responses"],
) -> tuple[Any, type[BaseModel] | None]:
    """Return (request_value, parser_model)."""
    if response_format is None:
        return None, None
    if isinstance(response_format, dict):
        return response_format, None
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        rf = pydantic_to_response_format(response_format)
        if endpoint == "responses":
            return {"format": rf}, response_format
        return rf, response_format
    raise TypeError(
        f"response_format must be a Pydantic model class or dict, got {type(response_format).__name__}"
    )


def _build_chat_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tool_dicts: list[dict[str, Any]] | None,
    response_format: Any,
    stream: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tool_dicts:
        payload["tools"] = tool_dicts
    if response_format is not None:
        payload["response_format"] = response_format
    if stream:
        payload["stream"] = True
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def _build_responses_payload(
    *,
    model: str,
    input_items: list[dict[str, Any]],
    tool_dicts: list[dict[str, Any]] | None,
    response_format: Any,
    stream: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "input": input_items}
    if tool_dicts:
        payload["tools"] = tool_dicts
    if response_format is not None:
        # response_format for /v1/responses lives under `text`
        payload["text"] = response_format
    if stream:
        payload["stream"] = True
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def _extract_chat_assistant(resp_body: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    choice = (resp_body.get("choices") or [{}])[0]
    return choice.get("message") or {}, choice.get("finish_reason")


def _extract_responses_assistant(
    resp_body: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (assistant_text, function_calls, all_output_items)."""
    output = resp_body.get("output") or []
    assistant_text_parts: list[str] = []
    function_calls: list[dict[str, Any]] = []
    for item in output:
        if item.get("type") == "message" and item.get("role") == "assistant":
            for part in item.get("content") or []:
                if part.get("type") == "output_text" and part.get("text"):
                    assistant_text_parts.append(part["text"])
        elif item.get("type") == "function_call":
            function_calls.append(item)
    text = "".join(assistant_text_parts) if assistant_text_parts else None
    return text, function_calls, output


# ---------------------------------------------------------------------------
# RelayHandle (sync)
# ---------------------------------------------------------------------------


@dataclass
class _RelayConfig:
    """Frozen call-time configuration shared by sync and async handles."""

    model: str
    messages: Any
    input: Any
    instructions: Any
    tools: Any
    tool_choice: Any
    response_format: Any
    web_search: bool
    max_tool_iterations: int | None
    stream: bool
    temperature: Any
    top_p: Any
    max_tokens: Any
    max_output_tokens: Any
    stop: Any
    presence_penalty: Any
    frequency_penalty: Any
    user: Any
    endpoint: Literal["chat", "responses"]
    extra_body: dict[str, Any]
    session: Session | None = None
    tool_executor: ToolExecutor | None = None


class RelayHandle:
    def __init__(
        self,
        client: Any,
        config: _RelayConfig,
        interceptors: list[InterceptCallback],
    ) -> None:
        self._client = client
        self._config = config
        self._interceptors = list(interceptors)
        self._memo: RelayResponse | None = None

    def intercept(self, callback: InterceptCallback) -> RelayHandle:
        self._interceptors.append(callback)
        return self

    def execute(self) -> RelayResponse:
        if self._memo is not None:
            return self._memo
        try:
            self._memo = _execute_sync(self._client, self._config, self._interceptors)
        except errors.MaxToolIterationsError as e:
            _absorb_into_messages(self._config.messages, e.partial)
            raise
        _absorb_into_messages(self._config.messages, self._memo)
        return self._memo

    @property
    def response(self) -> RelayResponse:
        return self.execute()

    def __iter__(self) -> Iterator[StreamEvent]:
        return _execute_sync_stream(self._client, self._config, self._interceptors)


class AsyncRelayHandle:
    def __init__(
        self,
        client: Any,
        config: _RelayConfig,
        interceptors: list[AsyncInterceptCallback],
    ) -> None:
        self._client = client
        self._config = config
        self._interceptors = list(interceptors)
        self._memo: RelayResponse | None = None

    def intercept(self, callback: AsyncInterceptCallback) -> AsyncRelayHandle:
        self._interceptors.append(callback)
        return self

    async def execute(self) -> RelayResponse:
        if self._memo is not None:
            return self._memo
        try:
            self._memo = await _execute_async(self._client, self._config, self._interceptors)
        except errors.MaxToolIterationsError as e:
            _absorb_into_messages(self._config.messages, e.partial)
            raise
        _absorb_into_messages(self._config.messages, self._memo)
        return self._memo

    async def get(self) -> RelayResponse:
        return await self.execute()

    def __await__(self):  # type: ignore[no-untyped-def]
        return self.execute().__await__()

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return _execute_async_stream(self._client, self._config, self._interceptors)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _gather_intercept(
    on_intercept: InterceptCallback | Sequence[InterceptCallback] | None,
) -> list[InterceptCallback]:
    if on_intercept is None:
        return []
    if callable(on_intercept):
        return [on_intercept]
    return list(on_intercept)


def relay(
    *,
    model: str,
    messages: Sequence[Any] | None = None,
    input: Any = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: type[BaseModel] | dict[str, Any] | None = None,
    web_search: bool = False,
    max_tool_iterations: int | None = None,
    stream: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    max_output_tokens: int | None = None,
    stop: str | list[str] | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    user: str | None = None,
    endpoint: Literal["chat", "responses", "auto"] = "auto",
    extra_body: dict[str, Any] | None = None,
    on_intercept: InterceptCallback | Sequence[InterceptCallback] | None = None,
    session: Session | None = None,
    tool_executor: ToolExecutor | None = None,
    client: Any = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = 60.0,
) -> RelayHandle:
    """Wrap /v1/chat/completions and /v1/responses with auto tool-call loops.

    See README for usage. Returns a :class:`RelayHandle`; access ``.response``
    to execute and get the typed :class:`RelayResponse`.

    When ``session`` is provided, events are emitted into the session's
    append-only log (``user.message``, ``assistant.message``, ``tool.call``,
    ``tool.result``, ``iteration.end``) and the initial conversation is
    hydrated from ``session.to_messages()`` (any ``messages=`` kwarg is
    appended as new user-message events first).

    When ``tool_executor`` is provided, tool calls are dispatched through it
    rather than the built-in in-process executor. Use to plug in remote
    dispatch, sandboxed execution, MCP, or sub-agent delegation. When
    omitted, a :class:`LocalToolExecutor` is built from ``tools=``.
    """
    if response_format is not None and stream:
        raise ValueError("response_format and stream cannot be combined")

    resolved_endpoint = _resolve_endpoint(
        endpoint, messages=messages, input=input, instructions=instructions
    )
    if client is None:
        from .client import Client, get_default_client

        if api_key or base_url or timeout != 60.0:
            client = Client(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            client = get_default_client()

    config = _RelayConfig(
        model=model,
        messages=messages,
        input=input,
        instructions=instructions,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        web_search=web_search,
        max_tool_iterations=max_tool_iterations,
        stream=stream,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_output_tokens=max_output_tokens,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        user=user,
        endpoint=resolved_endpoint,
        extra_body=extra_body or {},
        session=session,
        tool_executor=tool_executor,
    )
    return RelayHandle(client, config, _gather_intercept(on_intercept))


def relay_async(
    *,
    model: str,
    messages: Sequence[Any] | None = None,
    input: Any = None,
    instructions: str | None = None,
    tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: type[BaseModel] | dict[str, Any] | None = None,
    web_search: bool = False,
    max_tool_iterations: int | None = None,
    stream: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    max_output_tokens: int | None = None,
    stop: str | list[str] | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    user: str | None = None,
    endpoint: Literal["chat", "responses", "auto"] = "auto",
    extra_body: dict[str, Any] | None = None,
    on_intercept: AsyncInterceptCallback | Sequence[AsyncInterceptCallback] | None = None,
    session: Session | None = None,
    tool_executor: ToolExecutor | None = None,
    client: Any = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = 60.0,
) -> AsyncRelayHandle:
    if response_format is not None and stream:
        raise ValueError("response_format and stream cannot be combined")

    resolved_endpoint = _resolve_endpoint(
        endpoint, messages=messages, input=input, instructions=instructions
    )
    if client is None:
        from .client import AsyncClient, get_default_async_client

        if api_key or base_url or timeout != 60.0:
            client = AsyncClient(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            client = get_default_async_client()

    config = _RelayConfig(
        model=model,
        messages=messages,
        input=input,
        instructions=instructions,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        web_search=web_search,
        max_tool_iterations=max_tool_iterations,
        stream=stream,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_output_tokens=max_output_tokens,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        user=user,
        endpoint=resolved_endpoint,
        extra_body=extra_body or {},
        session=session,
        tool_executor=tool_executor,
    )
    return AsyncRelayHandle(client, config, _gather_intercept(on_intercept))


# ---------------------------------------------------------------------------
# Sync execution
# ---------------------------------------------------------------------------


def _common_extras(c: _RelayConfig) -> dict[str, Any]:
    keys = _PASSTHROUGH_CHAT if c.endpoint == "chat" else _PASSTHROUGH_RESPONSES
    extras = {k: getattr(c, k) for k in keys if hasattr(c, k)}
    extras.update(c.extra_body or {})
    return extras


def _hydrate_history(c: _RelayConfig) -> list[dict[str, Any]]:
    """Build the initial chat-style history for the loop.

    When ``session`` is provided, start from ``session.to_messages()`` and
    *append* any ``messages=`` kwarg as new user-message events. Otherwise
    just normalize ``messages=`` as before.
    """
    if c.session is None:
        return list(to_chat_messages(c.messages))
    history = list(c.session.to_messages())
    incoming = list(to_chat_messages(c.messages))
    for msg in incoming:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            c.session.emit(EventType.USER_MESSAGE, {"content": content})
        elif role == "system":
            c.session.emit(EventType.SYSTEM, {"content": content})
        elif role == "assistant":
            c.session.emit(
                EventType.ASSISTANT_MESSAGE,
                {"content": content, "tool_calls": msg.get("tool_calls")},
            )
        elif role == "tool":
            c.session.emit(
                EventType.TOOL_RESULT,
                {
                    "id": msg.get("tool_call_id", ""),
                    "result": content,
                    "result_serialized": content if isinstance(content, str) else "",
                    "error": None,
                    "duration_ms": 0.0,
                },
            )
        history.append(msg)
    return history


@dataclass
class _ToolBindings:
    """Mutable holder for the per-call tool state.

    The relay loop reads ``tool_dicts`` (sent to the model) and ``executor``
    (dispatches calls). When ``c.tools`` is a list that grows mid-loop —
    typically because it is ``session.tools`` and an intercept handler called
    ``event.register_tool(...)`` — :func:`_refresh_tools` rebuilds these
    on the next iteration so the new tool becomes visible.

    ``local`` is the in-process executor we own, if any: when present, we
    mutate its internal dispatch dict in place so its identity stays stable
    across iterations. User-supplied executors are not mutated.
    """

    tool_dicts: list[dict[str, Any]]
    executor: ToolExecutor
    local: LocalToolExecutor | None
    last_len: int


def _tools_len(c: _RelayConfig) -> int:
    return len(c.tools) if c.tools is not None else 0


def _make_bindings(c: _RelayConfig) -> _ToolBindings:
    """Initial build of the tool dispatch + schema list."""
    tool_dicts, tool_index = tools.build_tools(c.tools, web_search=c.web_search)
    if c.tool_executor is not None:
        return _ToolBindings(
            tool_dicts=tool_dicts,
            executor=c.tool_executor,
            local=None,
            last_len=_tools_len(c),
        )
    local = LocalToolExecutor(tool_index)
    return _ToolBindings(
        tool_dicts=tool_dicts,
        executor=local,
        local=local,
        last_len=_tools_len(c),
    )


def _refresh_tools(c: _RelayConfig, b: _ToolBindings) -> None:
    """Rebuild ``b.tool_dicts`` (and ``b.local`` dispatch) if ``c.tools`` grew.

    Append-only tool lists: a length change means new entries at the tail.
    We rebuild from scratch (cheap for typical tool counts) and mutate the
    in-process executor in place so iteration state stays consistent.
    """
    cur_len = _tools_len(c)
    if cur_len == b.last_len:
        return
    tool_dicts, tool_index = tools.build_tools(c.tools, web_search=c.web_search)
    b.tool_dicts = tool_dicts
    if b.local is not None:
        b.local._tools = tool_index
    b.last_len = cur_len


def _emit_assistant_event(
    c: _RelayConfig,
    content: str | None,
    tool_calls: list[dict[str, Any]] | None,
) -> None:
    if c.session is None:
        return
    c.session.emit(
        EventType.ASSISTANT_MESSAGE,
        {"content": content, "tool_calls": list(tool_calls) if tool_calls else None},
    )


def _emit_tool_call_event(
    c: _RelayConfig,
    *,
    id: str,
    name: str,
    arguments: dict[str, Any],
    iteration: int,
) -> None:
    if c.session is None:
        return
    c.session.emit(
        EventType.TOOL_CALL,
        {
            "id": id,
            "name": name,
            "arguments": dict(arguments),
            "iteration": iteration,
        },
    )


def _emit_tool_result_event(
    c: _RelayConfig,
    *,
    id: str,
    result: Any,
    result_serialized: str,
    error: str | None,
    duration_ms: float,
) -> None:
    if c.session is None:
        return
    c.session.emit(
        EventType.TOOL_RESULT,
        {
            "id": id,
            "result": result,
            "result_serialized": result_serialized,
            "error": error,
            "duration_ms": duration_ms,
        },
    )


def _emit_iteration_end_event(
    c: _RelayConfig,
    *,
    iteration: int,
    had_tool_calls: bool,
    finish_reason: str | None,
) -> None:
    if c.session is None:
        return
    c.session.emit(
        EventType.ITERATION_END,
        {
            "iteration": iteration,
            "had_tool_calls": had_tool_calls,
            "finish_reason": finish_reason,
        },
    )


def _read_back_intercept(
    event: InterceptEvent, c: _RelayConfig
) -> list[dict[str, Any]] | None:
    """If the intercept callback mutated ``messages``, return the new history.

    Also emits a ``context.modify`` event into the session when one is active.
    Returns ``None`` when nothing changed.
    """
    if not event.mutated:
        return None
    new_history = list(event.messages)
    if c.session is not None:
        c.session.emit(
            EventType.CONTEXT_MODIFY,
            {
                "by": "intercept",
                "summary": f"intercept mutated messages at iteration {event.iteration}",
                "before_len": len(event._initial_snapshot),
                "after_len": len(new_history),
            },
        )
    return new_history


def _fire_intercepts_sync(
    interceptors: Sequence[InterceptCallback], event: InterceptEvent
) -> bool:
    for cb in interceptors:
        try:
            result = cb(event)
            if inspect.iscoroutine(result):
                result.close()  # don't silently drop user awaitables
                raise TypeError(
                    "async interceptor passed to sync relay(); use relay_async() instead"
                )
        except TypeError:
            raise
        except Exception:  # noqa: BLE001 - intentional: keep loop running
            import logging

            logging.getLogger("encode").warning(
                "intercept callback raised; continuing", exc_info=True
            )
    return event.stopped


async def _fire_intercepts_async(
    interceptors: Sequence[AsyncInterceptCallback], event: InterceptEvent
) -> bool:
    for cb in interceptors:
        try:
            result = cb(event)
            if inspect.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger("encode").warning(
                "intercept callback raised; continuing", exc_info=True
            )
    return event.stopped


def _execute_sync(
    client: Any, c: _RelayConfig, interceptors: list[InterceptCallback]
) -> RelayResponse:
    bindings = _make_bindings(c)
    response_format_payload, parser_model = _build_response_format(
        c.response_format, c.endpoint
    )

    if c.endpoint == "chat":
        return _loop_chat_sync(
            client, c, interceptors, bindings, response_format_payload, parser_model
        )
    return _loop_responses_sync(
        client, c, interceptors, bindings, response_format_payload, parser_model
    )


def _loop_chat_sync(
    client: Any,
    c: _RelayConfig,
    interceptors: list[InterceptCallback],
    bindings: _ToolBindings,
    response_format_payload: Any,
    parser_model: type[BaseModel] | None,
) -> RelayResponse:
    history: list[dict[str, Any]] = _hydrate_history(c)
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    finish_reason: str | None = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        _refresh_tools(c, bindings)
        stream_this_iter = c.stream and not bindings.tool_dicts

        payload = _build_chat_payload(
            model=c.model,
            messages=history,
            tool_dicts=bindings.tool_dicts or None,
            response_format=response_format_payload,
            stream=stream_this_iter,
            extra=_common_extras(c),
        )
        resp_body = _post_chat(client, payload)
        last_raw = resp_body
        message, finish_reason = _extract_chat_assistant(resp_body)
        history.append(_clone_assistant(message))
        final_content = message.get("content")

        tcs = message.get("tool_calls") or []
        _emit_assistant_event(c, final_content, tcs or None)

        if not tcs:
            _emit_iteration_end_event(
                c, iteration=iteration, had_tool_calls=False, finish_reason=finish_reason
            )
            break

        records: list[ToolCallRecord] = []
        for tc in tcs:
            tc_id = tc.get("id") or ""
            fn_name = (tc.get("function") or {}).get("name") or ""
            args_raw = (tc.get("function") or {}).get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            _emit_tool_call_event(
                c, id=tc_id, name=fn_name, arguments=args, iteration=iteration
            )
            exec_result = bindings.executor.execute(fn_name, args)
            serialized = exec_result.result_serialized
            history.append(
                {"role": "tool", "tool_call_id": tc_id, "content": serialized}
            )
            _emit_tool_result_event(
                c,
                id=tc_id,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                duration_ms=exec_result.duration_ms,
            )
            rec = ToolCallRecord(
                id=tc_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                iteration=iteration,
                duration_ms=exec_result.duration_ms,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = c.max_tool_iterations is None or iterations < c.max_tool_iterations
        event = InterceptEvent(
            iteration=iteration,
            endpoint="chat",
            assistant_turn=AssistantTurn(content=message.get("content"), tool_calls=records),
            tool_calls=records,
            messages=Messages(history),
            raw_response=resp_body,
            will_continue=will_continue,
            _session=c.session,
        )
        stopped = _fire_intercepts_sync(interceptors, event)
        new_history = _read_back_intercept(event, c)
        if new_history is not None:
            history = new_history
        _emit_iteration_end_event(
            c, iteration=iteration, had_tool_calls=True, finish_reason=finish_reason
        )
        if stopped:
            break
        if c.max_tool_iterations is not None and iterations >= c.max_tool_iterations:
            partial = _build_response(
                "chat",
                c.model,
                final_content,
                history,
                all_tool_calls,
                iterations,
                finish_reason,
                last_raw,
                resp_body.get("usage"),
                None,
            )
            raise errors.MaxToolIterationsError(
                f"tool loop exceeded max_tool_iterations={c.max_tool_iterations}",
                partial=partial,
            )

    parsed = _maybe_parse(parser_model, final_content)
    return _build_response(
        "chat",
        c.model,
        final_content,
        history,
        all_tool_calls,
        iterations,
        finish_reason,
        last_raw,
        last_raw.get("usage") if isinstance(last_raw, dict) else None,
        parsed,
    )


def _loop_responses_sync(
    client: Any,
    c: _RelayConfig,
    interceptors: list[InterceptCallback],
    bindings: _ToolBindings,
    response_format_payload: Any,
    parser_model: type[BaseModel] | None,
) -> RelayResponse:
    if c.session is not None:
        history: list[dict[str, Any]] = _hydrate_history(c)
        input_items: list[dict[str, Any]] = list(
            to_responses_input(history, input=c.input)
        )
    else:
        input_items = list(to_responses_input(c.messages, input=c.input))
        history = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        _refresh_tools(c, bindings)
        stream_this_iter = c.stream and not bindings.tool_dicts

        payload = _build_responses_payload(
            model=c.model,
            input_items=input_items,
            tool_dicts=bindings.tool_dicts or None,
            response_format=response_format_payload,
            stream=stream_this_iter,
            extra=_common_extras(c),
        )
        resp_body = _post_responses(client, payload)
        last_raw = resp_body
        text, function_calls, output_items = _extract_responses_assistant(resp_body)
        final_content = text

        for item in output_items:
            input_items.append(item)
        if text is not None:
            history.append({"role": "assistant", "content": text})

        _emit_assistant_event(c, text, None)

        if not function_calls:
            _emit_iteration_end_event(
                c, iteration=iteration, had_tool_calls=False, finish_reason=None
            )
            break

        records: list[ToolCallRecord] = []
        for fc in function_calls:
            call_id = fc.get("call_id") or fc.get("id") or ""
            fn_name = fc.get("name") or ""
            args_raw = fc.get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            _emit_tool_call_event(
                c, id=call_id, name=fn_name, arguments=args, iteration=iteration
            )
            exec_result = bindings.executor.execute(fn_name, args)
            serialized = exec_result.result_serialized
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": serialized,
                }
            )
            history.append(
                {"role": "tool", "tool_call_id": call_id, "content": serialized}
            )
            _emit_tool_result_event(
                c,
                id=call_id,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                duration_ms=exec_result.duration_ms,
            )
            rec = ToolCallRecord(
                id=call_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                iteration=iteration,
                duration_ms=exec_result.duration_ms,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = c.max_tool_iterations is None or iterations < c.max_tool_iterations
        event = InterceptEvent(
            iteration=iteration,
            endpoint="responses",
            assistant_turn=AssistantTurn(content=text, tool_calls=records),
            tool_calls=records,
            messages=Messages(history),
            raw_response=resp_body,
            will_continue=will_continue,
            _session=c.session,
        )
        stopped = _fire_intercepts_sync(interceptors, event)
        new_history = _read_back_intercept(event, c)
        if new_history is not None:
            history = new_history
            input_items = list(to_responses_input(history))
        _emit_iteration_end_event(
            c, iteration=iteration, had_tool_calls=True, finish_reason=None
        )
        if stopped:
            break
        if c.max_tool_iterations is not None and iterations >= c.max_tool_iterations:
            partial = _build_response(
                "responses",
                c.model,
                final_content,
                history,
                all_tool_calls,
                iterations,
                None,
                last_raw,
                resp_body.get("usage"),
                None,
            )
            raise errors.MaxToolIterationsError(
                f"tool loop exceeded max_tool_iterations={c.max_tool_iterations}",
                partial=partial,
            )

    parsed = _maybe_parse(parser_model, final_content)
    return _build_response(
        "responses",
        c.model,
        final_content,
        history,
        all_tool_calls,
        iterations,
        None,
        last_raw,
        last_raw.get("usage") if isinstance(last_raw, dict) else None,
        parsed,
    )


# ---------------------------------------------------------------------------
# Async execution (mirrors sync)
# ---------------------------------------------------------------------------


async def _execute_async(
    client: Any, c: _RelayConfig, interceptors: list[AsyncInterceptCallback]
) -> RelayResponse:
    bindings = _make_bindings(c)
    response_format_payload, parser_model = _build_response_format(
        c.response_format, c.endpoint
    )

    if c.endpoint == "chat":
        return await _loop_chat_async(
            client, c, interceptors, bindings, response_format_payload, parser_model
        )
    return await _loop_responses_async(
        client, c, interceptors, bindings, response_format_payload, parser_model
    )


async def _loop_chat_async(
    client: Any,
    c: _RelayConfig,
    interceptors: list[AsyncInterceptCallback],
    bindings: _ToolBindings,
    response_format_payload: Any,
    parser_model: type[BaseModel] | None,
) -> RelayResponse:
    history: list[dict[str, Any]] = _hydrate_history(c)
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    finish_reason: str | None = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        _refresh_tools(c, bindings)
        stream_this_iter = c.stream and not bindings.tool_dicts
        payload = _build_chat_payload(
            model=c.model,
            messages=history,
            tool_dicts=bindings.tool_dicts or None,
            response_format=response_format_payload,
            stream=stream_this_iter,
            extra=_common_extras(c),
        )
        resp_body = await _post_chat_async(client, payload)
        last_raw = resp_body
        message, finish_reason = _extract_chat_assistant(resp_body)
        history.append(_clone_assistant(message))
        final_content = message.get("content")

        tcs = message.get("tool_calls") or []
        _emit_assistant_event(c, final_content, tcs or None)

        if not tcs:
            _emit_iteration_end_event(
                c, iteration=iteration, had_tool_calls=False, finish_reason=finish_reason
            )
            break

        records: list[ToolCallRecord] = []
        for tc in tcs:
            tc_id = tc.get("id") or ""
            fn_name = (tc.get("function") or {}).get("name") or ""
            args_raw = (tc.get("function") or {}).get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            _emit_tool_call_event(
                c, id=tc_id, name=fn_name, arguments=args, iteration=iteration
            )
            exec_result = await bindings.executor.execute_async(fn_name, args)
            serialized = exec_result.result_serialized
            history.append(
                {"role": "tool", "tool_call_id": tc_id, "content": serialized}
            )
            _emit_tool_result_event(
                c,
                id=tc_id,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                duration_ms=exec_result.duration_ms,
            )
            rec = ToolCallRecord(
                id=tc_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                iteration=iteration,
                duration_ms=exec_result.duration_ms,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = c.max_tool_iterations is None or iterations < c.max_tool_iterations
        event = InterceptEvent(
            iteration=iteration,
            endpoint="chat",
            assistant_turn=AssistantTurn(content=message.get("content"), tool_calls=records),
            tool_calls=records,
            messages=Messages(history),
            raw_response=resp_body,
            will_continue=will_continue,
            _session=c.session,
        )
        stopped = await _fire_intercepts_async(interceptors, event)
        new_history = _read_back_intercept(event, c)
        if new_history is not None:
            history = new_history
        _emit_iteration_end_event(
            c, iteration=iteration, had_tool_calls=True, finish_reason=finish_reason
        )
        if stopped:
            break
        if c.max_tool_iterations is not None and iterations >= c.max_tool_iterations:
            partial = _build_response(
                "chat",
                c.model,
                final_content,
                history,
                all_tool_calls,
                iterations,
                finish_reason,
                last_raw,
                resp_body.get("usage"),
                None,
            )
            raise errors.MaxToolIterationsError(
                f"tool loop exceeded max_tool_iterations={c.max_tool_iterations}",
                partial=partial,
            )

    parsed = _maybe_parse(parser_model, final_content)
    return _build_response(
        "chat",
        c.model,
        final_content,
        history,
        all_tool_calls,
        iterations,
        finish_reason,
        last_raw,
        last_raw.get("usage") if isinstance(last_raw, dict) else None,
        parsed,
    )


async def _loop_responses_async(
    client: Any,
    c: _RelayConfig,
    interceptors: list[AsyncInterceptCallback],
    bindings: _ToolBindings,
    response_format_payload: Any,
    parser_model: type[BaseModel] | None,
) -> RelayResponse:
    if c.session is not None:
        history: list[dict[str, Any]] = _hydrate_history(c)
        input_items: list[dict[str, Any]] = list(
            to_responses_input(history, input=c.input)
        )
    else:
        input_items = list(to_responses_input(c.messages, input=c.input))
        history = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        _refresh_tools(c, bindings)
        stream_this_iter = c.stream and not bindings.tool_dicts
        payload = _build_responses_payload(
            model=c.model,
            input_items=input_items,
            tool_dicts=bindings.tool_dicts or None,
            response_format=response_format_payload,
            stream=stream_this_iter,
            extra=_common_extras(c),
        )
        resp_body = await _post_responses_async(client, payload)
        last_raw = resp_body
        text, function_calls, output_items = _extract_responses_assistant(resp_body)
        final_content = text

        for item in output_items:
            input_items.append(item)
        if text is not None:
            history.append({"role": "assistant", "content": text})

        _emit_assistant_event(c, text, None)

        if not function_calls:
            _emit_iteration_end_event(
                c, iteration=iteration, had_tool_calls=False, finish_reason=None
            )
            break

        records: list[ToolCallRecord] = []
        for fc in function_calls:
            call_id = fc.get("call_id") or fc.get("id") or ""
            fn_name = fc.get("name") or ""
            args_raw = fc.get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            _emit_tool_call_event(
                c, id=call_id, name=fn_name, arguments=args, iteration=iteration
            )
            exec_result = await bindings.executor.execute_async(fn_name, args)
            serialized = exec_result.result_serialized
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": serialized,
                }
            )
            history.append(
                {"role": "tool", "tool_call_id": call_id, "content": serialized}
            )
            _emit_tool_result_event(
                c,
                id=call_id,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                duration_ms=exec_result.duration_ms,
            )
            rec = ToolCallRecord(
                id=call_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                iteration=iteration,
                duration_ms=exec_result.duration_ms,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = c.max_tool_iterations is None or iterations < c.max_tool_iterations
        event = InterceptEvent(
            iteration=iteration,
            endpoint="responses",
            assistant_turn=AssistantTurn(content=text, tool_calls=records),
            tool_calls=records,
            messages=Messages(history),
            raw_response=resp_body,
            will_continue=will_continue,
            _session=c.session,
        )
        stopped = await _fire_intercepts_async(interceptors, event)
        new_history = _read_back_intercept(event, c)
        if new_history is not None:
            history = new_history
            input_items = list(to_responses_input(history))
        _emit_iteration_end_event(
            c, iteration=iteration, had_tool_calls=True, finish_reason=None
        )
        if stopped:
            break
        if c.max_tool_iterations is not None and iterations >= c.max_tool_iterations:
            partial = _build_response(
                "responses",
                c.model,
                final_content,
                history,
                all_tool_calls,
                iterations,
                None,
                last_raw,
                resp_body.get("usage"),
                None,
            )
            raise errors.MaxToolIterationsError(
                f"tool loop exceeded max_tool_iterations={c.max_tool_iterations}",
                partial=partial,
            )

    parsed = _maybe_parse(parser_model, final_content)
    return _build_response(
        "responses",
        c.model,
        final_content,
        history,
        all_tool_calls,
        iterations,
        None,
        last_raw,
        last_raw.get("usage") if isinstance(last_raw, dict) else None,
        parsed,
    )


# ---------------------------------------------------------------------------
# Streaming entrypoints
#
# Each iteration opens an upstream SSE stream. Content tokens flow to the
# consumer immediately; tool_call deltas are buffered. When the upstream stream
# closes we either dispatch tools and loop, or yield a final ``finish`` event
# and stop.
# ---------------------------------------------------------------------------


def _execute_sync_stream(
    client: Any, c: _RelayConfig, interceptors: list[InterceptCallback]
) -> Iterator[StreamEvent]:
    bindings = _make_bindings(c)
    if c.endpoint == "chat":
        yield from _stream_chat_sync(client, c, interceptors, bindings)
    else:
        yield from _stream_responses_sync(client, c, interceptors, bindings)


def _stream_chat_sync(
    client: Any,
    c: _RelayConfig,
    interceptors: list[InterceptCallback],
    bindings: _ToolBindings,
) -> Iterator[StreamEvent]:
    history: list[dict[str, Any]] = _hydrate_history(c)
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    finish_reason: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        _refresh_tools(c, bindings)
        payload = _build_chat_payload(
            model=c.model,
            messages=history,
            tool_dicts=bindings.tool_dicts or None,
            response_format=None,
            stream=True,
            extra=_common_extras(c),
        )

        content_parts: list[str] = []
        tool_buf: dict[int, dict[str, Any]] = {}
        iter_finish: str | None = None
        last_chunk: Any = None

        with client._http.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            _http.raise_for_status(resp)
            for ev in iter_chat_completions(resp):
                last_chunk = ev.raw
                if ev.type == "content.delta":
                    content_parts.append(ev.data)
                    yield ev
                elif ev.type == "tool_calls.delta":
                    accumulate_chat_tool_calls(tool_buf, ev.data)
                    yield ev
                elif ev.type == "finish":
                    iter_finish = ev.data

        last_raw = last_chunk
        finish_reason = iter_finish
        content = "".join(content_parts) if content_parts else None
        final_content = content
        tcs = finalize_chat_tool_calls(tool_buf)

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tcs:
            assistant_msg["tool_calls"] = tcs
        history.append(assistant_msg)

        _emit_assistant_event(c, content, tcs or None)

        if not tcs:
            _emit_iteration_end_event(
                c, iteration=iteration, had_tool_calls=False, finish_reason=finish_reason
            )
            yield StreamEvent(type="finish", data=finish_reason, raw=last_chunk)
            break

        records: list[ToolCallRecord] = []
        for tc in tcs:
            tc_id = tc.get("id") or ""
            fn_name = (tc.get("function") or {}).get("name") or ""
            args_raw = (tc.get("function") or {}).get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            yield StreamEvent(
                type="tool_call.start",
                data={
                    "id": tc_id,
                    "name": fn_name,
                    "arguments": args,
                    "iteration": iteration,
                },
            )
            _emit_tool_call_event(
                c, id=tc_id, name=fn_name, arguments=args, iteration=iteration
            )
            exec_result = bindings.executor.execute(fn_name, args)
            serialized = exec_result.result_serialized
            if exec_result.error is None:
                yield StreamEvent(
                    type="tool_call.result",
                    data={
                        "id": tc_id,
                        "result": exec_result.result,
                        "result_serialized": serialized,
                        "duration_ms": exec_result.duration_ms,
                        "iteration": iteration,
                    },
                )
            else:
                yield StreamEvent(
                    type="tool_call.error",
                    data={"id": tc_id, "error": exec_result.error, "iteration": iteration},
                )
            history.append(
                {"role": "tool", "tool_call_id": tc_id, "content": serialized}
            )
            _emit_tool_result_event(
                c,
                id=tc_id,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                duration_ms=exec_result.duration_ms,
            )
            rec = ToolCallRecord(
                id=tc_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                iteration=iteration,
                duration_ms=exec_result.duration_ms,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = (
            c.max_tool_iterations is None or iterations < c.max_tool_iterations
        )
        intercept = InterceptEvent(
            iteration=iteration,
            endpoint="chat",
            assistant_turn=AssistantTurn(content=content, tool_calls=records),
            tool_calls=records,
            messages=Messages(history),
            raw_response=last_chunk if isinstance(last_chunk, dict) else {},
            will_continue=will_continue,
            _session=c.session,
        )
        stopped = _fire_intercepts_sync(interceptors, intercept)
        new_history = _read_back_intercept(intercept, c)
        if new_history is not None:
            history = new_history
        _emit_iteration_end_event(
            c, iteration=iteration, had_tool_calls=True, finish_reason=finish_reason
        )
        yield StreamEvent(
            type="iteration.end",
            data={"iteration": iteration, "had_tool_calls": True},
        )
        if stopped:
            yield StreamEvent(type="finish", data=finish_reason, raw=last_chunk)
            break
        if c.max_tool_iterations is not None and iterations >= c.max_tool_iterations:
            partial = _build_response(
                "chat",
                c.model,
                final_content,
                history,
                all_tool_calls,
                iterations,
                finish_reason,
                last_raw,
                None,
                None,
            )
            raise errors.MaxToolIterationsError(
                f"tool loop exceeded max_tool_iterations={c.max_tool_iterations}",
                partial=partial,
            )

    final_resp = _build_response(
        "chat",
        c.model,
        final_content,
        history,
        all_tool_calls,
        iterations,
        finish_reason,
        last_raw,
        None,
        None,
    )
    _absorb_into_messages(c.messages, final_resp)


def _stream_responses_sync(
    client: Any,
    c: _RelayConfig,
    interceptors: list[InterceptCallback],
    bindings: _ToolBindings,
) -> Iterator[StreamEvent]:
    if c.session is not None:
        history: list[dict[str, Any]] = _hydrate_history(c)
        input_items: list[dict[str, Any]] = list(
            to_responses_input(history, input=c.input)
        )
    else:
        input_items = list(to_responses_input(c.messages, input=c.input))
        history = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        _refresh_tools(c, bindings)
        payload = _build_responses_payload(
            model=c.model,
            input_items=input_items,
            tool_dicts=bindings.tool_dicts or None,
            response_format=None,
            stream=True,
            extra=_common_extras(c),
        )

        completed_response: Any = None
        with client._http.stream(
            "POST", "/v1/responses", json=payload
        ) as resp:
            _http.raise_for_status(resp)
            for ev in iter_responses(resp):
                if ev.type == "response.output_text.delta":
                    delta = (ev.data or {}).get("delta", "")
                    if delta:
                        yield StreamEvent(
                            type="content.delta", data=delta, raw=ev.raw
                        )
                if ev.type == "response.completed":
                    completed_response = (ev.data or {}).get("response")
                yield ev

        last_raw = completed_response or last_raw
        text, function_calls, output_items = _extract_responses_assistant(
            completed_response or {}
        )
        final_content = text

        for item in output_items:
            input_items.append(item)
        if text is not None:
            history.append({"role": "assistant", "content": text})

        _emit_assistant_event(c, text, None)

        if not function_calls:
            _emit_iteration_end_event(
                c, iteration=iteration, had_tool_calls=False, finish_reason=None
            )
            yield StreamEvent(type="finish", data="completed", raw=completed_response)
            break

        records: list[ToolCallRecord] = []
        for fc in function_calls:
            call_id = fc.get("call_id") or fc.get("id") or ""
            fn_name = fc.get("name") or ""
            args_raw = fc.get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            yield StreamEvent(
                type="tool_call.start",
                data={
                    "id": call_id,
                    "name": fn_name,
                    "arguments": args,
                    "iteration": iteration,
                },
            )
            _emit_tool_call_event(
                c, id=call_id, name=fn_name, arguments=args, iteration=iteration
            )
            exec_result = bindings.executor.execute(fn_name, args)
            serialized = exec_result.result_serialized
            if exec_result.error is None:
                yield StreamEvent(
                    type="tool_call.result",
                    data={
                        "id": call_id,
                        "result": exec_result.result,
                        "result_serialized": serialized,
                        "duration_ms": exec_result.duration_ms,
                        "iteration": iteration,
                    },
                )
            else:
                yield StreamEvent(
                    type="tool_call.error",
                    data={"id": call_id, "error": exec_result.error, "iteration": iteration},
                )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": serialized,
                }
            )
            history.append(
                {"role": "tool", "tool_call_id": call_id, "content": serialized}
            )
            _emit_tool_result_event(
                c,
                id=call_id,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                duration_ms=exec_result.duration_ms,
            )
            rec = ToolCallRecord(
                id=call_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                iteration=iteration,
                duration_ms=exec_result.duration_ms,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = (
            c.max_tool_iterations is None or iterations < c.max_tool_iterations
        )
        intercept = InterceptEvent(
            iteration=iteration,
            endpoint="responses",
            assistant_turn=AssistantTurn(content=text, tool_calls=records),
            tool_calls=records,
            messages=Messages(history),
            raw_response=completed_response if isinstance(completed_response, dict) else {},
            will_continue=will_continue,
            _session=c.session,
        )
        stopped = _fire_intercepts_sync(interceptors, intercept)
        new_history = _read_back_intercept(intercept, c)
        if new_history is not None:
            history = new_history
            input_items = list(to_responses_input(history))
        _emit_iteration_end_event(
            c, iteration=iteration, had_tool_calls=True, finish_reason=None
        )
        yield StreamEvent(
            type="iteration.end",
            data={"iteration": iteration, "had_tool_calls": True},
        )
        if stopped:
            yield StreamEvent(type="finish", data="stopped", raw=completed_response)
            break
        if c.max_tool_iterations is not None and iterations >= c.max_tool_iterations:
            partial = _build_response(
                "responses",
                c.model,
                final_content,
                history,
                all_tool_calls,
                iterations,
                None,
                last_raw,
                None,
                None,
            )
            raise errors.MaxToolIterationsError(
                f"tool loop exceeded max_tool_iterations={c.max_tool_iterations}",
                partial=partial,
            )

    final_resp = _build_response(
        "responses",
        c.model,
        final_content,
        history,
        all_tool_calls,
        iterations,
        None,
        last_raw,
        None,
        None,
    )
    _absorb_into_messages(c.messages, final_resp)


async def _execute_async_stream(
    client: Any, c: _RelayConfig, interceptors: list[AsyncInterceptCallback]
) -> AsyncIterator[StreamEvent]:
    bindings = _make_bindings(c)
    if c.endpoint == "chat":
        async for ev in _stream_chat_async(client, c, interceptors, bindings):
            yield ev
    else:
        async for ev in _stream_responses_async(client, c, interceptors, bindings):
            yield ev


async def _stream_chat_async(
    client: Any,
    c: _RelayConfig,
    interceptors: list[AsyncInterceptCallback],
    bindings: _ToolBindings,
) -> AsyncIterator[StreamEvent]:
    history: list[dict[str, Any]] = _hydrate_history(c)
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    finish_reason: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        _refresh_tools(c, bindings)
        payload = _build_chat_payload(
            model=c.model,
            messages=history,
            tool_dicts=bindings.tool_dicts or None,
            response_format=None,
            stream=True,
            extra=_common_extras(c),
        )

        content_parts: list[str] = []
        tool_buf: dict[int, dict[str, Any]] = {}
        iter_finish: str | None = None
        last_chunk: Any = None

        async with client._http.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            await _http.araise_for_status(resp)
            async for ev in aiter_chat_completions(resp):
                last_chunk = ev.raw
                if ev.type == "content.delta":
                    content_parts.append(ev.data)
                    yield ev
                elif ev.type == "tool_calls.delta":
                    accumulate_chat_tool_calls(tool_buf, ev.data)
                    yield ev
                elif ev.type == "finish":
                    iter_finish = ev.data

        last_raw = last_chunk
        finish_reason = iter_finish
        content = "".join(content_parts) if content_parts else None
        final_content = content
        tcs = finalize_chat_tool_calls(tool_buf)

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tcs:
            assistant_msg["tool_calls"] = tcs
        history.append(assistant_msg)

        _emit_assistant_event(c, content, tcs or None)

        if not tcs:
            _emit_iteration_end_event(
                c, iteration=iteration, had_tool_calls=False, finish_reason=finish_reason
            )
            yield StreamEvent(type="finish", data=finish_reason, raw=last_chunk)
            break

        records: list[ToolCallRecord] = []
        for tc in tcs:
            tc_id = tc.get("id") or ""
            fn_name = (tc.get("function") or {}).get("name") or ""
            args_raw = (tc.get("function") or {}).get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            yield StreamEvent(
                type="tool_call.start",
                data={
                    "id": tc_id,
                    "name": fn_name,
                    "arguments": args,
                    "iteration": iteration,
                },
            )
            _emit_tool_call_event(
                c, id=tc_id, name=fn_name, arguments=args, iteration=iteration
            )
            exec_result = await bindings.executor.execute_async(fn_name, args)
            serialized = exec_result.result_serialized
            if exec_result.error is None:
                yield StreamEvent(
                    type="tool_call.result",
                    data={
                        "id": tc_id,
                        "result": exec_result.result,
                        "result_serialized": serialized,
                        "duration_ms": exec_result.duration_ms,
                        "iteration": iteration,
                    },
                )
            else:
                yield StreamEvent(
                    type="tool_call.error",
                    data={"id": tc_id, "error": exec_result.error, "iteration": iteration},
                )
            history.append(
                {"role": "tool", "tool_call_id": tc_id, "content": serialized}
            )
            _emit_tool_result_event(
                c,
                id=tc_id,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                duration_ms=exec_result.duration_ms,
            )
            rec = ToolCallRecord(
                id=tc_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                iteration=iteration,
                duration_ms=exec_result.duration_ms,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = (
            c.max_tool_iterations is None or iterations < c.max_tool_iterations
        )
        intercept = InterceptEvent(
            iteration=iteration,
            endpoint="chat",
            assistant_turn=AssistantTurn(content=content, tool_calls=records),
            tool_calls=records,
            messages=Messages(history),
            raw_response=last_chunk if isinstance(last_chunk, dict) else {},
            will_continue=will_continue,
            _session=c.session,
        )
        stopped = await _fire_intercepts_async(interceptors, intercept)
        new_history = _read_back_intercept(intercept, c)
        if new_history is not None:
            history = new_history
        _emit_iteration_end_event(
            c, iteration=iteration, had_tool_calls=True, finish_reason=finish_reason
        )
        yield StreamEvent(
            type="iteration.end",
            data={"iteration": iteration, "had_tool_calls": True},
        )
        if stopped:
            yield StreamEvent(type="finish", data=finish_reason, raw=last_chunk)
            break
        if c.max_tool_iterations is not None and iterations >= c.max_tool_iterations:
            partial = _build_response(
                "chat",
                c.model,
                final_content,
                history,
                all_tool_calls,
                iterations,
                finish_reason,
                last_raw,
                None,
                None,
            )
            raise errors.MaxToolIterationsError(
                f"tool loop exceeded max_tool_iterations={c.max_tool_iterations}",
                partial=partial,
            )

    final_resp = _build_response(
        "chat",
        c.model,
        final_content,
        history,
        all_tool_calls,
        iterations,
        finish_reason,
        last_raw,
        None,
        None,
    )
    _absorb_into_messages(c.messages, final_resp)


async def _stream_responses_async(
    client: Any,
    c: _RelayConfig,
    interceptors: list[AsyncInterceptCallback],
    bindings: _ToolBindings,
) -> AsyncIterator[StreamEvent]:
    if c.session is not None:
        history: list[dict[str, Any]] = _hydrate_history(c)
        input_items: list[dict[str, Any]] = list(
            to_responses_input(history, input=c.input)
        )
    else:
        input_items = list(to_responses_input(c.messages, input=c.input))
        history = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        _refresh_tools(c, bindings)
        payload = _build_responses_payload(
            model=c.model,
            input_items=input_items,
            tool_dicts=bindings.tool_dicts or None,
            response_format=None,
            stream=True,
            extra=_common_extras(c),
        )

        completed_response: Any = None
        async with client._http.stream(
            "POST", "/v1/responses", json=payload
        ) as resp:
            await _http.araise_for_status(resp)
            async for ev in aiter_responses(resp):
                if ev.type == "response.output_text.delta":
                    delta = (ev.data or {}).get("delta", "")
                    if delta:
                        yield StreamEvent(
                            type="content.delta", data=delta, raw=ev.raw
                        )
                if ev.type == "response.completed":
                    completed_response = (ev.data or {}).get("response")
                yield ev

        last_raw = completed_response or last_raw
        text, function_calls, output_items = _extract_responses_assistant(
            completed_response or {}
        )
        final_content = text

        for item in output_items:
            input_items.append(item)
        if text is not None:
            history.append({"role": "assistant", "content": text})

        _emit_assistant_event(c, text, None)

        if not function_calls:
            _emit_iteration_end_event(
                c, iteration=iteration, had_tool_calls=False, finish_reason=None
            )
            yield StreamEvent(type="finish", data="completed", raw=completed_response)
            break

        records: list[ToolCallRecord] = []
        for fc in function_calls:
            call_id = fc.get("call_id") or fc.get("id") or ""
            fn_name = fc.get("name") or ""
            args_raw = fc.get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            yield StreamEvent(
                type="tool_call.start",
                data={
                    "id": call_id,
                    "name": fn_name,
                    "arguments": args,
                    "iteration": iteration,
                },
            )
            _emit_tool_call_event(
                c, id=call_id, name=fn_name, arguments=args, iteration=iteration
            )
            exec_result = await bindings.executor.execute_async(fn_name, args)
            serialized = exec_result.result_serialized
            if exec_result.error is None:
                yield StreamEvent(
                    type="tool_call.result",
                    data={
                        "id": call_id,
                        "result": exec_result.result,
                        "result_serialized": serialized,
                        "duration_ms": exec_result.duration_ms,
                        "iteration": iteration,
                    },
                )
            else:
                yield StreamEvent(
                    type="tool_call.error",
                    data={"id": call_id, "error": exec_result.error, "iteration": iteration},
                )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": serialized,
                }
            )
            history.append(
                {"role": "tool", "tool_call_id": call_id, "content": serialized}
            )
            _emit_tool_result_event(
                c,
                id=call_id,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                duration_ms=exec_result.duration_ms,
            )
            rec = ToolCallRecord(
                id=call_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=exec_result.result if exec_result.error is None else None,
                result_serialized=serialized,
                error=exec_result.error,
                iteration=iteration,
                duration_ms=exec_result.duration_ms,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = (
            c.max_tool_iterations is None or iterations < c.max_tool_iterations
        )
        intercept = InterceptEvent(
            iteration=iteration,
            endpoint="responses",
            assistant_turn=AssistantTurn(content=text, tool_calls=records),
            tool_calls=records,
            messages=Messages(history),
            raw_response=completed_response if isinstance(completed_response, dict) else {},
            will_continue=will_continue,
            _session=c.session,
        )
        stopped = await _fire_intercepts_async(interceptors, intercept)
        new_history = _read_back_intercept(intercept, c)
        if new_history is not None:
            history = new_history
            input_items = list(to_responses_input(history))
        _emit_iteration_end_event(
            c, iteration=iteration, had_tool_calls=True, finish_reason=None
        )
        yield StreamEvent(
            type="iteration.end",
            data={"iteration": iteration, "had_tool_calls": True},
        )
        if stopped:
            yield StreamEvent(type="finish", data="stopped", raw=completed_response)
            break
        if c.max_tool_iterations is not None and iterations >= c.max_tool_iterations:
            partial = _build_response(
                "responses",
                c.model,
                final_content,
                history,
                all_tool_calls,
                iterations,
                None,
                last_raw,
                None,
                None,
            )
            raise errors.MaxToolIterationsError(
                f"tool loop exceeded max_tool_iterations={c.max_tool_iterations}",
                partial=partial,
            )

    final_resp = _build_response(
        "responses",
        c.model,
        final_content,
        history,
        all_tool_calls,
        iterations,
        None,
        last_raw,
        None,
        None,
    )
    _absorb_into_messages(c.messages, final_resp)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post_chat(client: Any, payload: dict[str, Any]) -> dict[str, Any]:
    resp = _http.request_sync(
        client._http,
        "POST",
        "/v1/chat/completions",
        json=payload,
        max_retries=client.max_retries,
    )
    return resp.json()


async def _post_chat_async(client: Any, payload: dict[str, Any]) -> dict[str, Any]:
    resp = await _http.request_async(
        client._http,
        "POST",
        "/v1/chat/completions",
        json=payload,
        max_retries=client.max_retries,
    )
    return resp.json()


def _post_responses(client: Any, payload: dict[str, Any]) -> dict[str, Any]:
    resp = _http.request_sync(
        client._http,
        "POST",
        "/v1/responses",
        json=payload,
        max_retries=client.max_retries,
    )
    return resp.json()


async def _post_responses_async(client: Any, payload: dict[str, Any]) -> dict[str, Any]:
    resp = await _http.request_async(
        client._http,
        "POST",
        "/v1/responses",
        json=payload,
        max_retries=client.max_retries,
    )
    return resp.json()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _has_pending_tool_calls(history: list[dict[str, Any]]) -> bool:
    if not history:
        return False
    last = history[-1]
    return bool(last.get("tool_calls"))


def _clone_assistant(message: dict[str, Any]) -> dict[str, Any]:
    """Deep-clone an assistant message for safe append-into-history."""
    return json.loads(json.dumps(message))


def _absorb_into_messages(messages: Any, response: RelayResponse | None) -> None:
    """If `messages` is a Messages instance, replace its contents with the response history."""
    if response is None:
        return
    if isinstance(messages, Messages):
        messages._items = list(response.messages)


def _maybe_parse(model: type[BaseModel] | None, content: str | None) -> Any:
    if model is None or not content:
        return None
    try:
        return model.model_validate_json(content)
    except Exception:
        return None


def _build_response(
    endpoint: Literal["chat", "responses"],
    model: str,
    content: str | None,
    history: list[dict[str, Any]],
    tool_calls: list[ToolCallRecord],
    iterations: int,
    finish_reason: str | None,
    raw: Any,
    usage: Any,
    parsed: Any,
) -> RelayResponse:
    return RelayResponse(
        content=content,
        parsed=parsed,
        messages=history,
        tool_calls=tool_calls,
        iterations=iterations,
        finish_reason=finish_reason,
        endpoint=endpoint,
        model=model,
        raw=raw,
        usage=Usage(**usage) if isinstance(usage, dict) else None,
    )


__all__ = [
    "relay",
    "relay_async",
    "RelayHandle",
    "AsyncRelayHandle",
    "InterceptEvent",
    "InterceptCallback",
    "AsyncInterceptCallback",
]


