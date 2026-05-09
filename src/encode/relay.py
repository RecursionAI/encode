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
from .messages import Messages, to_chat_messages, to_responses_input
from .responses import AssistantTurn, RelayResponse, ToolCallRecord, Usage

InterceptCallback = Callable[["InterceptEvent"], Any]
AsyncInterceptCallback = Callable[["InterceptEvent"], Any]


@dataclass
class InterceptEvent:
    iteration: int
    endpoint: Literal["chat", "responses"]
    assistant_turn: AssistantTurn
    tool_calls: list[ToolCallRecord]
    messages_so_far: list[dict[str, Any]]
    raw_response: dict[str, Any]
    will_continue: bool
    _stopped: bool = field(default=False, repr=False)

    def stop(self) -> None:
        """Mark the loop for termination after the current iteration completes."""
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped


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
    client: Any = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = 60.0,
) -> RelayHandle:
    """Wrap /v1/chat/completions and /v1/responses with auto tool-call loops.

    See README for usage. Returns a :class:`RelayHandle`; access ``.response``
    to execute and get the typed :class:`RelayResponse`.
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
    tool_dicts, tool_index = tools.build_tools(c.tools, web_search=c.web_search)
    response_format_payload, parser_model = _build_response_format(
        c.response_format, c.endpoint
    )

    if c.endpoint == "chat":
        return _loop_chat_sync(
            client, c, interceptors, tool_dicts, tool_index, response_format_payload, parser_model
        )
    return _loop_responses_sync(
        client, c, interceptors, tool_dicts, tool_index, response_format_payload, parser_model
    )


def _loop_chat_sync(
    client: Any,
    c: _RelayConfig,
    interceptors: list[InterceptCallback],
    tool_dicts: list[dict[str, Any]],
    tool_index: dict[str, Callable[..., Any]],
    response_format_payload: Any,
    parser_model: type[BaseModel] | None,
) -> RelayResponse:
    history: list[dict[str, Any]] = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    finish_reason: str | None = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        stream_this_iter = c.stream and not tool_dicts

        payload = _build_chat_payload(
            model=c.model,
            messages=history,
            tool_dicts=tool_dicts or None,
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
        if not tcs:
            break

        records: list[ToolCallRecord] = []
        for tc in tcs:
            tc_id = tc.get("id") or ""
            fn_name = (tc.get("function") or {}).get("name") or ""
            args_raw = (tc.get("function") or {}).get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            fn = tool_index.get(fn_name)
            err: str | None
            if fn is None:
                result_obj: Any = {"error": f"no Python callable bound for tool '{fn_name}'"}
                err = result_obj["error"]
                duration = 0.0
            else:
                result_obj, err, duration = tools.safe_call(fn, args)
            serialized = tools.serialize_tool_result(result_obj)
            history.append(
                {"role": "tool", "tool_call_id": tc_id, "content": serialized}
            )
            rec = ToolCallRecord(
                id=tc_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=result_obj if err is None else None,
                result_serialized=serialized,
                error=err,
                iteration=iteration,
                duration_ms=duration,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = c.max_tool_iterations is None or iterations < c.max_tool_iterations
        event = InterceptEvent(
            iteration=iteration,
            endpoint="chat",
            assistant_turn=AssistantTurn(content=message.get("content"), tool_calls=records),
            tool_calls=records,
            messages_so_far=list(history),
            raw_response=resp_body,
            will_continue=will_continue,
        )
        if _fire_intercepts_sync(interceptors, event):
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
    tool_dicts: list[dict[str, Any]],
    tool_index: dict[str, Callable[..., Any]],
    response_format_payload: Any,
    parser_model: type[BaseModel] | None,
) -> RelayResponse:
    input_items: list[dict[str, Any]] = list(
        to_responses_input(c.messages, input=c.input)
    )
    history: list[dict[str, Any]] = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        stream_this_iter = c.stream and not tool_dicts

        payload = _build_responses_payload(
            model=c.model,
            input_items=input_items,
            tool_dicts=tool_dicts or None,
            response_format=response_format_payload,
            stream=stream_this_iter,
            extra=_common_extras(c),
        )
        resp_body = _post_responses(client, payload)
        last_raw = resp_body
        text, function_calls, output_items = _extract_responses_assistant(resp_body)
        final_content = text

        # echo assistant message + function_call items into input_items
        for item in output_items:
            input_items.append(item)
        if text is not None:
            history.append({"role": "assistant", "content": text})

        if not function_calls:
            break

        records: list[ToolCallRecord] = []
        for fc in function_calls:
            call_id = fc.get("call_id") or fc.get("id") or ""
            fn_name = fc.get("name") or ""
            args_raw = fc.get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            fn = tool_index.get(fn_name)
            err: str | None
            if fn is None:
                result_obj: Any = {"error": f"no Python callable bound for tool '{fn_name}'"}
                err = result_obj["error"]
                duration = 0.0
            else:
                result_obj, err, duration = tools.safe_call(fn, args)
            serialized = tools.serialize_tool_result(result_obj)
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
            rec = ToolCallRecord(
                id=call_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=result_obj if err is None else None,
                result_serialized=serialized,
                error=err,
                iteration=iteration,
                duration_ms=duration,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = c.max_tool_iterations is None or iterations < c.max_tool_iterations
        event = InterceptEvent(
            iteration=iteration,
            endpoint="responses",
            assistant_turn=AssistantTurn(content=text, tool_calls=records),
            tool_calls=records,
            messages_so_far=list(history),
            raw_response=resp_body,
            will_continue=will_continue,
        )
        if _fire_intercepts_sync(interceptors, event):
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
    tool_dicts, tool_index = tools.build_tools(c.tools, web_search=c.web_search)
    response_format_payload, parser_model = _build_response_format(
        c.response_format, c.endpoint
    )

    if c.endpoint == "chat":
        return await _loop_chat_async(
            client, c, interceptors, tool_dicts, tool_index, response_format_payload, parser_model
        )
    return await _loop_responses_async(
        client, c, interceptors, tool_dicts, tool_index, response_format_payload, parser_model
    )


async def _loop_chat_async(
    client: Any,
    c: _RelayConfig,
    interceptors: list[AsyncInterceptCallback],
    tool_dicts: list[dict[str, Any]],
    tool_index: dict[str, Callable[..., Any]],
    response_format_payload: Any,
    parser_model: type[BaseModel] | None,
) -> RelayResponse:
    history: list[dict[str, Any]] = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    finish_reason: str | None = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        stream_this_iter = c.stream and not tool_dicts
        payload = _build_chat_payload(
            model=c.model,
            messages=history,
            tool_dicts=tool_dicts or None,
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
        if not tcs:
            break

        records: list[ToolCallRecord] = []
        for tc in tcs:
            tc_id = tc.get("id") or ""
            fn_name = (tc.get("function") or {}).get("name") or ""
            args_raw = (tc.get("function") or {}).get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            fn = tool_index.get(fn_name)
            err: str | None
            if fn is None:
                result_obj: Any = {"error": f"no Python callable bound for tool '{fn_name}'"}
                err = result_obj["error"]
                duration = 0.0
            else:
                result_obj, err, duration = await tools.safe_call_async(fn, args)
            serialized = tools.serialize_tool_result(result_obj)
            history.append(
                {"role": "tool", "tool_call_id": tc_id, "content": serialized}
            )
            rec = ToolCallRecord(
                id=tc_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=result_obj if err is None else None,
                result_serialized=serialized,
                error=err,
                iteration=iteration,
                duration_ms=duration,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = c.max_tool_iterations is None or iterations < c.max_tool_iterations
        event = InterceptEvent(
            iteration=iteration,
            endpoint="chat",
            assistant_turn=AssistantTurn(content=message.get("content"), tool_calls=records),
            tool_calls=records,
            messages_so_far=list(history),
            raw_response=resp_body,
            will_continue=will_continue,
        )
        if await _fire_intercepts_async(interceptors, event):
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
    tool_dicts: list[dict[str, Any]],
    tool_index: dict[str, Callable[..., Any]],
    response_format_payload: Any,
    parser_model: type[BaseModel] | None,
) -> RelayResponse:
    input_items: list[dict[str, Any]] = list(
        to_responses_input(c.messages, input=c.input)
    )
    history: list[dict[str, Any]] = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        stream_this_iter = c.stream and not tool_dicts
        payload = _build_responses_payload(
            model=c.model,
            input_items=input_items,
            tool_dicts=tool_dicts or None,
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

        if not function_calls:
            break

        records: list[ToolCallRecord] = []
        for fc in function_calls:
            call_id = fc.get("call_id") or fc.get("id") or ""
            fn_name = fc.get("name") or ""
            args_raw = fc.get("arguments") or "{}"
            args = tools.parse_arguments(args_raw)
            fn = tool_index.get(fn_name)
            err: str | None
            if fn is None:
                result_obj: Any = {"error": f"no Python callable bound for tool '{fn_name}'"}
                err = result_obj["error"]
                duration = 0.0
            else:
                result_obj, err, duration = await tools.safe_call_async(fn, args)
            serialized = tools.serialize_tool_result(result_obj)
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
            rec = ToolCallRecord(
                id=call_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=result_obj if err is None else None,
                result_serialized=serialized,
                error=err,
                iteration=iteration,
                duration_ms=duration,
            )
            records.append(rec)
            all_tool_calls.append(rec)

        will_continue = c.max_tool_iterations is None or iterations < c.max_tool_iterations
        event = InterceptEvent(
            iteration=iteration,
            endpoint="responses",
            assistant_turn=AssistantTurn(content=text, tool_calls=records),
            tool_calls=records,
            messages_so_far=list(history),
            raw_response=resp_body,
            will_continue=will_continue,
        )
        if await _fire_intercepts_async(interceptors, event):
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
    tool_dicts, tool_index = tools.build_tools(c.tools, web_search=c.web_search)
    if c.endpoint == "chat":
        yield from _stream_chat_sync(client, c, interceptors, tool_dicts, tool_index)
    else:
        yield from _stream_responses_sync(client, c, interceptors, tool_dicts, tool_index)


def _stream_chat_sync(
    client: Any,
    c: _RelayConfig,
    interceptors: list[InterceptCallback],
    tool_dicts: list[dict[str, Any]],
    tool_index: dict[str, Callable[..., Any]],
) -> Iterator[StreamEvent]:
    history: list[dict[str, Any]] = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    finish_reason: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        payload = _build_chat_payload(
            model=c.model,
            messages=history,
            tool_dicts=tool_dicts or None,
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

        if not tcs:
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
            fn = tool_index.get(fn_name)
            err: str | None
            if fn is None:
                err = f"no Python callable bound for tool '{fn_name}'"
                result_obj: Any = {"error": err}
                duration = 0.0
                yield StreamEvent(
                    type="tool_call.error",
                    data={"id": tc_id, "error": err, "iteration": iteration},
                )
            else:
                result_obj, err, duration = tools.safe_call(fn, args)
                if err is None:
                    yield StreamEvent(
                        type="tool_call.result",
                        data={
                            "id": tc_id,
                            "result": result_obj,
                            "result_serialized": tools.serialize_tool_result(
                                result_obj
                            ),
                            "duration_ms": duration,
                            "iteration": iteration,
                        },
                    )
                else:
                    yield StreamEvent(
                        type="tool_call.error",
                        data={"id": tc_id, "error": err, "iteration": iteration},
                    )
            serialized = tools.serialize_tool_result(result_obj)
            history.append(
                {"role": "tool", "tool_call_id": tc_id, "content": serialized}
            )
            rec = ToolCallRecord(
                id=tc_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=result_obj if err is None else None,
                result_serialized=serialized,
                error=err,
                iteration=iteration,
                duration_ms=duration,
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
            messages_so_far=list(history),
            raw_response=last_chunk if isinstance(last_chunk, dict) else {},
            will_continue=will_continue,
        )
        stopped = _fire_intercepts_sync(interceptors, intercept)
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
    tool_dicts: list[dict[str, Any]],
    tool_index: dict[str, Callable[..., Any]],
) -> Iterator[StreamEvent]:
    input_items: list[dict[str, Any]] = list(
        to_responses_input(c.messages, input=c.input)
    )
    history: list[dict[str, Any]] = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        payload = _build_responses_payload(
            model=c.model,
            input_items=input_items,
            tool_dicts=tool_dicts or None,
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

        if not function_calls:
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
            fn = tool_index.get(fn_name)
            if fn is None:
                err: str | None = f"no Python callable bound for tool '{fn_name}'"
                result_obj: Any = {"error": err}
                duration = 0.0
                yield StreamEvent(
                    type="tool_call.error",
                    data={"id": call_id, "error": err, "iteration": iteration},
                )
            else:
                result_obj, err, duration = tools.safe_call(fn, args)
                if err is None:
                    yield StreamEvent(
                        type="tool_call.result",
                        data={
                            "id": call_id,
                            "result": result_obj,
                            "result_serialized": tools.serialize_tool_result(
                                result_obj
                            ),
                            "duration_ms": duration,
                            "iteration": iteration,
                        },
                    )
                else:
                    yield StreamEvent(
                        type="tool_call.error",
                        data={"id": call_id, "error": err, "iteration": iteration},
                    )
            serialized = tools.serialize_tool_result(result_obj)
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
            rec = ToolCallRecord(
                id=call_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=result_obj if err is None else None,
                result_serialized=serialized,
                error=err,
                iteration=iteration,
                duration_ms=duration,
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
            messages_so_far=list(history),
            raw_response=completed_response if isinstance(completed_response, dict) else {},
            will_continue=will_continue,
        )
        stopped = _fire_intercepts_sync(interceptors, intercept)
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
    tool_dicts, tool_index = tools.build_tools(c.tools, web_search=c.web_search)
    if c.endpoint == "chat":
        async for ev in _stream_chat_async(
            client, c, interceptors, tool_dicts, tool_index
        ):
            yield ev
    else:
        async for ev in _stream_responses_async(
            client, c, interceptors, tool_dicts, tool_index
        ):
            yield ev


async def _stream_chat_async(
    client: Any,
    c: _RelayConfig,
    interceptors: list[AsyncInterceptCallback],
    tool_dicts: list[dict[str, Any]],
    tool_index: dict[str, Callable[..., Any]],
) -> AsyncIterator[StreamEvent]:
    history: list[dict[str, Any]] = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    finish_reason: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        payload = _build_chat_payload(
            model=c.model,
            messages=history,
            tool_dicts=tool_dicts or None,
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
            _http.raise_for_status(resp)
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

        if not tcs:
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
            fn = tool_index.get(fn_name)
            err: str | None
            if fn is None:
                err = f"no Python callable bound for tool '{fn_name}'"
                result_obj: Any = {"error": err}
                duration = 0.0
                yield StreamEvent(
                    type="tool_call.error",
                    data={"id": tc_id, "error": err, "iteration": iteration},
                )
            else:
                result_obj, err, duration = await tools.safe_call_async(fn, args)
                if err is None:
                    yield StreamEvent(
                        type="tool_call.result",
                        data={
                            "id": tc_id,
                            "result": result_obj,
                            "result_serialized": tools.serialize_tool_result(
                                result_obj
                            ),
                            "duration_ms": duration,
                            "iteration": iteration,
                        },
                    )
                else:
                    yield StreamEvent(
                        type="tool_call.error",
                        data={"id": tc_id, "error": err, "iteration": iteration},
                    )
            serialized = tools.serialize_tool_result(result_obj)
            history.append(
                {"role": "tool", "tool_call_id": tc_id, "content": serialized}
            )
            rec = ToolCallRecord(
                id=tc_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=result_obj if err is None else None,
                result_serialized=serialized,
                error=err,
                iteration=iteration,
                duration_ms=duration,
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
            messages_so_far=list(history),
            raw_response=last_chunk if isinstance(last_chunk, dict) else {},
            will_continue=will_continue,
        )
        stopped = await _fire_intercepts_async(interceptors, intercept)
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
    tool_dicts: list[dict[str, Any]],
    tool_index: dict[str, Callable[..., Any]],
) -> AsyncIterator[StreamEvent]:
    input_items: list[dict[str, Any]] = list(
        to_responses_input(c.messages, input=c.input)
    )
    history: list[dict[str, Any]] = list(to_chat_messages(c.messages))
    all_tool_calls: list[ToolCallRecord] = []
    last_raw: Any = None
    final_content: str | None = None
    iterations = 0

    for iteration in itertools.count():
        iterations = iteration + 1
        payload = _build_responses_payload(
            model=c.model,
            input_items=input_items,
            tool_dicts=tool_dicts or None,
            response_format=None,
            stream=True,
            extra=_common_extras(c),
        )

        completed_response: Any = None
        async with client._http.stream(
            "POST", "/v1/responses", json=payload
        ) as resp:
            _http.raise_for_status(resp)
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

        if not function_calls:
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
            fn = tool_index.get(fn_name)
            err: str | None
            if fn is None:
                err = f"no Python callable bound for tool '{fn_name}'"
                result_obj: Any = {"error": err}
                duration = 0.0
                yield StreamEvent(
                    type="tool_call.error",
                    data={"id": call_id, "error": err, "iteration": iteration},
                )
            else:
                result_obj, err, duration = await tools.safe_call_async(fn, args)
                if err is None:
                    yield StreamEvent(
                        type="tool_call.result",
                        data={
                            "id": call_id,
                            "result": result_obj,
                            "result_serialized": tools.serialize_tool_result(
                                result_obj
                            ),
                            "duration_ms": duration,
                            "iteration": iteration,
                        },
                    )
                else:
                    yield StreamEvent(
                        type="tool_call.error",
                        data={"id": call_id, "error": err, "iteration": iteration},
                    )
            serialized = tools.serialize_tool_result(result_obj)
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
            rec = ToolCallRecord(
                id=call_id,
                name=fn_name,
                arguments=args,
                arguments_raw=args_raw,
                result=result_obj if err is None else None,
                result_serialized=serialized,
                error=err,
                iteration=iteration,
                duration_ms=duration,
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
            messages_so_far=list(history),
            raw_response=completed_response if isinstance(completed_response, dict) else {},
            will_continue=will_continue,
        )
        stopped = await _fire_intercepts_async(interceptors, intercept)
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


