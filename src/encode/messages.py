"""Message and content models for relay() inputs.

Pydantic v2 models for text/image/audio content plus a ``Message`` envelope.
The relay normalizers convert ``Sequence[Message | dict]`` to either:

- list[dict] for /v1/chat/completions
- list of typed input items for /v1/responses
"""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Iterable, Iterator, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .events import Event
    from .responses import RelayResponse


class TextContent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["text"] = "text"
    text: str


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: str
    detail: str | None = None


class ImageContent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL

    @classmethod
    def from_path(cls, path: str | PathLike[str], *, detail: str | None = None) -> ImageContent:
        p = Path(path)
        data = p.read_bytes()
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
        b64 = base64.b64encode(data).decode("ascii")
        return cls(image_url=ImageURL(url=f"data:{mime};base64,{b64}", detail=detail))

    @classmethod
    def from_url(cls, url: str, *, detail: str | None = None) -> ImageContent:
        return cls(image_url=ImageURL(url=url, detail=detail))


class InputAudio(BaseModel):
    model_config = ConfigDict(extra="allow")
    data: str  # base64
    format: Literal["wav", "mp3"]


class AudioContent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["input_audio"] = "input_audio"
    input_audio: InputAudio

    @classmethod
    def from_path(cls, path: str | PathLike[str]) -> AudioContent:
        p = Path(path)
        data = p.read_bytes()
        ext = p.suffix.lower().lstrip(".")
        fmt: Literal["wav", "mp3"] = "wav" if ext == "wav" else "mp3"
        b64 = base64.b64encode(data).decode("ascii")
        return cls(input_audio=InputAudio(data=b64, format=fmt))


ContentPart = TextContent | ImageContent | AudioContent


class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class Conversation(BaseModel):
    """Pydantic snapshot of a ``Messages`` container.

    Suitable for database persistence — call ``model.model_dump_json()`` to
    serialize, ``Conversation.model_validate_json(blob)`` to load, and
    ``Messages.from_pydantic(model)`` to rehydrate into a stateful container.
    """

    model_config = ConfigDict(extra="allow")
    messages: list[Message]


def _msg_to_dict(m: Message | dict) -> dict[str, Any]:
    if isinstance(m, Message):
        return m.model_dump(exclude_none=True)
    return dict(m)


def to_chat_messages(
    messages: Sequence[Message | dict] | None,
) -> list[dict[str, Any]]:
    """Normalize for /v1/chat/completions."""
    if not messages:
        return []
    return [_msg_to_dict(m) for m in messages]


def to_responses_input(
    messages: Sequence[Message | dict] | None,
    *,
    input: str | Sequence[dict] | None = None,
) -> list[dict[str, Any]]:
    """Normalize for /v1/responses.

    Combines an optional explicit ``input`` (string or list of typed items) with
    a converted ``messages`` list. Each message becomes a ``{"type":"message"}``
    item; tool messages become ``function_call_output`` items.
    """
    items: list[dict[str, Any]] = []
    if input is not None:
        if isinstance(input, str):
            items.append({"type": "message", "role": "user", "content": input})
        else:
            items.extend(dict(i) for i in input)
    if messages:
        for m in messages:
            d = _msg_to_dict(m)
            role = d.get("role")
            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": d.get("tool_call_id"),
                        "output": d.get("content") or "",
                    }
                )
            elif d.get("tool_calls"):
                # assistant turn with tool calls — emit message + function_call items
                if d.get("content"):
                    items.append(
                        {
                            "type": "message",
                            "role": role,
                            "content": d["content"],
                        }
                    )
                for tc in d["tool_calls"]:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        }
                    )
            else:
                items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": d.get("content"),
                    }
                )
    return items


def _coerce_content(content: Any) -> Any:
    """Pass through strings and lists; convert Pydantic content parts to dicts."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        return [c.model_dump(exclude_none=True) if isinstance(c, BaseModel) else c for c in content]
    return content


class Messages(Sequence[dict[str, Any]]):
    """Mutable conversation container.

    Pass to ``relay()`` / ``relay_async()`` as ``messages=`` and the SDK will
    append the new turns in place after the loop completes. Plain lists work
    too — they are not mutated.

    Implements :class:`collections.abc.Sequence`, so it satisfies
    ``Sequence[Any]`` parameter types and supports ``len()``, iteration,
    indexing, slicing, ``in``, ``index()``, ``count()``, etc.

    Example:
        m = (
            encode.Messages()
            .system("Be brief.")
            .user("name three colors")
        )
        encode.relay(model="...", messages=m).response
        m.user("now three more")
        encode.relay(model="...", messages=m).response   # carries prior turns
    """

    __slots__ = ("_items",)

    def __init__(self, messages: Iterable[Message | dict[str, Any]] | None = None) -> None:
        self._items: list[dict[str, Any]] = []
        if messages:
            self.extend(messages)

    # ------------------------- ergonomic adders -------------------------

    def system(self, content: str | list[Any]) -> Messages:
        self._items.append({"role": "system", "content": _coerce_content(content)})
        return self

    def user(self, content: str | list[Any]) -> Messages:
        self._items.append({"role": "user", "content": _coerce_content(content)})
        return self

    def assistant(
        self,
        content: str | None = None,
        *,
        tool_calls: list[dict[str, Any]] | list[ToolCall] | None = None,
    ) -> Messages:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = [
                tc.model_dump(exclude_none=True) if isinstance(tc, ToolCall) else dict(tc)
                for tc in tool_calls
            ]
        self._items.append(msg)
        return self

    def tool(self, content: str, *, tool_call_id: str) -> Messages:
        self._items.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )
        return self

    def add(self, message: Message | dict[str, Any]) -> Messages:
        self._items.append(_msg_to_dict(message))
        return self

    # ------------------------- list-like protocol -------------------------

    def append(self, message: Message | dict[str, Any]) -> None:
        self._items.append(_msg_to_dict(message))

    def extend(self, messages: Iterable[Message | dict[str, Any]]) -> None:
        for m in messages:
            self.append(m)

    def clear(self) -> None:
        self._items.clear()

    def copy(self) -> Messages:
        new: Messages = Messages.__new__(Messages)
        new._items = [dict(m) for m in self._items]
        return new

    def to_list(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._items)

    @overload
    def __getitem__(self, idx: int) -> dict[str, Any]: ...
    @overload
    def __getitem__(self, idx: slice) -> list[dict[str, Any]]: ...
    def __getitem__(
        self, idx: int | slice
    ) -> dict[str, Any] | list[dict[str, Any]]:
        return self._items[idx]

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return f"Messages({len(self._items)} items)"

    # ------------------------- absorb a response -------------------------

    def update(self, response: RelayResponse) -> None:
        """Replace contents with the conversation from a RelayResponse.

        Useful for streaming flows where auto-update doesn't fire, or any
        manual ingestion path. Replaces — does not merge — because
        ``RelayResponse.messages`` is the full history.
        """
        self._items = list(response.messages)

    # ------------------------- pydantic interop -------------------------

    def to_pydantic(self) -> Conversation:
        """Snapshot the current state as a validated Pydantic model.

        Useful for DB serialization (``model.model_dump_json()``) or anywhere
        else you want a typed, validated representation rather than raw dicts.
        """
        return Conversation(
            messages=[Message.model_validate(m) for m in self._items]
        )

    @classmethod
    def from_pydantic(cls, model: Conversation) -> Messages:
        """Rehydrate a Messages container from a Conversation model."""
        m = cls()
        for msg in model.messages:
            m._items.append(msg.model_dump(exclude_none=True))
        return m

    # ------------------------- session interop -------------------------

    @classmethod
    def from_events(cls, events: Iterable[Event]) -> Messages:
        """Project a sequence of :class:`Event` records into a Messages list.

        Standard projection:

        - ``user.message``      → ``{"role": "user", "content": ...}``
        - ``assistant.message`` → ``{"role": "assistant", "content": ..., "tool_calls": ...}``
        - ``tool.result``       → ``{"role": "tool", "tool_call_id": id, "content": serialized}``
        - ``system``            → ``{"role": "system", "content": ...}``

        Other event types (``tool.call`` standalone, ``iteration.end``,
        ``context.modify``, custom) are bookkeeping and are skipped — the
        assistant turn already carries the ``tool_calls`` it issued.

        Tool results without a preceding assistant tool_calls turn are still
        projected (so partial sessions remain usable); they end up as orphan
        tool messages.
        """
        m = cls()
        for ev in events:
            t = ev.type
            d = ev.data or {}
            if t == "user.message":
                m._items.append(
                    {"role": "user", "content": _coerce_content(d.get("content"))}
                )
            elif t == "assistant.message":
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": d.get("content"),
                }
                tcs = d.get("tool_calls")
                if tcs:
                    msg["tool_calls"] = [dict(tc) for tc in tcs]
                m._items.append(msg)
            elif t == "tool.result":
                serialized = d.get("result_serialized")
                if not serialized:
                    raw = d.get("result")
                    serialized = "" if raw is None else str(raw)
                m._items.append(
                    {
                        "role": "tool",
                        "tool_call_id": d.get("id", ""),
                        "content": serialized,
                    }
                )
            elif t == "system":
                m._items.append(
                    {"role": "system", "content": _coerce_content(d.get("content"))}
                )
            # tool.call / iteration.end / context.modify / custom: skipped
        return m


__all__ = [
    "TextContent",
    "ImageContent",
    "ImageURL",
    "AudioContent",
    "InputAudio",
    "ToolCall",
    "ToolCallFunction",
    "Message",
    "Messages",
    "Conversation",
    "ContentPart",
    "to_chat_messages",
    "to_responses_input",
]


