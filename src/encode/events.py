"""Event types for the Session event log.

The :class:`Event` model is a flat envelope ``{id, ts, type, data}``. The
``type`` is a string — one of the :class:`EventType` constants for standard
agent-loop events, or any user-defined string for custom events. The ``data``
payload is a plain dict whose shape depends on the type; standard shapes are
documented in :class:`EventType` and built via the classmethod factories on
:class:`Event`.

Designed so that Session events can express *everything that happened* in an
agent run (per the Managed Agents paper), while staying schema-light: users
can emit custom events without subclassing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType:
    """Standard event-type constants.

    Use these for the ``type`` field on :class:`Event` when emitting events
    that correspond to the SDK's built-in agent-loop semantics. Custom event
    types are also allowed — pass any string.
    """

    USER_MESSAGE = "user.message"
    ASSISTANT_MESSAGE = "assistant.message"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    TOOL_REGISTERED = "tool.registered"
    ITERATION_END = "iteration.end"
    CONTEXT_MODIFY = "context.modify"
    SYSTEM = "system"
    CUSTOM = "custom"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Event(BaseModel):
    """One entry in a Session's append-only log.

    Standard ``data`` shapes per type:

    - ``user.message``      : ``{"content": str | list[ContentPart]}``
    - ``assistant.message`` : ``{"content": str | None, "tool_calls": [...]}``
    - ``tool.call``         : ``{"id": str, "name": str, "arguments": dict, "iteration": int}``
    - ``tool.result``       : ``{"id": str, "result": Any, "result_serialized": str,
                                  "error": str | None, "duration_ms": float}``
    - ``tool.registered``   : ``{"name": str, "schema": dict, "by": str}``
    - ``iteration.end``     : ``{"iteration": int, "had_tool_calls": bool,
                                  "finish_reason": str | None}``
    - ``context.modify``    : ``{"by": str, "summary": str, ...}``
    - ``system``            : ``{"content": str}``
    - ``custom``            : anything

    ``id`` is assigned by the owning :class:`Session` on ``emit()`` — leave it
    at the default when constructing manually (the session will overwrite).
    """

    model_config = ConfigDict(extra="allow")

    id: int = -1
    ts: datetime = Field(default_factory=_now)
    type: str
    data: dict[str, Any] = Field(default_factory=dict)

    # ------------------------- type-safe factories -------------------------

    @classmethod
    def user_message(cls, content: Any) -> Event:
        return cls(type=EventType.USER_MESSAGE, data={"content": content})

    @classmethod
    def assistant_message(
        cls,
        content: str | None,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> Event:
        data: dict[str, Any] = {"content": content}
        if tool_calls:
            data["tool_calls"] = list(tool_calls)
        return cls(type=EventType.ASSISTANT_MESSAGE, data=data)

    @classmethod
    def tool_call(
        cls,
        *,
        id: str,
        name: str,
        arguments: dict[str, Any],
        iteration: int,
    ) -> Event:
        return cls(
            type=EventType.TOOL_CALL,
            data={
                "id": id,
                "name": name,
                "arguments": dict(arguments),
                "iteration": iteration,
            },
        )

    @classmethod
    def tool_result(
        cls,
        *,
        id: str,
        result: Any,
        result_serialized: str = "",
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> Event:
        return cls(
            type=EventType.TOOL_RESULT,
            data={
                "id": id,
                "result": result,
                "result_serialized": result_serialized,
                "error": error,
                "duration_ms": duration_ms,
            },
        )

    @classmethod
    def tool_registered(
        cls,
        *,
        name: str,
        schema: dict[str, Any],
        by: str = "user",
    ) -> Event:
        return cls(
            type=EventType.TOOL_REGISTERED,
            data={"name": name, "schema": dict(schema), "by": by},
        )

    @classmethod
    def iteration_end(
        cls,
        *,
        iteration: int,
        had_tool_calls: bool,
        finish_reason: str | None = None,
    ) -> Event:
        return cls(
            type=EventType.ITERATION_END,
            data={
                "iteration": iteration,
                "had_tool_calls": had_tool_calls,
                "finish_reason": finish_reason,
            },
        )

    @classmethod
    def context_modify(
        cls,
        *,
        by: str,
        summary: str = "",
        **extra: Any,
    ) -> Event:
        data: dict[str, Any] = {"by": by, "summary": summary}
        data.update(extra)
        return cls(type=EventType.CONTEXT_MODIFY, data=data)

    @classmethod
    def system(cls, content: str) -> Event:
        return cls(type=EventType.SYSTEM, data={"content": content})

    @classmethod
    def custom(cls, type: str, data: dict[str, Any]) -> Event:
        return cls(type=type, data=dict(data))


__all__ = [
    "Event",
    "EventType",
]
