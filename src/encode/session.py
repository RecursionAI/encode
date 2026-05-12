"""Sessions — append-only event logs that live outside the harness.

A :class:`Session` is a Pydantic model containing an append-only list of
:class:`Event` records. The SDK takes no opinion on persistence: serialize a
session with ``session.model_dump()`` and rehydrate elsewhere with
``Session.model_validate(...)``. Pair with any database, file, or transport.

The session is *not* Claude's context window. The harness reads events from
the session and projects them into a :class:`Messages` list (optionally via
a ``transform`` callable for compaction/trimming). This separation — durable
event log on one side, ephemeral context window on the other — is the core
move from Anthropic's Managed Agents paper.

Resume is plain Pydantic — no special API:

    raw = db.sessions.find_one({"id": sid})
    session = encode.Session.model_validate(raw)

Concurrency: a Session instance is owned by a single process. If you want
multiple writers against the same logical session id, reconcile at your DB
layer (atomic appends + rehydrate). The SDK does not coordinate across
processes.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from .events import Event, EventType

if TYPE_CHECKING:
    from .messages import Messages


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_session_id() -> str:
    return uuid.uuid4().hex


EventTransform = Callable[[list[Event]], list[Event]]


class Session(BaseModel):
    """Append-only event log for an agent run.

    Pure Pydantic — round-trips losslessly through ``model_dump()`` /
    ``model_validate()``. Persist however you like.

    Example:
        session = encode.Session.open()
        session.emit("user.message", {"content": "hi"})
        # ... pass to relay(session=...) ...
        db.save(session.model_dump())

        # later, anywhere:
        session = encode.Session.model_validate(db.load(sid))
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=_new_session_id)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    events: list[Event] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Runtime-only tool registry. Excluded from model_dump because Python
    # callables aren't JSON-serializable; the durable record lives in the
    # ``tool.registered`` events. Use ``register_tool`` / ``rebind_tools`` /
    # ``Session.resume(data, tools=...)`` to manage this list.
    tools: list[Any] = Field(default_factory=list, exclude=True)

    # ------------------------- write (append-only) -------------------------

    def emit(self, type: str | Event, data: dict[str, Any] | None = None) -> Event:
        """Append an event. Assigns ``id`` and refreshes ``updated_at``.

        Accepts either a ``(type, data)`` pair or a pre-built :class:`Event`
        (its ``id`` is overwritten with the next monotonic value).
        """
        if isinstance(type, Event):
            ev = type
            ev.id = len(self.events)
            if data is not None:
                raise TypeError("pass either an Event or (type, data), not both")
        else:
            ev = Event(
                id=len(self.events),
                ts=_now(),
                type=type,
                data=dict(data) if data is not None else {},
            )
        self.events.append(ev)
        self.updated_at = ev.ts
        return ev

    # ------------------------- tool registry (append-only) -------------------------

    def register_tool(self, tool: Any, *, by: str = "user") -> bool:
        """Register a tool on the session. Append-only and idempotent.

        Adds ``tool`` to ``self.tools`` and emits a ``tool.registered`` event
        carrying the model-facing schema. If a tool with the same name is
        already registered, the call is a no-op and returns ``False``.

        ``by`` records the origin of the registration in the event payload
        (``"user"``, ``"intercept"``, ``"resume"``, or any custom string).

        Returns ``True`` if newly registered, ``False`` if skipped.
        """
        from . import tools as _tools

        name = _tools.tool_name(tool)
        if not name:
            raise ValueError(
                "could not determine a name for the tool — pass a callable with __name__ or a dict with function.name / name"
            )
        for existing in self.tools:
            if _tools.tool_name(existing) == name:
                return False
        schema = _tools.tool_schema(tool)
        self.tools.append(tool)
        self.emit(EventType.TOOL_REGISTERED, {"name": name, "schema": schema, "by": by})
        return True

    def register_tools(self, tools: Iterable[Any], *, by: str = "user") -> int:
        """Bulk-register an iterable of tools. Returns the count newly added."""
        return sum(1 for t in tools if self.register_tool(t, by=by))

    def rebind_tools(self, tools: Iterable[Any]) -> list[str]:
        """Re-register callables matching ``tool.registered`` events in the log.

        Walks the log in order; for each unique registered name, looks up a
        callable in ``tools`` (matched via ``tool_name``) and registers it with
        ``by="resume"``. Idempotent — calling ``rebind_tools`` again on a
        session that's already bound is safe (same-name registrations skip).

        Returns the list of names from the event log that had no matching
        callable supplied — useful for surfacing missing bindings on resume.
        """
        from . import tools as _tools

        supplied: dict[str, Any] = {}
        for t in tools:
            name = _tools.tool_name(t)
            if name:
                supplied.setdefault(name, t)

        seen: set[str] = set()
        missing: list[str] = []
        for ev in self.events_by_type(EventType.TOOL_REGISTERED):
            name = str(ev.data.get("name") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            if name in supplied:
                self.register_tool(supplied[name], by="resume")
            else:
                missing.append(name)
        return missing

    # ------------------------- read -------------------------

    def events_since(self, n: int) -> list[Event]:
        """Return events with id > n. Cursor pattern for incremental reads."""
        return [e for e in self.events if e.id > n]

    def events_by_type(self, *types: str) -> list[Event]:
        """Return events whose ``type`` is in ``types``."""
        keep = set(types)
        return [e for e in self.events if e.type in keep]

    def events_slice(
        self, start: int, end: int | None = None
    ) -> list[Event]:
        """Return events with ``start <= id < end`` (or to the tail if ``end`` is None)."""
        if end is None:
            return [e for e in self.events if e.id >= start]
        return [e for e in self.events if start <= e.id < end]

    @property
    def last_event_id(self) -> int:
        """Highest event id present, or -1 if the log is empty."""
        return self.events[-1].id if self.events else -1

    # ------------------------- projection -------------------------

    def to_messages(
        self,
        *,
        transform: EventTransform | None = None,
    ) -> Messages:
        """Project events into a :class:`Messages` context window.

        Default projection: walks events in order and emits one Message per
        ``user.message`` / ``assistant.message`` / ``tool.result`` / ``system``
        event. Other event types (``tool.call``, ``iteration.end``,
        ``context.modify``, custom) are skipped — they are bookkeeping for the
        durable log, not part of the model's context window.

        ``transform`` is an optional callable applied to the event list before
        projection — use it to compact, trim, or rearrange. The transform
        operates on the raw event list; project happens after.
        """
        from .messages import Messages

        events = list(self.events)
        if transform is not None:
            events = list(transform(events))
        return Messages.from_events(events)

    # ------------------------- construction -------------------------

    @classmethod
    def open(
        cls,
        id: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        tools: Iterable[Any] | None = None,
    ) -> Session:
        """Start a fresh session with a generated (or provided) id.

        If ``tools`` is provided, each one is registered (and a
        ``tool.registered`` event is emitted) before the session is returned.
        """
        sess = cls(
            id=id if id is not None else _new_session_id(),
            metadata=dict(metadata) if metadata else {},
        )
        if tools:
            sess.register_tools(tools)
        return sess

    @classmethod
    def resume(
        cls,
        data: dict[str, Any],
        *,
        tools: Iterable[Any] = (),
    ) -> Session:
        """Round-trip helper: ``model_validate(data)`` + ``rebind_tools(tools)``.

        Convenience for restoring a session from a persisted dump and
        re-binding callables to the names captured in the event log. Names
        that have no matching callable in ``tools`` are silently skipped — use
        :meth:`rebind_tools` directly if you need the list of missing names.
        """
        sess = cls.model_validate(data)
        sess.rebind_tools(tools)
        return sess


class AsyncSession(Session):
    """Async-friendly Session.

    Identical model and read API; async ``emit`` mirrors sync ``emit`` for
    use inside async harnesses. State is in-process; persistence is the
    caller's responsibility (await your DB driver after each ``emit``).
    """

    async def aemit(
        self, type: str | Event, data: dict[str, Any] | None = None
    ) -> Event:
        return self.emit(type, data)

    async def aregister_tool(self, tool: Any, *, by: str = "user") -> bool:
        return self.register_tool(tool, by=by)

    async def aregister_tools(
        self, tools: Iterable[Any], *, by: str = "user"
    ) -> int:
        return self.register_tools(tools, by=by)

    async def arebind_tools(self, tools: Iterable[Any]) -> list[str]:
        return self.rebind_tools(tools)


__all__ = [
    "Event",
    "EventType",
    "EventTransform",
    "Session",
    "AsyncSession",
]
