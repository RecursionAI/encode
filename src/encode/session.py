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
    ) -> Session:
        """Start a fresh session with a generated (or provided) id."""
        return cls(
            id=id if id is not None else _new_session_id(),
            metadata=dict(metadata) if metadata else {},
        )


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


__all__ = [
    "Event",
    "EventType",
    "EventTransform",
    "Session",
    "AsyncSession",
]
