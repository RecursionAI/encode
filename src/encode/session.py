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

import importlib
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .events import Event, EventType

if TYPE_CHECKING:
    from .messages import Messages


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_session_id() -> str:
    return uuid.uuid4().hex


EventTransform = Callable[[list[Event]], list[Event]]


def _capture_import_path(tool: Any) -> str | None:
    """Return ``"module:qualname"`` if a callable can plausibly be re-imported.

    Returns ``None`` for raw dict tools, lambdas, inner functions (qualnames
    containing ``<lambda>`` or ``<locals>``), and callables defined in
    ``__main__`` (which won't round-trip across processes). These will land in
    :attr:`Session.unresolved_tools` on resume — users supply them manually if
    needed via :meth:`Session.rebind_tools`.
    """
    if not callable(tool) or isinstance(tool, dict):
        return None
    try:
        module = tool.__module__
        qualname = tool.__qualname__
    except AttributeError:
        return None
    if not module or not qualname:
        return None
    if "<" in qualname:
        return None
    if module == "__main__":
        return None
    return f"{module}:{qualname}"


def _resolve_import_path(path: str) -> Any | None:
    """Resolve ``"module:qualname"`` to a callable. Returns ``None`` on failure."""
    if ":" not in path:
        return None
    module_name, _, qualname = path.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError):
        return None
    return obj if callable(obj) else None


def _try_bind_from_event(ev: Event) -> Any | None:
    """Best-effort rebinding of a tool entry from its ``tool.registered`` event.

    Dispatches on the event's ``is_callable`` flag (set at registration time):

    - **Callable origin** + ``import_path`` resolves → return the callable.
    - **Callable origin** without a usable ``import_path`` (lambda, closure,
      ``__main__``, moved module) → return ``None`` so the name lands in
      :attr:`Session.unresolved_tools`.
    - **Dict origin** → return a fresh copy of the captured schema dict so
      raw-dict tool registrations round-trip exactly.

    Returns ``None`` when the event payload is malformed.
    """
    data = ev.data or {}
    is_callable = bool(data.get("is_callable", False))
    if is_callable:
        import_path = data.get("import_path")
        if import_path:
            return _resolve_import_path(str(import_path))
        return None
    schema = data.get("schema")
    if isinstance(schema, dict) and schema:
        return dict(schema)
    return None


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
    # ``tool.registered`` events. On ``model_validate``, a post-validator
    # walks the event log and re-imports each callable via its captured
    # ``import_path`` — so ``Session.model_validate(db.load(sid))`` returns a
    # session with ``tools`` fully populated for the common case.
    tools: list[Any] = Field(default_factory=list, exclude=True)

    # Names from ``tool.registered`` events that couldn't be auto-re-bound on
    # resume (typically lambdas, closures, ``__main__`` functions, or moved
    # callables). Excluded from model_dump — purely a diagnostic surface.
    _unresolved_tool_names: list[str] = PrivateAttr(default_factory=list)

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

    # ------------------------- resume / auto-rebind -------------------------

    @model_validator(mode="after")
    def _auto_rebind_tools(self) -> Session:
        """Re-populate ``self.tools`` from ``tool.registered`` events.

        Runs after every Session construction (both ``model_validate`` and a
        direct ``__init__``). Walks the event log in order; for each unique
        registered name not already bound, tries to resolve a callable from
        the event's ``import_path`` (or falls back to the captured schema
        dict for raw-dict tools). Names that can't be resolved go into
        :attr:`unresolved_tools` — the validator never raises.

        Idempotent: if a name is already present in ``self.tools`` (typically
        because the user just called ``register_tool`` during normal runtime
        flow), it is skipped.
        """
        from . import tools as _tools

        bound: set[str] = {_tools.tool_name(t) for t in self.tools}
        unresolved: list[str] = []
        seen: set[str] = set()
        for ev in self.events:
            if ev.type != EventType.TOOL_REGISTERED:
                continue
            name = str((ev.data or {}).get("name") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            if name in bound:
                continue
            tool = _try_bind_from_event(ev)
            if tool is not None:
                self.tools.append(tool)
                bound.add(name)
            else:
                unresolved.append(name)
        self._unresolved_tool_names = unresolved
        return self

    @property
    def unresolved_tools(self) -> list[str]:
        """Names from the ``tool.registered`` log that couldn't be auto-bound.

        Typically lambdas, closures, ``__main__``-scope functions, or
        callables whose module path changed since registration. Supply them
        manually via :meth:`rebind_tools` if you need them.
        """
        return list(self._unresolved_tool_names)

    # ------------------------- tool registry (append-only) -------------------------

    def register_tool(self, tool: Any, *, by: str = "user") -> bool:
        """Register a tool on the session. Append-only and idempotent.

        Adds ``tool`` to ``self.tools`` and emits a ``tool.registered`` event
        carrying the model-facing schema and (for cleanly importable
        callables) an ``import_path`` of the form ``"module:qualname"``. That
        path is what enables :meth:`Session.model_validate` to automatically
        rebind callables on resume.

        If a tool with the same name is already registered, the call is a
        no-op and returns ``False``.

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
        is_callable = callable(tool) and not isinstance(tool, dict)
        payload: dict[str, Any] = {
            "name": name,
            "schema": schema,
            "by": by,
            "is_callable": is_callable,
        }
        if is_callable:
            import_path = _capture_import_path(tool)
            if import_path is not None:
                payload["import_path"] = import_path
        self.emit(EventType.TOOL_REGISTERED, payload)
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
        """Restore a session from a persisted dump.

        For the common case (module-level callable tools), this is equivalent
        to ``cls.model_validate(data)`` — the post-validator auto-rebinds
        tools from each ``tool.registered`` event's ``import_path``. Pass
        ``tools=[...]`` to supply callables that can't be auto-resolved
        (lambdas, closures, ``__main__`` functions) or to override specific
        names with a different implementation.
        """
        sess = cls.model_validate(data)
        if tools:
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
