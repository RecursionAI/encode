"""Session-owned tool registry: register_tool, register_tools, rebind_tools, resume."""

from __future__ import annotations

import json

import pytest

import encode
from encode import EventType, Session


def search(query: str) -> dict:
    """Search."""
    return {"q": query}


def fetch(url: str) -> dict:
    """Fetch."""
    return {"url": url}


def list_tools() -> list[dict]:
    """List tools."""
    return []


def test_open_with_tools_registers_and_emits_events():
    s = Session.open(tools=[search, fetch])
    assert [getattr(t, "__name__", "") for t in s.tools] == ["search", "fetch"]
    types = [e.type for e in s.events]
    assert types == [EventType.TOOL_REGISTERED, EventType.TOOL_REGISTERED]
    assert s.events[0].data["name"] == "search"
    assert s.events[0].data["by"] == "user"
    assert s.events[0].data["schema"]["type"] == "function"


def test_register_tool_is_idempotent_by_name():
    s = Session.open()
    assert s.register_tool(search) is True
    assert s.register_tool(search) is False  # same name → skip
    assert len(s.tools) == 1
    # only one tool.registered event
    assert len(s.events_by_type(EventType.TOOL_REGISTERED)) == 1


def test_register_tools_bulk_returns_newly_added_count():
    s = Session.open()
    added = s.register_tools([search, fetch, search])  # duplicate skipped
    assert added == 2
    assert len(s.tools) == 2


def test_register_tool_accepts_dict_schemas():
    s = Session.open()
    spec = {
        "type": "function",
        "function": {
            "name": "custom_op",
            "description": "do a thing",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }
    assert s.register_tool(spec) is True
    assert s.register_tool(spec) is False  # idempotent on dict name
    # tool list contains the dict; schema event captures it
    assert isinstance(s.tools[0], dict)
    ev = s.events_by_type(EventType.TOOL_REGISTERED)[0]
    assert ev.data["name"] == "custom_op"


def test_register_tool_rejects_unnamed_input():
    s = Session.open()
    with pytest.raises(ValueError):
        s.register_tool({"type": "function", "function": {}})


def test_tools_field_excluded_from_model_dump_but_auto_rebound():
    s = Session.open(tools=[search, fetch])
    raw = s.model_dump()
    assert "tools" not in raw
    # JSON-serializable end-to-end
    blob = json.dumps(raw, default=str)
    parsed = json.loads(blob)
    s2 = Session.model_validate(parsed)
    # ↓ THE headline UX: model_validate auto-rebinds callables from the event log
    assert [t.__name__ for t in s2.tools] == ["search", "fetch"]
    # identity preserved when the original module-level callable is importable
    assert s2.tools[0] is search
    assert s2.tools[1] is fetch
    # original tool.registered events survive (2); auto-rebind doesn't emit new ones
    assert len(s2.events_by_type(EventType.TOOL_REGISTERED)) == 2
    assert s2.unresolved_tools == []


def test_auto_rebind_runs_on_plain_model_validate():
    s = Session.open(tools=[search, fetch])
    raw = json.loads(json.dumps(s.model_dump(), default=str))
    s2 = Session.model_validate(raw)
    assert [t.__name__ for t in s2.tools] == ["search", "fetch"]


def test_session_resume_no_tools_arg_just_works():
    s = Session.open(tools=[search, fetch])
    raw = json.loads(json.dumps(s.model_dump(), default=str))
    s2 = Session.resume(raw)
    assert [t.__name__ for t in s2.tools] == ["search", "fetch"]
    assert s2.id == s.id


def test_unresolved_tools_for_unimportable_callable():
    s = Session.open()
    anon = lambda x: x  # noqa: E731
    anon.__name__ = "anon"
    s.register_tool(anon)
    raw = json.loads(json.dumps(s.model_dump(), default=str))
    s2 = Session.model_validate(raw)
    # lambda can't be re-imported — lands in unresolved_tools, not tools
    assert s2.tools == []
    assert s2.unresolved_tools == ["anon"]


def test_unresolved_tools_recovered_via_rebind_tools_override():
    s = Session.open()
    anon = lambda x: x  # noqa: E731
    anon.__name__ = "anon"
    s.register_tool(anon)
    raw = json.loads(json.dumps(s.model_dump(), default=str))
    # Supply the callable manually — closures, instance methods, etc.
    s2 = Session.resume(raw, tools=[anon])
    assert [getattr(t, "__name__", None) for t in s2.tools] == ["anon"]


def test_raw_dict_tool_round_trips_through_auto_rebind():
    s = Session.open()
    spec = {
        "type": "function",
        "function": {
            "name": "custom_op",
            "description": "x",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }
    s.register_tool(spec)
    raw = json.loads(json.dumps(s.model_dump(), default=str))
    s2 = Session.model_validate(raw)
    # dict-origin tool comes back as a dict — model still sees the schema
    assert len(s2.tools) == 1
    assert s2.tools[0]["function"]["name"] == "custom_op"
    assert s2.unresolved_tools == []


def test_rebind_tools_is_idempotent_against_auto_rebind():
    s = Session.open(tools=[search, fetch])
    raw = json.loads(json.dumps(s.model_dump(), default=str))
    s2 = Session.model_validate(raw)
    # auto-rebind already populated tools; explicit rebind is a no-op
    missing = s2.rebind_tools([search, fetch, list_tools])
    assert missing == []  # everything was already bound by auto-rebind
    assert [t.__name__ for t in s2.tools] == ["search", "fetch"]


def test_double_validate_does_not_duplicate_tools():
    s = Session.open(tools=[search, fetch])
    raw = json.loads(json.dumps(s.model_dump(), default=str))
    s2 = Session.model_validate(raw)
    raw2 = json.loads(json.dumps(s2.model_dump(), default=str))
    s3 = Session.model_validate(raw2)
    assert [t.__name__ for t in s3.tools] == ["search", "fetch"]


def test_async_session_register_tool():
    import asyncio

    async def run():
        s = encode.AsyncSession.open()
        added = await s.aregister_tool(search)
        assert added is True
        assert await s.aregister_tool(search) is False
        # bulk
        added2 = await s.aregister_tools([fetch, search])
        assert added2 == 1
        # rebind
        missing = await s.arebind_tools([search, fetch])
        assert missing == []

    asyncio.run(run())
