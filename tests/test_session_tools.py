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


def test_tools_field_excluded_from_model_dump():
    s = Session.open(tools=[search, fetch])
    raw = s.model_dump()
    assert "tools" not in raw
    # JSON-serializable end-to-end
    blob = json.dumps(raw, default=str)
    parsed = json.loads(blob)
    s2 = Session.model_validate(parsed)
    # tools list is empty after resume; events survived
    assert s2.tools == []
    assert len(s2.events_by_type(EventType.TOOL_REGISTERED)) == 2


def test_rebind_tools_repopulates_from_event_log():
    s = Session.open(tools=[search, fetch])
    raw = json.loads(json.dumps(s.model_dump(), default=str))
    s2 = Session.model_validate(raw)
    missing = s2.rebind_tools([search, fetch, list_tools])
    assert missing == []
    assert [t.__name__ for t in s2.tools] == ["search", "fetch"]
    # rebind itself emits new events with by="resume"
    resume_events = [
        e for e in s2.events_by_type(EventType.TOOL_REGISTERED) if e.data["by"] == "resume"
    ]
    assert len(resume_events) == 2


def test_rebind_tools_returns_missing_names():
    s = Session.open(tools=[search, fetch])
    raw = json.loads(json.dumps(s.model_dump(), default=str))
    s2 = Session.model_validate(raw)
    missing = s2.rebind_tools([search])  # fetch not supplied
    assert missing == ["fetch"]
    assert [t.__name__ for t in s2.tools] == ["search"]


def test_resume_classmethod_round_trip():
    s = Session.open(tools=[search, fetch])
    raw = json.loads(json.dumps(s.model_dump(), default=str))
    s2 = Session.resume(raw, tools=[search, fetch])
    assert s2.id == s.id
    assert [t.__name__ for t in s2.tools] == ["search", "fetch"]


def test_double_resume_is_safe():
    s = Session.open(tools=[search])
    raw = json.loads(json.dumps(s.model_dump(), default=str))
    s2 = Session.resume(raw, tools=[search])
    # call rebind again — should be a no-op for the bound tool
    missing = s2.rebind_tools([search])
    assert missing == []
    # tools list still has exactly one entry
    assert len(s2.tools) == 1


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
