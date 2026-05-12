"""Tests for the new Session event-log primitive (Pydantic-only)."""

from __future__ import annotations

import json
from datetime import datetime

import encode
from encode import Event, EventType, Session


def test_open_creates_session_with_id_and_empty_events():
    s = Session.open()
    assert s.id
    assert isinstance(s.id, str)
    assert s.events == []
    assert s.last_event_id == -1


def test_open_accepts_custom_id_and_metadata():
    s = Session.open("my-session", metadata={"owner": "jackson"})
    assert s.id == "my-session"
    assert s.metadata == {"owner": "jackson"}


def test_emit_assigns_monotonic_ids():
    s = Session.open()
    e1 = s.emit("user.message", {"content": "hello"})
    e2 = s.emit("assistant.message", {"content": "hi"})
    e3 = s.emit("user.message", {"content": "again"})
    assert e1.id == 0
    assert e2.id == 1
    assert e3.id == 2
    assert s.last_event_id == 2


def test_emit_sets_timestamp_and_data():
    s = Session.open()
    before = datetime.now().astimezone().timestamp()
    e = s.emit("user.message", {"content": "hi"})
    after = datetime.now().astimezone().timestamp()
    assert before <= e.ts.timestamp() <= after + 1
    assert e.type == "user.message"
    assert e.data == {"content": "hi"}


def test_emit_with_prebuilt_event():
    s = Session.open()
    ev = Event.user_message("hi")
    appended = s.emit(ev)
    assert appended.id == 0
    assert appended.type == EventType.USER_MESSAGE


def test_emit_rejects_event_plus_data():
    s = Session.open()
    import pytest

    with pytest.raises(TypeError):
        s.emit(Event.user_message("hi"), {"extra": True})


def test_events_since_returns_only_newer():
    s = Session.open()
    for i in range(5):
        s.emit("custom", {"i": i})
    after = s.events_since(2)
    assert [e.id for e in after] == [3, 4]


def test_events_by_type_filters():
    s = Session.open()
    s.emit("user.message", {"content": "a"})
    s.emit("assistant.message", {"content": "b"})
    s.emit("user.message", {"content": "c"})
    users = s.events_by_type("user.message")
    assert [e.data["content"] for e in users] == ["a", "c"]


def test_events_slice_positional_window():
    s = Session.open()
    for i in range(10):
        s.emit("custom", {"i": i})
    window = s.events_slice(3, 6)
    assert [e.id for e in window] == [3, 4, 5]


def test_model_dump_validate_round_trip_preserves_log():
    s = Session.open("sid-1", metadata={"k": "v"})
    s.emit(*("user.message", {"content": "hi"}))
    s.emit(*("assistant.message", {"content": "yo"}))
    raw = s.model_dump()
    # serialize through JSON the way a DB would
    blob = json.dumps(raw, default=str)
    parsed = json.loads(blob)
    s2 = Session.model_validate(parsed)
    assert s2.id == s.id
    assert s2.metadata == s.metadata
    assert [(e.id, e.type, e.data) for e in s2.events] == [
        (e.id, e.type, e.data) for e in s.events
    ]


def test_to_messages_projects_standard_events():
    s = Session.open()
    s.emit("system", {"content": "be brief"})
    s.emit("user.message", {"content": "hi"})
    s.emit("assistant.message", {"content": "hello", "tool_calls": None})
    msgs = s.to_messages()
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "hello"


def test_to_messages_skips_bookkeeping_events():
    s = Session.open()
    s.emit("user.message", {"content": "hi"})
    s.emit("tool.call", {"id": "c1", "name": "f", "arguments": {}, "iteration": 0})
    s.emit("iteration.end", {"iteration": 0, "had_tool_calls": True, "finish_reason": None})
    s.emit("context.modify", {"by": "intercept", "summary": "trimmed"})
    msgs = s.to_messages()
    # only the user message survives
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_to_messages_projects_tool_result_with_id():
    s = Session.open()
    s.emit("assistant.message", {
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
    })
    s.emit("tool.result", {"id": "c1", "result": {"ok": True}, "result_serialized": '{"ok": true}'})
    msgs = s.to_messages()
    assert len(msgs) == 2
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_call_id"] == "c1"
    assert msgs[1]["content"] == '{"ok": true}'


def test_to_messages_transform_runs_before_projection():
    s = Session.open()
    for i in range(5):
        s.emit("user.message", {"content": f"msg {i}"})

    def keep_last_two(events):
        return events[-2:]

    msgs = s.to_messages(transform=keep_last_two)
    assert len(msgs) == 2
    assert msgs[0]["content"] == "msg 3"
    assert msgs[1]["content"] == "msg 4"


def test_event_factories_build_correct_shapes():
    e = Event.tool_call(id="c1", name="f", arguments={"x": 1}, iteration=2)
    assert e.type == "tool.call"
    assert e.data == {"id": "c1", "name": "f", "arguments": {"x": 1}, "iteration": 2}

    e2 = Event.tool_result(id="c1", result={"ok": True}, result_serialized='{"ok": true}', duration_ms=12.5)
    assert e2.type == "tool.result"
    assert e2.data["duration_ms"] == 12.5
    assert e2.data["error"] is None


def test_session_is_exported_from_top_level():
    assert encode.Session is Session
    assert encode.Event is Event
    assert encode.EventType is EventType
