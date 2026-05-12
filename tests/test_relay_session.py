"""Tests for relay(session=...) — event emission, hydration, Pydantic resume."""

from __future__ import annotations

import json
from itertools import cycle

import httpx

import encode


def _single_chat(base_url, respx_mock, content="hello back"):
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
            },
        )
    )


def _two_step_with_tool(base_url, respx_mock):
    """A tool-call iteration followed by a final assistant turn."""
    iterator = cycle([
        httpx.Response(
            200,
            json={
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            },
        ),
        httpx.Response(
            200,
            json={
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "result is foo"},
                    "finish_reason": "stop",
                }],
            },
        ),
    ])
    captures: list[list[dict]] = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        captures.append(list(body["messages"]))
        return next(iterator)

    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=handler)
    return captures


def lookup(q: str) -> dict:
    """Lookup something."""
    return {"q": q, "answer": "foo"}


def test_session_emits_basic_events(respx_mock, base_url):
    _single_chat(base_url, respx_mock, content="hi back")
    session = encode.Session.open()
    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        session=session,
    ).response

    types = [e.type for e in session.events]
    assert "user.message" in types
    assert "assistant.message" in types
    assert "iteration.end" in types
    # ids are monotonic
    assert [e.id for e in session.events] == list(range(len(session.events)))


def test_session_emits_tool_call_and_result_events(respx_mock, base_url):
    _two_step_with_tool(base_url, respx_mock)
    session = encode.Session.open()
    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "look up x"}],
        tools=[lookup],
        session=session,
    ).response

    types = [e.type for e in session.events]
    assert types.count("user.message") == 1
    assert types.count("assistant.message") == 2  # tool-call turn + final
    assert types.count("tool.call") == 1
    assert types.count("tool.result") == 1
    assert types.count("iteration.end") == 2

    tool_call = session.events_by_type("tool.call")[0]
    assert tool_call.data["name"] == "lookup"
    assert tool_call.data["arguments"] == {"q": "x"}

    tool_result = session.events_by_type("tool.result")[0]
    assert tool_result.data["id"] == "c1"
    assert tool_result.data["error"] is None


def test_session_resume_via_model_validate(respx_mock, base_url):
    """A session can be serialized, sent through JSON, rehydrated, and continued."""
    _single_chat(base_url, respx_mock, content="first reply")
    session = encode.Session.open()
    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "first"}],
        session=session,
    ).response

    # Serialize like a DB would
    raw = json.loads(json.dumps(session.model_dump(), default=str))
    resumed = encode.Session.model_validate(raw)

    # Resumed session has the same events
    assert len(resumed.events) == len(session.events)
    assert resumed.id == session.id

    # Continue the conversation — context is hydrated from events
    _single_chat(base_url, respx_mock, content="second reply")
    captures: list[list[dict]] = []
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        side_effect=lambda req: (
            captures.append(json.loads(req.content)["messages"])
            or httpx.Response(
                200,
                json={
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "second reply"},
                        "finish_reason": "stop",
                    }],
                },
            )
        )
    )
    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "second"}],
        session=resumed,
    ).response

    # Second relay call must have seen the first conversation's history
    first_call_msgs = captures[0]
    roles = [m.get("role") for m in first_call_msgs]
    # Hydrated history: user (first), assistant (first reply), user (second)
    assert "user" in roles
    assert "assistant" in roles
    # The last message should be the new user message
    assert first_call_msgs[-1] == {"role": "user", "content": "second"}


def test_session_without_messages_kwarg_uses_only_events(respx_mock, base_url):
    """If a session already has events and no messages= is passed, hydration uses events alone."""
    _single_chat(base_url, respx_mock, content="response")
    session = encode.Session.open()
    session.emit("user.message", {"content": "from event"})

    captures: list[list[dict]] = []
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        side_effect=lambda req: (
            captures.append(json.loads(req.content)["messages"])
            or httpx.Response(
                200,
                json={
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "response"},
                        "finish_reason": "stop",
                    }],
                },
            )
        )
    )

    encode.relay(model="m", session=session, messages=[]).response
    assert captures[0] == [{"role": "user", "content": "from event"}]


def test_session_default_path_unchanged(respx_mock, base_url):
    """relay() without session= behaves exactly as before."""
    _single_chat(base_url, respx_mock, content="ok")
    out = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    ).response
    assert out.content == "ok"


def test_emit_tool_call_argument_iteration_index(respx_mock, base_url):
    _two_step_with_tool(base_url, respx_mock)
    session = encode.Session.open()
    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        tools=[lookup],
        session=session,
    ).response

    tool_call = session.events_by_type("tool.call")[0]
    assert tool_call.data["iteration"] == 0
