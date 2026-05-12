"""Tests for the mutable InterceptEvent API.

Intercept callbacks can now mutate the conversation via append/insert/replace/
edit_last_tool_result/compact. Mutations apply to the next iteration and (when
a Session is active) emit context.modify events.
"""

from __future__ import annotations

from itertools import cycle

import httpx
import pytest

import encode


def _two_step_chat_with_capture(base_url, respx_mock):
    """Return (capture_list, route_setup). Captures each request's `messages`."""
    captures: list[list[dict]] = []

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
                            "function": {"name": "noop", "arguments": "{}"},
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
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }],
            },
        ),
    ])

    def handler(request: httpx.Request):
        import json
        body = json.loads(request.content)
        captures.append(list(body["messages"]))
        return next(iterator)

    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=handler)
    return captures


def _noop_tool() -> dict:
    """No-op tool."""
    return {"ok": True}


def noop() -> dict:
    """No-op."""
    return {"ok": True}


def test_intercept_append_propagates_to_next_iteration(respx_mock, base_url):
    captures = _two_step_chat_with_capture(base_url, respx_mock)

    def watcher(event):
        event.append({"role": "system", "content": "stay focused"})

    out = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "go"}],
        tools=[noop],
        on_intercept=watcher,
    ).response

    assert out.content == "done"
    # second request must include the injected system message
    second_req_messages = captures[1]
    assert any(
        m.get("role") == "system" and m.get("content") == "stay focused"
        for m in second_req_messages
    )


def test_intercept_replace_swaps_whole_history(respx_mock, base_url):
    captures = _two_step_chat_with_capture(base_url, respx_mock)

    def watcher(event):
        event.replace([{"role": "user", "content": "new prompt"}])

    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "original"}],
        tools=[noop],
        on_intercept=watcher,
    ).response

    second_req_messages = captures[1]
    assert second_req_messages == [{"role": "user", "content": "new prompt"}]


def test_intercept_compact_applies_fn(respx_mock, base_url):
    captures = _two_step_chat_with_capture(base_url, respx_mock)

    def watcher(event):
        def keep_last(msgs):
            return msgs[-1:]
        event.compact(keep_last)

    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "u"}],
        tools=[noop],
        on_intercept=watcher,
    ).response

    second_req_messages = captures[1]
    assert len(second_req_messages) == 1


def test_intercept_edit_last_tool_result(respx_mock, base_url):
    captures = _two_step_chat_with_capture(base_url, respx_mock)

    def watcher(event):
        event.edit_last_tool_result(lambda c: "REDACTED")

    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "u"}],
        tools=[noop],
        on_intercept=watcher,
    ).response

    second_req_messages = captures[1]
    last_tool = [m for m in second_req_messages if m.get("role") == "tool"][-1]
    assert last_tool["content"] == "REDACTED"


def test_intercept_no_mutation_leaves_history_alone(respx_mock, base_url):
    captures = _two_step_chat_with_capture(base_url, respx_mock)

    def watcher(event):
        # Only read; do not mutate
        assert event.iteration == 0
        assert event.mutated is False

    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "u"}],
        tools=[noop],
        on_intercept=watcher,
    ).response

    # Second request should have the normal sequence: user, assistant(tool_calls), tool
    second_req_messages = captures[1]
    roles = [m.get("role") for m in second_req_messages]
    assert roles == ["user", "assistant", "tool"]


def test_intercept_mutated_flag_reflects_mutation():
    """Unit-level test of InterceptEvent.mutated without going through relay."""
    from encode.relay import InterceptEvent
    from encode.responses import AssistantTurn

    ev = InterceptEvent(
        iteration=0,
        endpoint="chat",
        assistant_turn=AssistantTurn(content=None, tool_calls=[]),
        tool_calls=[],
        raw_response={},
        will_continue=True,
        messages=encode.Messages([{"role": "user", "content": "hi"}]),
    )
    assert ev.mutated is False
    ev.append({"role": "system", "content": "reminder"})
    assert ev.mutated is True


def test_intercept_session_emits_context_modify(respx_mock, base_url):
    """When a session is active, mutation emits a context.modify event."""
    _two_step_chat_with_capture(base_url, respx_mock)

    session = encode.Session.open()

    def watcher(event):
        event.append({"role": "system", "content": "extra"})

    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "u"}],
        tools=[noop],
        session=session,
        on_intercept=watcher,
    ).response

    modify_events = session.events_by_type("context.modify")
    assert len(modify_events) == 1
    assert modify_events[0].data["by"] == "intercept"
