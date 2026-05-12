"""Intercept-driven tool discovery: event.register_tool(...) → next iter dispatches."""

from __future__ import annotations

import json
from itertools import cycle

import httpx
import pytest

import encode


def list_tools() -> list[dict]:
    """Discover available tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch",
                "description": "fetch a URL",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "additionalProperties": False,
                },
            }
        }
    ]


def fetch(url: str) -> dict:
    """Fetch a URL."""
    return {"url": url, "status": 200}


def _discovery_responses():
    """Three-iter sequence: list_tools call → fetch call → final answer."""
    return cycle(
        [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "list_tools",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "c2",
                                        "type": "function",
                                        "function": {
                                            "name": "fetch",
                                            "arguments": '{"url":"https://example.com"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "all done"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ),
        ]
    )


def test_intercept_register_tool_makes_tool_visible_next_iter(respx_mock, base_url):
    captures: list[list[dict]] = []
    iterator = _discovery_responses()

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        captures.append(list(body.get("tools") or []))
        return next(iterator)

    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=handler)

    session = encode.Session.open(tools=[list_tools])

    def discover(event):
        for tc in event.tool_calls:
            if tc.name == "list_tools":
                for spec in (tc.result or []):
                    event.register_tool(spec)
                # also register the python callable so dispatch works
                event.register_tool(fetch)

    out = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "discover and use a tool"}],
        session=session,
        tools=session.tools,
        on_intercept=discover,
    ).response

    assert out.content == "all done"
    # iteration 0: only list_tools
    assert {t["function"]["name"] for t in captures[0]} == {"list_tools"}
    # iteration 1: list_tools + fetch (registered via intercept)
    assert "fetch" in {t["function"]["name"] for t in captures[1]}
    # registration was logged
    registered = session.events_by_type("tool.registered")
    names = {ev.data["name"] for ev in registered}
    assert "fetch" in names
    by_intercept = [ev for ev in registered if ev.data["by"] == "intercept"]
    assert any(ev.data["name"] == "fetch" for ev in by_intercept)


def test_register_tool_without_session_raises():
    iterator = cycle(
        [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hi"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        ]
    )

    # No mock needed — exception fires from the intercept before any request matters,
    # but we still need to build an InterceptEvent without a session to exercise it.
    event = encode.InterceptEvent(
        iteration=0,
        endpoint="chat",
        assistant_turn=encode.AssistantTurn(content=None, tool_calls=[]),
        tool_calls=[],
        raw_response={},
        will_continue=False,
    )
    with pytest.raises(RuntimeError, match="session"):
        event.register_tool(fetch)


def test_direct_list_append_also_works_with_session_tools(respx_mock, base_url):
    """session.tools is a real list — mutating it directly is also picked up,
    though it bypasses the audit-log event (use event.register_tool for that)."""
    iterator = _discovery_responses()
    captures: list[list[dict]] = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        captures.append(list(body.get("tools") or []))
        return next(iterator)

    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=handler)

    session = encode.Session.open(tools=[list_tools])

    def discover(event):
        for tc in event.tool_calls:
            if tc.name == "list_tools":
                # bypass the verb: append directly
                session.tools.append(fetch)

    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "go"}],
        session=session,
        tools=session.tools,
        on_intercept=discover,
    ).response

    # second iter saw the appended tool — schema is rebuilt from session.tools
    assert "fetch" in {t["function"]["name"] for t in captures[1]}


def test_async_register_tool_visible_next_iter(respx_mock, base_url):
    import asyncio

    captures: list[list[dict]] = []
    iterator = _discovery_responses()

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        captures.append(list(body.get("tools") or []))
        return next(iterator)

    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=handler)

    session = encode.AsyncSession.open(tools=[list_tools])

    async def discover(event):
        for tc in event.tool_calls:
            if tc.name == "list_tools":
                event.register_tool(fetch)

    async def run():
        return await encode.relay_async(
            model="m",
            messages=[{"role": "user", "content": "go"}],
            session=session,
            tools=session.tools,
            on_intercept=discover,
        )

    out = asyncio.run(run())
    assert out.content == "all done"
    assert "fetch" in {t["function"]["name"] for t in captures[1]}
