"""Tool-call loop: multi-iter execution, intercept, event.stop(), tool errors."""

from __future__ import annotations

from itertools import cycle

import httpx
import pytest

import encode


def get_weather(city: str) -> dict:
    """Get current weather by city."""
    return {"city": city, "temp_f": 72}


def explode(reason: str = "boom") -> dict:
    """A tool that always raises."""
    raise RuntimeError(reason)


def _two_step_chat(base_url, respx_mock, *, second_content="all done"):
    iterator = cycle(
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
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":"Denver"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": second_content},
                            "finish_reason": "stop",
                        }
                    ],
                },
            ),
        ]
    )
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=lambda *a, **k: next(iterator))


def test_tool_loop_executes_callable_and_returns_final(respx_mock, base_url):
    _two_step_chat(base_url, respx_mock, second_content="It's 72F in Denver.")
    out = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "weather in denver?"}],
        tools=[get_weather],
    ).response
    assert out.content == "It's 72F in Denver."
    assert out.iterations == 2
    assert len(out.tool_calls) == 1
    rec = out.tool_calls[0]
    assert rec.name == "get_weather"
    assert rec.arguments == {"city": "Denver"}
    assert rec.result == {"city": "Denver", "temp_f": 72}
    assert rec.error is None
    # history should contain user, assistant(tool_calls), tool, assistant(final)
    roles = [m["role"] for m in out.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_intercept_method_fires_with_event(respx_mock, base_url):
    _two_step_chat(base_url, respx_mock)
    seen: list[encode.InterceptEvent] = []

    def cb(event):
        seen.append(event)

    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[get_weather],
    ).intercept(cb).response

    assert len(seen) == 1
    assert seen[0].iteration == 0
    assert seen[0].endpoint == "chat"
    assert seen[0].tool_calls[0].name == "get_weather"
    assert seen[0].will_continue is True


def test_on_intercept_kwarg_works(respx_mock, base_url):
    _two_step_chat(base_url, respx_mock)
    seen: list[int] = []
    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[get_weather],
        on_intercept=lambda ev: seen.append(ev.iteration),
    ).response
    assert seen == [0]


def test_event_stop_terminates_loop_after_iteration(respx_mock, base_url):
    # Both responses return tool calls; if event.stop() didn't work, max_tool_iterations would trigger.
    iterator = cycle(
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
                                            "name": "get_weather",
                                            "arguments": '{"city":"X"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        ]
    )
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=lambda *a, **k: next(iterator))

    def stopper(event):
        event.stop()

    out = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[get_weather],
        on_intercept=stopper,
        max_tool_iterations=5,
    ).response
    assert out.iterations == 1
    # tool was still executed and recorded for the stopped iteration
    assert len(out.tool_calls) == 1


def test_tool_exception_captured_and_fed_to_model(respx_mock, base_url):
    iterator = cycle(
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
                                        "function": {"name": "explode", "arguments": "{}"},
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
                            "message": {"role": "assistant", "content": "tool failed but I'm ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ),
        ]
    )
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=lambda *a, **k: next(iterator))

    out = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[explode],
    ).response
    assert out.iterations == 2
    rec = out.tool_calls[0]
    assert rec.error is not None and "RuntimeError" in rec.error
    # tool message content carries the error envelope so the model can recover
    tool_msg = next(m for m in out.messages if m["role"] == "tool")
    assert "error" in tool_msg["content"].lower()


def test_max_tool_iterations_raises(respx_mock, base_url):
    # Always return a tool call — never converges.
    iterator = cycle(
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
                                            "name": "get_weather",
                                            "arguments": '{"city":"Denver"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        ]
    )
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=lambda *a, **k: next(iterator))

    with pytest.raises(encode.MaxToolIterationsError) as exc_info:
        encode.relay(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[get_weather],
            max_tool_iterations=2,
        ).response
    assert exc_info.value.partial is not None
    assert exc_info.value.partial.iterations >= 2


def test_unbounded_default_does_not_cap(respx_mock, base_url):
    """With no max_tool_iterations set, the loop runs as long as the model keeps calling tools."""
    # Mock returns 12 tool-call rounds, then a final answer. v0.1.0 would have raised at 8.
    seq = []
    for _ in range(12):
        seq.append(
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
                                        "id": "c",
                                        "type": "function",
                                        "function": {"name": "get_weather", "arguments": '{"city":"X"}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        )
    seq.append(
        httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
                ]
            },
        )
    )
    iterator = iter(seq)
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=lambda *a, **k: next(iterator))

    out = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[get_weather],
    ).response
    assert out.content == "done"
    assert out.iterations == 13
    assert len(out.tool_calls) == 12


def test_web_search_appends_shorthand_when_enabled(respx_mock, base_url):
    captured = {}

    def grab(request):
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]},
        )

    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=grab)
    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        web_search=True,
    ).response
    body = captured["body"].decode()
    assert '"web_search"' in body
