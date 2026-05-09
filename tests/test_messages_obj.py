"""Integration: passing a Messages object to relay() auto-updates it; lists do not."""

from __future__ import annotations

from itertools import cycle

import httpx
import pytest

import encode


def _stop_response():
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi back"},
                    "finish_reason": "stop",
                }
            ]
        },
    )


def test_messages_auto_updates_after_relay(respx_mock, base_url):
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(return_value=_stop_response())
    m = encode.Messages().system("be brief").user("hi")
    out = encode.relay(model="m", messages=m).response
    assert out.content == "hi back"
    # m should now contain the appended assistant turn
    assert len(m) == 3
    assert m[-1]["role"] == "assistant"
    assert m[-1]["content"] == "hi back"


def test_messages_carries_history_across_calls(respx_mock, base_url):
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(return_value=_stop_response())
    m = encode.Messages().user("hi")
    encode.relay(model="m", messages=m).response
    assert len(m) == 2

    # Second call: m already has [user, assistant]; we add another user turn and relay again.
    m.user("again")
    encode.relay(model="m", messages=m).response
    # Now should be [user, assistant, user, assistant]
    assert len(m) == 4
    assert [t["role"] for t in m] == ["user", "assistant", "user", "assistant"]


def test_plain_list_is_not_mutated(respx_mock, base_url):
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(return_value=_stop_response())
    plain = [{"role": "user", "content": "hi"}]
    encode.relay(model="m", messages=plain).response
    assert plain == [{"role": "user", "content": "hi"}]


def test_messages_auto_update_after_tool_loop(respx_mock, base_url):
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
                                        "function": {"name": "ping", "arguments": "{}"},
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
                        {"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
                    ]
                },
            ),
        ]
    )
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=lambda *a, **k: next(iterator))

    def ping() -> str:
        """No-op tool."""
        return "pong"

    m = encode.Messages().user("call the tool")
    encode.relay(model="m", messages=m, tools=[ping]).response
    # final history: user, assistant(tool_calls), tool, assistant(content)
    assert [t["role"] for t in m] == ["user", "assistant", "tool", "assistant"]
    assert m[-1]["content"] == "done"


def test_messages_updates_even_on_max_iterations_error(respx_mock, base_url):
    """When the loop raises MaxToolIterationsError, the partial history still gets absorbed."""
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
                                        "function": {"name": "ping", "arguments": "{}"},
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

    def ping() -> str:
        return "pong"

    m = encode.Messages().user("hi")
    with pytest.raises(encode.MaxToolIterationsError):
        encode.relay(model="m", messages=m, tools=[ping], max_tool_iterations=2).response
    # Even on raise, m should reflect the partial conversation.
    assert len(m) > 1
    roles = [t["role"] for t in m]
    assert "tool" in roles
