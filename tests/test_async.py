"""relay_async() and AsyncClient parity with sync."""

from __future__ import annotations

from itertools import cycle

import httpx
import pytest

import encode


async def get_weather(city: str) -> dict:
    """Async tool example."""
    return {"city": city, "temp_f": 70}


@pytest.mark.asyncio
async def test_relay_async_basic(respx_mock, base_url):
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ahoy"}, "finish_reason": "stop"}
                ]
            },
        )
    )
    out = await encode.relay_async(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert out.content == "ahoy"


@pytest.mark.asyncio
async def test_relay_async_tool_loop_with_async_callable(respx_mock, base_url):
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
                                        "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
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
                        {"index": 0, "message": {"role": "assistant", "content": "70F"}, "finish_reason": "stop"}
                    ]
                },
            ),
        ]
    )
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=lambda *a, **k: next(iterator))

    seen: list[int] = []

    async def cb(event):
        seen.append(event.iteration)

    out = await encode.relay_async(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[get_weather],
    ).intercept(cb)
    assert out.content == "70F"
    assert out.iterations == 2
    assert seen == [0]
    assert out.tool_calls[0].result == {"city": "NYC", "temp_f": 70}


@pytest.mark.asyncio
async def test_async_client_context_manager():
    async with encode.AsyncClient() as c:
        assert c.api_key == "test-key"
        assert c.base_url == "https://test.courier.local"
