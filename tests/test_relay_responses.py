"""/v1/responses paths: auto-routing, function_call loop, intercept parity."""

from __future__ import annotations

from itertools import cycle

import httpx

import encode


def search(query: str) -> dict:
    """Search the index."""
    return {"hits": [{"title": query.upper()}]}


def test_responses_endpoint_auto_selected_from_input(respx_mock, base_url):
    captured = {}

    def grab(request):
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "object": "response",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello world"}],
                    }
                ],
            },
        )

    respx_mock.post(f"{base_url}/v1/responses").mock(side_effect=grab)
    out = encode.relay(
        model="m",
        input="say hi",
    ).response
    assert out.endpoint == "responses"
    assert out.content == "hello world"
    assert "/v1/responses" in captured["url"]


def test_responses_tool_loop(respx_mock, base_url):
    iterator = cycle(
        [
            httpx.Response(
                200,
                json={
                    "object": "response",
                    "output": [
                        {
                            "id": "fc_1",
                            "type": "function_call",
                            "call_id": "call_abc",
                            "name": "search",
                            "arguments": '{"query":"sdk"}',
                        }
                    ],
                },
            ),
            httpx.Response(
                200,
                json={
                    "object": "response",
                    "output": [
                        {
                            "id": "msg_2",
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "found 1 hit"}],
                        }
                    ],
                },
            ),
        ]
    )
    respx_mock.post(f"{base_url}/v1/responses").mock(side_effect=lambda *a, **k: next(iterator))

    seen = []
    out = encode.relay(
        model="m",
        input="search for sdk",
        tools=[search],
        on_intercept=lambda ev: seen.append(ev),
    ).response
    assert out.endpoint == "responses"
    assert out.content == "found 1 hit"
    assert out.iterations == 2
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "search"
    assert out.tool_calls[0].result == {"hits": [{"title": "SDK"}]}
    assert len(seen) == 1
    assert seen[0].endpoint == "responses"


def test_responses_explicit_endpoint_with_messages(respx_mock, base_url):
    """Forcing endpoint='responses' while passing messages converts them to input items."""
    captured = {}

    def grab(request):
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "object": "response",
                "output": [
                    {
                        "id": "m",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            },
        )

    respx_mock.post(f"{base_url}/v1/responses").mock(side_effect=grab)
    encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        endpoint="responses",
    ).response
    body = captured["body"].decode()
    # input items should contain a message-typed entry
    assert '"input"' in body
    assert '"type":"message"' in body or '"type": "message"' in body
