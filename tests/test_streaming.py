"""SSE accumulators for both endpoints."""

from __future__ import annotations

import httpx

import encode
from encode._streaming import iter_chat_completions, iter_responses


def _sse_response(chunks: list[bytes]) -> httpx.Response:
    return httpx.Response(200, content=b"".join(chunks))


def test_chat_completions_accumulator():
    body = b"".join(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":" there"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    resp = httpx.Response(200, content=body)
    events = list(iter_chat_completions(resp))
    types = [e.type for e in events]
    assert "content.delta" in types
    assert events[-1].type == "finish"
    assert events[-1].data == "stop"
    assert "".join(e.data for e in events if e.type == "content.delta") == "hi there"


def test_responses_accumulator_passthrough_event_types():
    body = b"".join(
        [
            b'data: {"type":"response.created","response":{}}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n',
            b'data: {"type":"response.completed","response":{}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    resp = httpx.Response(200, content=body)
    events = list(iter_responses(resp))
    types = [e.type for e in events]
    assert "response.created" in types
    assert "response.output_text.delta" in types
    assert "response.completed" in types


def test_relay_stream_iterates(respx_mock, base_url):
    body = b"".join(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"b"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=body)
    )
    handle = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    chunks = [ev for ev in handle if ev.type == "content.delta"]
    assert "".join(e.data for e in chunks) == "ab"
