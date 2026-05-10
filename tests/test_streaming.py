"""SSE accumulators for both endpoints."""

from __future__ import annotations

import httpx
import pytest

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


# ---------------------------------------------------------------------------
# Streaming + tool-call loop
# ---------------------------------------------------------------------------

from itertools import cycle  # noqa: E402


def _get_weather(city: str) -> dict:
    """Get current weather by city."""
    return {"city": city, "temp_f": 72}


def _explode(reason: str = "boom") -> dict:
    raise RuntimeError(reason)


_TOOL_CALL_STREAM = b"".join(
    [
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"_get_weather","arguments":""}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"Denver\\"}"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
)

_FINAL_CONTENT_STREAM = b"".join(
    [
        b'data: {"choices":[{"delta":{"content":"It\'s "}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"72F."}}]}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
)


def _stream_two_step(respx_mock, base_url):
    iterator = cycle(
        [
            httpx.Response(200, content=_TOOL_CALL_STREAM),
            httpx.Response(200, content=_FINAL_CONTENT_STREAM),
        ]
    )
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        side_effect=lambda *a, **k: next(iterator)
    )


def test_chat_stream_with_tools_two_iterations(respx_mock, base_url):
    _stream_two_step(respx_mock, base_url)
    handle = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "weather?"}],
        tools=[_get_weather],
        stream=True,
    )
    events = list(handle)
    types = [e.type for e in events]
    # Tool dispatch happens before final content
    assert "tool_call.start" in types
    assert "tool_call.result" in types
    assert "iteration.end" in types
    assert types.index("tool_call.start") < types.index("content.delta")
    # Final answer arrives as content deltas
    text = "".join(e.data for e in events if e.type == "content.delta")
    assert text == "It's 72F."
    assert events[-1].type == "finish"
    # Tool was actually called with parsed args, accumulated across deltas
    start = next(e for e in events if e.type == "tool_call.start")
    assert start.data["name"] == "_get_weather"
    assert start.data["arguments"] == {"city": "Denver"}
    result = next(e for e in events if e.type == "tool_call.result")
    assert result.data["result"] == {"city": "Denver", "temp_f": 72}


def test_chat_stream_with_tools_mutates_messages(respx_mock, base_url):
    _stream_two_step(respx_mock, base_url)
    m = encode.Messages().user("weather?")
    handle = encode.relay(model="m", messages=m, tools=[_get_weather], stream=True)
    list(handle)  # drain
    # After streaming completes, history was absorbed into Messages
    assert len(m) >= 4  # user + assistant(tool_call) + tool + assistant(final)
    roles = [msg["role"] for msg in m]
    assert roles[0] == "user"
    assert "tool" in roles
    assert roles[-1] == "assistant"
    assert m[-1]["content"] == "It's 72F."


def test_chat_stream_with_tools_max_iterations(respx_mock, base_url):
    # Always return a tool-call stream so the loop never finishes naturally
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_TOOL_CALL_STREAM)
    )
    handle = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "loop"}],
        tools=[_get_weather],
        stream=True,
        max_tool_iterations=2,
    )
    with pytest.raises(encode.MaxToolIterationsError) as excinfo:
        list(handle)
    partial = excinfo.value.partial
    assert partial is not None
    assert partial.iterations >= 2


def test_chat_stream_tool_error_emits_error_event(respx_mock, base_url):
    explode_stream = b"".join(
        [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"_explode","arguments":"{}"}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    iterator = cycle(
        [
            httpx.Response(200, content=explode_stream),
            httpx.Response(200, content=_FINAL_CONTENT_STREAM),
        ]
    )
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        side_effect=lambda *a, **k: next(iterator)
    )
    handle = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "go"}],
        tools=[_explode],
        stream=True,
    )
    events = list(handle)
    types = [e.type for e in events]
    assert "tool_call.error" in types
    err = next(e for e in events if e.type == "tool_call.error")
    assert "RuntimeError" in err.data["error"]
    # Loop continues to the next iteration after the error
    assert "It's 72F." in "".join(e.data for e in events if e.type == "content.delta")


async def test_chat_stream_with_tools_async(respx_mock, base_url):
    _stream_two_step(respx_mock, base_url)
    handle = encode.relay_async(
        model="m",
        messages=[{"role": "user", "content": "weather?"}],
        tools=[_get_weather],
        stream=True,
    )
    types: list[str] = []
    text_parts: list[str] = []
    async for ev in handle:
        types.append(ev.type)
        if ev.type == "content.delta":
            text_parts.append(ev.data)
    assert "tool_call.start" in types
    assert "tool_call.result" in types
    assert "".join(text_parts) == "It's 72F."


def test_responses_stream_with_tools_two_iterations(respx_mock, base_url):
    tool_call_resp_stream = b"".join(
        [
            b'data: {"type":"response.created","response":{}}\n\n',
            b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"_get_weather","arguments":""}}\n\n',
            b'data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\\"city\\":\\"Denver\\"}"}\n\n',
            b'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"_get_weather","arguments":"{\\"city\\":\\"Denver\\"}"}}\n\n',
            b'data: {"type":"response.completed","response":{"output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"_get_weather","arguments":"{\\"city\\":\\"Denver\\"}"}]}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    final_resp_stream = b"".join(
        [
            b'data: {"type":"response.created","response":{}}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"It\'s 72F."}\n\n',
            b'data: {"type":"response.completed","response":{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"It\'s 72F."}]}]}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    iterator = cycle(
        [
            httpx.Response(200, content=tool_call_resp_stream),
            httpx.Response(200, content=final_resp_stream),
        ]
    )
    respx_mock.post(f"{base_url}/v1/responses").mock(
        side_effect=lambda *a, **k: next(iterator)
    )
    handle = encode.relay(
        model="m",
        input="weather?",
        tools=[_get_weather],
        stream=True,
        endpoint="responses",
    )
    events = list(handle)
    types = [e.type for e in events]
    assert "tool_call.start" in types
    assert "tool_call.result" in types
    assert "content.delta" in types
    text = "".join(e.data for e in events if e.type == "content.delta")
    assert text == "It's 72F."


# ---------------------------------------------------------------------------
# Streaming + error envelope (regression: httpx.ResponseNotRead)
# ---------------------------------------------------------------------------


_AUTH_ENVELOPE = {"error": {"message": "bad key", "type": "auth_error"}}


def test_stream_chat_sync_error_status_parses_envelope(respx_mock, base_url):
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(401, json=_AUTH_ENVELOPE)
    )
    handle = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    with pytest.raises(encode.errors.AuthError):
        list(handle)


def test_stream_responses_sync_error_status_parses_envelope(respx_mock, base_url):
    respx_mock.post(f"{base_url}/v1/responses").mock(
        return_value=httpx.Response(401, json=_AUTH_ENVELOPE)
    )
    handle = encode.relay(
        model="m",
        input="hi",
        stream=True,
        endpoint="responses",
    )
    with pytest.raises(encode.errors.AuthError):
        list(handle)


@pytest.mark.asyncio
async def test_stream_chat_async_error_status_parses_envelope(respx_mock, base_url):
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(401, json=_AUTH_ENVELOPE)
    )
    handle = encode.relay_async(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    with pytest.raises(encode.errors.AuthError):
        async for _ in handle:
            pass


@pytest.mark.asyncio
async def test_stream_responses_async_error_status_parses_envelope(respx_mock, base_url):
    respx_mock.post(f"{base_url}/v1/responses").mock(
        return_value=httpx.Response(401, json=_AUTH_ENVELOPE)
    )
    handle = encode.relay_async(
        model="m",
        input="hi",
        stream=True,
        endpoint="responses",
    )
    with pytest.raises(encode.errors.AuthError):
        async for _ in handle:
            pass
