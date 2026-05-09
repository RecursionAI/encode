"""Basic /v1/chat/completions paths: simple call, message echo, error envelopes."""

from __future__ import annotations

import httpx
import pytest

import encode


def test_basic_chat_returns_content(respx_mock, base_url):
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi there"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )
    )
    out = encode.relay(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    ).response
    assert out.content == "hi there"
    assert out.endpoint == "chat"
    assert out.model == "test-model"
    assert out.iterations == 1
    assert out.finish_reason == "stop"
    assert out.usage and out.usage.total_tokens == 7
    # last entry of messages should be the assistant turn
    assert out.messages[-1]["role"] == "assistant"
    assert out.messages[-1]["content"] == "hi there"


def test_explicit_kwargs_override_env(respx_mock, monkeypatch):
    monkeypatch.setenv("ENCODE_BASE_URL", "https://wrong.example")
    route = respx_mock.post("https://right.example/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
            },
        )
    )
    out = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        api_key="other-key",
        base_url="https://right.example",
    ).response
    assert out.content == "ok"
    assert route.called


def test_error_envelope_maps_to_typed_exception(respx_mock, base_url):
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "missing model",
                    "type": "invalid_request_error",
                    "code": "invalid_request_error",
                }
            },
        )
    )
    with pytest.raises(encode.InvalidRequestError) as exc_info:
        encode.relay(model="m", messages=[{"role": "user", "content": "hi"}]).response
    assert "missing model" in str(exc_info.value)


def test_auth_error_on_401(respx_mock, base_url):
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key", "type": "authentication_error"}}),
    )
    with pytest.raises(encode.AuthError):
        encode.relay(model="m", messages=[{"role": "user", "content": "hi"}]).response


def test_missing_api_key_raises_auth_error(monkeypatch):
    monkeypatch.delenv("ENCODE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from encode import client as _c

    _c._default_client = None
    with pytest.raises(encode.AuthError):
        encode.relay(model="m", messages=[{"role": "user", "content": "hi"}]).response


def test_handle_memoizes_response(respx_mock, base_url):
    route = respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]},
        )
    )
    h = encode.relay(model="m", messages=[{"role": "user", "content": "hi"}])
    out1 = h.response
    out2 = h.response
    assert out1 is out2
    assert route.call_count == 1


def test_response_format_with_stream_raises():
    from pydantic import BaseModel

    class S(BaseModel):
        x: int

    with pytest.raises(ValueError, match="response_format and stream"):
        encode.relay(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            response_format=S,
            stream=True,
        )
