"""response_format with Pydantic models and dict pass-through."""

from __future__ import annotations

import json

import httpx
from pydantic import BaseModel

import encode


class Sentiment(BaseModel):
    reasoning: str
    sentiment: str


def test_pydantic_response_format_parsed(respx_mock, base_url):
    captured = {}

    def grab(request):
        captured["body"] = json.loads(request.read())
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"reasoning":"It is happy","sentiment":"positive"}',
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=grab)

    out = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "Classify: 'Great product!'"}],
        response_format=Sentiment,
    ).response

    rf = captured["body"]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Sentiment"
    assert "schema" in rf["json_schema"]

    assert isinstance(out.parsed, Sentiment)
    assert out.parsed.sentiment == "positive"


def test_dict_response_format_passthrough_no_parsed(respx_mock, base_url):
    schema = {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}
                ]
            },
        )
    )
    out = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        response_format=schema,
    ).response
    assert out.parsed is None
    assert out.content == "{}"
