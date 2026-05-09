"""Multimodal content normalization (text/image/audio) and Messages container."""

from __future__ import annotations

import base64

import encode
from encode.messages import (
    AudioContent,
    ImageContent,
    Message,
    Messages,
    TextContent,
    to_chat_messages,
    to_responses_input,
)
from encode.responses import RelayResponse


def test_image_from_url():
    c = ImageContent.from_url("https://x/y.png", detail="high")
    d = c.model_dump()
    assert d["type"] == "image_url"
    assert d["image_url"]["url"] == "https://x/y.png"
    assert d["image_url"]["detail"] == "high"


def test_image_from_path(tmp_path):
    p = tmp_path / "pixel.png"
    raw = b"\x89PNG\r\n\x1a\n"
    p.write_bytes(raw)
    c = ImageContent.from_path(p)
    url = c.image_url.url
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == raw


def test_audio_from_path(tmp_path):
    p = tmp_path / "sample.wav"
    p.write_bytes(b"RIFF\x00")
    c = AudioContent.from_path(p)
    assert c.input_audio.format == "wav"
    assert base64.b64decode(c.input_audio.data) == b"RIFF\x00"


def test_to_chat_messages_with_pydantic_model():
    msg = Message(role="user", content=[TextContent(text="hi"), ImageContent.from_url("http://x")])
    out = to_chat_messages([msg, {"role": "system", "content": "be brief"}])
    assert out[0]["role"] == "user"
    assert isinstance(out[0]["content"], list)
    assert out[0]["content"][0]["text"] == "hi"
    assert out[1]["role"] == "system"


def test_messages_chainable_adders():
    m = (
        Messages()
        .system("be brief")
        .user("hi")
        .assistant("hello")
    )
    assert len(m) == 3
    assert m[0]["role"] == "system"
    assert m[1]["content"] == "hi"
    assert m[2]["role"] == "assistant"


def test_messages_assistant_with_tool_calls():
    m = Messages().assistant(
        None,
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    )
    assert m[0]["tool_calls"][0]["id"] == "c1"


def test_messages_tool_message():
    m = Messages().tool('{"ok":true}', tool_call_id="c1")
    assert m[0] == {"role": "tool", "tool_call_id": "c1", "content": '{"ok":true}'}


def test_messages_list_protocol():
    m = Messages([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    assert len(m) == 2
    assert bool(m) is True
    assert [d["role"] for d in m] == ["user", "assistant"]
    m.append({"role": "user", "content": "c"})
    m.extend([{"role": "assistant", "content": "d"}])
    assert len(m) == 4
    m.clear()
    assert len(m) == 0
    assert bool(m) is False


def test_messages_copy_is_deep_enough():
    m = Messages().user("hi")
    n = m.copy()
    n.user("again")
    assert len(m) == 1
    assert len(n) == 2


def test_messages_to_list_round_trips_through_chat_normalizer():
    m = Messages().system("s").user("u")
    out = to_chat_messages(m)
    assert out == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_messages_update_from_response_replaces_contents():
    m = Messages().user("old")
    fake = RelayResponse(
        content="hi",
        messages=[{"role": "user", "content": "u"}, {"role": "assistant", "content": "hi"}],
        endpoint="chat",
        model="m",
    )
    m.update(fake)
    assert len(m) == 2
    assert m[1]["role"] == "assistant"


def test_messages_accepts_pydantic_message():
    m = Messages([Message(role="user", content="hi")])
    assert m[0]["role"] == "user"
    assert m[0]["content"] == "hi"


def test_messages_exported_from_top_level():
    assert encode.Messages is Messages


def test_to_pydantic_returns_validated_conversation():
    m = encode.Messages().system("be brief").user("hi").assistant("hello")
    conv = m.to_pydantic()
    assert isinstance(conv, encode.Conversation)
    assert len(conv.messages) == 3
    assert all(isinstance(msg, encode.Message) for msg in conv.messages)
    assert conv.messages[0].role == "system"
    assert conv.messages[1].content == "hi"


def test_pydantic_round_trip_via_json():
    m = (
        encode.Messages()
        .system("be brief")
        .user("hi")
        .assistant(None, tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ])
        .tool('{"ok":true}', tool_call_id="c1")
    )
    blob = m.to_pydantic().model_dump_json()
    # Round trip through JSON like a DB read/write would
    restored = encode.Messages.from_pydantic(encode.Conversation.model_validate_json(blob))
    assert len(restored) == len(m)
    assert restored[0]["role"] == "system"
    assert restored[2]["tool_calls"][0]["id"] == "c1"
    assert restored[3]["tool_call_id"] == "c1"


def test_from_pydantic_creates_independent_messages():
    conv = encode.Conversation(messages=[encode.Message(role="user", content="hi")])
    m = encode.Messages.from_pydantic(conv)
    m.user("again")
    assert len(conv.messages) == 1   # original untouched
    assert len(m) == 2


def test_to_pydantic_with_image_content_round_trips():
    img = encode.ImageContent.from_url("https://example.com/x.png", detail="high")
    m = encode.Messages().user([encode.TextContent(text="caption?"), img])
    conv = m.to_pydantic()
    blob = conv.model_dump_json()
    restored = encode.Conversation.model_validate_json(blob)
    parts = restored.messages[0].content
    assert isinstance(parts, list)
    assert any(getattr(p, "type", None) == "image_url" for p in parts)


def test_to_responses_input_converts_messages_and_tool_results():
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "{\"ok\":true}"},
    ]
    items = to_responses_input(msgs)
    types = [i["type"] for i in items]
    assert "message" in types
    assert "function_call" in types
    assert "function_call_output" in types
    fco = next(i for i in items if i["type"] == "function_call_output")
    assert fco["call_id"] == "c1"
    assert fco["output"] == "{\"ok\":true}"
