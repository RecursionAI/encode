"""Whisper multipart upload, transcribe vs translate, error mapping."""

from __future__ import annotations

import httpx
import pytest

import encode


def test_transcribe_json(respx_mock, base_url, tmp_path):
    p = tmp_path / "audio.wav"
    p.write_bytes(b"RIFFsome-bytes")

    captured = {}

    def grab(request):
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers.get("content-type", "")
        return httpx.Response(200, json={"text": "Hello world."})

    respx_mock.post(f"{base_url}/v1/audio/transcriptions").mock(side_effect=grab)
    out = encode.whisper(file=p)
    assert out.text == "Hello world."
    assert "/v1/audio/transcriptions" in captured["url"]
    assert captured["content_type"].startswith("multipart/form-data")


def test_translate_routes_to_translations(respx_mock, base_url, tmp_path):
    p = tmp_path / "audio.mp3"
    p.write_bytes(b"\x00\x00\x00")
    captured = {}

    def grab(request):
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"text": "translated"})

    respx_mock.post(f"{base_url}/v1/audio/translations").mock(side_effect=grab)
    out = encode.whisper(file=p, mode="translate")
    assert out.text == "translated"
    assert "/v1/audio/translations" in captured["url"]


def test_unsupported_extension_rejected(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_bytes(b"x")
    with pytest.raises(encode.InvalidAudioError):
        encode.whisper(file=p)


def test_too_large_rejected(tmp_path):
    p = tmp_path / "audio.wav"
    # 26MB
    p.write_bytes(b"\x00" * (26 * 1024 * 1024))
    with pytest.raises(encode.InvalidRequestError):
        encode.whisper(file=p)


def test_timestamp_granularities_requires_verbose_json(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"R")
    with pytest.raises(encode.InvalidRequestError):
        encode.whisper(file=p, response_format="json", timestamp_granularities=["word"])


def test_verbose_json_returns_structured(respx_mock, base_url, tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"R")
    respx_mock.post(f"{base_url}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            200,
            json={
                "text": "hi",
                "language": "en",
                "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi"}],
            },
        )
    )
    out = encode.whisper(file=p, response_format="verbose_json")
    assert out.text == "hi"
    assert out.language == "en"
    assert out.segments and out.segments[0]["text"] == "hi"
