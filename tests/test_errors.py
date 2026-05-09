"""Error envelope -> exception class mapping."""

from __future__ import annotations

from encode import errors


def test_envelope_invalid_tool_call():
    e = errors.from_envelope(
        {"error": {"message": "bad", "type": "invalid_request_error", "code": "invalid_tool_call"}},
        status=400,
    )
    assert isinstance(e, errors.InvalidToolCallError)
    assert e.code == "invalid_tool_call"


def test_envelope_invalid_audio():
    e = errors.from_envelope(
        {"error": {"message": "bad audio", "type": "invalid_audio"}},
        status=400,
    )
    assert isinstance(e, errors.InvalidAudioError)


def test_envelope_falls_back_to_status_class():
    e = errors.from_envelope({"error": {"message": "rate limited"}}, status=429)
    assert isinstance(e, errors.RateLimitError)


def test_detail_envelope_falls_back():
    e = errors.from_envelope({"detail": "nope"}, status=400)
    assert isinstance(e, errors.InvalidRequestError)
    assert "nope" in str(e)


def test_string_body_falls_back():
    e = errors.from_envelope("Internal Server Error", status=500)
    assert isinstance(e, errors.ServerError)
