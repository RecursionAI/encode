"""whisper() / whisper_async() — multipart transcription and translation."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from . import _http, errors
from .responses import WhisperResponse

ALLOWED_EXTS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
MAX_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_FORMATS = {"json", "verbose_json", "text", "srt", "vtt"}


def _resolve_file(
    file: str | PathLike[str] | bytes | tuple[str, bytes],
) -> tuple[str, bytes]:
    if isinstance(file, tuple) and len(file) == 2:
        name, data = file
        if not isinstance(data, (bytes, bytearray)):
            raise errors.InvalidAudioError(
                "file tuple must be (filename, bytes)", code="invalid_audio"
            )
        return name, bytes(data)
    if isinstance(file, (bytes, bytearray)):
        return "audio.wav", bytes(file)
    p = Path(file)  # type: ignore[arg-type]
    if not p.exists():
        raise errors.InvalidAudioError(f"file not found: {p}", code="invalid_audio")
    return p.name, p.read_bytes()


def _validate(
    name: str,
    data: bytes,
    response_format: str,
    timestamp_granularities: Sequence[str] | None,
) -> None:
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in ALLOWED_EXTS:
        raise errors.InvalidAudioError(
            f"unsupported audio extension '{suffix}'; allowed: {sorted(ALLOWED_EXTS)}",
            code="invalid_audio",
        )
    if len(data) > MAX_BYTES:
        raise errors.InvalidRequestError(
            f"file too large: {len(data)} bytes (max {MAX_BYTES})",
            code="invalid_request_error",
        )
    if response_format not in ALLOWED_FORMATS:
        raise errors.CourierError(
            f"invalid response_format '{response_format}'; allowed: {sorted(ALLOWED_FORMATS)}",
            code="invalid_response_format",
            type="invalid_response_format",
        )
    if timestamp_granularities and response_format != "verbose_json":
        raise errors.InvalidRequestError(
            "timestamp_granularities is only allowed with response_format='verbose_json'",
            code="invalid_request_error",
        )


def _build_form(
    *,
    model: str,
    response_format: str,
    language: str | None,
    timestamp_granularities: Sequence[str] | None,
    prompt: str | None,
    temperature: float | None,
) -> list[tuple[str, tuple[None, str]]]:
    form: list[tuple[str, tuple[None, str]]] = [
        ("model", (None, model)),
        ("response_format", (None, response_format)),
    ]
    if language:
        form.append(("language", (None, language)))
    if prompt:
        form.append(("prompt", (None, prompt)))
    if temperature is not None:
        form.append(("temperature", (None, str(temperature))))
    if timestamp_granularities:
        for g in timestamp_granularities:
            form.append(("timestamp_granularities[]", (None, g)))
    return form


def _path_for(mode: str) -> str:
    return "/v1/audio/translations" if mode == "translate" else "/v1/audio/transcriptions"


def _parse_response(body: Any, response_format: str) -> WhisperResponse:
    if response_format in ("text", "srt", "vtt"):
        text = body if isinstance(body, str) else str(body)
        return WhisperResponse(text=text, raw=text)
    if not isinstance(body, dict):
        return WhisperResponse(text=str(body), raw=body)
    return WhisperResponse(
        text=str(body.get("text", "")),
        language=body.get("language"),
        duration=body.get("duration"),
        segments=body.get("segments"),
        words=body.get("words"),
        raw=body,
    )


def whisper(
    *,
    file: str | PathLike[str] | bytes | tuple[str, bytes],
    mode: Literal["transcribe", "translate"] = "transcribe",
    model: str = "whisper-1",
    response_format: Literal["json", "verbose_json", "text", "srt", "vtt"] = "json",
    language: str | None = None,
    timestamp_granularities: Sequence[Literal["word", "segment"]] | None = None,
    prompt: str | None = None,
    temperature: float | None = None,
    client: Any = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = 120.0,
) -> WhisperResponse:
    name, data = _resolve_file(file)
    _validate(name, data, response_format, timestamp_granularities)

    if client is None:
        from .client import Client, get_default_client

        if api_key or base_url or (timeout is not None and timeout != 60.0):
            client = Client(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            client = get_default_client()

    files = {"file": (name, data)}
    form = _build_form(
        model=model,
        response_format=response_format,
        language=language,
        timestamp_granularities=timestamp_granularities,
        prompt=prompt,
        temperature=temperature,
    )
    resp = _http.request_sync(
        client._http,
        "POST",
        _path_for(mode),
        files=files,
        data=dict(form),  # httpx accepts list-of-tuples too via 'data'
        max_retries=client.max_retries,
    )
    return _parse_response(_http.parse_body(resp), response_format)


async def whisper_async(
    *,
    file: str | PathLike[str] | bytes | tuple[str, bytes],
    mode: Literal["transcribe", "translate"] = "transcribe",
    model: str = "whisper-1",
    response_format: Literal["json", "verbose_json", "text", "srt", "vtt"] = "json",
    language: str | None = None,
    timestamp_granularities: Sequence[Literal["word", "segment"]] | None = None,
    prompt: str | None = None,
    temperature: float | None = None,
    client: Any = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = 120.0,
) -> WhisperResponse:
    name, data = _resolve_file(file)
    _validate(name, data, response_format, timestamp_granularities)

    if client is None:
        from .client import AsyncClient, get_default_async_client

        if api_key or base_url or (timeout is not None and timeout != 60.0):
            client = AsyncClient(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            client = get_default_async_client()

    files = {"file": (name, data)}
    form = _build_form(
        model=model,
        response_format=response_format,
        language=language,
        timestamp_granularities=timestamp_granularities,
        prompt=prompt,
        temperature=temperature,
    )
    resp = await _http.request_async(
        client._http,
        "POST",
        _path_for(mode),
        files=files,
        data=dict(form),
        max_retries=client.max_retries,
    )
    return _parse_response(_http.parse_body(resp), response_format)
