"""Typed exception hierarchy for encode.

Maps the OpenAI-compatible error envelope ``{"error": {"message", "type", "code"}}``
to specific exception classes. Falls back to ``CourierError`` for non-envelope bodies
(some compatible servers return ``{"detail": "..."}`` or plain text).
"""

from __future__ import annotations

from typing import Any


class CourierError(Exception):
    def __init__(
        self,
        message: str,
        *,
        type: str | None = None,
        code: str | None = None,
        status: int | None = None,
        raw: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.type = type
        self.code = code
        self.status = status
        self.raw = raw

    def __repr__(self) -> str:  # pragma: no cover - trivial
        parts = [f"message={self.message!r}"]
        if self.type:
            parts.append(f"type={self.type!r}")
        if self.code:
            parts.append(f"code={self.code!r}")
        if self.status is not None:
            parts.append(f"status={self.status}")
        return f"{type(self).__name__}({', '.join(parts)})"


class AuthError(CourierError):
    """401 / missing credentials."""


class InvalidRequestError(CourierError):
    """400-level client error."""


class InvalidToolCallError(InvalidRequestError):
    """code=invalid_tool_call — arguments failed strict JSON validation."""


class InvalidToolChoiceError(InvalidRequestError):
    """code=invalid_tool_choice — forced tool name not in tools list."""


class InvalidResponseFormatError(InvalidRequestError):
    """type=invalid_response_format — unsupported response_format."""


class InvalidAudioError(InvalidRequestError):
    """type=invalid_audio — bad multipart upload, unsupported format, etc."""


class RateLimitError(CourierError):
    """429."""


class ServerError(CourierError):
    """5xx."""


class TransportError(CourierError):
    """Network / timeout failure that exhausted retries."""


class MaxToolIterationsError(CourierError):
    """SDK-side: tool loop exceeded max_tool_iterations.

    Carries the partial RelayResponse so callers can inspect what happened.
    """

    def __init__(self, message: str, *, partial: Any = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.partial = partial


class SessionError(CourierError):
    """Persistent shell session failure (spawn, dead process, unsupported platform)."""


class SessionTimeoutError(SessionError):
    """A session command exceeded its timeout.

    Carries ``partial_output`` (whatever was emitted before the cutoff) and
    ``command`` so callers can salvage what happened.
    """

    def __init__(
        self,
        message: str,
        *,
        partial_output: str = "",
        command: str = "",
        **kw: Any,
    ) -> None:
        super().__init__(message, **kw)
        self.partial_output = partial_output
        self.command = command


_CODE_MAP: dict[str, type[CourierError]] = {
    "invalid_tool_call": InvalidToolCallError,
    "invalid_tool_choice": InvalidToolChoiceError,
    "invalid_response_format": InvalidResponseFormatError,
    "invalid_audio": InvalidAudioError,
}

_TYPE_MAP: dict[str, type[CourierError]] = {
    "invalid_request_error": InvalidRequestError,
    "invalid_audio": InvalidAudioError,
    "invalid_response_format": InvalidResponseFormatError,
}


def from_envelope(
    body: Any, status: int | None = None
) -> CourierError:
    """Build a typed exception from a parsed response body and HTTP status.

    Tries the OpenAI envelope first, then falls back to ``{"detail": ...}``,
    then to a plain string body.
    """
    err: dict[str, Any] | None = None
    detail: str | None = None
    if isinstance(body, dict):
        if isinstance(body.get("error"), dict):
            err = body["error"]
        elif "detail" in body:
            detail = str(body["detail"])

    if err is not None:
        message = str(err.get("message") or "request failed")
        code = err.get("code")
        type_ = err.get("type")
        cls = (
            _CODE_MAP.get(str(code) if code else "")
            or _TYPE_MAP.get(str(type_) if type_ else "")
            or _from_status(status)
        )
        return cls(message, type=type_, code=code, status=status, raw=body)

    message = detail or (body if isinstance(body, str) else f"HTTP {status}")
    cls = _from_status(status)
    return cls(message, status=status, raw=body)


def _from_status(status: int | None) -> type[CourierError]:
    if status is None:
        return CourierError
    if status == 401 or status == 403:
        return AuthError
    if status == 429:
        return RateLimitError
    if 400 <= status < 500:
        return InvalidRequestError
    if 500 <= status < 600:
        return ServerError
    return CourierError
