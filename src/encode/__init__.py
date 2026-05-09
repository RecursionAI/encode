"""encode — Python SDK for OpenAI-compatible inference endpoints.

Designed for the self-hosted Courier inference service but works against any
OpenAI-compatible endpoint. Point ``base_url`` and ``api_key`` at your provider
via ``.env`` (auto-loaded) or pass them directly.

Example:

    import encode

    out = encode.relay(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    ).response
    print(out.content)
"""

from __future__ import annotations

from . import _config

# Auto-load .env once on import (opt-out via ENCODE_DISABLE_DOTENV=1).
_config.load_dotenv_once()

from ._version import __version__
from .client import AsyncClient, Client
from .errors import (
    AuthError,
    CourierError,
    InvalidAudioError,
    InvalidRequestError,
    InvalidResponseFormatError,
    InvalidToolCallError,
    InvalidToolChoiceError,
    MaxToolIterationsError,
    RateLimitError,
    ServerError,
    SessionError,
    SessionTimeoutError,
    TransportError,
)
from .messages import (
    AudioContent,
    Conversation,
    ImageContent,
    ImageURL,
    InputAudio,
    Message,
    Messages,
    TextContent,
    ToolCall,
    ToolCallFunction,
)
from .relay import (
    AsyncRelayHandle,
    InterceptEvent,
    RelayHandle,
    relay,
    relay_async,
)
from .responses import (
    AssistantTurn,
    RelayResponse,
    ToolCallRecord,
    Usage,
    WhisperResponse,
)
from .session import AsyncSession, CommandResult, Session
from .whisper import whisper, whisper_async

__all__ = [
    "__version__",
    "Client",
    "AsyncClient",
    "relay",
    "relay_async",
    "RelayHandle",
    "AsyncRelayHandle",
    "InterceptEvent",
    "whisper",
    "whisper_async",
    "Message",
    "Messages",
    "Conversation",
    "TextContent",
    "ImageContent",
    "ImageURL",
    "AudioContent",
    "InputAudio",
    "ToolCall",
    "ToolCallFunction",
    "RelayResponse",
    "WhisperResponse",
    "ToolCallRecord",
    "AssistantTurn",
    "Usage",
    "CourierError",
    "AuthError",
    "InvalidRequestError",
    "InvalidToolCallError",
    "InvalidToolChoiceError",
    "InvalidResponseFormatError",
    "InvalidAudioError",
    "RateLimitError",
    "ServerError",
    "TransportError",
    "MaxToolIterationsError",
    "Session",
    "AsyncSession",
    "CommandResult",
    "SessionError",
    "SessionTimeoutError",
]
