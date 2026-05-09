"""Resolve api_key and base_url for any OpenAI-compatible endpoint.

Resolution order (highest wins):

1. Explicit kwarg on relay() / whisper() / Client(...).
2. ``ENCODE_API_KEY`` / ``ENCODE_BASE_URL`` env vars.
3. ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` env vars.

A ``.env`` file is auto-loaded once on package import via python-dotenv with
``override=False`` so real shell env always wins. Disable with
``ENCODE_DISABLE_DOTENV=1``.
"""

from __future__ import annotations

import os

from .errors import AuthError, InvalidRequestError

_DOTENV_LOADED = False


def load_dotenv_once() -> None:
    """Auto-load .env exactly once. Idempotent.

    Walks up from CWD. Does not override existing environment variables.
    Skipped entirely if ENCODE_DISABLE_DOTENV is set.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    if os.getenv("ENCODE_DISABLE_DOTENV"):
        return
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


def resolve_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    key = os.getenv("ENCODE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise AuthError(
            "missing API key — pass api_key=... or set ENCODE_API_KEY (or OPENAI_API_KEY)"
        )
    return key


def resolve_base_url(explicit: str | None) -> str:
    url = explicit or os.getenv("ENCODE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not url:
        raise InvalidRequestError(
            "missing base_url — pass base_url=... or set ENCODE_BASE_URL (or OPENAI_BASE_URL)"
        )
    return url.rstrip("/")
