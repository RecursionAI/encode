"""Shared fixtures: respx mock, isolated client, fake credentials."""

from __future__ import annotations

import os

import pytest
import respx

import encode


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Force every test to use predictable env credentials.

    Clears any real keys the user might have set, then injects test values.
    Disables dotenv autoload so a stray .env doesn't pollute tests.
    """
    for var in (
        "ENCODE_API_KEY",
        "ENCODE_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ENCODE_DISABLE_DOTENV", "1")
    monkeypatch.setenv("ENCODE_API_KEY", "test-key")
    monkeypatch.setenv("ENCODE_BASE_URL", "https://test.courier.local")
    # reset module-level caches so each test gets a fresh client
    from encode import client as _client

    _client._default_client = None
    _client._default_async_client = None


@pytest.fixture
def base_url() -> str:
    return os.environ["ENCODE_BASE_URL"]


@pytest.fixture
def respx_mock():
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def fresh_client():
    c = encode.Client()
    yield c
    c.close()
