"""Client / AsyncClient — thin wrappers around httpx with config resolution."""

from __future__ import annotations

from typing import Any

import httpx

from . import _config, _http
from ._version import __version__


class Client:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        _config.load_dotenv_once()
        self._api_key = _config.resolve_api_key(api_key)
        self._base_url = _config.resolve_base_url(base_url)
        self._timeout = _http.build_timeout(timeout)
        self.max_retries = max_retries
        headers = _http.default_headers(self._api_key, version=__version__)
        if default_headers:
            headers.update(default_headers)
        self._owns_client = http_client is None
        self._http: httpx.Client = http_client or httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=headers,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key(self) -> str:
        return self._api_key

    def relay(self, **kwargs: Any) -> Any:
        from .relay import relay

        return relay(client=self, **kwargs)

    def whisper(self, **kwargs: Any) -> Any:
        from .whisper import whisper

        return whisper(client=self, **kwargs)

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        _config.load_dotenv_once()
        self._api_key = _config.resolve_api_key(api_key)
        self._base_url = _config.resolve_base_url(base_url)
        self._timeout = _http.build_timeout(timeout)
        self.max_retries = max_retries
        headers = _http.default_headers(self._api_key, version=__version__)
        if default_headers:
            headers.update(default_headers)
        self._owns_client = http_client is None
        self._http: httpx.AsyncClient = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=headers,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key(self) -> str:
        return self._api_key

    def relay(self, **kwargs: Any) -> Any:
        from .relay import relay_async

        return relay_async(client=self, **kwargs)

    async def whisper(self, **kwargs: Any) -> Any:
        from .whisper import whisper_async

        return await whisper_async(client=self, **kwargs)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


_default_client: Client | None = None
_default_async_client: AsyncClient | None = None


def get_default_client() -> Client:
    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


def get_default_async_client() -> AsyncClient:
    global _default_async_client
    if _default_async_client is None:
        _default_async_client = AsyncClient()
    return _default_async_client
