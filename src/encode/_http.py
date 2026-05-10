"""Low-level HTTP helpers shared by sync and async clients.

Pure helpers: build httpx Requests, parse responses, map error envelopes,
provide retry policy.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from . import errors

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)
RETRY_STATUS = {502, 503, 504}
RETRYABLE_EXC = (httpx.TransportError, httpx.ReadTimeout)


def default_headers(api_key: str, *, version: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"encode-python/{version}",
        "Accept": "application/json",
    }


def build_timeout(timeout: float | httpx.Timeout | None) -> httpx.Timeout:
    if timeout is None:
        return DEFAULT_TIMEOUT
    if isinstance(timeout, httpx.Timeout):
        return timeout
    return httpx.Timeout(connect=10.0, read=timeout, write=timeout, pool=10.0)


def parse_body(resp: httpx.Response) -> Any:
    ctype = resp.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            return resp.json()
        except Exception:
            return resp.text
    return resp.text


def raise_for_status(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    # Idempotent for already-buffered responses; required for sync streaming.
    resp.read()
    body = parse_body(resp)
    raise errors.from_envelope(body, status=resp.status_code)


async def araise_for_status(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    # Required for responses obtained via async_client.stream(...).
    await resp.aread()
    body = parse_body(resp)
    raise errors.from_envelope(body, status=resp.status_code)


def request_sync(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    json: Any = None,
    files: Any = None,
    data: Any = None,
    params: Any = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 2,
) -> httpx.Response:
    backoff = 0.25
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.request(
                method,
                url,
                json=json,
                files=files,
                data=data,
                params=params,
                headers=headers,
            )
        except RETRYABLE_EXC as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise errors.TransportError(f"transport error after {attempt + 1} attempts: {exc!r}") from exc

        if resp.status_code in RETRY_STATUS and attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code == 429 and attempt < max_retries:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
            time.sleep(wait)
            backoff *= 2
            continue
        raise_for_status(resp)
        return resp
    # unreachable; loop above either returns or raises
    raise errors.TransportError(f"request failed: {last_exc!r}")


async def request_async(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json: Any = None,
    files: Any = None,
    data: Any = None,
    params: Any = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 2,
) -> httpx.Response:
    import asyncio

    backoff = 0.25
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.request(
                method,
                url,
                json=json,
                files=files,
                data=data,
                params=params,
                headers=headers,
            )
        except RETRYABLE_EXC as exc:
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            raise errors.TransportError(f"transport error after {attempt + 1} attempts: {exc!r}") from exc

        if resp.status_code in RETRY_STATUS and attempt < max_retries:
            await asyncio.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code == 429 and attempt < max_retries:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
            await asyncio.sleep(wait)
            backoff *= 2
            continue
        raise_for_status(resp)
        return resp
    raise errors.TransportError(f"request failed: {last_exc!r}")
