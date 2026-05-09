"""SSE accumulators for /v1/chat/completions and /v1/responses streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class StreamEvent:
    type: str
    data: Any
    raw: dict[str, Any] | None = None


def _iter_sse_lines(resp: httpx.Response) -> Iterator[str]:
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            payload = line[5:].lstrip()
            if payload == "[DONE]":
                return
            yield payload


async def _aiter_sse_lines(resp: httpx.Response) -> AsyncIterator[str]:
    async for raw in resp.aiter_lines():
        if not raw:
            continue
        line = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            payload = line[5:].lstrip()
            if payload == "[DONE]":
                return
            yield payload


def iter_chat_completions(resp: httpx.Response) -> Iterator[StreamEvent]:
    for payload in _iter_sse_lines(resp):
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
        if "content" in delta and delta["content"] is not None:
            yield StreamEvent(type="content.delta", data=delta["content"], raw=chunk)
        if "tool_calls" in delta and delta["tool_calls"]:
            yield StreamEvent(type="tool_calls.delta", data=delta["tool_calls"], raw=chunk)
        finish = (chunk.get("choices") or [{}])[0].get("finish_reason")
        if finish:
            yield StreamEvent(type="finish", data=finish, raw=chunk)


def iter_responses(resp: httpx.Response) -> Iterator[StreamEvent]:
    for payload in _iter_sse_lines(resp):
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        etype = event.get("type", "unknown")
        yield StreamEvent(type=etype, data=event, raw=event)


async def aiter_chat_completions(resp: httpx.Response) -> AsyncIterator[StreamEvent]:
    async for payload in _aiter_sse_lines(resp):
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
        if "content" in delta and delta["content"] is not None:
            yield StreamEvent(type="content.delta", data=delta["content"], raw=chunk)
        if "tool_calls" in delta and delta["tool_calls"]:
            yield StreamEvent(type="tool_calls.delta", data=delta["tool_calls"], raw=chunk)
        finish = (chunk.get("choices") or [{}])[0].get("finish_reason")
        if finish:
            yield StreamEvent(type="finish", data=finish, raw=chunk)


async def aiter_responses(resp: httpx.Response) -> AsyncIterator[StreamEvent]:
    async for payload in _aiter_sse_lines(resp):
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        etype = event.get("type", "unknown")
        yield StreamEvent(type=etype, data=event, raw=event)
