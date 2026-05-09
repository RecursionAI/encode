"""Pydantic response models returned by relay() and whisper()."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    id: str
    name: str
    arguments: dict[str, Any]
    arguments_raw: str
    result: Any = None
    result_serialized: str = ""
    error: str | None = None
    iteration: int = 0
    duration_ms: float = 0.0


class AssistantTurn(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCallRecord] = []


class RelayResponse(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    content: str | None = None
    parsed: Any = None
    messages: list[dict[str, Any]] = []
    tool_calls: list[ToolCallRecord] = []
    iterations: int = 1
    finish_reason: str | None = None
    endpoint: Literal["chat", "responses"] = "chat"
    model: str = ""
    raw: Any = None
    usage: Usage | None = None


class WhisperResponse(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict[str, Any]] | None = None
    words: list[dict[str, Any]] | None = None
    raw: Any = None


__all__ = [
    "Usage",
    "ToolCallRecord",
    "AssistantTurn",
    "RelayResponse",
    "WhisperResponse",
]
