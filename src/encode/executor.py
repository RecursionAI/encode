"""Tool execution seam — the brain↔hands boundary.

Per the Managed Agents paper, each hand is reachable through a uniform
``execute(name, input) → string`` interface. The :class:`ToolExecutor`
Protocol expresses that interface; :class:`LocalToolExecutor` is the default
implementation that runs Python callables in-process (the SDK's current
behavior).

Users can swap in their own executors for:

- remote dispatch (HTTP/gRPC to a tools service)
- containerized execution (run in a sandbox per call)
- MCP server proxying
- sub-agent delegation (each tool call spawns a child agent)

``relay()`` accepts ``tool_executor=`` and routes tool calls through whatever
executor is provided. When omitted, relay constructs a ``LocalToolExecutor``
from ``tools=`` and behavior is unchanged.

The :class:`CredentialProvider` Protocol is a stub for the future
``VaultedToolExecutor`` work — credentials never need to be visible to the
harness or the tool callables themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from . import tools as _tools


@runtime_checkable
class ToolExecutor(Protocol):
    """Uniform interface for executing a named tool call.

    Implementations must accept a ``name`` and an ``input`` dict (the
    parsed arguments from the model's tool call) and return a JSON-serialized
    string (the content placed in the ``tool`` / ``function_call_output``
    message). Errors should be returned as a serialized error payload rather
    than raised — the harness treats raised exceptions as fatal.
    """

    def execute(self, name: str, input: dict[str, Any]) -> ExecutionResult: ...

    async def execute_async(
        self, name: str, input: dict[str, Any]
    ) -> ExecutionResult: ...


class ExecutionResult:
    """Outcome of a single tool execution.

    Carries the raw ``result`` (for the response object), the serialized
    string sent back to the model, optional ``error`` text, and a
    ``duration_ms`` measurement. Lightweight class (not Pydantic) because
    it's a per-call ephemeral value, not something to persist.
    """

    __slots__ = ("result", "result_serialized", "error", "duration_ms")

    def __init__(
        self,
        *,
        result: Any = None,
        result_serialized: str = "",
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        self.result = result
        self.result_serialized = result_serialized
        self.error = error
        self.duration_ms = duration_ms


class LocalToolExecutor:
    """Default executor — runs Python callables in-process.

    Wraps the existing ``tools.safe_call`` / ``tools.safe_call_async`` so
    behavior is identical to ``relay()`` without an explicit executor. Pass a
    ``{name: callable}`` mapping at construction.
    """

    def __init__(self, tools: dict[str, Callable[..., Any]]) -> None:
        self._tools = dict(tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def has(self, name: str) -> bool:
        return name in self._tools

    def execute(self, name: str, input: dict[str, Any]) -> ExecutionResult:
        fn = self._tools.get(name)
        if fn is None:
            err = f"no Python callable bound for tool {name!r}"
            return ExecutionResult(
                result={"error": err},
                result_serialized=_tools.serialize_tool_result({"error": err}),
                error=err,
                duration_ms=0.0,
            )
        result, error, duration = _tools.safe_call(fn, input)
        return ExecutionResult(
            result=result,
            result_serialized=_tools.serialize_tool_result(result),
            error=error,
            duration_ms=duration,
        )

    async def execute_async(
        self, name: str, input: dict[str, Any]
    ) -> ExecutionResult:
        fn = self._tools.get(name)
        if fn is None:
            err = f"no Python callable bound for tool {name!r}"
            return ExecutionResult(
                result={"error": err},
                result_serialized=_tools.serialize_tool_result({"error": err}),
                error=err,
                duration_ms=0.0,
            )
        result, error, duration = await _tools.safe_call_async(fn, input)
        return ExecutionResult(
            result=result,
            result_serialized=_tools.serialize_tool_result(result),
            error=error,
            duration_ms=duration,
        )


class CredentialProvider(Protocol):
    """Stub Protocol for future vault-backed credential injection.

    A ``VaultedToolExecutor`` will resolve credentials at call time from a
    provider rather than letting them sit in the harness's memory. Not used
    in v1; documented here so the seam is visible.
    """

    def get(self, key: str) -> str: ...


__all__ = [
    "ToolExecutor",
    "LocalToolExecutor",
    "ExecutionResult",
    "CredentialProvider",
]
