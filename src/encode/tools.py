"""Tool registry helpers for the relay() loop.

Splits the user's ``tools=`` list into:

- ``tool_dicts``: list[dict] sent to the model
- ``tool_index``: name -> Python callable (for execution)
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

from . import errors
from ._schema import callable_to_tool_dict


def tool_name(tool: Any) -> str:
    """Extract the registered name for a tool entry.

    Accepts a Python callable (uses ``__name__``) or a raw tool dict in either
    OpenAI's wrapped form (``{"type": "function", "function": {"name": ...}}``)
    or a flat ``{"name": ...}`` shape. Returns ``""`` if no name can be found.
    """
    if callable(tool) and not isinstance(tool, dict):
        return getattr(tool, "__name__", "") or ""
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"])
        if tool.get("name"):
            return str(tool["name"])
    return ""


def tool_schema(tool: Any) -> dict[str, Any]:
    """Return the model-facing tool schema for a single tool entry.

    For callables, introspects the signature via :func:`callable_to_tool_dict`.
    For dicts, returns a shallow copy.
    """
    if callable(tool) and not isinstance(tool, dict):
        return callable_to_tool_dict(tool)
    if isinstance(tool, dict):
        return dict(tool)
    raise TypeError(f"unsupported tool type: {type(tool).__name__}")


def build_tools(
    tools: Sequence[Callable[..., Any] | dict[str, Any]] | None,
    *,
    web_search: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
    tool_dicts: list[dict[str, Any]] = []
    tool_index: dict[str, Callable[..., Any]] = {}

    if tools:
        for t in tools:
            if callable(t) and not isinstance(t, dict):
                d = tool_schema(t)
                tool_dicts.append(d)
                tool_index[d["function"]["name"]] = t
            else:
                td = tool_schema(t)
                tool_dicts.append(td)
                # raw dict tools have no callable; nothing to register

    if web_search:
        if not any(td.get("type") == "web_search" for td in tool_dicts):
            tool_dicts.append({"type": "web_search"})

    return tool_dicts, tool_index


def safe_call(
    fn: Callable[..., Any], args: dict[str, Any]
) -> tuple[Any, str | None, float]:
    """Run a tool function, capturing any exception as ``{"error": repr(e)}``.

    Returns ``(result_or_error_dict, error_repr_or_none, duration_ms)``.
    """
    if inspect.iscoroutinefunction(fn):
        raise TypeError(
            f"async tool function {fn.__qualname__} cannot be used in sync relay(); use relay_async()"
        )
    t0 = time.perf_counter()
    try:
        result = fn(**args)
        return result, None, (time.perf_counter() - t0) * 1000
    except Exception as e:
        return {"error": repr(e)}, repr(e), (time.perf_counter() - t0) * 1000


async def safe_call_async(
    fn: Callable[..., Any], args: dict[str, Any]
) -> tuple[Any, str | None, float]:
    t0 = time.perf_counter()
    try:
        if inspect.iscoroutinefunction(fn):
            result = await fn(**args)
        else:
            result = fn(**args)
        return result, None, (time.perf_counter() - t0) * 1000
    except Exception as e:
        return {"error": repr(e)}, repr(e), (time.perf_counter() - t0) * 1000


def serialize_tool_result(result: Any) -> str:
    """JSON-serialize a tool result for the tool/function_call_output message."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"failed to serialize tool result: {e!r}"})


def parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise errors.InvalidToolCallError(
            f"tool call arguments are not valid JSON: {e!r}",
            code="invalid_tool_call",
            raw=raw,
        ) from e
    if not isinstance(parsed, dict):
        raise errors.InvalidToolCallError(
            "tool call arguments must decode to a JSON object",
            code="invalid_tool_call",
            raw=raw,
        )
    return parsed
