"""Convert a Python callable into an OpenAI tool schema dict.

Pipeline: ``inspect.signature`` -> per-param annotations + defaults -> dynamic
``pydantic.create_model`` -> ``model_json_schema()`` -> OpenAI tool dict shape.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, Field, create_model


class EncodeUserWarning(UserWarning):
    """Warning emitted when a tool callable has unannotated args, *args, or **kwargs."""


_warned: set[str] = set()


def _docstring_summary(fn: Callable[..., Any]) -> str:
    doc = inspect.getdoc(fn) or ""
    return doc.split("\n\n", 1)[0].strip() if doc else ""


def _docstring_param_descriptions(fn: Callable[..., Any]) -> dict[str, str]:
    """Best-effort parse of Google-style 'Args:' or NumPy-style 'Parameters' blocks."""
    doc = inspect.getdoc(fn) or ""
    out: dict[str, str] = {}
    if not doc:
        return out
    lines = doc.splitlines()
    in_args = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower in ("args:", "arguments:", "parameters:", "parameters", "args"):
            in_args = True
            continue
        if in_args:
            if not stripped:
                in_args = False
                continue
            if ":" in stripped:
                name, _, desc = stripped.partition(":")
                name = name.strip().split(" ")[0].lstrip("*")
                if name and name.isidentifier():
                    out[name] = desc.strip()
    return out


def callable_to_tool_dict(fn: Callable[..., Any]) -> dict[str, Any]:
    """Produce ``{"type": "function", "function": {...}}`` for an arbitrary callable."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    descriptions = _docstring_param_descriptions(fn)
    fields: dict[str, Any] = {}
    saw_var = False

    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            saw_var = True
            continue
        annotation = hints.get(pname, param.annotation)
        if annotation is inspect.Parameter.empty:
            key = f"{fn.__module__}.{fn.__qualname__}:{pname}"
            if key not in _warned:
                _warned.add(key)
                warnings.warn(
                    f"parameter '{pname}' on {fn.__qualname__} has no type annotation; defaulting to str",
                    EncodeUserWarning,
                    stacklevel=2,
                )
            annotation = str
        default = param.default if param.default is not inspect.Parameter.empty else ...
        desc = descriptions.get(pname)
        field = Field(default=default, description=desc) if desc else Field(default=default)
        fields[pname] = (annotation, field)

    if saw_var:
        warnings.warn(
            f"{fn.__qualname__} uses *args/**kwargs; variadic parameters are ignored in the tool schema",
            EncodeUserWarning,
            stacklevel=2,
        )

    if fields:
        Model: type[BaseModel] = create_model(f"_{fn.__name__}Args", **fields)  # type: ignore[call-overload]
        schema = Model.model_json_schema()
    else:
        schema = {"type": "object", "properties": {}}

    schema = _clean_schema(schema)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema["additionalProperties"] = False

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": _docstring_summary(fn),
            "parameters": schema,
        },
    }


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove Pydantic's auto-generated ``title`` keys recursively."""
    if isinstance(schema, dict):
        out = {k: _clean_schema(v) for k, v in schema.items() if k != "title"}
        return out
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]  # type: ignore[return-value]
    return schema


def pydantic_to_response_format(model: type[BaseModel]) -> dict[str, Any]:
    """Build the chat-completions ``response_format`` payload for a Pydantic model."""
    schema = _clean_schema(model.model_json_schema())
    schema.setdefault("type", "object")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": schema,
        },
    }
