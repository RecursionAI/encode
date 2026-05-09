"""Callable -> OpenAI tool schema conversion."""

from __future__ import annotations

import warnings

from encode._schema import EncodeUserWarning, callable_to_tool_dict


def test_simple_function_schema():
    def get_weather(city: str, units: str = "fahrenheit") -> dict:
        """Get current weather by city.

        Args:
            city: City name to look up.
            units: Temperature units.
        """
        return {}

    d = callable_to_tool_dict(get_weather)
    assert d["type"] == "function"
    fn = d["function"]
    assert fn["name"] == "get_weather"
    assert fn["description"] == "Get current weather by city."
    schema = fn["parameters"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "city" in schema["properties"]
    assert "units" in schema["properties"]
    assert schema["properties"]["city"].get("description") == "City name to look up."
    assert schema["required"] == ["city"]


def test_no_args_function():
    def ping() -> str:
        """No args."""
        return "pong"

    d = callable_to_tool_dict(ping)
    assert d["function"]["parameters"]["properties"] == {}


def test_unannotated_param_warns_and_defaults_to_str():
    def loose(x):
        return x

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = callable_to_tool_dict(loose)
        assert any(issubclass(item.category, EncodeUserWarning) for item in w)
    assert d["function"]["parameters"]["properties"]["x"]["type"] == "string"


def test_variadic_params_ignored_with_warning():
    def variadic(a: int, *args, **kwargs):
        return a

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = callable_to_tool_dict(variadic)
        assert any(issubclass(item.category, EncodeUserWarning) for item in w)
    props = d["function"]["parameters"]["properties"]
    assert "a" in props
    assert "args" not in props and "kwargs" not in props
