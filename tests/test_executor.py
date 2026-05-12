"""Tests for the ToolExecutor protocol and LocalToolExecutor."""

from __future__ import annotations

import pytest

import encode
from encode import LocalToolExecutor


def test_local_executor_runs_callable():
    def add(a: int, b: int) -> int:
        return a + b

    ex = LocalToolExecutor({"add": add})
    result = ex.execute("add", {"a": 2, "b": 3})
    assert result.error is None
    assert result.result == 5
    assert result.result_serialized == "5"
    assert result.duration_ms >= 0


def test_local_executor_serializes_dict_result():
    def hello(name: str) -> dict:
        return {"greeting": f"hi {name}"}

    ex = LocalToolExecutor({"hello": hello})
    result = ex.execute("hello", {"name": "jackson"})
    assert result.error is None
    assert result.result == {"greeting": "hi jackson"}
    assert "hi jackson" in result.result_serialized


def test_local_executor_captures_tool_exception():
    def boom(reason: str) -> None:
        raise ValueError(reason)

    ex = LocalToolExecutor({"boom": boom})
    result = ex.execute("boom", {"reason": "kapow"})
    assert result.error is not None
    assert "kapow" in result.error
    assert result.result == {"error": result.error}


def test_local_executor_unknown_name_returns_error_result():
    ex = LocalToolExecutor({})
    result = ex.execute("missing", {})
    assert result.error is not None
    assert "missing" in result.error
    assert result.result == {"error": result.error}


async def test_local_executor_async_runs_async_callable():
    async def acompute(x: int) -> int:
        return x * 2

    ex = LocalToolExecutor({"acompute": acompute})
    result = await ex.execute_async("acompute", {"x": 21})
    assert result.error is None
    assert result.result == 42


async def test_local_executor_async_runs_sync_callable():
    def sync_fn(s: str) -> str:
        return s.upper()

    ex = LocalToolExecutor({"sync_fn": sync_fn})
    result = await ex.execute_async("sync_fn", {"s": "hello"})
    assert result.error is None
    assert result.result == "HELLO"


def test_local_executor_has_and_names():
    def f(): return 0
    def g(): return 0

    ex = LocalToolExecutor({"f": f, "g": g})
    assert ex.has("f")
    assert ex.has("g")
    assert not ex.has("h")
    assert set(ex.names) == {"f", "g"}


def test_tool_executor_protocol_runtime_check():
    """LocalToolExecutor should satisfy the ToolExecutor Protocol."""
    ex = LocalToolExecutor({})
    assert isinstance(ex, encode.ToolExecutor)


def test_relay_uses_custom_tool_executor(respx_mock, base_url):
    """relay(tool_executor=...) routes dispatch through the provided executor."""
    import httpx
    from itertools import cycle

    iterator = cycle([
        httpx.Response(
            200,
            json={
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"v":"hello"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            },
        ),
        httpx.Response(
            200,
            json={
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "got echo"},
                    "finish_reason": "stop",
                }],
            },
        ),
    ])
    respx_mock.post(f"{base_url}/v1/chat/completions").mock(side_effect=lambda *a, **k: next(iterator))

    # Schema-only tool (for the model)
    def echo(v: str) -> str:
        """Echo a value."""
        return ""  # never called — executor routes elsewhere

    # Custom executor returns a marker we can detect
    class Marker:
        def __init__(self): self.calls = []
        def execute(self, name, input):
            self.calls.append((name, dict(input)))
            from encode.executor import ExecutionResult
            return ExecutionResult(result="ROUTED", result_serialized='"ROUTED"', duration_ms=1.0)
        async def execute_async(self, name, input):
            return self.execute(name, input)

    marker = Marker()
    out = encode.relay(
        model="m",
        messages=[{"role": "user", "content": "echo hello"}],
        tools=[echo],
        tool_executor=marker,
    ).response
    assert out.content == "got echo"
    assert marker.calls == [("echo", {"v": "hello"})]
    # tool record reflects routed result
    assert out.tool_calls[0].result == "ROUTED"
