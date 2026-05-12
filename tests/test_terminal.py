"""Real bash integration tests for Terminal and AsyncTerminal.

No mocks — these spawn actual /bin/bash processes. Skipped on Windows.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

import encode

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="encode.Terminal is macOS/Linux only",
)


# --------------------------- sync ---------------------------


def test_run_basic():
    with encode.Terminal() as sh:
        r = sh.run("echo hello")
    assert r.output == "hello"
    assert r.exit_code == 0
    assert r.timed_out is False
    assert r.duration_ms >= 0


def test_state_persists_cwd():
    target = os.path.realpath("/tmp")
    with encode.Terminal() as sh:
        sh.run("cd /tmp")
        r = sh.run("pwd")
    assert os.path.realpath(r.output) == target
    assert os.path.realpath(r.cwd) == target


def test_state_persists_env():
    with encode.Terminal() as sh:
        sh.run("export FOO=bar")
        r = sh.run("echo $FOO")
    assert r.output == "bar"
    assert r.exit_code == 0


def test_nonzero_exit_code():
    with encode.Terminal() as sh:
        r = sh.run("false")
    assert r.exit_code == 1


def test_constructor_cwd_honored(tmp_path):
    with encode.Terminal(cwd=tmp_path) as sh:
        r = sh.run("pwd")
    assert os.path.realpath(r.output) == os.path.realpath(str(tmp_path))


def test_run_timeout_raises():
    with encode.Terminal() as sh:
        with pytest.raises(encode.TerminalTimeoutError) as excinfo:
            sh.run("sleep 5", timeout=0.5)
    assert excinfo.value.command == "sleep 5"
    assert excinfo.value.partial_output is not None


def test_kill_is_idempotent():
    sh = encode.Terminal()
    assert sh.alive is True
    sh.kill()
    sh.kill()  # second call must not raise
    assert sh.alive is False


def test_context_manager_terminates():
    with encode.Terminal() as sh:
        assert sh.alive is True
    assert sh.alive is False
    assert sh.pid is None


def test_start_and_read_background_process():
    with encode.Terminal() as sh:
        sh.start("for i in 1 2 3; do echo line$i; done")
        # The loop runs to completion quickly; give bash a moment to flush.
        time.sleep(0.3)
        out = sh.read(timeout=0.5)
    assert "line1" in out
    assert "line3" in out


def test_alive_pid_reflect_state():
    sh = encode.Terminal()
    try:
        assert sh.alive is True
        assert isinstance(sh.pid, int)
    finally:
        sh.kill()
    assert sh.alive is False
    assert sh.pid is None


def test_snapshot_reflects_state():
    with encode.Terminal() as sh:
        snap = sh.snapshot()
        assert snap.alive is True
        assert isinstance(snap.pid, int)
        assert snap.cwd
        sh.run("cd /tmp")
        snap2 = sh.snapshot()
        assert os.path.realpath(snap2.cwd) == os.path.realpath("/tmp")


def test_terminal_error_is_courier_error():
    assert issubclass(encode.TerminalError, encode.CourierError)
    assert issubclass(encode.TerminalTimeoutError, encode.TerminalError)


# --------------------------- async ---------------------------


async def test_async_run_basic():
    async with encode.AsyncTerminal() as sh:
        r = await sh.run("echo hello")
    assert r.output == "hello"
    assert r.exit_code == 0


async def test_async_state_persists():
    async with encode.AsyncTerminal() as sh:
        await sh.run("export X=ok")
        r = await sh.run("echo $X")
    assert r.output == "ok"


async def test_async_run_timeout_raises():
    async with encode.AsyncTerminal() as sh:
        with pytest.raises(encode.TerminalTimeoutError):
            await sh.run("sleep 5", timeout=0.5)


async def test_async_context_manager():
    sh = encode.AsyncTerminal()
    async with sh:
        await sh.run("echo hi")
        assert sh.alive is True
    assert sh.alive is False
