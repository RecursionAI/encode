"""Persistent shell sessions.

A ``Session`` wraps a single long-lived bash process. State (cwd, env vars,
sourced venvs) persists across ``run()`` calls because every command executes
in the same shell. Use ``CommandResult`` to inspect output, exit code, and the
post-command cwd.

Backed by ``pexpect`` (sync) and ``asyncio.create_subprocess_shell`` (async).
macOS and Linux only.

Sentinel protocol: each ``run()`` appends a unique echo command after the
user's command. Bash emits ``---ENCODE_DONE_<nonce>:<exit>:<cwd>---`` on its
own line, which is how the SDK detects completion without prompt matching.
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import sys
import time
from os import PathLike
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from .errors import SessionError, SessionTimeoutError

if TYPE_CHECKING:
    import pexpect as _pexpect_module


_SENTINEL_TEMPLATE = '; echo "---ENCODE_DONE_{nonce}:$?:$(pwd)---"'


def _sentinel_for(nonce: str) -> str:
    return _SENTINEL_TEMPLATE.format(nonce=nonce)


def _sentinel_str_re(nonce: str) -> re.Pattern[str]:
    # Trailing \r?\n? is consumed so the next command's read buffer starts clean.
    return re.compile(rf"---ENCODE_DONE_{nonce}:(-?\d+):([^\r\n]*)---\r?\n?")


def _sentinel_bytes_re(nonce: str) -> re.Pattern[bytes]:
    return re.compile(
        rb"---ENCODE_DONE_" + nonce.encode("ascii") + rb":(-?\d+):([^\r\n]*)---\r?\n?"
    )


def _check_platform() -> None:
    if sys.platform.startswith("win"):
        raise SessionError(
            "encode.Session is only supported on macOS and Linux"
        )


def _shell_env() -> dict[str, str]:
    """Inherit os.environ but suppress prompt and prompt-side-effects."""
    env = dict(os.environ)
    env["PS1"] = ""
    env["PS2"] = ""
    env["PROMPT_COMMAND"] = ""
    env["TERM"] = env.get("TERM", "dumb")
    return env


class CommandResult(BaseModel):
    """Result of a single ``Session.run()`` call.

    ``exit_code`` is ``None`` only when the command timed out before the
    sentinel was emitted (paired with ``timed_out=True``).
    """

    model_config = ConfigDict(extra="allow")

    command: str
    output: str
    exit_code: int | None
    cwd: str
    duration_ms: float
    timed_out: bool = False


class Session:
    """Persistent bash session backed by ``pexpect.spawn``.

    State (cwd, env vars, activated venvs) persists across ``run()`` calls.

    Example:
        with encode.Session() as sh:
            sh.run("export FOO=bar")
            r = sh.run("echo $FOO")
            assert r.output == "bar"
    """

    def __init__(
        self,
        cwd: str | PathLike[str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        _check_platform()
        try:
            import pexpect
        except ImportError as e:  # pragma: no cover - declared dep
            raise SessionError(f"pexpect is required for sessions: {e}") from e
        self._pexpect: Any = pexpect
        self._cwd = os.fspath(cwd) if cwd is not None else os.getcwd()
        self._default_timeout = timeout
        self._nonce = secrets.token_hex(8)
        self._sentinel = _sentinel_for(self._nonce)
        self._sentinel_re = _sentinel_str_re(self._nonce)
        self._proc: _pexpect_module.spawn | None = None
        self._spawn()
        try:
            self.run(":", timeout=timeout)
        except SessionTimeoutError as e:
            self.kill()
            raise SessionError(
                f"bash session failed to initialize within {timeout}s"
            ) from e

    def _spawn(self) -> None:
        try:
            self._proc = self._pexpect.spawn(
                "/bin/bash",
                ["--norc", "--noprofile"],
                cwd=self._cwd,
                env=_shell_env(),
                encoding="utf-8",
                echo=False,
                timeout=self._default_timeout,
                dimensions=(24, 200),
            )
        except Exception as e:
            raise SessionError(f"failed to spawn bash: {e}") from e

    @property
    def alive(self) -> bool:
        return self._proc is not None and bool(self._proc.isalive())

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def run(self, command: str, *, timeout: float | None = None) -> CommandResult:
        if self._proc is None or not self._proc.isalive():
            raise SessionError(
                "session is not running; spawn was killed or never started"
            )
        t = self._default_timeout if timeout is None else timeout
        full = command + self._sentinel
        start = time.perf_counter()
        self._proc.sendline(full)
        try:
            idx = self._proc.expect(
                [self._sentinel_re, self._pexpect.TIMEOUT, self._pexpect.EOF],
                timeout=t,
            )
        except Exception as e:  # pragma: no cover - expect normally returns an index
            raise SessionError(f"session read failed: {e}") from e
        duration_ms = (time.perf_counter() - start) * 1000.0
        if idx == 1:
            partial = self._proc.before or ""
            raise SessionTimeoutError(
                f"command timed out after {t}s",
                partial_output=partial,
                command=command,
            )
        if idx == 2:
            raise SessionError(
                f"bash exited unexpectedly during command: {command!r}"
            )
        match = self._proc.match
        exit_code = int(match.group(1))
        cwd = match.group(2)
        before = self._proc.before or ""
        output = before.replace("\r\n", "\n").rstrip()
        return CommandResult(
            command=command,
            output=output,
            exit_code=exit_code,
            cwd=cwd,
            duration_ms=duration_ms,
            timed_out=False,
        )

    def start(self, command: str) -> None:
        """Send a command without a sentinel — fire and forget.

        Useful for long-running processes (servers, watchers). After ``start()``
        the shell is busy with the foreground command; subsequent ``run()``
        calls will block until that command exits, so background with ``&`` if
        you want to interleave.
        """
        if self._proc is None or not self._proc.isalive():
            raise SessionError("session is not running")
        self._proc.sendline(command)

    def read(self, timeout: float = 1.0) -> str:
        """Drain currently available output (non-blocking after the first read)."""
        if self._proc is None or not self._proc.isalive():
            raise SessionError("session is not running")
        chunks: list[str] = []
        sub_t = timeout
        while True:
            try:
                data = self._proc.read_nonblocking(size=4096, timeout=sub_t)
            except self._pexpect.TIMEOUT:
                break
            except self._pexpect.EOF:
                break
            if not data:
                break
            chunks.append(data)
            sub_t = 0
        return "".join(chunks)

    def kill(self) -> None:
        """Terminate the underlying bash process. Idempotent."""
        if self._proc is None:
            return
        try:
            if self._proc.isalive():
                self._proc.terminate(force=True)
        except Exception:
            pass
        self._proc = None

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.kill()


class AsyncSession:
    """Persistent bash session backed by ``asyncio.create_subprocess_shell``.

    Same surface as :class:`Session` but async. Spawns lazily on the first
    ``run()`` / ``start()`` / ``read()`` because ``__init__`` cannot await.
    """

    def __init__(
        self,
        cwd: str | PathLike[str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        _check_platform()
        self._cwd = os.fspath(cwd) if cwd is not None else os.getcwd()
        self._default_timeout = timeout
        self._nonce = secrets.token_hex(8)
        self._sentinel = _sentinel_for(self._nonce)
        self._sentinel_bytes_re = _sentinel_bytes_re(self._nonce)
        self._proc: asyncio.subprocess.Process | None = None
        self._lock: asyncio.Lock | None = None
        self._buffer: bytes = b""
        self._started = False

    async def _ensure_started(self) -> None:
        if self._started:
            return
        try:
            self._proc = await asyncio.create_subprocess_shell(
                "/bin/bash --norc --noprofile",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._cwd,
                env=_shell_env(),
            )
        except Exception as e:
            raise SessionError(f"failed to spawn bash: {e}") from e
        self._lock = asyncio.Lock()
        self._started = True
        try:
            await self.run(":", timeout=self._default_timeout)
        except SessionTimeoutError as e:
            await self.kill()
            raise SessionError(
                f"bash session failed to initialize within {self._default_timeout}s"
            ) from e

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    async def run(self, command: str, *, timeout: float | None = None) -> CommandResult:
        await self._ensure_started()
        proc = self._proc
        lock = self._lock
        if proc is None or lock is None or proc.returncode is not None:
            raise SessionError("session is not running")
        if proc.stdin is None or proc.stdout is None:
            raise SessionError("session pipes are not available")
        t = self._default_timeout if timeout is None else timeout
        async with lock:
            full = (command + self._sentinel + "\n").encode()
            start = time.perf_counter()
            proc.stdin.write(full)
            await proc.stdin.drain()
            buffer = self._buffer
            self._buffer = b""
            deadline = time.monotonic() + t
            while True:
                m = self._sentinel_bytes_re.search(buffer)
                if m:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SessionTimeoutError(
                        f"command timed out after {t}s",
                        partial_output=buffer.decode("utf-8", errors="replace"),
                        command=command,
                    )
                try:
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(4096), timeout=remaining
                    )
                except asyncio.TimeoutError as e:
                    raise SessionTimeoutError(
                        f"command timed out after {t}s",
                        partial_output=buffer.decode("utf-8", errors="replace"),
                        command=command,
                    ) from e
                if not chunk:
                    raise SessionError(
                        f"bash exited unexpectedly during command: {command!r}"
                    )
                buffer += chunk
            duration_ms = (time.perf_counter() - start) * 1000.0
            exit_code = int(m.group(1))
            cwd = m.group(2).decode("utf-8", errors="replace")
            line_start = buffer.rfind(b"\n", 0, m.start())
            line_start = 0 if line_start == -1 else line_start
            line_end_idx = buffer.find(b"\n", m.end())
            if line_end_idx == -1:
                line_end = len(buffer)
            else:
                line_end = line_end_idx + 1
            before = buffer[:line_start]
            self._buffer = buffer[line_end:]
            output = (
                before.decode("utf-8", errors="replace")
                .replace("\r\n", "\n")
                .rstrip()
            )
            return CommandResult(
                command=command,
                output=output,
                exit_code=exit_code,
                cwd=cwd,
                duration_ms=duration_ms,
                timed_out=False,
            )

    async def start(self, command: str) -> None:
        """Send a command without a sentinel — fire and forget."""
        await self._ensure_started()
        proc = self._proc
        if proc is None or proc.returncode is not None:
            raise SessionError("session is not running")
        if proc.stdin is None:
            raise SessionError("session stdin is not available")
        proc.stdin.write((command + "\n").encode())
        await proc.stdin.drain()

    async def read(self, timeout: float = 1.0) -> str:
        """Drain currently available output."""
        await self._ensure_started()
        proc = self._proc
        if proc is None:
            raise SessionError("session is not running")
        if proc.stdout is None:
            raise SessionError("session stdout is not available")
        buf = self._buffer
        self._buffer = b""
        try:
            chunk = await asyncio.wait_for(proc.stdout.read(4096), timeout=timeout)
            if chunk:
                buf += chunk
        except asyncio.TimeoutError:
            pass
        while True:
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(4096), timeout=0.01)
            except asyncio.TimeoutError:
                break
            if not chunk:
                break
            buf += chunk
        return buf.decode("utf-8", errors="replace")

    async def kill(self) -> None:
        """Terminate the underlying bash process. Idempotent."""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.close()
        except Exception:
            pass
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        # Close the subprocess transport so its pipes don't try to clean up
        # against an already-closed event loop at GC time.
        try:
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                transport.close()
        except Exception:
            pass
        self._proc = None

    async def __aenter__(self) -> AsyncSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.kill()


__all__ = [
    "CommandResult",
    "Session",
    "AsyncSession",
]
