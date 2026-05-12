# Terminal

`encode.Terminal` is a persistent bash subprocess wrapped as a Python class. State (cwd, env vars, sourced venvs) survives across `run()` calls because every command executes in the same shell. Wrap one in a closure to expose it as a tool — that's the canonical **sandbox-as-tool** pattern.

```python
with encode.Terminal() as sh:
    sh.run("export FOO=bar")
    r = sh.run("echo $FOO")
    assert r.output == "bar"
    r = sh.run("pwd")
    print(r.cwd)
```

Async equivalent:

```python
async with encode.AsyncTerminal() as sh:
    await sh.run("cd /tmp && touch demo.txt")
    r = await sh.run("ls demo.txt")
    print(r.output)
```

Platform: macOS and Linux only.

## API

### `Terminal(cwd=None, timeout=30.0)`

- `cwd`: starting directory; defaults to `os.getcwd()`.
- `timeout`: default per-command timeout in seconds; overridable per-call.

### `run(command, *, timeout=None) → CommandResult`

Run a command and wait for it to finish. Uses a unique sentinel echo to detect completion, so it never confuses real output with the prompt.

```python
r = sh.run("echo hi")
r.command    # "echo hi"
r.output     # "hi"
r.exit_code  # 0
r.cwd        # the cwd after the command
r.duration_ms
r.timed_out  # False
```

### `start(command) → None`

Send a command without a sentinel — fire and forget. Useful for long-running processes (servers, watchers). After `start()` the shell is busy with the foreground command; subsequent `run()` calls will block until that command exits, so background with `&` if you want to interleave.

### `read(timeout=1.0) → str`

Drain currently-buffered output (non-blocking after the first read). Pair with `start()`.

### `kill() → None`

Terminate the underlying bash process. Idempotent. The context manager calls this on exit.

### `snapshot() → TerminalSnapshot`

Point-in-time view — handy for logging or persisting alongside a Session.

```python
class TerminalSnapshot(BaseModel):
    cwd: str
    alive: bool
    pid: int | None
    nonce: str
```

### Properties

- `alive: bool` — process is still running.
- `pid: int | None` — child PID.

## `CommandResult`

```python
class CommandResult(BaseModel):
    command: str
    output: str          # stripped trailing newline, \r\n → \n
    exit_code: int | None  # None only when timed_out=True
    cwd: str             # cwd after the command (post-cd, etc.)
    duration_ms: float
    timed_out: bool
```

## Timeouts

A command that doesn't finish before `timeout` raises `TerminalTimeoutError` with whatever output was captured so far:

```python
try:
    sh.run("sleep 5", timeout=1.0)
except encode.TerminalTimeoutError as e:
    print(e.command, e.partial_output)
```

## Sandbox-as-tool

`Terminal` is just a stateful Python object. Wrap it in a closure to give a model a shell:

```python
import encode

class BashSandbox:
    """Owns a long-lived bash terminal and exposes a tool-shaped callable."""

    def __init__(self) -> None:
        self._terminal: encode.Terminal | None = None

    def _ensure(self) -> encode.Terminal:
        if self._terminal is None or not self._terminal.alive:
            self._terminal = encode.Terminal()
        return self._terminal

    def teardown(self) -> None:
        if self._terminal is not None:
            self._terminal.kill()
            self._terminal = None

    def as_tool(self):
        sandbox = self

        def bash(command: str) -> dict:
            """Run a bash command in a persistent shell.

            Args:
                command: A bash command. State (cwd, env vars) persists across calls.
            """
            result = sandbox._ensure().run(command, timeout=10.0)
            return {
                "command": command,
                "output": result.output,
                "exit_code": result.exit_code,
                "cwd": result.cwd,
            }

        return bash


sandbox = BashSandbox()
try:
    out = encode.relay(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
                   "Create /tmp/encode_demo, write hello.py that prints 'hello', and run it."}],
        tools=[sandbox.as_tool()],
        max_tool_iterations=6,
    ).response
    print(out.content)
finally:
    sandbox.teardown()
```

Two design notes:

- **Closures, not `__call__`.** A function returned from `as_tool()` has a stable `__name__` and `__doc__` that encode's schema introspection can read. An instance with `__call__` doesn't.
- **Lazy provisioning.** Spawning a bash process takes ~50ms. Lazy `_ensure()` keeps TTFT low for conversations that never need a shell.

## Why Terminal isn't a "Sandbox primitive"

Per the [Managed Agents](https://www.anthropic.com/engineering/managed-agents) paper, sandboxes are tools. They reach Claude through the same `execute(name, input) → string` interface as anything else. encode follows that convention — no separate `Sandbox` protocol, no special harness wiring. `Terminal` is a useful primitive *because* it's a process wrapper, not because it's plumbed differently.

See [executors.md](./executors.md) for the seam itself and [concepts.md](./concepts.md) for the architectural rationale.

## See also

- [tools.md](./tools.md) — function → schema introspection
- [executors.md](./executors.md) — the `ToolExecutor` seam
- [errors.md](./errors.md) — `TerminalError`, `TerminalTimeoutError`
- [cookbook.md](./cookbook.md) — full sandbox-as-tool recipe
