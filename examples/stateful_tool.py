"""Sandbox-as-tool pattern.

Per the Managed Agents paper: "each hand is a tool with execute(name, input)
→ string." Sandboxes — bash, browsers, REPLs — are just stateful tools. No
separate Sandbox protocol needed; bind a function to instance state and pass
it via ``tools=``.

This example wraps a Terminal as a tool so Claude can run shell commands. The
Terminal's state (cwd, env vars, sourced venvs) persists across calls because
the same bash process is reused.
"""

from __future__ import annotations

import encode


class BashSandbox:
    """Owns a long-lived bash terminal and exposes a tool-shaped callable.

    The Terminal is provisioned lazily on the first call — improves TTFT for
    conversations that don't end up needing a shell.
    """

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
        """Return a callable suitable for ``tools=`` with stable name + schema."""

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


def main() -> None:
    sandbox = BashSandbox()
    session = encode.Session.open()
    try:
        out = encode.relay(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "Create /tmp/encode_demo, write a 'hello.py' inside it that "
                    "prints 'hello', and run it."
                ),
            }],
            tools=[sandbox.as_tool()],
            session=session,
            max_tool_iterations=6,
        ).response
        print("answer:", out.content)
        print(f"shell commands run: {len(session.events_by_type('tool.call'))}")
    finally:
        sandbox.teardown()


if __name__ == "__main__":
    main()
