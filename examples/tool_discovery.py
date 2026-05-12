"""Auto tool discovery via Session-owned tools + intercept.

Pattern: the model has a single bootstrap tool (`list_tools`) that returns
specs for additional tools. An intercept handler reads those specs, calls
``event.register_tool(...)`` to append them to the session's append-only
tool registry, and the *next* iteration of the relay loop sees them — the
model can immediately call them.

Run against any OpenAI-compatible endpoint with ENCODE_API_KEY +
ENCODE_BASE_URL set; pick a model that supports tool-calling. The print
statements below show the discovery flow.
"""

from __future__ import annotations

import encode

# --- the bootstrap tool ---


def list_tools() -> list[dict]:
    """List tools that can be registered on this session.

    Returns OpenAI-style tool schemas. Pair with an intercept handler that
    calls ``event.register_tool(spec)`` so the model sees them next turn.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch",
                "description": "Fetch a URL and return its body.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize",
                "description": "Summarize a block of text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
    ]


# --- the actual implementations (registered as callables once discovered) ---


def fetch(url: str) -> dict:
    """Fetch a URL."""
    # placeholder — wire up httpx in real code
    return {"url": url, "status": 200, "body": "..."}


def summarize(text: str) -> dict:
    """Summarize text."""
    return {"summary": text[:80]}


IMPLS = {"fetch": fetch, "summarize": summarize}


def discover(event: encode.InterceptEvent) -> None:
    """Intercept handler: register discovered tools on the session."""
    for tc in event.tool_calls:
        if tc.name != "list_tools":
            continue
        for spec in tc.result or []:
            name = spec.get("function", {}).get("name") or ""
            impl = IMPLS.get(name)
            if impl is None:
                # we don't have a Python implementation — skip
                continue
            registered = event.register_tool(impl)
            if registered:
                print(f"  ↳ discovered + registered: {name}")


def main() -> None:
    session = encode.Session.open(tools=[list_tools])

    out = encode.relay(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "Discover the tools available to you, then fetch "
                    "https://example.com and summarize the body."
                ),
            }
        ],
        session=session,
        tools=session.tools,
        on_intercept=discover,
        max_tool_iterations=10,
    ).response

    print()
    print("final answer:", out.content)
    print("iterations:", out.iterations)
    print(
        "registered tools (final):",
        [ev.data["name"] for ev in session.events_by_type("tool.registered")],
    )


if __name__ == "__main__":
    main()
