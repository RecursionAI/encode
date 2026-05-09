"""Observe each tool-call iteration; stop the loop when a sentinel tool is called."""

from __future__ import annotations

import encode


def search(query: str) -> dict:
    """Search the index."""
    return {"hits": [{"title": query.upper()}]}


def submit_final(answer: str) -> dict:
    """Submit the final answer when done. Calling this ends the agent."""
    return {"submitted": answer}


def main() -> None:
    def watcher(event: encode.InterceptEvent) -> None:
        names = [tc.name for tc in event.tool_calls]
        print(f"iter {event.iteration}: {names}")
        if any(tc.name == "submit_final" for tc in event.tool_calls):
            event.stop()

    # Both forms are equivalent — pick whichever reads better at the call site.
    out = encode.relay(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Search 'python' then submit the title as your final answer."}],
        tools=[search, submit_final],
    ).intercept(watcher).response
    # ...or:
    # encode.relay(..., on_intercept=watcher).response

    print("Final content:", out.content)
    print("Iterations:", out.iterations)


if __name__ == "__main__":
    main()
