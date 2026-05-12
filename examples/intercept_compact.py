"""Mid-loop context engineering via the mutable InterceptEvent API.

The intercept callback can rewrite the conversation that goes into the *next*
iteration — inject reminders, compact history, redact tool output, branch.
When a Session is active, mutations also emit ``context.modify`` events into
the durable log.
"""

from __future__ import annotations

import encode


def search(query: str) -> dict:
    """Fake search tool.

    Args:
        query: Search string.
    """
    # Pretend the result is huge and noisy.
    return {
        "query": query,
        "results": [f"hit-{i}: lots of irrelevant text here" for i in range(50)],
    }


def summarize_tool_results(content: str) -> str:
    """Trim a noisy tool result before the model sees it."""
    if len(content) > 200:
        return content[:200] + "  …[trimmed]"
    return content


def watcher(event: encode.InterceptEvent) -> None:
    print(f"iteration {event.iteration}: {[tc.name for tc in event.tool_calls]}")
    # Trim the last tool result so the next iteration's context stays small.
    event.edit_last_tool_result(summarize_tool_results)
    # And nudge the model to wrap up.
    if event.iteration >= 1:
        event.append(
            {"role": "system", "content": "You have enough info — answer now."}
        )


def main() -> None:
    session = encode.Session.open()
    out = encode.relay(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "search 'glacier melt rates 2026'"}],
        tools=[search],
        session=session,
        on_intercept=watcher,
    ).response

    print("final:", out.content)
    print(f"iterations: {out.iterations}")
    print(f"context.modify events emitted: {len(session.events_by_type('context.modify'))}")


if __name__ == "__main__":
    main()
