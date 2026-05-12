"""Durable agent runs via Session (event log) + your own persistence.

Session is a pure Pydantic model — serialize with ``model_dump()`` and load
back with ``model_validate()``. This example uses a JSON file on disk; in a
real app you'd persist into Postgres, Redis, S3, or whatever DB you already
have. The SDK takes no opinion.
"""

from __future__ import annotations

import json
from pathlib import Path

import encode


def lookup(q: str) -> dict:
    """Pretend to look up a fact.

    Args:
        q: Query string.
    """
    facts = {"capital_of_france": "Paris", "highest_mountain": "Everest"}
    return {"q": q, "answer": facts.get(q, "unknown")}


def run_first_turn(store_path: Path) -> str:
    session = encode.Session.open()
    out = encode.relay(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "look up the capital_of_france"}],
        tools=[lookup],
        session=session,
    ).response
    print(f"first turn → {out.content}")
    # Persist however you like — here, a JSON file.
    store_path.write_text(json.dumps(session.model_dump(), default=str))
    return session.id


def run_second_turn(store_path: Path) -> None:
    # Resume from disk. In a real app this could be a different process /
    # machine / day.
    raw = json.loads(store_path.read_text())
    session = encode.Session.model_validate(raw)
    out = encode.relay(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "what about highest_mountain?"}],
        tools=[lookup],
        session=session,
    ).response
    print(f"second turn → {out.content}")
    print(f"total events on session: {len(session.events)}")
    # Save again for the next round.
    store_path.write_text(json.dumps(session.model_dump(), default=str))


def main() -> None:
    store = Path("/tmp/encode_session_demo.json")
    sid = run_first_turn(store)
    print(f"session id: {sid}")
    run_second_turn(store)


if __name__ == "__main__":
    main()
