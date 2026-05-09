"""Structured JSON output with a Pydantic response_format."""

from __future__ import annotations

from pydantic import BaseModel

import encode


class Sentiment(BaseModel):
    reasoning: str
    sentiment: str  # positive | negative | neutral


def main() -> None:
    out = encode.relay(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Classify: 'I absolutely loved it.'"}],
        response_format=Sentiment,
    ).response

    assert isinstance(out.parsed, Sentiment)
    print("reasoning:", out.parsed.reasoning)
    print("sentiment:", out.parsed.sentiment)


if __name__ == "__main__":
    main()
