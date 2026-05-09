"""Use the /v1/responses endpoint by passing input= or instructions=."""

from __future__ import annotations

import encode


def main() -> None:
    out = encode.relay(
        model="gpt-4o-mini",
        instructions="You are a concise summarizer.",
        input="Summarize: encode is a Python SDK for OpenAI-compatible inference endpoints.",
    ).response
    print("endpoint:", out.endpoint)  # "responses"
    print(out.content)


if __name__ == "__main__":
    main()
