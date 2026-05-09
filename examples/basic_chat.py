"""Basic chat completion. Set ENCODE_API_KEY and ENCODE_BASE_URL in .env."""

from __future__ import annotations

import encode


def main() -> None:
    out = encode.relay(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Greet me in one sentence."},
        ],
    ).response
    print(out.content)


if __name__ == "__main__":
    main()
