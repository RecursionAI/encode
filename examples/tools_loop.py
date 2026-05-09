"""Tool-call loop with auto-execution of Python callables."""

from __future__ import annotations

import encode


def get_weather(city: str) -> dict:
    """Return the (fake) current weather for a city.

    Args:
        city: The city to look up.
    """
    fake = {"Denver": 72, "Tokyo": 65, "Paris": 60}
    return {"city": city, "temp_f": fake.get(city, 70)}


def main() -> None:
    out = encode.relay(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in Tokyo and Paris?"}],
        tools=[get_weather],
    ).response

    print("Final content:", out.content)
    print("Iterations:", out.iterations)
    for tc in out.tool_calls:
        print(f"  -> {tc.name}({tc.arguments}) = {tc.result}")


if __name__ == "__main__":
    main()
