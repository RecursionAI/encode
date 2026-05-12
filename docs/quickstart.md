# Quickstart

Get from `pip install` to a durable agent in five minutes.

## Install

```bash
pip install encode
```

Requires Python 3.10+. Runtime deps: `httpx`, `pydantic`, `python-dotenv`, `typing-extensions`. The `Terminal` primitive needs `pexpect` on macOS or Linux.

## Configure

encode reads credentials from (highest wins):

1. Explicit `api_key=` / `base_url=` kwargs.
2. `ENCODE_API_KEY` / `ENCODE_BASE_URL` env vars.
3. `OPENAI_API_KEY` / `OPENAI_BASE_URL` env vars.

A `.env` in the cwd (or any parent) is auto-loaded once on import. Disable with `ENCODE_DISABLE_DOTENV=1`.

```bash
# .env
ENCODE_API_KEY=sk-your-key
ENCODE_BASE_URL=https://api.openai.com/v1
```

Works against OpenAI, Courier, vLLM, LM Studio, Ollama, Together, Groq, and anything else that speaks the OpenAI-compatible wire format.

## Hello world

```python
import encode

out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
).response

print(out.content)        # "Hello!"
print(out.iterations)     # 1
print(out.usage)          # Usage(prompt_tokens=..., completion_tokens=...)
```

## First tool

Drop a Python function into `tools=`. encode introspects the signature + docstring and builds the schema for you. The loop runs until the model stops calling tools.

```python
def get_weather(city: str, units: str = "fahrenheit") -> dict:
    """Get current weather for a city.

    Args:
        city: City name to look up.
        units: Temperature units.
    """
    return {"city": city, "temp_f": 72}

out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Weather in Denver?"}],
    tools=[get_weather],
).response

print(out.content)              # "It's 72°F in Denver."
print(out.iterations)           # 2 (one tool call + final)
print(out.tool_calls[0].name)   # "get_weather"
print(out.tool_calls[0].result) # {"city": "Denver", "temp_f": 72}
```

→ More: [tools.md](./tools.md)

## First Session

Sessions are how you make a run durable. Pass `session=` to `relay()` and every iteration appends events you can persist anywhere.

```python
import json
import encode

session = encode.Session.open()
encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "remember: my name is Alex"}],
    session=session,
).response

# Persist — any DB, file, or transport
blob = json.dumps(session.model_dump(), default=str)
open("/tmp/agent.json", "w").write(blob)

# Resume in a fresh process
resumed = encode.Session.model_validate(json.loads(open("/tmp/agent.json").read()))
encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "what's my name?"}],
    session=resumed,
).response  # → "Alex"
```

→ More: [sessions.md](./sessions.md)

## What to read next

- [concepts.md](./concepts.md) — the mental model behind the SDK
- [relay.md](./relay.md) — full `relay()` reference
- [messages.md](./messages.md) — stateful conversations and multimodal content
- [cookbook.md](./cookbook.md) — runnable recipes
