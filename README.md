# encode

> One Python entry-point for any OpenAI-compatible LLM, with an auto-agent loop, durable sessions, and the brain/hands/session primitives from Anthropic's [Managed Agents](https://www.anthropic.com/engineering/managed-agents) post baked in.

```python
import encode

def get_weather(city: str) -> dict:
    """Get weather by city."""
    return {"city": city, "temp_f": 72}

out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Weather in Denver?"}],
    tools=[get_weather],
).response

print(out.content)   # "It's 72°F in Denver."
```

That's the whole "hello world." The model called `get_weather`, encode dispatched it, fed the result back, and returned the final answer.

## What encode is

A Python SDK for any OpenAI-compatible inference endpoint — Courier, OpenAI, vLLM, LM Studio, Ollama, Together, Groq, and more. One `relay()` function spans `/v1/chat/completions` and `/v1/responses`. Pass tools and it runs the loop to completion. Pass a `Session` and the run is durable. Pass a `Terminal` and your agent has a shell.

## Why encode

- **Auto tool loop.** Drop a Python function in `tools=`; encode introspects the signature, builds the schema, runs the loop, and feeds results back. No decorators, no manual loop scaffolding.
- **Both endpoints, one API.** `relay()` auto-routes between `/v1/chat/completions` and `/v1/responses` — same handle, same tool loop, same intercept, same streaming consumer.
- **Sessions as Pydantic event logs.** Append-only, BYO storage. `session.model_dump()` to your DB; `Session.model_validate()` to resume — across processes, machines, or days.
- **Mid-loop context engineering.** Intercept callbacks can `append`, `insert`, `replace`, `edit_last_tool_result`, or `compact` the conversation that goes into the *next* iteration. Real context engineering in the harness, not just observation.
- **`ToolExecutor` seam.** Swap dispatch (local → remote → MCP → sub-agent) without touching the harness.
- **Terminal as a first-class primitive.** Persistent bash subprocess that retains cwd/env/venvs across calls. Wrap one in a closure and your agent has a shell.
- **Full sync/async parity.** Every helper has a `*_async` twin; async tool callables and async intercept callbacks just work.
- **Streaming with the loop intact.** Stream tokens *and* tool calls in real time across both endpoints.

## 60-second tour

```python
import json
import encode

# 1. Plain chat
encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
).response.content

# 2. With a tool — auto-loop runs until the model stops calling tools
def lookup(q: str) -> dict:
    """Look something up."""
    return {"q": q, "answer": "42"}

encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the meaning of life?"}],
    tools=[lookup],
).response.content

# 3. With a Session — durable, resumable, BYO persistence
session = encode.Session.open()
encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "remember: my name is Alex"}],
    session=session,
).response

# Persist anywhere — file, Postgres, Redis, S3...
open("/tmp/agent.json", "w").write(json.dumps(session.model_dump(), default=str))

# Resume anywhere
resumed = encode.Session.model_validate(json.loads(open("/tmp/agent.json").read()))
encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "what's my name?"}],
    session=resumed,
).response.content   # → "Alex"
```

## Install

```bash
pip install encode
```

Python 3.10+. Configure with a `.env` (auto-loaded) or kwargs:

```bash
# .env
ENCODE_API_KEY=sk-your-key
ENCODE_BASE_URL=https://api.openai.com/v1
```

`OPENAI_API_KEY` / `OPENAI_BASE_URL` are also picked up if `ENCODE_*` aren't set.

## What you can build

**Multi-turn chat with stateful `Messages`** — pass a `Messages` instance and it grows itself across `relay()` calls.

```python
m = encode.Messages().system("Be brief.")
m.user("name three colors")
encode.relay(model="m", messages=m).response   # m now has the assistant turn too
```

**An agent that stops when it submits a final answer.**

```python
def watcher(event):
    if any(tc.name == "submit_final" for tc in event.tool_calls):
        event.stop()

encode.relay(..., tools=[search, submit_final]).intercept(watcher).response
```

**Mid-loop context compaction** — trim, summarize, redact, or rewrite history without subclassing anything.

```python
def trim(event):
    event.edit_last_tool_result(lambda c: c[:1000])

encode.relay(..., tools=[noisy_tool], on_intercept=trim).response
```

**A bash sandbox tool.**

```python
class BashSandbox:
    def __init__(self):
        self._term = None
    def as_tool(self):
        sandbox = self
        def bash(command: str) -> dict:
            """Run a bash command in a persistent shell."""
            self._term = self._term or encode.Terminal()
            r = self._term.run(command, timeout=10.0)
            return {"output": r.output, "exit_code": r.exit_code, "cwd": r.cwd}
        return bash

encode.relay(..., tools=[BashSandbox().as_tool()]).response
```

## Docs

→ [docs.md](./docs.md) is the entry point — concept map + a link grid into focused topic pages:

[quickstart](./docs/quickstart.md) · [concepts](./docs/concepts.md) · [relay()](./docs/relay.md) · [messages](./docs/messages.md) · [tools](./docs/tools.md) · [intercept](./docs/intercept.md) · [sessions](./docs/sessions.md) · [executors](./docs/executors.md) · [terminal](./docs/terminal.md) · [streaming](./docs/streaming.md) · [structured output](./docs/structured-output.md) · [whisper](./docs/whisper.md) · [async](./docs/async.md) · [errors](./docs/errors.md) · [cookbook](./docs/cookbook.md)

## Status & compatibility

- Python 3.10+
- macOS / Linux for `Terminal` (pexpect-backed)
- OpenAI-compatible endpoints: OpenAI, [Courier](https://getcourier.ai), vLLM, LM Studio, Ollama, Together, Groq, and others

## License

MIT
