# Cookbook

Runnable recipes. Each one is self-contained and lifts from the SDK's [examples/](../examples/) directory where applicable.

---

## Multi-turn chat with `Messages`

```python
import encode

m = encode.Messages().system("You are concise.")

while True:
    user = input("> ")
    if not user:
        break
    m.user(user)
    out = encode.relay(model="gpt-4o-mini", messages=m).response
    print(out.content)    # m already has the assistant turn appended
```

→ [messages.md](./messages.md)

---

## Agent with a final-answer tool + `event.stop()`

```python
import encode

def search(query: str) -> dict:
    """Search the index."""
    return do_search(query)

def submit_final(answer: str) -> dict:
    """Submit the final answer when done."""
    return {"submitted": answer}

def watcher(event):
    if any(tc.name == "submit_final" for tc in event.tool_calls):
        event.stop()

out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Find the latest python release."}],
    tools=[search, submit_final],
    max_tool_iterations=20,
).intercept(watcher).response

final = next(tc for tc in out.tool_calls if tc.name == "submit_final")
print(final.arguments["answer"])
```

→ [intercept.md](./intercept.md)

---

## Durable run — save and resume across processes

```python
import json
from pathlib import Path
import encode

def lookup(q: str) -> dict:
    """Look something up."""
    facts = {"capital_of_france": "Paris", "highest_mountain": "Everest"}
    return {"q": q, "answer": facts.get(q, "unknown")}

# --- first turn ---
session = encode.Session.open()
out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "look up the capital_of_france"}],
    tools=[lookup],
    session=session,
).response
print("first turn →", out.content)

# Persist however you like
Path("/tmp/agent.json").write_text(json.dumps(session.model_dump(), default=str))

# --- later, in a fresh process ---
raw = json.loads(Path("/tmp/agent.json").read_text())
session = encode.Session.model_validate(raw)

out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "what about highest_mountain?"}],
    tools=[lookup],
    session=session,
).response
print("second turn →", out.content)
print("events on session:", len(session.events))
```

Source: [`examples/session_resume.py`](../examples/session_resume.py) — [sessions.md](./sessions.md)

---

## Mid-loop context engineering — trim noisy output + nudge to wrap up

```python
import encode

def search(query: str) -> dict:
    """Fake search tool.

    Args:
        query: Search string.
    """
    return {
        "query": query,
        "results": [f"hit-{i}: lots of irrelevant text" for i in range(50)],
    }

def summarize_tool_results(content):
    if len(content) > 200:
        return content[:200] + "  …[trimmed]"
    return content

def watcher(event):
    event.edit_last_tool_result(summarize_tool_results)
    if event.iteration >= 1:
        event.append({"role": "system",
                      "content": "You have enough info — answer now."})

session = encode.Session.open()
out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "search 'glacier melt rates'"}],
    tools=[search],
    session=session,
    on_intercept=watcher,
).response

print("final:", out.content)
print("context.modify events:",
      len(session.events_by_type("context.modify")))
```

Source: [`examples/intercept_compact.py`](../examples/intercept_compact.py) — [intercept.md](./intercept.md)

---

## Bash sandbox tool

```python
import encode

class BashSandbox:
    def __init__(self) -> None:
        self._terminal: encode.Terminal | None = None

    def _ensure(self) -> encode.Terminal:
        if self._terminal is None or not self._terminal.alive:
            self._terminal = encode.Terminal()
        return self._terminal

    def teardown(self) -> None:
        if self._terminal is not None:
            self._terminal.kill()
            self._terminal = None

    def as_tool(self):
        sandbox = self

        def bash(command: str) -> dict:
            """Run a bash command in a persistent shell.

            Args:
                command: A bash command. State (cwd, env vars) persists across calls.
            """
            r = sandbox._ensure().run(command, timeout=10.0)
            return {"command": command, "output": r.output,
                    "exit_code": r.exit_code, "cwd": r.cwd}

        return bash


sandbox = BashSandbox()
session = encode.Session.open()
try:
    out = encode.relay(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
                   "Create /tmp/encode_demo, write hello.py that prints 'hello', and run it."}],
        tools=[sandbox.as_tool()],
        session=session,
        max_tool_iterations=6,
    ).response
    print("answer:", out.content)
    print("shell commands:", len(session.events_by_type("tool.call")))
finally:
    sandbox.teardown()
```

Source: [`examples/stateful_tool.py`](../examples/stateful_tool.py) — [terminal.md](./terminal.md)

---

## Async pipeline — persist mid-loop

```python
import asyncio
import encode

async def persist(event):
    # event.messages_so_far is a snapshot — safe to ship to DB
    await db.write(conversation_id, event.messages_so_far)

async def fetch(url: str) -> dict:
    """Fetch a URL."""
    async with httpx.AsyncClient() as c:
        r = await c.get(url)
        return {"status": r.status_code, "body": r.text[:200]}

async def main():
    await encode.relay_async(
        model="m",
        messages=[{"role": "user", "content": "Fetch and summarize example.com."}],
        tools=[fetch],
        on_intercept=persist,
    )

asyncio.run(main())
```

→ [async.md](./async.md), [intercept.md](./intercept.md)

---

## Streaming agent UI events

```python
import encode

def get_weather(city: str) -> dict:
    """Get weather by city."""
    return {"city": city, "temp_f": 72}

for ev in encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Denver?"}],
    tools=[get_weather],
    stream=True,
):
    match ev.type:
        case "content.delta":
            print(ev.data, end="", flush=True)
        case "tool_call.start":
            print(f"\n[calling {ev.data['name']}({ev.data['arguments']})]")
        case "tool_call.result":
            print(f"[result: {ev.data['result']}]")
        case "tool_call.error":
            print(f"[tool error: {ev.data['error']}]")
        case "iteration.end":
            ...
        case "finish":
            print(f"\n[done: {ev.data}]")
```

→ [streaming.md](./streaming.md)

---

## Structured output (Pydantic in / Pydantic out)

```python
from pydantic import BaseModel
import encode

class Triage(BaseModel):
    reasoning: str
    priority: str   # urgent | normal | low
    tags: list[str]

out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user",
               "content": "Triage: 'Server is on fire, customers can't log in.'"}],
    response_format=Triage,
).response

t = out.parsed
print(t.priority, t.tags)
```

→ [structured-output.md](./structured-output.md)

---

## Whisper — verbose JSON with word timestamps

```python
import encode

out = encode.whisper(
    file="./meeting.m4a",
    response_format="verbose_json",
    timestamp_granularities=["word", "segment"],
)
print(out.language, out.duration)
for w in out.words or []:
    print(f"{w['start']:>6.2f} {w['word']}")
```

→ [whisper.md](./whisper.md)

---

## Branching conversations

```python
import encode

shared = encode.Messages().system("Be helpful.").user("Topic: birds")

owls = shared.copy().user("Tell me about owls")
hawks = shared.copy().user("Tell me about hawks")

encode.relay(model="m", messages=owls).response
encode.relay(model="m", messages=hawks).response

# owls and hawks now have independent assistant turns appended
```

→ [messages.md](./messages.md)
