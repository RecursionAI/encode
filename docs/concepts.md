# Concepts

encode is shaped around the three primitives in Anthropic's [Scaling Managed Agents](https://www.anthropic.com/engineering/managed-agents) post: a **Session** that owns *what happened*, a **Harness** that owns *what to do next*, and **Hands** that own *how to run a tool*. Understanding which lives where is the difference between fighting the SDK and flowing with it.

## The three primitives

```
┌──────────────────────────────┐
│  Session  (durable, Pydantic)│   ← what happened (append-only event log)
└──────────────────────────────┘
              │  to_messages()
              ▼
┌──────────────────────────────┐
│  Harness  (relay(), stateless)│  ← what to do next (loop + dispatch)
└──────────────────────────────┘
              │  ToolExecutor.execute(name, input)
              ▼
┌──────────────────────────────┐
│  Hands    (ToolExecutor)     │   ← how to run a tool (local / remote / sub-agent)
└──────────────────────────────┘
```

### Session — durable, append-only

`encode.Session` is a Pydantic model containing a list of `Event` records. Events are immutable once appended; ids are monotonic per-session. The SDK takes **zero opinion** on storage — `session.model_dump()` gives you a dict to put in your DB, and `Session.model_validate(raw)` brings it back.

```python
session = encode.Session.open()
session.emit("user.message", {"content": "hi"})
# later:
db.save(session.model_dump())
# even later, anywhere:
session = encode.Session.model_validate(db.load(sid))
```

The session is **not** the context window. The model never sees raw events — the harness projects them via `session.to_messages()` (optionally through a `transform=` callable for compaction). This separation is the whole point: durable record on one side, ephemeral prompt on the other.

→ Full reference: [sessions.md](./sessions.md)

### Harness — stateless, swappable, cattle

`encode.relay()` is the harness. It:

1. Resolves which endpoint to hit (`/v1/chat/completions` or `/v1/responses`).
2. Builds the request from `messages` / `input` / `instructions`, plus `tools`.
3. Hits the model.
4. If the model called tools, dispatches them through the executor, appends results, and loops.
5. Emits events into the session along the way.

The harness holds no state across calls — everything it needs lives in the session (if you gave it one) or in the `Messages` / `list` you passed. Crash mid-loop and the durable log is intact; rebuild a fresh harness and resume.

→ Full reference: [relay.md](./relay.md) and [intercept.md](./intercept.md)

### Hands — uniform `execute(name, input)`

`ToolExecutor` is a [Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol):

```python
class ToolExecutor(Protocol):
    def execute(self, name: str, input: dict) -> ExecutionResult: ...
    async def execute_async(self, name: str, input: dict) -> ExecutionResult: ...
```

The default `LocalToolExecutor` runs Python callables in-process. Swap in a remote executor (HTTP/gRPC dispatch), an MCP client, a sub-agent (a tool whose body is another `relay()`), or a vault-backed executor — the harness doesn't know or care.

```python
encode.relay(model="m", messages=[...], tool_executor=MyRemoteExecutor()).response
```

→ Full reference: [executors.md](./executors.md)

## What's *not* a primitive

A few things deliberately stay below the primitive line:

- **`Messages`** is the ephemeral context window — a mutable list you can build, mutate, branch, and persist. It is **not** the session. You can use one without the other. → [messages.md](./messages.md)
- **`Terminal`** is just a stateful tool. A persistent bash subprocess wrapped as a callable that goes through the `ToolExecutor` seam like any other tool. No separate sandbox abstraction — sandboxes are tools. → [terminal.md](./terminal.md)
- **Intercept callbacks** are mutation points *inside* the harness loop — not a fourth primitive. They sit between iterations and can rewrite the next prompt or `stop()` the loop. → [intercept.md](./intercept.md)

## Putting it together

The minimum durable agent:

```python
import encode

def lookup(q: str) -> dict:
    """Look something up."""
    return {"q": q, "answer": "42"}

session = encode.Session.open()
out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "look up the meaning of life"}],
    tools=[lookup],
    session=session,
).response

# Persist anywhere — JSON file, Postgres, Redis, S3...
open(f"/tmp/{session.id}.json", "w").write(session.model_dump_json())

# Later, anywhere:
raw = open(f"/tmp/{session.id}.json").read()
resumed = encode.Session.model_validate_json(raw)
encode.relay(model="gpt-4o-mini",
             messages=[{"role": "user", "content": "and what about death?"}],
             tools=[lookup],
             session=resumed).response
```

That's the whole shape. One harness call per turn, the session carries the history, tools dispatch through whatever executor you wire in.

## See also

- [sessions.md](./sessions.md) — event log API
- [relay.md](./relay.md) — the harness call
- [executors.md](./executors.md) — swapping the hands
- [cookbook.md](./cookbook.md) — full runnable recipes
