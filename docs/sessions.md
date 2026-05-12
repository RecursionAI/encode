# Sessions

A `Session` is an append-only event log that lives **outside** the harness. The SDK takes zero opinion on storage — serialize with `model_dump()`, rehydrate with `model_validate()`, plug into whatever DB you already use.

```python
session = encode.Session.open()

encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
    session=session,
).response

# Persist anywhere
db.save(session.model_dump())

# Resume anywhere, anytime
session = encode.Session.model_validate(db.load(sid))
```

## Why sessions are separate from `Messages`

| Concern             | `Messages`                        | `Session`                                 |
| ------------------- | --------------------------------- | ----------------------------------------- |
| What it holds       | OpenAI-format messages            | Typed events (`user.message`, `tool.call`, `iteration.end`, …) |
| Mutability          | Mutable, list-like                | Append-only                                |
| Persistence         | `to_pydantic()` → `Conversation`  | `model_dump()` → dict                      |
| Granularity         | Coarse (one turn = one entry)     | Fine (every tool call, every iteration)    |
| Concept             | Context window                    | Audit log                                  |

You can use either, both, or neither. They compose — pass `session=` to `relay()` and the harness hydrates the context window from the session at the start of each iteration.

## Session model

```python
class Session(BaseModel):
    id: str                        # UUID by default
    created_at: datetime           # UTC
    updated_at: datetime           # bumped on each emit
    events: list[Event]
    metadata: dict[str, Any]       # user-extensible
```

### Constructing

```python
session = encode.Session.open()                          # fresh UUID
session = encode.Session.open(id="my-session-id")        # specific id
session = encode.Session.open(metadata={"user": "alex"}) # arbitrary metadata
```

## Events

```python
class Event(BaseModel):
    id: int           # monotonic per-session, assigned on emit
    ts: datetime      # UTC, set on emit
    type: str         # one of EventType OR any user string
    data: dict        # shape depends on type
```

Standard types (`EventType.*`):

| Constant            | String           | `data` shape                                                 |
| ------------------- | ---------------- | ------------------------------------------------------------ |
| `USER_MESSAGE`      | `user.message`   | `{"content": str | list[ContentPart]}`                       |
| `ASSISTANT_MESSAGE` | `assistant.message` | `{"content": str | None, "tool_calls": [...] | None}`     |
| `TOOL_CALL`         | `tool.call`      | `{"id": str, "name": str, "arguments": dict, "iteration": int}` |
| `TOOL_RESULT`       | `tool.result`    | `{"id": str, "result": Any, "result_serialized": str, "error": str | None, "duration_ms": float}` |
| `TOOL_REGISTERED`   | `tool.registered` | `{"name": str, "schema": dict, "by": str}` (emitted by `register_tool` / `Session.open(tools=...)` / `rebind_tools`) |
| `ITERATION_END`     | `iteration.end`  | `{"iteration": int, "had_tool_calls": bool, "finish_reason": str | None}` |
| `CONTEXT_MODIFY`    | `context.modify` | `{"by": str, "summary": str, ...}` (emitted by Intercept mutations) |
| `SYSTEM`            | `system`         | `{"content": str}`                                           |
| `CUSTOM`            | `custom`         | anything                                                     |

Custom event types are also fine — any string works.

### Type-safe construction

The factories on `Event` build correctly-shaped payloads for you:

```python
ev = encode.Event.user_message("hi")
ev = encode.Event.assistant_message("done", tool_calls=[{"id": "c1", ...}])
ev = encode.Event.tool_call(id="c1", name="search", arguments={"q": "x"}, iteration=0)
ev = encode.Event.tool_result(id="c1", result={"hits": 3}, result_serialized='{"hits":3}')
ev = encode.Event.iteration_end(iteration=0, had_tool_calls=True, finish_reason="tool_calls")
ev = encode.Event.system("Reminder: be brief.")
ev = encode.Event.custom("metric.tokens", {"prompt": 42, "completion": 7})
```

## Writing (append-only)

```python
session.emit("user.message", {"content": "hi"})
session.emit(encode.Event.user_message("hi"))   # via factory
session.emit(EventType.USER_MESSAGE, {"content": "hi"})

# Append a custom analytics event
session.emit("metric.latency", {"ms": 142})
```

`emit()` assigns the next monotonic `id`, sets `ts`, refreshes `updated_at`, and appends. Past events are never mutated; ids never reused.

Async sessions add `aemit` for parity:

```python
session = encode.AsyncSession.open()
await session.aemit("user.message", {"content": "hi"})
```

## Reading

```python
session.events                          # list[Event] — full log
session.last_event_id                   # int (-1 if empty)
session.events_since(n)                 # events with id > n (cursor pattern)
session.events_by_type("tool.call")     # filter by type
session.events_by_type("tool.call", "tool.result")
session.events_slice(start, end)        # half-open positional slice
```

The cursor pattern is useful for tailing into UIs or downstream processors:

```python
cursor = -1
while True:
    new = session.events_since(cursor)
    for e in new:
        publish(e)
        cursor = e.id
```

## Projection — `to_messages()`

The model never sees events. The harness projects them into a [Messages](./messages.md) context window via `session.to_messages()`.

```python
m = session.to_messages()
# → Messages projected from user.message / assistant.message / tool.result / system events
```

Default projection skips bookkeeping events (`tool.call` is redundant with `assistant.message.tool_calls`; `iteration.end` / `context.modify` / `custom` aren't part of the context window).

Pass a `transform` callable to compact or filter before projection:

```python
def trim(events):
    return events[-50:]   # last 50 events only

m = session.to_messages(transform=trim)
```

## Resume

There's no special "resume" API. Pydantic round-trip is the resume:

```python
# Save
db.save(session.model_dump())                # dict
db.save(session.model_dump_json())           # str

# Load
session = encode.Session.model_validate(db.load(sid))
session = encode.Session.model_validate_json(db.load_str(sid))

# Continue
encode.relay(model="m", messages=[...], session=session).response
```

Works across processes, machines, and days — `Session` is pure data.

## `relay(session=...)` semantics

When you pass `session=` to `relay()`:

1. **Hydration.** `session.to_messages()` builds the initial context window. Any `messages=` you also passed is *appended* as new `user.message` / `system` / `assistant.message` / `tool.result` events first (so multi-turn flows just keep adding to the log).
2. **Per-iteration emission.** Each loop iteration emits `assistant.message`, `tool.call` (one per call), `tool.result` (one per call), and an `iteration.end`.
3. **Intercept mutations.** If an intercept callback mutates `event.messages`, a `context.modify` event is emitted as well.

After the loop the session is up-to-date; persist whenever you like.

```python
session = encode.Session.open()
encode.relay(model="m",
             messages=[{"role": "user", "content": "hi"}],
             tools=[my_tool],
             session=session).response

types = [e.type for e in session.events]
# ['user.message', 'assistant.message', 'tool.call', 'tool.result',
#  'iteration.end', 'assistant.message', 'iteration.end']
```

## Concurrency

A `Session` instance is owned by **one process**. The SDK does not coordinate across writers — if two processes mutate the same logical session id, you'll get id collisions on whichever one persists last.

Recommended patterns:

- **Single writer**: one process owns the in-memory `Session`; persist after each `relay()` call.
- **Atomic append at the DB layer**: write events one at a time (e.g. `INSERT` with serial id from a sequence) and rehydrate before each `relay()`.

## Session-owned tools

A Session can also own an **append-only tool registry** — pass `tools=session.tools` to `relay()` and you can grow the registry mid-loop (typically from an intercept callback) so a single agent run can discover and start using new tools without re-launching.

```python
def search(query: str) -> dict:
    """Search the index."""
    return ...

def list_tools() -> list[dict]:
    """Discover available tools."""
    return [...]   # tool schemas as raw dicts

session = encode.Session.open(tools=[search, list_tools])

encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "discover, then use what you find"}],
    session=session,
    tools=session.tools,        # ← the live registry
    on_intercept=lambda ev: [ev.register_tool(s) for tc in ev.tool_calls
                             if tc.name == "list_tools" for s in (tc.result or [])],
).response
```

API:

```python
session.register_tool(fn_or_dict)           # idempotent: same name → no-op, returns False
session.register_tools([fn1, fn2, ...])     # bulk, returns count newly added
session.tools                               # list[Any] — the live registry
```

Each `register_tool` call emits a `tool.registered` event into the durable log; the `by` field records the origin (`"user"`, `"intercept"`, `"resume"`).

### How auto-discovery flows through the loop

The relay loop re-reads `tools=` at the top of each iteration. When it's the same list reference as `session.tools`, additions made during an intercept callback show up on the **next** iteration's request to the model. The in-process executor's dispatch table is also rebuilt automatically — your registered callable is callable next turn.

If you pass a different list as `tools=` (i.e. not `session.tools`), `event.register_tool(...)` still appends to the session and emits the audit-log event, but the new tool won't appear in the model's next request — pass `tools=session.tools` to opt in.

### Idempotency

Same-name registrations are silently skipped. This makes auto-discovery loops safe to re-trigger: if the model calls `list_tools` twice and returns overlapping specs, only the new names are registered. Both `register_tool(fn)` and `register_tools([...])` return how many entries were *newly* added.

### Resuming with tools

`session.model_dump()` does **not** include the live `tools` list — Python callables aren't JSON-serializable. The `tool.registered` events are part of the log, though, so they survive the round-trip. Use `rebind_tools` (or the `Session.resume` convenience class method) to bind your callables back to the session on the other side:

```python
# round-trip — anything that survives JSON works
raw = json.dumps(session.model_dump(), default=str)

# Method 1: one-liner via Session.resume
session = encode.Session.resume(json.loads(raw), tools=[search, list_tools])

# Method 2: split into validate + rebind
session = encode.Session.model_validate(json.loads(raw))
missing = session.rebind_tools([search, list_tools])
if missing:
    print(f"missing callables for: {missing}")   # names from the log without a binding

# Continue the run
encode.relay(model="m", messages=[...], session=session, tools=session.tools).response
```

`rebind_tools` walks `tool.registered` events in order, matches each name against the callables you supplied (by `__name__` for functions, `function.name` / `name` for dicts), and registers them with `by="resume"`. It's idempotent: calling it twice on the same session is safe.

Async parity: `aregister_tool` / `aregister_tools` / `arebind_tools` on `AsyncSession`, and `AsyncSession.resume`.

## Custom events

Emit whatever you want. Useful for application-level audit:

```python
session.emit("billing.charge", {"user_id": "u_1", "cents": 42})
session.emit("safety.flag",    {"reason": "PII", "redacted_count": 2})
```

`Messages.from_events` and `session.to_messages()` skip unknown event types, so custom events don't leak into the model's context.

## See also

- [concepts.md](./concepts.md) — why session is a primitive
- [messages.md](./messages.md) — the context window the session projects into
- [intercept.md](./intercept.md) — `context.modify` events
- [cookbook.md](./cookbook.md) — durable-resume recipe
