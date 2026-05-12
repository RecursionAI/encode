# Intercept

`intercept` is the callback that fires between iterations of the tool loop. Use it to observe (log, persist, audit) or to **mutate** the next iteration's input (inject reminders, compact history, trim noisy tool output, branch).

```python
def watcher(event):
    print(f"iter {event.iteration}: {[tc.name for tc in event.tool_calls]}")

encode.relay(...).intercept(watcher).response
# or:
encode.relay(..., on_intercept=watcher).response
```

Both forms append to the same internal list — chain multiple:

```python
encode.relay(...).intercept(log).intercept(audit).response
```

Interceptors fire **only on iterations that had tool calls**. The final iteration where the model returns plain text doesn't trigger them.

If a callback raises, encode logs at WARNING and continues — one bad observer doesn't kill the loop.

## The event

```python
@dataclass
class InterceptEvent:
    iteration: int                       # 0-based loop index that just completed
    endpoint: Literal["chat", "responses"]
    assistant_turn: AssistantTurn        # what the model said this iteration
    tool_calls: list[ToolCallRecord]     # parsed calls + executed results
    raw_response: dict                   # the raw API response body
    will_continue: bool                  # True iff another API call is queued
    messages: Messages                   # LIVE, mutable conversation view

    @property
    def messages_so_far(self) -> list[dict]: ...   # read-only snapshot
    @property
    def mutated(self) -> bool: ...                  # True if you changed messages
    @property
    def stopped(self) -> bool: ...

    def stop(self) -> None: ...

    # mutation helpers
    def append(self, msg) -> None: ...
    def insert(self, idx, msg) -> None: ...
    def replace(self, msgs) -> None: ...
    def edit_last_tool_result(self, fn) -> None: ...
    def compact(self, fn) -> None: ...
```

## Observing

Pure read — log, persist, audit. No mutation = no change to the loop.

```python
async def persist(event):
    await db.write(conversation_id, event.messages_so_far)

await encode.relay_async(
    model="m",
    messages=[...],
    tools=[my_tool],
    on_intercept=persist,
)
```

## Stopping early

`event.stop()` is the **only** way to break out — there's no `return False` sentinel. Makes intent obvious in code review and discoverable via autocomplete.

```python
def watcher(event):
    if any(tc.name == "submit_final" for tc in event.tool_calls):
        event.stop()    # loop exits cleanly after this iteration

encode.relay(..., tools=[search, submit_final]).intercept(watcher).response
```

## Mutating

`event.messages` is a **live** [Messages](./messages.md) view of the conversation as it stands now. Anything you change is picked up by the harness and becomes the input to the next iteration.

### Append — inject a reminder

```python
def nudge(event):
    if event.iteration >= 2:
        event.append({"role": "system", "content": "You have enough info — answer now."})
```

### Edit the last tool result — trim noisy output

```python
def trim(event):
    event.edit_last_tool_result(lambda content: content[:1000])
```

### Compact — replace history with a summary

```python
def summarize_old_turns(messages):
    if len(messages) <= 6:
        return messages
    older = messages[:-6]
    summary = my_summarizer.run(older)
    return [{"role": "system", "content": f"Earlier: {summary}"}, *messages[-6:]]

def compact(event):
    if len(event.messages) > 20:
        event.compact(summarize_old_turns)
```

### Replace — branch wholesale

```python
def reset(event):
    event.replace([
        {"role": "system", "content": "Try a different approach."},
        {"role": "user", "content": "Re-attempt the task."},
    ])
```

### Insert — slot a message at a position

```python
event.insert(0, {"role": "system", "content": "Reminder: use markdown."})
```

## Mutation + Sessions

When a `Session` is active, every mutation also emits a `context.modify` event into the durable log:

```python
session = encode.Session.open()

def watcher(event):
    event.append({"role": "system", "content": "stay focused"})

encode.relay(model="m", messages=[...], tools=[t], session=session,
             on_intercept=watcher).response

# Audit trail of what the harness did to context:
for e in session.events_by_type("context.modify"):
    print(e.data)
# → [{"by": "intercept", "summary": "...", ...}]
```

→ [sessions.md](./sessions.md)

## `messages` vs `messages_so_far`

| Attribute              | Type                        | Mutates the loop?              |
| ---------------------- | --------------------------- | ------------------------------ |
| `event.messages`       | `Messages` (live view)      | Yes — changes feed the next iteration |
| `event.messages_so_far` | `list[dict]` (snapshot copy) | No — read-only                 |

Use `messages_so_far` when you want a frozen snapshot to log or persist. Use `messages` (or the helpers) when you want to **change what the model sees next**.

## Responses endpoint caveat

For `/v1/responses`, mutating `messages` causes the harness to reproject `input` items from the chat-style history on the next iteration. This **loses any non-message typed items** the model emitted (e.g. `reasoning` items). For most context-engineering use cases that's fine. If you need to preserve typed items, leave `messages` alone and use `stop()` instead.

## Async

Async callbacks are awaited:

```python
async def cb(event):
    await persist_to_db(event.messages_so_far)

await encode.relay_async(...).intercept(cb)
```

Async callbacks can still call the sync mutation helpers — they operate on the in-memory `Messages` view.

## Cookbook — full agent + stop()

```python
def search(q: str) -> dict:
    """Search the index."""
    return do_search(q)

def submit_final(answer: str) -> dict:
    """Submit when done."""
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

## See also

- [messages.md](./messages.md) — the container `event.messages` is a view of
- [sessions.md](./sessions.md) — `context.modify` events when sessions are active
- [cookbook.md](./cookbook.md) — mid-loop compaction recipe
