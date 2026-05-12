# Streaming

Set `stream=True` and iterate the handle. The handle yields parsed `StreamEvent` objects — not raw SSE bytes — so a single consumer can render both endpoints uniformly.

```python
handle = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a haiku."}],
    stream=True,
)

for event in handle:
    if event.type == "content.delta":
        print(event.data, end="", flush=True)
```

The upstream HTTP connection is held open with `httpx.stream()`, so backpressure works end-to-end. Drop the iterator and the connection drops with it.

## `StreamEvent`

```python
@dataclass
class StreamEvent:
    type: str             # event kind
    data: Any             # parsed payload (str for text deltas, dict otherwise)
    raw: dict | None      # full upstream chunk as parsed JSON (use to proxy verbatim)
```

## Event types — `/v1/chat/completions`

| `type`               | `data`                                                                |
| -------------------- | --------------------------------------------------------------------- |
| `content.delta`      | `str` — next token of assistant text                                  |
| `tool_calls.delta`   | `list[dict]` — partial tool-call fragments (raw upstream deltas)      |
| `tool_call.start`    | `{id, name, arguments: dict, iteration}` — assembled, about to dispatch |
| `tool_call.result`   | `{id, result, result_serialized, duration_ms, iteration}` — succeeded |
| `tool_call.error`    | `{id, error, iteration}` — tool raised; loop continues                |
| `iteration.end`      | `{iteration, had_tool_calls}` — one loop iteration completed          |
| `finish`             | `str` — final finish reason (`"stop"`, `"length"`, `"tool_calls"`, …) |

The `tool_call.*` events only fire when `tools=` is set. Without tools, you see `content.delta` / `finish` only. The raw `tool_calls.delta` events still fire when the model emits fragments — most chat-UI consumers can ignore them and key off `tool_call.start` / `tool_call.result` instead.

## Event types — `/v1/responses`

Upstream events are passed through with their `type` intact (`response.output_text.delta`, `response.completed`, …) and `data` set to the entire parsed event dict.

When `tools=` is set, encode also synthesizes the same `content.delta` / `tool_call.start` / `tool_call.result` / `tool_call.error` / `iteration.end` events as chat, so a single consumer works across both endpoints.

```python
for event in encode.relay(model="m", input="hi", stream=True):
    print(event.type, event.data)
```

## Streaming with tools (auto-loop)

`stream=True` and `tools=` work together. The SDK runs the same auto-tool-loop as `stream=False` but yields events as the iteration proceeds:

```python
def get_weather(city: str) -> dict:
    """Get current weather by city."""
    return {"city": city, "temp_f": 72}

for ev in encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Denver?"}],
    tools=[get_weather],
    stream=True,
):
    if ev.type == "content.delta":
        print(ev.data, end="", flush=True)
    elif ev.type == "tool_call.start":
        print(f"\n[calling {ev.data['name']}({ev.data['arguments']})]")
    elif ev.type == "tool_call.result":
        print(f"[result: {ev.data['result']}]")
    elif ev.type == "tool_call.error":
        print(f"[tool error: {ev.data['error']}]")
```

`max_tool_iterations` still works — exceeding the cap raises `MaxToolIterationsError` with `.partial` carrying the streamed-so-far state.

## Messages auto-update

`Messages` containers passed as `messages=` are mutated **when the stream finishes**, just like the non-stream path. If the consumer abandons the iterator early (`break`), the container stays unchanged — drain the iterator if you want the absorption.

```python
m = encode.Messages().user("Write a haiku.")

for ev in encode.relay(model="m", messages=m, stream=True):
    if ev.type == "content.delta":
        print(ev.data, end="", flush=True)
# m now has the assistant turn appended
```

## Async streaming

`relay_async` with `async for`:

```python
handle = encode.relay_async(
    model="m",
    messages=[...],
    tools=[get_weather],
    stream=True,
)
async for event in handle:
    if event.type == "content.delta":
        print(event.data, end="", flush=True)
```

→ [async.md](./async.md)

## Restrictions

- **`response_format` is not supported when streaming.** Combining them raises `ValueError` immediately — structured output isn't meaningful mid-stream.

## See also

- [relay.md](./relay.md) — `RelayHandle` iteration protocol
- [tools.md](./tools.md) — what produces the `tool_call.*` events
- [intercept.md](./intercept.md) — intercept still fires between iterations during streaming
