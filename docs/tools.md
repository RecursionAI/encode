# Tools

Drop Python functions into `tools=`. encode introspects them, builds OpenAI tool schemas, runs the loop, and feeds results back to the model — no decorators, no manual schema, no loop scaffolding.

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
```

That's the whole API. The loop runs until the model stops calling tools.

## What encode reads from your function

| Source                       | Becomes                                          |
| ---------------------------- | ------------------------------------------------ |
| Function name                | `function.name` in the tool schema               |
| Docstring summary (1st line) | `function.description`                           |
| `Args:` block                | Per-parameter `description` in the JSON schema   |
| Type annotations             | JSON Schema types via Pydantic                   |
| Default values               | `default` in the JSON schema (parameter optional) |
| No default                   | Parameter goes in `required`                     |

Example function and the schema it produces (sketch):

```python
def search_docs(query: str, top_k: int = 5) -> list[dict]:
    """Search project docs.

    Args:
        query: Search string.
        top_k: How many results to return.
    """
```

```json
{
  "type": "function",
  "function": {
    "name": "search_docs",
    "description": "Search project docs.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search string."},
        "top_k": {"type": "integer", "description": "How many results to return.", "default": 5}
      },
      "required": ["query"]
    }
  }
}
```

## What the loop does

1. Sends the request with your tools.
2. If the assistant turn has no tool calls, returns the response. Done.
3. Otherwise, parses each tool call's `arguments` as JSON, calls your function with kwargs, serializes the return value (`json.dumps(default=str)`), appends a `tool` message, and loops back to step 1.

## Tool exceptions are non-fatal

If your function raises, encode catches the exception and feeds back `{"error": "ExceptionType('msg')"}` as the tool result. The model gets a chance to recover — the loop never crashes from a buggy tool.

```python
def flaky(x: int) -> int:
    """Sometimes fails."""
    if x < 0:
        raise ValueError("x must be non-negative")
    return x * 2

out = encode.relay(model="m", messages=[...], tools=[flaky]).response

for tc in out.tool_calls:
    if tc.error:
        print(f"{tc.name} failed: {tc.error}")
```

`tc.error` is the `repr()` of the exception; `tc.result` is `{"error": "..."}`.

## Mixing callables and raw dicts

```python
encode.relay(
    model="m",
    messages=[...],
    tools=[
        get_weather,                              # callable
        {"type": "function", "function": {...}},  # pre-built dict (server-side or remote tool)
    ],
).response
```

A raw dict has no Python callable behind it — useful for server-resolved tools (e.g. `web_search`) or when the executor handles dispatch itself ([executors.md](./executors.md)).

## Web search

If your provider supports the shorthand:

```python
encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's trending today?"}],
    web_search=True,    # appends {"type": "web_search"} to tools
).response
```

The provider resolves the search server-side; the SDK doesn't see the tool call as a function-call to dispatch.

## Capping iterations

The loop is **unbounded by default**. Add a safety net when the model can't be trusted to converge:

```python
try:
    out = encode.relay(model="m", messages=[...], tools=[t], max_tool_iterations=10).response
except encode.MaxToolIterationsError as e:
    partial: encode.RelayResponse = e.partial   # everything that ran so far
    print(f"got {partial.iterations} iters and {len(partial.tool_calls)} tool calls")
```

`MaxToolIterationsError.partial` is a `RelayResponse` with `messages`/`tool_calls`/`iterations` filled in — recover or report as you like.

## tool_choice

Force the model to call a specific tool or none at all:

```python
encode.relay(..., tools=[a, b, c], tool_choice="required").response
encode.relay(..., tools=[a, b, c], tool_choice="none").response
encode.relay(..., tools=[a, b, c],
             tool_choice={"type": "function", "function": {"name": "a"}}).response
```

Forced names must exist in `tools` — the server returns `InvalidToolChoiceError` otherwise.

## Async tools

Async callables are accepted in `relay_async()`. Sync `relay()` rejects them with `TypeError`.

```python
async def fetch(url: str) -> dict:
    """Fetch a URL."""
    async with httpx.AsyncClient() as c:
        r = await c.get(url)
        return {"status": r.status_code, "body": r.text[:200]}

await encode.relay_async(model="m", messages=[...], tools=[fetch])
```

→ [async.md](./async.md)

## Return values

Anything `json.dumps(default=str)` can handle works. Strings are passed through unchanged; everything else is JSON-serialized. If serialization itself fails, encode sends back `{"error": "failed to serialize tool result: ..."}` and lets the model see it.

```python
def query(sql: str) -> list[dict]:        # → JSON array
    ...
def stringy(q: str) -> str:                # → passed through
    return "ok"
def with_objects(q: str) -> dict:          # default=str handles datetimes etc.
    return {"created_at": datetime.now(), "rows": 3}
```

## See also

- [intercept.md](./intercept.md) — observe / mutate between iterations
- [executors.md](./executors.md) — swap dispatch (remote, MCP, sub-agent)
- [terminal.md](./terminal.md) — the canonical stateful tool pattern
- [errors.md](./errors.md) — `MaxToolIterationsError`, `InvalidToolCallError`
