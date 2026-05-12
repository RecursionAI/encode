# relay()

`encode.relay()` is the harness. One function spans both `/v1/chat/completions` and `/v1/responses`, runs the tool loop, handles streaming, and emits events into a session when you give it one.

## Signature

```python
encode.relay(
    *,
    model: str,
    messages: Sequence[Message | dict] | None = None,
    input: Any = None,
    instructions: str | None = None,
    tools: Sequence[Callable | dict] | None = None,
    tool_choice: str | dict | None = None,
    response_format: type[BaseModel] | dict | None = None,
    web_search: bool = False,
    max_tool_iterations: int | None = None,
    stream: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    max_output_tokens: int | None = None,
    stop: str | list[str] | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    user: str | None = None,
    endpoint: Literal["chat", "responses", "auto"] = "auto",
    extra_body: dict | None = None,
    on_intercept: Callable | Sequence[Callable] | None = None,
    session: Session | None = None,
    tool_executor: ToolExecutor | None = None,
    client: Client | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = 60.0,
) -> RelayHandle
```

`relay_async()` has the same signature and returns `AsyncRelayHandle`.

## Endpoint auto-routing

| You pass                   | Endpoint              |
| -------------------------- | --------------------- |
| `messages=`                | `/v1/chat/completions` |
| `input=` or `instructions=` | `/v1/responses`        |
| Force it                    | `endpoint="chat" \| "responses"` |

```python
# chat
encode.relay(model="m", messages=[{"role": "user", "content": "hi"}]).response

# responses
encode.relay(model="m", instructions="Be concise.", input="Summarize: ...").response

# force chat even with input=
encode.relay(model="m", input="hi", endpoint="chat").response
```

Tool loop, intercept, streaming, and session behavior are identical on both — the differences in wire format (`tool_calls` vs `function_call`, `role: "tool"` vs `function_call_output`) are handled internally.

## RelayHandle

`relay()` returns a handle; nothing fires until you ask for the result.

```python
handle = encode.relay(model="m", messages=[...])

handle.intercept(my_cb)   # chain — returns the handle
out = handle.execute()    # run, return RelayResponse
out = handle.response     # property: run + memoize (same object on repeat reads)

for event in handle:      # stream events (only when stream=True or no-tools path)
    ...
```

Async variant:

```python
handle = encode.relay_async(model="m", messages=[...])
out = await handle              # awaitable directly
out = await handle.execute()    # same thing
out = await handle.get()        # same thing
async for event in handle: ...  # streaming
```

## RelayResponse

```python
class RelayResponse(BaseModel):
    content: str | None              # final assistant text
    parsed: Any = None               # populated when response_format=PydanticModel
    messages: list[dict]             # full conversation in OpenAI format
    tool_calls: list[ToolCallRecord] # flat across all iterations
    iterations: int                  # number of API round trips
    finish_reason: str | None        # last server-side finish_reason
    endpoint: Literal["chat", "responses"]
    model: str
    raw: Any                         # last raw API response body
    usage: Usage | None              # token counts when available
```

`messages` is ready to feed back into the next `relay()` call as conversation history. Round-tripping is lossless on both endpoints.

## What relay does for you

- Normalizes `messages=` (mix of dicts and `Message` Pydantic models) to OpenAI shape.
- Builds OpenAI tool schemas from any `tools=` callables ([tools.md](./tools.md)).
- Runs the tool loop until the model stops calling tools (or `max_tool_iterations` is hit).
- Captures tool exceptions and feeds them back as `{"error": "..."}` so the model can recover.
- Auto-mutates a `Messages` container if you passed one ([messages.md](./messages.md)).
- Emits events into a `Session` if you passed one ([sessions.md](./sessions.md)).
- Memoizes `.response` so reading it twice is free.

## Capping iterations

The loop is unbounded by default. Pass `max_tool_iterations=N` to bail on stuck loops:

```python
try:
    out = encode.relay(model="m", messages=[...], tools=[t], max_tool_iterations=3).response
except encode.MaxToolIterationsError as e:
    partial: encode.RelayResponse = e.partial   # everything that happened so far
```

→ More: [errors.md](./errors.md)

## See also

- [intercept.md](./intercept.md) — observe and mutate between iterations
- [streaming.md](./streaming.md) — `stream=True` mode
- [structured-output.md](./structured-output.md) — `response_format=`
- [async.md](./async.md) — `relay_async()` specifics
