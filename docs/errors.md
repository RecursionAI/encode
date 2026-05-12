# Errors

All exceptions inherit from `encode.CourierError`. The OpenAI error envelope (`{"error": {"message", "type", "code"}}`) is mapped to specific classes so you can `except` the case you actually care about.

```python
try:
    encode.relay(model="m", messages=[...]).response
except encode.InvalidRequestError as e:
    print(e.code, e.type, e.status)
    print(e.raw)         # original parsed body
```

## Class table

| Class                          | When                                       |
| ------------------------------ | ------------------------------------------ |
| `AuthError`                    | 401, 403, missing API key                  |
| `InvalidRequestError`          | 400 generic                                |
| `InvalidToolCallError`         | `code=invalid_tool_call`                   |
| `InvalidToolChoiceError`       | `code=invalid_tool_choice`                 |
| `InvalidResponseFormatError`   | `type=invalid_response_format`             |
| `InvalidAudioError`            | `type=invalid_audio` (whisper)             |
| `RateLimitError`               | 429 (retried automatically before raising) |
| `ServerError`                  | 5xx (502/503/504 retried first)            |
| `TransportError`               | network / timeout, retries exhausted       |
| `MaxToolIterationsError`       | tool loop exceeded `max_tool_iterations`   |
| `TerminalError`                | bash subprocess failure                    |
| `TerminalTimeoutError`         | command exceeded its timeout               |

## Common attributes

Every `CourierError` carries:

- `.message: str` — human-readable
- `.type: str | None` — server-supplied error type
- `.code: str | None` — server-supplied error code
- `.status: int | None` — HTTP status
- `.raw: Any` — original parsed body (dict, str, or None)

## `MaxToolIterationsError`

Carries the in-progress `RelayResponse` so you can recover what happened so far:

```python
try:
    out = encode.relay(
        model="m",
        messages=[...],
        tools=[my_tool],
        max_tool_iterations=5,
    ).response
except encode.MaxToolIterationsError as e:
    partial: encode.RelayResponse = e.partial
    print(f"got {partial.iterations} iters before cap")
    print(f"messages so far: {len(partial.messages)}")
    print(f"tool calls: {[tc.name for tc in partial.tool_calls]}")
```

If you passed a `Messages` container, it's still mutated to the partial state on this error so you can resume cleanly.

## `TerminalTimeoutError`

Carries whatever output landed before the cutoff:

```python
try:
    sh.run("sleep 5", timeout=1.0)
except encode.TerminalTimeoutError as e:
    print(e.command)         # "sleep 5"
    print(e.partial_output)  # whatever bash flushed before the timeout
```

## Retry policy

- `502 / 503 / 504` and transport errors retry up to `max_retries=2` times with exponential backoff (0.25 / 0.5 / 1s).
- `429` honors the `Retry-After` header.
- POST bodies are kept in memory so retries are safe.
- Whisper reads the file upfront (single read), so its retries are safe too.

Once retries are exhausted, the appropriate error class is raised.

## Customizing retries

Build your own client:

```python
with encode.Client(api_key="...", base_url="...", max_retries=5) as client:
    out = client.relay(model="m", messages=[...]).response
```

## See also

- [relay.md](./relay.md) — `max_tool_iterations`
- [tools.md](./tools.md) — tool exceptions are non-fatal (fed back as `{"error": "..."}`)
- [terminal.md](./terminal.md) — `TerminalError` / `TerminalTimeoutError`
