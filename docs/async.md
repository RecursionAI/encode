# Async

Every sync helper has an `*_async` twin. Async-callable tools, async intercept callbacks, async sessions, async terminals — all first-class.

```python
import asyncio
import encode

async def main():
    out = await encode.relay_async(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
    print(out.content)

asyncio.run(main())
```

## `relay_async()`

Same signature as `relay()`; returns `AsyncRelayHandle`.

```python
handle = encode.relay_async(model="m", messages=[...])

out = await handle              # awaitable directly
out = await handle.execute()
out = await handle.get()
async for event in handle: ...   # streaming
```

The handle is awaitable — `await handle` and `await handle.execute()` are equivalent.

## Async tool callables

`relay_async` accepts both sync and async tool functions. Sync `relay()` rejects async callables with `TypeError`:

```python
import httpx

async def fetch(url: str) -> dict:
    """Fetch a URL."""
    async with httpx.AsyncClient() as c:
        r = await c.get(url)
        return {"status": r.status_code, "body": r.text[:200]}

await encode.relay_async(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Fetch example.com."}],
    tools=[fetch],
)
```

## Async intercept

```python
async def persist(event):
    await db.write(conversation_id, event.messages_so_far)

await encode.relay_async(model="m", messages=[...], tools=[t], on_intercept=persist)
```

Sync callbacks work too — encode awaits async ones, calls sync ones directly.

## Async streaming

```python
async for event in encode.relay_async(model="m", messages=[...], stream=True):
    if event.type == "content.delta":
        print(event.data, end="", flush=True)
```

Streaming with tools auto-loops the same way as sync.

## Async clients and sessions

```python
async with encode.AsyncClient(api_key="...", base_url="...") as client:
    out = await client.relay(model="m", messages=[...])

session = encode.AsyncSession.open()
await session.aemit("user.message", {"content": "hi"})

async with encode.AsyncTerminal() as sh:
    r = await sh.run("ls")
```

`AsyncSession` is just `Session` with an `aemit()` method for symmetry — there's no async-only state machine to worry about.

→ [sessions.md](./sessions.md), [terminal.md](./terminal.md)

## See also

- [relay.md](./relay.md) — full kwarg reference
- [streaming.md](./streaming.md) — `StreamEvent` shape
- [intercept.md](./intercept.md) — observe + mutate
