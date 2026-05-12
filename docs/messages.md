# Messages

`Messages` is the SDK's stateful conversation container. It's a `list[dict]` plus chainable builders, multimodal helpers, branching, Pydantic interop, and a Session bridge — and it grows itself when you pass it to `relay()`.

```python
m = encode.Messages().system("Be brief.").user("name three colors")
encode.relay(model="gpt-4o-mini", messages=m).response
# m now contains [system, user, assistant] — the assistant turn was appended for you

m.user("now three more")
encode.relay(model="gpt-4o-mini", messages=m).response
# m is now [system, user, assistant, user, assistant]
```

Pass a plain `list[dict]` instead and nothing gets mutated (back-compat). The **type of the argument** decides whether auto-update fires.

## Building

```python
m = encode.Messages()                        # empty
m = encode.Messages([{"role": "user", "content": "hi"}, ...])   # seed from anything
m = encode.Messages([encode.Message(role="user", content="hi")])  # Message objects fine too
```

Chainable adders (each returns `self`):

```python
m.system("You are concise.")
m.user("name three colors")
m.assistant("Red, blue, green.")
m.assistant(None, tool_calls=[{"id": "c1", "type": "function",
                                "function": {"name": "f", "arguments": "{}"}}])
m.tool("result here", tool_call_id="c1")
m.add({"role": "user", "content": "another"})
m.add(encode.Message(role="user", content="and another"))
```

## List-like

```python
len(m)            # 5
m[0]              # {"role": "system", ...}
m[-1]             # last message
m[1:3]            # slice → list[dict]
for msg in m: ...

m.append({"role": "user", "content": "..."})
m.extend([m1, m2, m3])
m.clear()
m.copy()                # deep-ish copy for branching
m.to_list()             # plain list[dict] copy
```

`Messages` implements `collections.abc.Sequence` so it satisfies any `Sequence[...]` parameter and supports `in`, `index()`, `count()`, etc.

## Multimodal

`Messages.user(...)` and `.system(...)` accept either a `str` or a list of content parts.

### Images

```python
from encode import ImageContent, TextContent, Messages

img = ImageContent.from_path("./photo.jpg", detail="high")
# or
img = ImageContent.from_url("https://example.com/photo.jpg")

m = Messages().user([
    TextContent(text="What's in this image?"),
    img,
])
encode.relay(model="gpt-4o-mini", messages=m).response
```

### Audio (for `gpt-4o-audio` and similar)

```python
from encode import AudioContent, TextContent, Messages

audio = AudioContent.from_path("./clip.wav")    # or .mp3

m = Messages().user([
    TextContent(text="Transcribe and summarize."),
    audio,
])
encode.relay(model="gpt-4o-audio-preview", messages=m).response
```

`AudioContent.from_path` reads the file and base64-encodes it into the OpenAI `input_audio` shape.

## Branching

```python
shared = encode.Messages().system("Be helpful.").user("Topic: birds")

owls   = shared.copy().user("Tell me about owls")
hawks  = shared.copy().user("Tell me about hawks")

encode.relay(model="m", messages=owls).response   # owls grows independently
encode.relay(model="m", messages=hawks).response  # hawks grows independently
```

`copy()` returns a fresh `Messages` with a shallow-copied list of message dicts — branches don't bleed into the parent.

## When auto-update fires

After `handle.execute()` (or `.response`, which calls execute). Memoized — calling `.response` twice on the same handle does not double-append. If the loop raises `MaxToolIterationsError`, the *partial* conversation is still absorbed so you can recover state.

**Streaming does not auto-update** — there's no final `RelayResponse` until the stream drains. Once you've consumed the stream, call `m.update(response)` yourself if you reconstructed one. See [streaming.md](./streaming.md).

If you abandon a streaming iterator early (`break`), the container stays unchanged.

## Persisting messages

`Messages` rides on a Pydantic snapshot (`Conversation`) for serialization.

```python
m = encode.Messages().system("Be brief.").user("hi")
encode.relay(model="gpt-4o-mini", messages=m).response

# Save
blob = m.to_pydantic().model_dump_json()
db.save(conversation_id, blob)

# Load
loaded = encode.Conversation.model_validate_json(db.load(conversation_id))
m2 = encode.Messages.from_pydantic(loaded)
encode.relay(model="gpt-4o-mini", messages=m2).response   # carries prior turns
```

`Conversation` is a Pydantic `BaseModel` wrapping `list[Message]`. Pick the shape that fits your storage layer.

→ For durable agent runs with event-level granularity (tool calls, iteration boundaries, custom events), use [sessions.md](./sessions.md). `Messages` is the *context window*; `Session` is the *audit log*.

## Bridging from a Session

If you stored a `Session` and want to seed a `Messages` from it:

```python
session = encode.Session.model_validate(db.load(sid))
m = encode.Messages.from_events(session.events)
encode.relay(model="m", messages=m).response
```

Or just pass `session=` directly to `relay()` — it calls `session.to_messages()` for you.

## Manual ingestion

```python
m = encode.Messages([{"role": "user", "content": "hi"}])
resp = encode.relay(model="m", messages=encode.Messages(m)).response
# … or for streaming, after draining:
m.update(resp)
```

`m.update(response)` replaces the contents with `response.messages` — it doesn't merge, because `RelayResponse.messages` is the full history.

## Mixing dicts and Message objects

Both shapes are interchangeable everywhere `messages=` is accepted:

```python
encode.relay(
    model="m",
    messages=[
        {"role": "system", "content": "Be brief."},
        encode.Message(role="user", content="hi"),
    ],
).response
```

`encode.Message` validates the shape and gives you a typed object; raw dicts are fine if you don't need validation.

## API summary

```python
m = encode.Messages([...])

# adders
m.system(text_or_parts)
m.user(text_or_parts)
m.assistant(text, tool_calls=[...])
m.tool(content, tool_call_id="...")
m.add(message_or_dict)

# list-like
len(m); m[i]; for ... in m; m in ...; m.index(...); m.count(...)
m.append(...); m.extend(...); m.clear()
m.copy(); m.to_list()

# response ingestion
m.update(response)

# pydantic interop
m.to_pydantic()                           # -> Conversation
encode.Messages.from_pydantic(conv)       # rebuild

# session interop
encode.Messages.from_events(session.events)   # project events → messages
```

## See also

- [sessions.md](./sessions.md) — durable event log (the persistence story)
- [tools.md](./tools.md) — how `tools=` interacts with the assistant turn that lands in `Messages`
- [intercept.md](./intercept.md) — mutate the message list mid-loop
