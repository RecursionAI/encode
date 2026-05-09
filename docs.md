# encode — usage guide

A Python SDK for any OpenAI-compatible inference endpoint. Designed for [Courier](https://getcourier.ai), works equally well against OpenAI, vLLM, LM Studio, Ollama, Together, Groq, etc.

```python
import encode

out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
).response

print(out.content)        # "Hello!"
print(out.iterations)     # 1
print(out.usage)          # Usage(prompt_tokens=..., completion_tokens=...)
```

That's the whole "hello world." Everything below shows how each feature layers on top of `relay()`.

---

## Install

```bash
pip install encode           # once published
# or, from source:
uv sync --extra dev
```

Requires Python 3.10+. Runtime deps: `httpx`, `pydantic`, `python-dotenv`, `typing-extensions`.

---

## Configuration

encode resolves credentials in this order (highest wins):

1. Explicit `api_key=` / `base_url=` arg on `relay()`, `whisper()`, or `Client(...)`.
2. `ENCODE_API_KEY` / `ENCODE_BASE_URL` environment variables.
3. `OPENAI_API_KEY` / `OPENAI_BASE_URL` environment variables.

A `.env` file in the current directory (or any parent) is auto-loaded once on import via `python-dotenv` with `override=False`, so real shell env always wins. Disable with `ENCODE_DISABLE_DOTENV=1`.

**`.env` example:**

```
ENCODE_API_KEY=sk-your-key-here
ENCODE_BASE_URL=https://your-courier-instance.example/
```

**Or pass directly:**

```python
encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
).response
```

Trailing slashes on `base_url` are normalized — `https://api.openai.com/v1` and `https://api.openai.com/v1/` are equivalent.

---

## Messages

`messages=` accepts dicts (OpenAI shape) or `encode.Message` Pydantic models — mix freely.

```python
encode.relay(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Greet me."},
    ],
).response.content
```

### Multimodal: images

```python
from encode import Message, TextContent, ImageContent

# Local file (base64'd into a data URL):
img = ImageContent.from_path("./photo.jpg", detail="high")

# Or a URL:
img = ImageContent.from_url("https://example.com/photo.jpg")

out = encode.relay(
    model="gpt-4o-mini",
    messages=[
        Message(role="user", content=[
            TextContent(text="What's in this image?"),
            img,
        ])
    ],
).response
```

### Multimodal: audio

```python
from encode import AudioContent, Message, TextContent

audio = AudioContent.from_path("./clip.wav")  # or .mp3

encode.relay(
    model="gpt-4o-audio-preview",
    messages=[
        Message(role="user", content=[
            TextContent(text="Transcribe and summarize."),
            audio,
        ])
    ],
).response
```

---

## Messages — stateful conversations

Plain `list[dict]` works fine. For multi-turn flows, `encode.Messages` is a stateful container that grows itself across `relay()` calls.

```python
m = (
    encode.Messages()
    .system("Be brief.")
    .user("name three colors")
)
encode.relay(model="gpt-4o-mini", messages=m).response
# m now contains [system, user, assistant] — the assistant turn was appended for you

m.user("now three more")
encode.relay(model="gpt-4o-mini", messages=m).response
# m is now [system, user, assistant, user, assistant]
```

**Auto-update is opt-in via the type you pass.** Pass a `Messages` instance → it gets mutated in place after the loop. Pass a plain `list` → no mutation (back-compat).

### API

```python
m = encode.Messages()                          # empty
m = encode.Messages([...])                     # seed from a list of dicts or Message objects

# Chainable adders (each returns self)
m.system("...")
m.user("...")               # str OR list of TextContent / ImageContent / AudioContent
m.assistant("...", tool_calls=[...])
m.tool(content, tool_call_id="c1")
m.add(some_message_or_dict)

# List-like
len(m); m[0]; for msg in m: ...
m.append(msg); m.extend([...]); m.clear()
m.copy()                    # branch a conversation
m.to_list()                 # plain list[dict] copy

# Manual ingestion (e.g. after streaming, or when you have a RelayResponse already)
m.update(response)          # replaces contents with response.messages

# Pydantic interop (for DB persistence, validation, etc.)
m.to_pydantic()             # -> Conversation (Pydantic BaseModel)
encode.Messages.from_pydantic(conv)   # rebuild a Messages from a Conversation
```

### Saving and loading from a database

```python
m = encode.Messages().system("Be brief.").user("hi")
encode.relay(model="gpt-4o-mini", messages=m).response

# Save
blob = m.to_pydantic().model_dump_json()
db.save(conversation_id, blob)

# Load
blob = db.load(conversation_id)
m = encode.Messages.from_pydantic(encode.Conversation.model_validate_json(blob))
encode.relay(model="gpt-4o-mini", messages=m).response   # picks up where you left off
```

`Conversation` is a thin Pydantic wrapper around `list[Message]` — pick whichever shape you want for your storage layer.

`Messages` works for both endpoints — contents stay in OpenAI chat format internally and are converted to typed input items when routing to `/v1/responses`.

### Branching a conversation

```python
shared = encode.Messages().system("Be helpful.").user("Topic: birds")

a = shared.copy().user("Tell me about owls")
b = shared.copy().user("Tell me about hawks")

encode.relay(model="m", messages=a).response   # a grows independently
encode.relay(model="m", messages=b).response   # b grows independently
```

### When auto-update fires

After `handle.execute()` (or `.response`, which calls execute). Memoized — calling `.response` twice on the same handle does not double-append. If the loop raises `MaxToolIterationsError`, the *partial* conversation is still absorbed so you can recover state.

Streaming does not auto-update in v0.1 (no final response object). Once you've consumed the stream, call `m.update(response)` yourself if you've reconstructed one.

---

## Tool calling

Pass plain Python functions; encode introspects their signatures, builds a JSON schema (via Pydantic), and runs the loop for you.

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

print(out.content)             # "It's 72°F in Denver."
print(out.iterations)          # 2 (one tool call, then final answer)
print(out.tool_calls[0].name)  # "get_weather"
print(out.tool_calls[0].result)  # {"city": "Denver", "temp_f": 72}
```

The loop runs until the model stops calling tools (or hits `max_tool_iterations`, default `8`).

### What the SDK does for you

- Reads function name, docstring summary, and `Args:` block for descriptions.
- Builds an OpenAI tool schema (`{"type": "function", "function": {...}}`).
- Parses tool-call arguments as JSON, validates against the schema, calls your function with kwargs.
- Serializes the return value (via `json.dumps(default=str)`), appends as a `tool` message.
- Captures exceptions and feeds them back as `{"error": "..."}` so the model can recover (the loop never crashes from a buggy tool).

### Mixing callables and raw dicts

```python
encode.relay(
    model="gpt-4o-mini",
    messages=[...],
    tools=[
        get_weather,                          # callable
        {"type": "function", "function": ...} # raw dict (server-side or pre-built)
    ],
).response
```

### Web search

If your provider supports the shorthand:

```python
encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's trending today?"}],
    web_search=True,         # appends {"type": "web_search"} to tools
).response
```

### Capping iterations (optional)

By default the tool loop is **unbounded** — it runs as long as the model keeps requesting tools. Pass `max_tool_iterations=N` as a safety net when you want to bail on stuck loops:

```python
encode.relay(..., tools=[my_tool], max_tool_iterations=3).response
```

If the model is still calling tools after `max_tool_iterations` API calls, encode raises `MaxToolIterationsError` with a `.partial` attribute holding the partial `RelayResponse`. Without a cap, a misbehaving model can loop forever — pick a number when you don't trust the model to converge.

---

## Intercept — observe and interrupt

`intercept` lets you run code between tool calls. Two equivalent forms:

```python
def watcher(event):
    print(f"iter {event.iteration}: {[tc.name for tc in event.tool_calls]}")

# Chained method:
encode.relay(...).intercept(watcher).response

# Kwarg sugar:
encode.relay(..., on_intercept=watcher).response
```

Both append to the same internal list — chain multiple if you want:

```python
encode.relay(...).intercept(log).intercept(audit).response
```

### Stopping the loop early

By default, `intercept` is observe-only — the loop keeps running. To stop early, call `event.stop()`:

```python
def watcher(event):
    if any(tc.name == "submit_final" for tc in event.tool_calls):
        event.stop()  # loop exits cleanly after this iteration

encode.relay(..., tools=[search, submit_final]).intercept(watcher).response
```

`event.stop()` is the only way to break out — there's no magic "return False" sentinel. This makes the intent obvious in code review and discoverable via autocomplete.

### What's on the event

```python
@dataclass
class InterceptEvent:
    iteration: int                       # 0-based loop index just completed
    endpoint: Literal["chat", "responses"]
    assistant_turn: AssistantTurn        # what the model said this iteration
    tool_calls: list[ToolCallRecord]     # parsed calls + executed results
    messages_so_far: list[dict]          # full conversation in OpenAI format
    raw_response: dict                   # the just-completed API response body
    will_continue: bool                  # True iff another API call is queued

    def stop(self) -> None: ...
    @property
    def stopped(self) -> bool: ...
```

Interceptors only fire on iterations that *had* tool calls. The final iteration (where the model returns a plain answer) doesn't trigger them.

If a callback raises, encode logs at WARNING and continues — one bad observer doesn't kill the loop.

---

## Structured output

Pass a Pydantic model as `response_format`. encode generates the JSON schema, sends it to the model, and parses the response back into your model.

```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    reasoning: str
    sentiment: str  # positive | negative | neutral

out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Classify: 'I loved it.'"}],
    response_format=Sentiment,
).response

assert isinstance(out.parsed, Sentiment)
print(out.parsed.sentiment)   # "positive"
print(out.parsed.reasoning)
```

Or pass a raw schema dict if you'd rather control the wire format yourself:

```python
encode.relay(
    ...,
    response_format={"type": "json_schema", "json_schema": {"schema": {...}}},
)
```

In that case `out.parsed` stays `None` (encode only auto-parses when you give it a Pydantic class).

> `response_format` and `stream=True` together raise `ValueError` immediately — structured output isn't compatible with streaming on most servers.

---

## /v1/responses endpoint

encode auto-routes to `/v1/responses` when you pass `input=` or `instructions=`. Otherwise it routes to `/v1/chat/completions`.

```python
out = encode.relay(
    model="gpt-4o-mini",
    instructions="You are a concise summarizer.",
    input="Summarize: encode is a Python SDK for OpenAI-compatible endpoints.",
).response

print(out.endpoint)   # "responses"
print(out.content)
```

Force a specific endpoint with `endpoint="chat"` or `endpoint="responses"`:

```python
encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hi"}],
    endpoint="responses",   # converts messages to typed input items
).response
```

The tool loop and intercept work identically across both endpoints — `function_call` / `function_call_output` items vs `tool_calls` / `role: "tool"` is handled internally.

---

## Streaming

Set `stream=True` and iterate the handle:

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

For `/v1/responses`, events use the typed names from the spec (`response.output_text.delta`, `response.completed`, etc.):

```python
for event in encode.relay(model="m", input="hi", stream=True):
    print(event.type, event.data)
```

> Streaming with auto-tool-loop is not supported in v0.1.0 — you can only iterate streams when `tools=None`. Use `stream=False` with tools, then re-issue a streaming call yourself if you want to stream the final answer.

---

## Whisper — transcription and translation

```python
out = encode.whisper(file="./recording.wav")
print(out.text)
```

### Translate (any language → English)

```python
encode.whisper(file="./spanish.mp3", mode="translate").text
```

### Verbose JSON with word-level timestamps

```python
out = encode.whisper(
    file="./meeting.m4a",
    response_format="verbose_json",
    timestamp_granularities=["word", "segment"],
)
print(out.language)          # "en"
print(out.duration)          # seconds
for seg in out.segments:
    print(seg["start"], seg["text"])
```

### Other inputs

```python
# Bytes:
encode.whisper(file=open("a.wav", "rb").read())

# (filename, bytes) tuple:
encode.whisper(file=("clip.mp3", audio_bytes))
```

Allowed extensions: `.mp3 .mp4 .mpeg .mpga .m4a .wav .webm`. Max size: 25 MB. encode reads the file into bytes upfront so transient retries are safe.

---

## Async

Every sync helper has an `*_async` twin.

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

Async tool callables are supported in `relay_async` (rejected in sync `relay` with a clear `TypeError`):

```python
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

Async interceptors are awaited:

```python
async def cb(event):
    await persist_to_db(event.messages_so_far)

await encode.relay_async(...).intercept(cb)
```

`AsyncRelayHandle` is awaitable — you can `await handle` directly or call `await handle.execute()`.

---

## Long-lived clients

Module-level `encode.relay()` lazily constructs and reuses a process-wide `Client`. For finer control (custom headers, retries, connection pooling), instantiate one yourself:

```python
with encode.Client(api_key="...", base_url="...", max_retries=3) as client:
    out1 = client.relay(model="m", messages=[...]).response
    out2 = client.whisper(file="./a.wav")
```

Async equivalent:

```python
async with encode.AsyncClient() as client:
    out = await client.relay(model="m", messages=[...])
```

You can also bring your own httpx client (useful for proxies, custom transports, etc.):

```python
import httpx
encode.Client(http_client=httpx.Client(proxies="http://localhost:8888"))
```

---

## Errors

All exceptions inherit from `encode.CourierError`. The OpenAI error envelope (`{"error": {"message", "type", "code"}}`) is mapped to specific classes:

| Class                          | When                                       |
|--------------------------------|--------------------------------------------|
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

Inspect the original payload via `.raw`, `.code`, `.type`, `.status`:

```python
try:
    encode.relay(model="m", messages=[...]).response
except encode.InvalidRequestError as e:
    print(e.code, e.type, e.status)
    print(e.raw)
```

Retries: `502/503/504` and transport errors retry up to `max_retries=2` times with exponential backoff (0.25/0.5/1s). `429` honors the `Retry-After` header. POST bodies are kept in memory so retries are safe (whisper reads the file upfront).

---

## RelayResponse — what you get back

```python
class RelayResponse(BaseModel):
    content: str | None              # final assistant text
    parsed: Any = None               # populated when response_format=PydanticModel
    messages: list[dict]             # full conversation in OpenAI format
    tool_calls: list[ToolCallRecord] # flat list across all iterations
    iterations: int                  # number of round trips
    finish_reason: str | None        # last finish_reason from the server
    endpoint: Literal["chat", "responses"]
    model: str
    raw: Any                         # last raw API response
    usage: Usage | None              # token counts when the server provides them
```

`messages` is ready to feed back into the next `relay()` call as conversation history — round-tripping is lossless for both endpoints.

---

## Cookbook

### Multi-turn conversation

```python
m = encode.Messages().system("You are concise.")

while True:
    user = input("> ")
    if not user:
        break
    m.user(user)
    out = encode.relay(model="gpt-4o-mini", messages=m).response
    print(out.content)   # m already has the assistant turn appended
```

### Agent that runs until it submits a final answer

```python
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

### Persisting messages mid-loop

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

### Inspecting tool failures

```python
out = encode.relay(model="m", messages=[...], tools=[flaky_tool]).response
for tc in out.tool_calls:
    if tc.error:
        print(f"{tc.name} failed: {tc.error}")
```

The model still saw an `{"error": "..."}` envelope as the tool result, so it had a chance to recover.

---

## Public API at a glance

```python
# Top-level functions
encode.relay(...)         -> RelayHandle
encode.relay_async(...)   -> AsyncRelayHandle
encode.whisper(...)       -> WhisperResponse
encode.whisper_async(...) -> WhisperResponse

# Clients
encode.Client(api_key=..., base_url=...)
encode.AsyncClient(api_key=..., base_url=...)

# Handles
RelayHandle.intercept(cb)        # chain
RelayHandle.execute()            # run, return RelayResponse
RelayHandle.response             # property: run + memoize
iter(RelayHandle)                # stream events (no-tools path)

AsyncRelayHandle.intercept(cb)
await AsyncRelayHandle.execute()
await AsyncRelayHandle.get()
async for event in AsyncRelayHandle: ...

# Models
encode.Message, Messages, Conversation, TextContent, ImageContent, AudioContent
encode.RelayResponse, WhisperResponse, ToolCallRecord, AssistantTurn, Usage
encode.InterceptEvent

# Errors (all inherit CourierError)
encode.AuthError, InvalidRequestError, InvalidToolCallError, InvalidToolChoiceError,
encode.InvalidResponseFormatError, InvalidAudioError, RateLimitError, ServerError,
encode.TransportError, MaxToolIterationsError

encode.__version__
```
