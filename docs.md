# encode docs

encode is a Python SDK for any OpenAI-compatible inference endpoint, shaped around the three primitives from Anthropic's [Scaling Managed Agents](https://www.anthropic.com/engineering/managed-agents) post: a durable **Session**, a stateless **Harness** (`relay()`), and pluggable **Hands** (`ToolExecutor`). Drop a Python function in `tools=` and the harness runs the loop. Pass a `Session` and the run is durable. Pass a `Terminal` and your agent has a shell.

## Mental model — three primitives

```
┌──────────────────────────────┐
│ Session   — what happened    │   append-only event log (Pydantic, BYO DB)
└──────────────────────────────┘
              │  to_messages()
              ▼
┌──────────────────────────────┐
│ Harness   — what to do next  │   relay() — stateless loop, two endpoints, intercept
└──────────────────────────────┘
              │  ToolExecutor.execute()
              ▼
┌──────────────────────────────┐
│ Hands     — how to run a tool│   local by default; swap for remote / MCP / sub-agent
└──────────────────────────────┘
```

`Messages` is the *ephemeral context window*, not the session — a mutable list you build, mutate, branch, and persist. `Terminal` is just a stateful tool that goes through the same `ToolExecutor` seam as anything else. Intercept is a mutation point *inside* the harness loop. → [concepts.md](./docs/concepts.md)

## 30-second quickstart

```python
import encode

def get_weather(city: str) -> dict:
    """Get weather by city."""
    return {"city": city, "temp_f": 72}

out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Weather in Denver?"}],
    tools=[get_weather],
).response

print(out.content)              # "It's 72°F in Denver."
print(out.iterations)           # 2
print(out.tool_calls[0].name)   # "get_weather"
```

Full walkthrough: [docs/quickstart.md](./docs/quickstart.md)

## Reference

| Topic                                            | What's in it                                                      |
| ------------------------------------------------ | ----------------------------------------------------------------- |
| [quickstart](./docs/quickstart.md)               | install, configure, hello world, first tool, first session        |
| [concepts](./docs/concepts.md)                   | Managed Agents primitives mapped to encode types                  |
| [relay()](./docs/relay.md)                       | the harness — signature, handles, response object                 |
| [messages](./docs/messages.md)                   | stateful conversations, multimodal, branching, persistence        |
| [tools](./docs/tools.md)                         | callables → schemas, web_search, max iterations, raw tool dicts   |
| [intercept](./docs/intercept.md)                 | observe + mutate the loop, helpers, `stop()`, `context.modify`    |
| [sessions](./docs/sessions.md)                   | event log, types, persistence, resume                             |
| [executors](./docs/executors.md)                 | `ToolExecutor` protocol, remote/MCP/sub-agent patterns            |
| [terminal](./docs/terminal.md)                   | persistent bash, snapshots, sandbox-as-tool                       |
| [streaming](./docs/streaming.md)                 | `StreamEvent`, with-tools auto-loop                               |
| [structured output](./docs/structured-output.md) | Pydantic `response_format`                                        |
| [whisper](./docs/whisper.md)                     | transcription & translation                                       |
| [async](./docs/async.md)                         | `relay_async`, async tools / intercepts / sessions                |
| [errors](./docs/errors.md)                       | exception hierarchy, retries, partial recovery                    |
| [cookbook](./docs/cookbook.md)                   | runnable end-to-end recipes                                       |

## Compatibility

Python 3.10+. Works against any OpenAI-compatible endpoint: OpenAI, [Courier](https://getcourier.ai), vLLM, LM Studio, Ollama, Together, Groq, and more. `Terminal` requires macOS or Linux (uses pexpect).

## Public API at a glance

```python
# Entry points
encode.relay(...)         -> RelayHandle
encode.relay_async(...)   -> AsyncRelayHandle
encode.whisper(...)       -> WhisperResponse
encode.whisper_async(...) -> WhisperResponse

# Clients
encode.Client(api_key=..., base_url=..., max_retries=2, timeout=60.0)
encode.AsyncClient(...)

# Handles
RelayHandle.intercept(cb)        # chain
RelayHandle.execute()            # run, return RelayResponse
RelayHandle.response             # property: run + memoize
iter(RelayHandle)                # stream events

# Intercept (mutable mid-loop)
event.append/.insert/.replace/.compact/.edit_last_tool_result   # mutate messages
event.stop()                                                     # halt the loop
event.register_tool(fn_or_dict)                                  # session-required

AsyncRelayHandle.intercept(cb)
await AsyncRelayHandle           # awaitable directly
await AsyncRelayHandle.execute()
async for event in AsyncRelayHandle: ...

# Sessions (durable event log)
encode.Session.open(id=None, metadata=None, tools=None)
encode.Session.resume(data, tools=())     # model_validate + rebind_tools
encode.AsyncSession.open(...)
encode.AsyncSession.resume(data, tools=())
# Per-session tool registry (append-only, idempotent by name)
session.tools                              # list[Any], excluded from model_dump
session.register_tool(fn_or_dict)          # returns True if newly added
session.register_tools([...])              # bulk, returns count newly added
session.rebind_tools([...])                # returns list[str] of unmatched names
encode.Event              # factory classmethods: user_message, assistant_message,
                          # tool_call, tool_result, tool_registered, iteration_end,
                          # system, custom, ...
encode.EventType          # USER_MESSAGE, ASSISTANT_MESSAGE, TOOL_CALL, TOOL_RESULT,
                          # TOOL_REGISTERED, ITERATION_END, CONTEXT_MODIFY, SYSTEM,
                          # CUSTOM

# Executors (the brain↔hands seam)
encode.ToolExecutor                       # Protocol
encode.LocalToolExecutor(tools={...})     # default
encode.ExecutionResult
encode.CredentialProvider                 # stub Protocol for future vault work

# Terminals (stateful bash)
encode.Terminal(cwd=None, timeout=30.0)
encode.AsyncTerminal(...)
encode.TerminalSnapshot
encode.CommandResult

# Content / Messages
encode.Message, Messages, Conversation
encode.TextContent, ImageContent, ImageURL, AudioContent, InputAudio
encode.ToolCall, ToolCallFunction

# Response types
encode.RelayResponse, WhisperResponse, ToolCallRecord, AssistantTurn, Usage
encode.InterceptEvent, StreamEvent

# Errors (all inherit CourierError)
encode.AuthError, InvalidRequestError, InvalidToolCallError, InvalidToolChoiceError,
encode.InvalidResponseFormatError, InvalidAudioError, RateLimitError, ServerError,
encode.TransportError, MaxToolIterationsError,
encode.TerminalError, TerminalTimeoutError

encode.__version__
```
