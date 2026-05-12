# Tool executors

`ToolExecutor` is the brain↔hands seam. Every tool call goes through it. The default executor runs Python callables in-process; swap it for remote dispatch, MCP, sub-agents, or anything else — without touching the harness.

```python
class ToolExecutor(Protocol):
    def execute(self, name: str, input: dict) -> ExecutionResult: ...
    async def execute_async(self, name: str, input: dict) -> ExecutionResult: ...
```

```python
class ExecutionResult:
    result: Any                # raw return value (for the response object)
    result_serialized: str     # JSON string sent back to the model
    error: str | None          # repr(exception) if the tool raised
    duration_ms: float
```

## Default — `LocalToolExecutor`

When you pass `tools=[fn, ...]` and no `tool_executor=`, encode builds a `LocalToolExecutor` from the callables for you. Behavior is identical to the SDK without an explicit executor.

```python
# implicit
encode.relay(model="m", messages=[...], tools=[my_fn]).response

# explicit, equivalent
executor = encode.LocalToolExecutor({"my_fn": my_fn})
encode.relay(model="m", messages=[...], tools=[my_fn], tool_executor=executor).response
```

`LocalToolExecutor`:

- Looks up `name` in its `{name: callable}` mapping.
- Calls the function (sync or via `safe_call_async`).
- Captures exceptions, serializes the result with `json.dumps(default=str)`.
- Returns an `ExecutionResult`.

If the name isn't in the mapping, the executor returns an error result (`"no Python callable bound for tool 'X'"`) so the model can recover instead of crashing.

## Swapping

Pass any `ToolExecutor`-compatible object via `tool_executor=`:

```python
encode.relay(
    model="m",
    messages=[...],
    tools=[{"type": "function", "function": {"name": "search", "parameters": {...}}}],
    tool_executor=MyRemoteExecutor(),
).response
```

When `tool_executor=` is set, the `tools=` argument still controls **what the model sees** (the schemas) — the executor controls **what runs**. Pass schemas as raw dicts (no Python callables needed) or as callables (their schemas will be introspected, the callables ignored).

## Pattern — Remote executor

Run tools in a separate service:

```python
import httpx

class RemoteToolExecutor:
    def __init__(self, base_url: str):
        self._url = base_url
        self._client = httpx.Client()

    def execute(self, name, input):
        r = self._client.post(f"{self._url}/tools/{name}", json=input, timeout=30.0)
        r.raise_for_status()
        body = r.json()
        return encode.ExecutionResult(
            result=body.get("result"),
            result_serialized=r.text,
            error=body.get("error"),
            duration_ms=body.get("duration_ms", 0.0),
        )

    async def execute_async(self, name, input):
        ...  # mirror with httpx.AsyncClient
```

## Pattern — MCP executor

Wire an MCP client behind the same protocol:

```python
class MCPToolExecutor:
    def __init__(self, mcp_session):
        self._mcp = mcp_session

    def execute(self, name, input):
        result = self._mcp.call_tool(name, input)
        return encode.ExecutionResult(
            result=result,
            result_serialized=json.dumps(result, default=str),
        )

    async def execute_async(self, name, input):
        result = await self._mcp.call_tool_async(name, input)
        return encode.ExecutionResult(
            result=result,
            result_serialized=json.dumps(result, default=str),
        )
```

## Pattern — Sub-agent executor

A "tool" whose body is another `relay()`:

```python
class SubAgentExecutor:
    """Delegate to a child agent. The child's tool calls run in its own loop."""

    def __init__(self, model, child_tools):
        self._model = model
        self._tools = child_tools

    def execute(self, name, input):
        if name != "sub_agent":
            return encode.ExecutionResult(error=f"unknown tool {name!r}")
        out = encode.relay(
            model=self._model,
            messages=[{"role": "user", "content": input["task"]}],
            tools=self._tools,
        ).response
        return encode.ExecutionResult(
            result=out.content,
            result_serialized=out.content or "",
        )

    async def execute_async(self, name, input):
        ...  # mirror via relay_async
```

Wire it up:

```python
sub = SubAgentExecutor(
    model="gpt-4o-mini",
    child_tools=[file_search, read_file],
)

encode.relay(
    model="gpt-4o-mini",
    messages=[...],
    tools=[{
        "type": "function",
        "function": {
            "name": "sub_agent",
            "description": "Run a sub-task in a child agent",
            "parameters": {"type": "object",
                           "properties": {"task": {"type": "string"}},
                           "required": ["task"]},
        },
    }],
    tool_executor=sub,
).response
```

## Stateful tools without an executor

The simpler way to keep state across calls is to put state in a Python object and pass a method (or closure) as a tool. The default `LocalToolExecutor` is enough.

→ See [terminal.md](./terminal.md) for the canonical example — wrapping a persistent bash subprocess as a tool.

## `CredentialProvider` (roadmap)

A stub `Protocol` exists for future vault-backed credential injection:

```python
class CredentialProvider(Protocol):
    def get(self, key: str) -> str: ...
```

The intent is a future `VaultedToolExecutor` that resolves credentials at call time rather than letting them sit in the harness's memory. Not used in v1; documented here so the seam is visible.

## See also

- [tools.md](./tools.md) — what `tools=` accepts and how schemas are built
- [terminal.md](./terminal.md) — stateful tool pattern
- [concepts.md](./concepts.md) — why this seam exists
