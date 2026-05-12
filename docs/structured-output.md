# Structured output

Pass a Pydantic model as `response_format`. encode generates the JSON schema, sends it to the model, and parses the response back into your model.

```python
from pydantic import BaseModel
import encode

class Sentiment(BaseModel):
    reasoning: str
    sentiment: str       # positive | negative | neutral

out = encode.relay(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Classify: 'I loved it.'"}],
    response_format=Sentiment,
).response

assert isinstance(out.parsed, Sentiment)
print(out.parsed.sentiment)    # "positive"
print(out.parsed.reasoning)
```

`out.content` still contains the raw JSON string. `out.parsed` is the validated Pydantic instance.

## Raw schema dicts

Pass a dict if you'd rather control the wire format directly. `out.parsed` stays `None` in this case — encode only auto-parses when you give it a Pydantic class.

```python
encode.relay(
    model="m",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "answer": {"type": "number"},
                },
                "required": ["thought", "answer"],
            },
        },
    },
).response
```

## Endpoint handling

- `/v1/chat/completions`: sent as `response_format=…`.
- `/v1/responses`: wrapped under `text: {format: …}` automatically.

You don't need to think about it — same kwarg, both endpoints.

## Nested models, enums, arrays

Pydantic v2 generates the right schema for nested models, enums, optional fields, and constraints. encode passes the schema through unchanged.

```python
from typing import Literal
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str
    quantity: int = Field(ge=1)

class Order(BaseModel):
    items: list[Item]
    status: Literal["new", "paid", "shipped"]
    note: str | None = None

out = encode.relay(
    model="m",
    messages=[{"role": "user", "content": "Create an order for 3 apples and 1 banana."}],
    response_format=Order,
).response

order = out.parsed                # Order
for it in order.items:
    print(it.name, it.quantity)
```

## Restrictions

- `response_format` and `stream=True` together raise `ValueError` immediately. Structured output isn't meaningful mid-stream.
- Servers that rely on FSM-based logit masking (Outlines, etc.) may have a ~0.1–1s cold start the first time a schema is compiled. Subsequent calls hit a cache.
- Deeply nested schemas may slow compilation or fail entirely on some servers. Keep schemas shallow when you can.

## Errors

If the server rejects the format (unsupported by the model, malformed schema), encode raises `InvalidResponseFormatError`. → [errors.md](./errors.md)

## See also

- [relay.md](./relay.md) — full kwarg reference
- [errors.md](./errors.md) — `InvalidResponseFormatError`
