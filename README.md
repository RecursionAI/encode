# Courier — ENCODE

Courier — ENCODE is an SDK specifically designed for developing AI apps and agents with open source
LLMs.

## Features

- Python SDK
- Request wrapper that sends messages to v1/chat/completions and v1/responses, configurable for each, auto formatting,
  and returns a pydantic response with props the end user can use in their code.
    - The request wrapper includes agentic processing with an optional 'tools' prop. Python functions can be passed and
      automatically converted into tools for an agent. The SDK should parse out tool responses and call the functions,
      and automatically send a follow up message to the agent. The SDK should be considered an option that runs until
      it's completed it's tool calling, and then return a full list of messages in openai format.
    - The request wrapper should optionally accept Pydantic models as a response_format for structured JSON responses
      and automatically apply the format. The response object should be auto formatted to that pydantic model and usable
      as such.
    - The request wrapper should accept a pydantic model of messages. Images and audio should be included as well.
    - a model name should be able to be selected.
- A whisper request wrapper that allows for transcriptions and translations

## Notes

**encode should be a `uv` project and structured to be deployed on PyPi as an open source pip package.**

Everywhere that pydantic is accepted JSON should be accepted as well.

The tool call loop runs until the model doesn't make a tool call (or a hard capped limit that can be specified). So, if
a tool call is present, the code should execute and the response appended to the messages, and then another API requests
sent to the model. Once a response comes in with no acions the code continues past the relay loop.

Web search should be a boolean to enable. If enabled it should automatically be appended using the shorthand schema.

### function names

- `encode.relay()` — the chat completions and responses wrapper
    - `.intercept() ` — listener that can be attached to a `relay()` call and execute code whenever a tool call loop is
      engaged, where the model is processing requests until the loop ends. Intercept runs every time a tool call
      finishes even if the model is continuing the loop.
- `encode.whisper()` — whisper function. Accepts audio and can translate or transcribe.

### Courier Docs

[Courier](https://getcourier.ai/docs)

#### API docs

## API Docs

### Courier Inference API

Courier provides a custom inference API optimized for n8n and other workflows.

#### POST /inference/

```json
{
  "model_name": "Solar Open 100B",
  "model_id": "Model_UUID",
  "model_type": "text-text",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant"
    },
    {
      "role": "user",
      "content": "hello"
    }
  ],
  "temperature": 0.7
}
```

#### Authentication

For the Courier **/inference/** endpoint, use token authentication:

```
Authorization: API_KEY
```

### OpenAI Compatible Endpoints

Courier supports OpenAI-compatible APIs for completions and responses workflows.

#### POST /v1/chat/completions

```json
{
  "model": "Solar Open 100B",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant"
    },
    {
      "role": "user",
      "content": "hello"
    }
  ],
  "temperature": 0.7
}
```

#### GET /v1/models

```json
{
  "object": "list",
  "data": [
    {
      "id": "model_name",
      "object": "model",
      "created": 1686935092,
      "owned_by": "recursion-ai"
    }
  ]
}
```

#### POST /v1/responses

```json
{
  "model": "my-shared-model",
  "input": "Summarize this in one sentence.",
  "instructions": "optional system/developer instruction",
  "tools": [],
  "tool_choice": "auto",
  "text": {
    "format": {
      "type": "text"
    }
  },
  "stream": false,
  "max_output_tokens": 256
}
```

#### Authentication (OpenAI Endpoints)

Use Bearer authentication:

```
Authorization: Bearer API_KEY
```

#### Tool Calling

## Tool Calling API

Industry-leading tool calling for self-hosted AI stacks, with production-ready reliability for text and fused modality (
vision) models. One of the most robust OpenAI-compatible tool-calling implementations available on an API platform you
can own.

### Global OpenAI Compatibility

- Auth header: `Authorization: Bearer <api_key>` is required.
- Model matching is case-insensitive against workbench `name` or `nickname`, and only models available to the API key
  are usable.

#### Error Envelope

```json
{
  "error": {
    "message": "....",
    "type": "invalid_request_error",
    "code": "...."
  }
}
```

### POST /v1/chat/completions

#### Supported Request Fields

`model`, `messages`, `tools`, `tool_choice`, `stream`, `response_format`, `stop`, `max_tokens`, `temperature`, `top_p`,
`presence_penalty`, `frequency_penalty`, `user`. `n` is accepted but currently returns one choice (`index: 0`).

#### Request Body Example

```json
{
  "model": "Solar Open 100B",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the weather in Denver?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather by city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            }
          },
          "required": [
            "city"
          ]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "stream": false
}
```

#### Tool Support Rules

- Only tools with `type: "function"` are used.
- Forced `tool_choice` names must exist in `tools`, or requests fail with `400 invalid_tool_choice`.
- Text and fused modality (image-text-text) models support the tool-calling pipeline. Audio models do not support tools
  or streaming.
- Tool arguments are normalized to JSON strings; strict invalid arguments fail with `400 invalid_tool_call`.

#### Response Body Example

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Denver\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

#### Streaming Behavior (SSE)

1. Emits `chat.completion.chunk` events.
2. Streams content via `choices[0].delta.content`.
3. Tool calls stream incrementally through `delta.tool_calls` argument chunks.
4. Final chunk sets `finish_reason` to `tool_calls` or `stop`, then emits `[DONE]`.

### POST /v1/responses

#### Supported Request Fields

`model`, `input`, `messages`, `input_content`, `instructions`, `text`, `tools`, `tool_choice`, `stream`, `stop`,
`max_tokens`, `max_output_tokens`, `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `user`. `n`,
`logit_bias`, and `input_type` are accepted but not used in generation logic.

#### Request Body Example

```json
{
  "model": "Solar Open 100B",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": "Find today's top headline"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_news",
        "description": "Search current headlines",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string"
            }
          },
          "required": [
            "query"
          ]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "stream": false
}
```

#### Input Normalization

- `input` can be a string or a list of typed items.
- Supported item types: `message`, `input_text`, `function_call`, `function_call_output`.
- `reasoning` input items are rejected with `400 invalid_input`.

#### Response Body Example

```json
{
  "object": "response",
  "output": [
    {
      "id": "msg_123",
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Searching now...",
          "annotations": []
        }
      ]
    },
    {
      "id": "fc_123",
      "type": "function_call",
      "status": "completed",
      "call_id": "call_abc123",
      "name": "search_news",
      "arguments": "{\"query\":\"top headline today\"}"
    }
  ]
}
```

#### Streaming Behavior (Responses SSE Events)

`response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`,
`response.output_text.delta`, `response.output_text.done`, `response.content_part.done`,
`response.function_call_arguments.delta`, `response.function_call_arguments.done`, `response.output_item.done`,
`response.completed`, `error`

Streams end with `data: [DONE]`.

#### Multimodal Restrictions on /v1/responses

- For `audio-*` models, `stream` and `tools` are not supported.
- When provided for audio models, `text.format` must be plain text.

#### Tool Calling Parity Notes

- OpenAI-style behavior with function tools only.
- Tool arguments must normalize to valid JSON strings.
- Audio models do not support tools and/or streaming. Text and fused modality (image-text-text) models have full
  tool-calling parity.

#### Whisper API

## Whisper API

OpenAI-compatible Whisper transcription and translation endpoints built for production automation workflows.

### Implemented Endpoints

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `/v1/audio/speech` is not currently implemented.

### Common Request Behavior

- Multipart upload with required `file` and `model`.
- Allowed extensions: `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, `.webm`. Max size: 25 MB.
- Error mapping: unsupported format → `invalid_audio`, too large → `invalid_request_error`, invalid format value →
  `invalid_response_format`.
- `model=whisper-1` maps to `UCE_WHISPER_MODEL` (default `mlx-community/whisper-large-v3-turbo`); other model names pass
  through unchanged.

### POST /v1/audio/transcriptions

#### Request Example (multipart)

```bash
curl -X POST "$BASE_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word"
```

#### Response Example (json)

```json
{
  "text": "Hello from Courier Whisper."
}
```

#### Response Example (verbose_json)

```json
{
  "text": "Hello from Courier Whisper.",
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 1.8,
      "text": "Hello from Courier Whisper."
    }
  ]
}
```

> `timestamp_granularities` values `segment` and `word` are only allowed when `response_format=verbose_json`.

### POST /v1/audio/translations

#### Request Example (multipart)

```bash
curl -X POST "$BASE_URL/v1/audio/translations" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@audio-es.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

#### Response Example

```json
{
  "text": "This audio was translated into English."
}
```

### Behavior Notes

- Uses translate operation internally.
- `word_timestamps` is disabled for translations.
- For `verbose_json`, language defaults to `en` if upstream language is missing.

#### JSON Response Formatting

## JSON Response Formatting

### Structured JSON Outputs with Outlines

Courier supports guaranteed structured JSON outputs using the [Outlines](https://github.com/outlines-dev/outlines)
library. This feature enables models to generate responses that strictly adhere to a provided JSON schema through
FSM-based logit masking.

### Overview

Courier's structured JSON output feature uses Outlines to ensure models generate responses that strictly follow your
JSON schema. This is achieved through Finite State Machine (FSM) based logit masking that constrains token generation to
only produce valid JSON matching your schema.

### Technical Architecture

- **FSM-Based Logit Masking:** Outlines builds a Finite State Machine from your JSON schema that constrains token
  generation to only produce valid JSON matching the schema.
- **Generator Caching:** The first time a schema is used, there's a 0.1-1s cold start while the FSM is compiled.
  Subsequent uses are instant (cached in memory per worker).
- **Thought Field Pattern:** To prevent "probability tunneling", schemas are automatically enhanced with a "thought"or "
  reasoning" field if one isn't present, allowing natural language processing before data constraints.
- **Zero-Copy Integration:** The Outlines wrapper shares the same MLX model weights in memory, providing minimal
  overhead when structured output is requested.

### Usage

Both **/v1/chat/completions** and **/inference/** endpoints use the OpenAI-compatible **response_format** parameter:

#### POST /v1/chat/completions

```json
{
  "model": "Solar Open 100B",
  "messages": [
    {
      "role": "user",
      "content": "What is 123 * 456?"
    }
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "schema": {
        "type": "object",
        "properties": {
          "thought": {
            "type": "string"
          },
          "answer": {
            "type": "number"
          }
        },
        "required": [
          "thought",
          "answer"
        ]
      }
    }
  }
}
```

#### POST /inference/

```json
{
  "model_id": "uuid-here",
  "model_name": "your-model",
  "model_type": "text-text",
  "messages": [
    {
      "role": "user",
      "content": "Classify: 'Great product!'"
    }
  ],
  "temperature": 0.7,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "schema": {
        "type": "object",
        "properties": {
          "reasoning": {
            "type": "string"
          },
          "sentiment": {
            "type": "string",
            "enum": [
              "positive",
              "negative",
              "neutral"
            ]
          }
        },
        "required": [
          "reasoning",
          "sentiment"
        ]
      }
    }
  }
}
```

#### Authentication

For the Courier **/inference/** endpoint, use token authentication:

```
Authorization: API_KEY
```

For the OpenAI **/v1/chat/completions** endpoint, use bearer authentication:

```
Authorization: Bearer API_KEY
```

#### Example Structured Response

```json
{
  "content": "{\n    \"reasoning\": \"The user sent a positive sentiment message: 'Great product!'. I need to classify this as positive sentiment.\",\n    \"sentiment\": \"positive\"\n  }"
}
```

As you can see, the LLM responded in the specified JSON structure with both reasoning and sentiment fields. You can
parse this response using `JSON.parse()` without any extra formatting or validation needed.

### Schema Examples

#### Simple Classification

```json
{
  "type": "object",
  "properties": {
    "thought": {
      "type": "string"
    },
    "classification": {
      "type": "string",
      "enum": [
        "urgent",
        "normal",
        "low_priority"
      ]
    }
  },
  "required": [
    "thought",
    "classification"
  ]
}
```

#### Nested Objects

```json
{
  "type": "object",
  "properties": {
    "analysis": {
      "type": "string"
    },
    "person": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string"
        },
        "age": {
          "type": "integer"
        },
        "email": {
          "type": "string"
        }
      },
      "required": [
        "name",
        "age"
      ]
    }
  },
  "required": [
    "analysis",
    "person"
  ]
}
```

#### Arrays and Lists

```json
{
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string"
    },
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          },
          "quantity": {
            "type": "integer"
          }
        },
        "required": [
          "name",
          "quantity"
        ]
      }
    }
  },
  "required": [
    "reasoning",
    "items"
  ]
}
```

#### Enums and Constraints

```json
{
  "type": "object",
  "properties": {
    "thought": {
      "type": "string"
    },
    "rating": {
      "type": "integer",
      "minimum": 1,
      "maximum": 5
    },
    "category": {
      "type": "string",
      "enum": [
        "electronics",
        "clothing",
        "food",
        "other"
      ]
    }
  },
  "required": [
    "thought",
    "rating",
    "category"
  ]
}
```

### Limitations

- **No Streaming:** Structured output currently doesn't support streaming. The response is returned as a complete JSON
  object.
- **Schema Complexity:** Extremely complex schemas with deep nesting may take longer to compile or potentially fail.
- **Model Capabilities:** The underlying model must be capable of understanding and following instructions. Smaller
  models may struggle with complex schemas.
- **Text & Vision Models:** Supported for `text-text` and `image-text-text` (fused modality) model types. Audio and
  image generation models use standard unconstrained generation.

### Backward Compatibility

When no `response_format` is provided, the system works exactly as it did before. There is zero impact on existing
inference flows. This feature is purely opt-in.

```json
// This still works exactly as before
{
  "model": "your-model",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ]
  // No response_format = standard unconstrained generation
}
```

#### Web search

## Web Search

Built-in web search that lets models automatically ground responses with real-time information from the web. Powered by
the Brave Search API.

When enabled, models can decide to search the web mid-inference. The server handles the search transparently and returns
a grounded response - no extra client-side logic required.

### Setup

#### 1. Get a Brave Search API Key

Sign up at [brave.com/search/api](https://brave.com/search/api/) to get your API key. New accounts receive $5/month in
free credits (~1,000 searches).

#### 2. Configure the Key

- **Option A: Courier TUI Installer** - Run `courier` and enter your key in the "Brave Search API Key" field (in the
  System Configuration section, below the ngrok fields).
- **Option B: Manual** - Add to your `~/.courier/.env`:

```bash
BRAVE_SEARCH_API_KEY=your_key_here
```

Then restart Courier.

### Usage

Include `web_search` in your request's `tools` array. Two formats are supported:

#### Shorthand Format

```json
{
  "model": "Qwen3 30B",
  "messages": [
    {
      "role": "user",
      "content": "What happened in the news today?"
    }
  ],
  "tools": [
    {
      "type": "web_search"
    }
  ]
}
```

#### Standard Function Format

```json
{
  "model": "Qwen3 30B",
  "messages": [
    {
      "role": "user",
      "content": "What is the current price of Bitcoin?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string"
            }
          },
          "required": [
            "query"
          ]
        }
      }
    }
  ]
}
```

#### cURL Example

```bash
curl -X POST http://localhost:9100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "Qwen3 30B",
    "messages": [{"role": "user", "content": "What are the latest developments in AI?"}],
    "tools": [{"type": "web_search"}]
  }'
```

#### Streaming

```bash
curl -X POST http://localhost:9100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "Qwen3 30B",
    "messages": [{"role": "user", "content": "What are the latest developments in AI?"}],
    "tools": [{"type": "web_search"}],
    "stream": true
  }'
```

Streaming works with web search enabled. Add `"stream": true` to your request. The server resolves all searches before
streaming the final grounded response.

### Mixing Web Search with Other Tools

Web search works alongside your own function tools. The server executes `web_search` calls automatically while returning
your custom tool calls to the client as normal.

#### Example

```json
{
  "model": "Qwen3 30B",
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in Denver and what's trending on Hacker News?"
    }
  ],
  "tools": [
    {
      "type": "web_search"
    },
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather by city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            }
          },
          "required": [
            "city"
          ]
        }
      }
    }
  ]
}
```

In this case, the model may call both tools. `web_search` is resolved server-side and `get_weather` is returned to the
client for execution.

### How It Works

1. The model decides whether a search is needed based on the user's question
2. If the model calls `web_search`, the server intercepts the call and queries the Brave Search API
3. The top 5 results (title, URL, description) are injected back as context
4. The model generates a final grounded response using the search results
5. The client receives only the final answer — the search loop is invisible

The server caps search iterations at 3 per request to prevent runaway loops.

### Behavior Without a Key

If `BRAVE_SEARCH_API_KEY` is not configured and a request includes `web_search`, the tool is silently dropped. The
request proceeds normally as if no tools were provided. A warning is logged server-side.

### Pricing

Brave Search API charges $5 per 1,000 queries. New accounts get $5/month in free credits. There is no markup from
Courier — you pay Brave directly for what you use.