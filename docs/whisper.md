# Whisper — transcription & translation

Multipart audio uploads against `/v1/audio/transcriptions` and `/v1/audio/translations`.

```python
out = encode.whisper(file="./recording.wav")
print(out.text)
```

## Translate (any language → English)

```python
encode.whisper(file="./spanish.mp3", mode="translate").text
```

## Verbose JSON with word timestamps

```python
out = encode.whisper(
    file="./meeting.m4a",
    response_format="verbose_json",
    timestamp_granularities=["word", "segment"],
)
print(out.language)     # "en"
print(out.duration)     # seconds
for seg in out.segments:
    print(seg["start"], seg["text"])
for w in out.words or []:
    print(w["start"], w["end"], w["word"])
```

`timestamp_granularities` requires `response_format="verbose_json"` — encode raises `InvalidRequestError` otherwise.

## Input shapes

```python
# Path-like:
encode.whisper(file="./audio.wav")
encode.whisper(file=Path("./audio.wav"))

# Bytes (filename defaults to "audio.wav"):
encode.whisper(file=open("a.wav", "rb").read())

# (filename, bytes) tuple — controls the extension the server sees:
encode.whisper(file=("clip.mp3", audio_bytes))
```

## Constraints

- Allowed extensions: `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, `.webm`.
- Max size: 25 MB.
- Allowed `response_format`: `"json"` (default), `"verbose_json"`, `"text"`, `"srt"`, `"vtt"`.

Violations raise `InvalidAudioError` / `InvalidRequestError`.

## Other parameters

```python
encode.whisper(
    file="./audio.wav",
    model="whisper-1",            # or any model your provider routes to
    language="en",                 # ISO code hint
    prompt="Acme Inc., Q4, EBITDA", # bias the decoder
    temperature=0.0,
)
```

## Async

```python
out = await encode.whisper_async(file="./audio.wav")
```

## Retry safety

encode reads the file into bytes upfront — transient retries (5xx, network errors) are safe and don't re-read the file.

## Response shape

```python
class WhisperResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict] | None = None
    words: list[dict] | None = None
    raw: Any = None
```

For non-JSON response formats (`text`, `srt`, `vtt`), `text` contains the body verbatim and everything else is `None`.

## See also

- [errors.md](./errors.md) — `InvalidAudioError`
- [relay.md](./relay.md) — for audio input to a chat model, use `AudioContent` in messages
