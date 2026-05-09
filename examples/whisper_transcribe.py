"""Transcribe an audio file via the Whisper-compatible endpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import encode


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python whisper_transcribe.py <audio.wav>")
        return
    audio = Path(sys.argv[1])
    out = encode.whisper(file=audio, response_format="verbose_json")
    print(out.text)
    if out.segments:
        print(f"({len(out.segments)} segments, language={out.language})")


if __name__ == "__main__":
    main()
