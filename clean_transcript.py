"""Run the local LLM cleanup pass over an existing session transcript.

Writes <stem>-cleaned.md and <stem>-cleaned.changes.md next to the input.
See docs/session-transcription.md ("LLM cleanup").
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from snoozy_scribe import paths
from snoozy_scribe.config import load_config

DERIVED_SUFFIXES = ("-cleaned.md", ".changes.md", ".pruned.md", ".errors.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcript", type=Path, help="Transcript .md written by transcribe.py")
    parser.add_argument(
        "--config",
        type=Path,
        default=paths.DEFAULT_CONFIG_PATH,
        help=f"Path to config YAML (default: {paths.DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()
    # Show the package's INFO lines (e.g. which cleanup prompt is used), not
    # every library's.
    logging.basicConfig(format="%(message)s")
    logging.getLogger("snoozy_scribe").setLevel(logging.INFO)

    transcript: Path = args.transcript
    if not transcript.is_file():
        sys.exit(f"error: transcript not found: {transcript}")
    if transcript.name.endswith(DERIVED_SUFFIXES):
        sys.exit(f"error: {transcript.name} is a derived file; pass the original transcript .md")

    # Imported after argument checks: these pull in MLX.
    from snoozy_scribe.local_llm import DEFAULT_CLEANUP_MODEL, LocalLLM
    from snoozy_scribe.models import resolve_model_path
    from snoozy_scribe.transcript_cleanup import (
        DEFAULT_CHUNK_LINES,
        DEFAULT_MAX_DROP_WORDS,
        DEFAULT_MIN_SIMILARITY,
        clean_transcript,
        parse_transcript,
    )

    try:
        parse_transcript(transcript.read_text())
    except ValueError as e:
        sys.exit(f"error: {transcript.name}: {e}")

    paths.ensure_user_dirs()
    config = load_config(args.config)
    settings = config.get("transcript_cleanup") or {}
    model = settings.get("model", DEFAULT_CLEANUP_MODEL)
    model_dir = paths.resolve(config.get("model_dir", paths.DEFAULT_MODEL_DIR))

    start_time = time.monotonic()
    print(f"Loading {model}…")
    llm = LocalLLM(resolve_model_path(model, model_dir, on_download=lambda: print(f"Downloading {model}…")))
    result = clean_transcript(
        llm,
        transcript,
        model,
        chunk_lines=settings.get("chunk_lines", DEFAULT_CHUNK_LINES),
        min_similarity=settings.get("min_similarity", DEFAULT_MIN_SIMILARITY),
        max_drop_words=settings.get("max_drop_words", DEFAULT_MAX_DROP_WORDS),
    )
    elapsed = time.monotonic() - start_time

    print(f"Wrote {result['cleaned_path']}")
    print(f"Wrote {result['changes_path']}")
    print(
        f"Lines sent: {result['lines_sent']}, dropped: {result['dropped']}, "
        f"rewritten: {result['rewritten']}, guard fallbacks: {result['guard_fallbacks']}, "
        f"drops refused: {result['drop_refused']} ({elapsed:.0f}s)"
    )


if __name__ == "__main__":
    main()
