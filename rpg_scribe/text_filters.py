"""Text filters shared by session transcription and dictation."""

from __future__ import annotations

import re

# Stock phrases Whisper emits from its training data on near-silent or noisy
# audio, removed as case-insensitive substrings from each segment before
# merging. Kept narrow on purpose: only phrases nobody says at a game table
# (so not "See you next time!" — that belongs in `filler_words`). The
# `hallucination_phrases` config key replaces this list rather than extending
# it — see docs/design/transcription-pipeline.md.
DEFAULT_HALLUCINATION_PHRASES = [
    r"Subtitles by the Amara\.org community",
    r"Thanks? (you )?for watching[.!]?",
    r"www\.\S+",
]


def compile_hallucination_patterns(phrases: list[str]) -> list[re.Pattern]:
    """Compile `hallucination_phrases` entries as case-insensitive regexes.

    Raises `ValueError` naming the first entry that isn't a valid regex.
    """
    patterns = []
    for phrase in phrases:
        try:
            patterns.append(re.compile(phrase, re.IGNORECASE))
        except re.error as err:
            raise ValueError(f"hallucination_phrases entry {phrase!r} is not a valid regex: {err}") from err
    return patterns


def strip_hallucination_phrases(text: str, patterns: list[re.Pattern]) -> tuple[str, int]:
    """Remove every `patterns` match from `text`.

    Returns the cleaned text (whitespace collapsed, stripped) and the hit
    count. With no hits, `text` comes back unchanged.
    """
    cleaned = text
    hits = 0
    for pattern in patterns:
        cleaned, count = pattern.subn("", cleaned)
        hits += count
    if hits:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, hits


def strip_inline_fillers(text: str, words: list[str]) -> str:
    """Remove filler words (whole words, any case) wherever they appear.

    A comma right after a filler goes with it ("Um, so" -> "so"). If the
    text started with a filler, the new first letter is capitalized.
    """
    if not words or not text:
        return text
    alternation = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    pattern = re.compile(rf"\b(?:{alternation})\b,?", re.IGNORECASE)
    started_with_filler = pattern.match(text.lstrip()) is not None
    cleaned = pattern.sub("", text)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # A filler alone before punctuation can leave it leading ("Um." -> ".").
    cleaned = re.sub(r"^[.,;:]\s*", "", cleaned)
    if started_with_filler and cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned
