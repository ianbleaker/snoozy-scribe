"""Dictated clip → text: VAD gate, Whisper, then regex filters and vocabulary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from snoozy_scribe import whisper_stubs  # noqa: F401  (must precede mlx_whisper)

import mlx_whisper
import numpy as np

from snoozy_scribe import paths, silero
from snoozy_scribe.dictation.recorder import SAMPLE_RATE
from snoozy_scribe.models import resolve_model_path
from snoozy_scribe.text_filters import (
    DEFAULT_HALLUCINATION_PHRASES,
    compile_hallucination_patterns,
    strip_hallucination_phrases,
    strip_inline_fillers,
)
from snoozy_scribe.vocabulary import apply_vocabulary, build_combined_prompt, merge_vocabularies

# Shorter clips are accidental taps, not speech.
MIN_CLIP_SECONDS = 0.3

DEFAULT_INLINE_FILLERS = ["um", "uh", "er", "erm", "ah", "hmm", "mm"]

# Whisper copies the style of its initial prompt. A prompt with written-out
# disfluencies makes it transcribe fillers as "um"/"uh" (which
# strip_inline_fillers removes) instead of guessing real words like "a" or
# "I'm". It goes first so prompt truncation drops it before vocabulary terms.
FILLER_PREAMBLE = "Um, uh, so, I think we should, uh, hmm, go."


def _initial_prompt(vocabs: list[dict]) -> str:
    vocab_prompt = build_combined_prompt(vocabs, None)
    return f"{FILLER_PREAMBLE} {vocab_prompt}" if vocab_prompt else FILLER_PREAMBLE


@dataclass
class PipelineSettings:
    whisper_model_dir: Path
    language: str | None
    hallucination_patterns: list[re.Pattern]
    inline_fillers: list[str]
    vad_min_silence_ms: int
    vad_min_speech_ms: int


def load_settings(config: dict) -> PipelineSettings:
    """Build settings from the loaded config, downloading the Whisper model if needed."""
    dictation = config.get("dictation") or {}
    model_dir = paths.resolve(config.get("model_dir", "./data/models"))
    return PipelineSettings(
        whisper_model_dir=resolve_model_path(config["whisper_model"], model_dir),
        language=config.get("language"),
        hallucination_patterns=compile_hallucination_patterns(
            config.get("hallucination_phrases", DEFAULT_HALLUCINATION_PHRASES)
        ),
        inline_fillers=dictation.get("inline_fillers", DEFAULT_INLINE_FILLERS),
        vad_min_silence_ms=int(config.get("vad_min_silence_ms", 800)),
        vad_min_speech_ms=int(config.get("vad_min_speech_ms", 250)),
    )


_vad_model = None


def _has_speech(audio: np.ndarray, settings: PipelineSettings) -> bool:
    global _vad_model
    if _vad_model is None:
        _vad_model = silero.SileroVADSequence()
    return bool(
        silero.get_speech_timestamps(
            audio,
            _vad_model,
            min_silence_duration_ms=settings.vad_min_silence_ms,
            min_speech_duration_ms=settings.vad_min_speech_ms,
        )
    )


def warm_up(settings: PipelineSettings) -> None:
    """Load the Whisper weights and VAD model so the first dictation is fast."""
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    _has_speech(silence, settings)
    mlx_whisper.transcribe(
        silence, path_or_hf_repo=str(settings.whisper_model_dir), verbose=None, language=settings.language
    )


def transcribe_clip(audio: np.ndarray, vocabs: list[dict], settings: PipelineSettings) -> str:
    """Transcribe one dictated clip. `vocabs` is the selected parsed
    vocabularies in alphabetical order, which drives both the prompt and the
    merge. Returns `""` for a too-short or speechless clip."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < MIN_CLIP_SECONDS * SAMPLE_RATE:
        return ""
    # Whisper hallucinates "Thank you." on silence, so don't let it see any.
    if not _has_speech(audio, settings):
        return ""
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=str(settings.whisper_model_dir),
        verbose=None,
        language=settings.language,
        initial_prompt=_initial_prompt(vocabs),
        condition_on_previous_text=False,
    )
    text = result["text"].strip()
    text, _ = strip_hallucination_phrases(text, settings.hallucination_patterns)
    text = strip_inline_fillers(text, settings.inline_fillers)
    return apply_vocabulary(text, merge_vocabularies(vocabs))
