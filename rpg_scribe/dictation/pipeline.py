"""Dictated clip → text: VAD gate, Whisper, then regex filters and vocabulary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mlx_whisper
import numpy as np

from rpg_scribe import paths, silero_onnx
from rpg_scribe.dictation.recorder import SAMPLE_RATE
from rpg_scribe.models import resolve_model_path
from rpg_scribe.text_filters import (
    DEFAULT_HALLUCINATION_PHRASES,
    compile_hallucination_patterns,
    strip_hallucination_phrases,
    strip_inline_fillers,
)
from rpg_scribe.vocabulary import apply_vocabulary, build_initial_prompt

# Shorter clips are accidental taps, not speech.
MIN_CLIP_SECONDS = 0.3

DEFAULT_INLINE_FILLERS = ["um", "uh", "er", "erm", "ah", "hmm", "mm"]


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
        _vad_model = silero_onnx.SileroVADSequence()
    return bool(
        silero_onnx.get_speech_timestamps(
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


def initial_prompt(vocab: dict) -> str:
    """Base terms, then the campaign's. mlx-whisper keeps the *last* ~224
    tokens of a long prompt, so the campaign terms go last."""
    parts = [build_initial_prompt(vocab.get("base") or {}, None),
             build_initial_prompt(vocab.get("campaign") or {}, None)]
    return ", ".join(p for p in parts if p)


def merged_vocab(vocab: dict) -> dict:
    return {**(vocab.get("base") or {}), **(vocab.get("campaign") or {})}


def transcribe_clip(audio: np.ndarray, vocab: dict, settings: PipelineSettings) -> str:
    """Transcribe one dictated clip. `vocab` is `{"base": ..., "campaign": ...}`
    parsed vocabularies. Returns `""` for a too-short or speechless clip."""
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
        initial_prompt=initial_prompt(vocab) or None,
        condition_on_previous_text=False,
    )
    text = result["text"].strip()
    text, _ = strip_hallucination_phrases(text, settings.hallucination_patterns)
    text = strip_inline_fillers(text, settings.inline_fillers)
    return apply_vocabulary(text, merged_vocab(vocab))
