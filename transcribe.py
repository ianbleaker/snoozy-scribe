"""Transcribe a Craig-recorded Discord TTRPG session into a Markdown file."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import random
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import subprocess

import mlx_whisper
import numpy as np
import silero_vad
import torch
import tqdm
import yaml
from huggingface_hub import snapshot_download
from mlx_whisper.audio import HOP_LENGTH
from mlx_whisper.tokenizer import LANGUAGES

VAD_SAMPLE_RATE = 16000

AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".aac"}

# Bump when a change alters what a .partial/ directory's saved segments mean
# (e.g. #20: per-region transcription calls instead of one batched call per
# track) so stale partials from before the change are discarded rather than
# resumed as-is.
PARTIAL_SCHEMA_VERSION = 2

# Matches mlx_whisper.transcribe's own default: a segment above this gzip
# compression ratio is repetitive enough that mlx_whisper already retried it
# at higher temperatures internally. If it's still above threshold here,
# every retry failed and mlx_whisper fell back to the last (still bad)
# decode rather than dropping it — see docs/design/transcription-pipeline.md.
COMPRESSION_RATIO_THRESHOLD = 2.4

_vad_model = None


def _strip_discriminator(username: str) -> str:
    return username.split("#", 1)[0]


def _slugify(session_title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", session_title.lower()).strip("-")


def _safe_track_filename(username: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", username)


def partial_dir_for(output_dir: Path, slug: str) -> Path:
    return output_dir / ".partial" / slug


def write_partial_meta(
    partial_dir: Path,
    model_repo: str,
    vad_min_silence_ms: int,
    initial_prompt: str,
    language: str | None,
    compression_ratio_threshold: float,
    best_of: int | None,
) -> None:
    partial_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": PARTIAL_SCHEMA_VERSION,
        "model_repo": model_repo,
        "vad_min_silence_ms": vad_min_silence_ms,
        "initial_prompt_hash": hashlib.sha256(initial_prompt.encode()).hexdigest(),
        "language": language,
        "compression_ratio_threshold": compression_ratio_threshold,
        "best_of": best_of,
    }
    (partial_dir / "_meta.json").write_text(json.dumps(meta))


def load_partial_meta(partial_dir: Path) -> dict | None:
    try:
        return json.loads((partial_dir / "_meta.json").read_text())
    except (json.JSONDecodeError, OSError):
        return None


def meta_matches(
    meta: dict | None,
    model_repo: str,
    vad_min_silence_ms: int,
    initial_prompt: str,
    language: str | None,
    compression_ratio_threshold: float,
    best_of: int | None,
) -> bool:
    if meta is None:
        return False
    return (
        meta.get("schema_version") == PARTIAL_SCHEMA_VERSION
        and meta.get("model_repo") == model_repo
        and meta.get("vad_min_silence_ms") == vad_min_silence_ms
        and meta.get("initial_prompt_hash")
        == hashlib.sha256(initial_prompt.encode()).hexdigest()
        and meta.get("language") == language
        and meta.get("compression_ratio_threshold") == compression_ratio_threshold
        and meta.get("best_of") == best_of
    )


def write_track_partial(
    partial_dir: Path,
    username: str,
    audio_path: Path,
    segments: list[tuple[float, float, str, float]],
    language: str | None,
) -> None:
    stat = audio_path.stat()
    data = {
        "audio_mtime": stat.st_mtime,
        "audio_size": stat.st_size,
        "language": language,
        "segments": segments,
    }
    partial_path = partial_dir / f"{_safe_track_filename(username)}.json"
    partial_path.write_text(json.dumps(data))


def load_track_partial(
    partial_dir: Path, username: str, audio_path: Path, progress: tqdm.tqdm | None = None
) -> tuple[list[tuple[float, float, str, float]], str | None] | None:
    partial_path = partial_dir / f"{_safe_track_filename(username)}.json"
    try:
        data = json.loads(partial_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    stat = audio_path.stat()
    if data["audio_mtime"] != stat.st_mtime or data["audio_size"] != stat.st_size:
        message = f"!! warning: partial for {username} is stale (audio file changed), retranscribing"
        progress.write(message) if progress is not None else print(message)
        return None

    segments = [
        (start, end, text, compression_ratio)
        for start, end, text, compression_ratio in data["segments"]
    ]
    return segments, data["language"]


def parse_manifest(craig_folder: Path) -> tuple[datetime, list[str]]:
    """Parse a Craig `info.txt` manifest into (start time, usernames)."""
    info_path = craig_folder / "info.txt"
    text = info_path.read_text()

    start_match = re.search(r"^Start time:\s*(\S+)", text, re.MULTILINE)
    if not start_match:
        raise ValueError(f"no 'Start time:' line found in {info_path}")
    start_time = datetime.fromisoformat(start_match.group(1).replace("Z", "+00:00"))

    tracks_match = re.search(r"^Tracks:\s*\n(.+)", text, re.MULTILINE | re.DOTALL)
    if not tracks_match:
        raise ValueError(f"no 'Tracks:' section found in {info_path}")

    usernames = []
    for line in tracks_match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        # Line format: "username#discriminator (user_id)"
        raw_username = line.split(" ", 1)[0]
        usernames.append(_strip_discriminator(raw_username))

    return start_time, usernames


def match_tracks(craig_folder: Path, usernames: list[str]) -> dict[str, Path]:
    """Map each manifest username to its audio file under `craig_folder`."""
    audio_files = [
        path
        for path in craig_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]

    matched: dict[str, Path] = {}
    for username in usernames:
        for path in audio_files:
            if username in path.name:
                matched[username] = path
                break
        else:
            print(f"!! warning: no audio file found for {username}, skipping")

    return matched


def resolve_model_path(model_repo: str, model_dir: Path) -> Path:
    """Resolve (downloading if needed) the local directory for a model repo."""
    target_dir = model_dir / model_repo.split("/")[-1]
    if not target_dir.exists() or not any(target_dir.iterdir()):
        snapshot_download(repo_id=model_repo, local_dir=str(target_dir))
    return target_dir


def find_default_zip(input_dir: Path) -> Path | None:
    """Return the oldest-by-mtime `*.zip` directly inside `input_dir`, if any."""
    if not input_dir.exists():
        return None
    zips = list(input_dir.glob("*.zip"))
    if not zips:
        return None
    return min(zips, key=lambda path: path.stat().st_mtime)


def extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract `zip_path` into `dest_dir` and return the Craig export root."""
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)

    entries = list(dest_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest_dir


def mark_zip_processed(zip_path: Path, input_dir: Path) -> Path:
    """Move `zip_path` into `<input_dir>/processed/`, creating it if needed."""
    processed_dir = input_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    new_path = processed_dir / zip_path.name
    shutil.move(str(zip_path), str(new_path))
    return new_path


def _decode_audio_mono16k(audio_path: Path) -> torch.Tensor:
    """Load audio as a mono float32 tensor resampled to `VAD_SAMPLE_RATE`.

    Shared decode helper for both VAD and transcription (Whisper itself
    expects 16kHz mono input, same as VAD)."""
    # Shells out to ffmpeg rather than soundfile/torchaudio; see
    # docs/design/transcription-pipeline.md for why.
    cmd = [
        "ffmpeg", "-nostdin", "-i", str(audio_path),
        "-threads", "0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(VAD_SAMPLE_RATE),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"failed to decode {audio_path}: {e.stderr.decode()}") from e

    samples = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(samples)


def _probe_duration(audio_path: Path) -> float:
    """Return an audio file's duration in seconds via `ffprobe`."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(audio_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"failed to probe duration of {audio_path}: {e.stderr.decode()}") from e
    return float(out.decode().strip())


def _parse_hms(value: str) -> float:
    """Parse an `HH:MM:SS` string into seconds."""
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"expected HH:MM:SS, got {value!r}")
    try:
        hours, minutes, seconds = (float(part) for part in parts)
    except ValueError as e:
        raise ValueError(f"expected HH:MM:SS, got {value!r}") from e
    return hours * 3600 + minutes * 60 + seconds


def crop_track(audio_path: Path, start: float, length: float, dest_dir: Path, username: str) -> Path:
    """Crop `audio_path` to `[start, start + length)` into a wav under `dest_dir`."""
    dest_path = dest_dir / f"{_safe_track_filename(username)}.wav"
    cmd = [
        "ffmpeg", "-nostdin",
        "-ss", str(start),
        "-t", str(length),
        "-i", str(audio_path),
        "-acodec", "pcm_s16le",
        str(dest_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"failed to crop {audio_path}: {e.stderr.decode()}") from e
    return dest_path


def detect_speech_regions(
    audio_path: Path,
    min_silence_ms: int,
    min_speech_ms: int = 250,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[list[float], float]:
    """Return (flat [start1, end1, start2, end2, ...] speech regions, raw duration), in seconds.

    `min_silence_ms` sets how long a pause must be before it splits speech
    into separate regions; see docs/design/transcription-pipeline.md for why
    the default (800ms) overrides Silero's own 100ms.

    `min_speech_ms` drops detected speech regions shorter than this (coughs,
    breath, mic pops) before they reach Whisper; defaults to Silero's own
    250ms default. See docs/design/transcription-pipeline.md.
    """
    global _vad_model
    if _vad_model is None:
        _vad_model = silero_vad.load_silero_vad()

    wav = _decode_audio_mono16k(audio_path)
    timestamps = silero_vad.get_speech_timestamps(
        wav,
        _vad_model,
        sampling_rate=VAD_SAMPLE_RATE,
        return_seconds=True,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        progress_tracking_callback=progress_callback,
    )

    regions: list[float] = []
    for entry in timestamps:
        regions.append(entry["start"])
        regions.append(entry["end"])
    duration = len(wav) / VAD_SAMPLE_RATE
    return regions, duration


def speech_seconds(regions: list[float]) -> float:
    """Sum of region durations in a flat [start1, end1, start2, end2, ...] list."""
    return sum(end - start for start, end in zip(regions[::2], regions[1::2]))


def speech_frames(regions: list[float]) -> int:
    """Whisper mel-frame count Whisper will actually decode for these regions.

    Mirrors `transcribe_track`'s own sample-based slicing (seconds -> samples
    -> mel frames) so this estimate and mlx_whisper's reported frame count
    for each region agree exactly; see docs/design/transcription-pipeline.md.
    """
    samples = [round(ts * VAD_SAMPLE_RATE) for ts in regions]
    return sum(
        (end - start) // HOP_LENGTH for start, end in zip(samples[::2], samples[1::2])
    )


class _ForwardingPbar:
    """Stand-in for `tqdm.tqdm` that forwards `.update()` calls to a shared bar."""

    def __init__(self, master: tqdm.tqdm):
        self._master = master

    def update(self, n: int = 1) -> None:
        self._master.update(min(n, self._master.total - self._master.n))

    def __enter__(self) -> "_ForwardingPbar":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _TqdmShim:
    """Replaces the `tqdm` name mlx_whisper.transcribe's module sees; see
    docs/design/transcription-pipeline.md for why."""

    def __init__(self, master: tqdm.tqdm):
        self._master = master

    def tqdm(self, *args: object, **kwargs: object) -> _ForwardingPbar:
        return _ForwardingPbar(self._master)


@contextlib.contextmanager
def _redirect_mlx_whisper_progress(master: tqdm.tqdm):
    transcribe_module = importlib.import_module("mlx_whisper.transcribe")
    original_tqdm = transcribe_module.tqdm
    transcribe_module.tqdm = _TqdmShim(master)
    try:
        yield
    finally:
        transcribe_module.tqdm = original_tqdm


def _tail_words(text: str, max_chars: int = 200) -> str:
    """Return the last `max_chars` characters of `text`, trimmed to a word boundary."""
    tail = text[-max_chars:]
    if len(tail) == len(text):
        return tail
    return tail.split(" ", 1)[-1] if " " in tail else tail


def transcribe_track(
    audio_path: Path,
    model_path: Path,
    speech_regions: list[float],
    progress: tqdm.tqdm,
    initial_prompt: str | None = None,
    language: str | None = None,
    compression_ratio_threshold: float = COMPRESSION_RATIO_THRESHOLD,
    best_of: int | None = None,
) -> tuple[list[tuple[float, float, str, float]], str | None]:
    """Transcribe one audio track into ((segment_start_offset, segment_end_offset,
    text, compression_ratio) tuples, detected language).

    Issues one mlx_whisper.transcribe() call per VAD speech region (not one
    batched call for the whole track): mlx_whisper's own `clip_timestamps`
    handling never resets its internal `seek` pointer between clips, so a
    single batched call decodes straight through the silence between
    regions. A single-clip-per-call shape sidesteps that entirely. See
    docs/design/transcription-pipeline.md.

    `language`, if given, is forced for every region's call instead of being
    auto-detected from the first region. Auto-detection runs Whisper's own
    language guess on whatever audio happens to be in that first region; a
    short, noisy, or ambiguous one (a grunt, a game sound effect) can guess
    wrong, and that wrong guess then gets pinned for the rest of the track,
    producing hallucinated non-English text throughout. Forcing a known
    language (e.g. "en" for an English-speaking group) skips that guess
    entirely.

    `compression_ratio_threshold` controls how eagerly mlx_whisper retries a
    region's decode at a higher temperature: a segment whose gzip
    compression ratio exceeds it is treated as a repetition failure and
    retried. Lowering it triggers more retries (slower, but catches more
    repetitive/garbled decodes); it also becomes the threshold `merge_tracks`
    uses to flag a segment for human review, since a segment still above it
    after retrying means every retry failed. See
    docs/design/transcription-pipeline.md.

    `best_of` only affects those same retries: mlx_whisper's first decode
    attempt is always greedy (temperature 0), and `best_of` is silently
    disabled at temperature 0, so it has no effect unless a segment already
    failed and is being retried at a higher temperature. When it does apply,
    mlx_whisper samples `best_of` independent decodes at that temperature and
    keeps the one it ranks highest, instead of a single resample — mlx_whisper
    doesn't implement beam search (`beam_size` raises `NotImplementedError`),
    so this is the available way to spend extra compute on a hard segment.
    """
    # verbose=None (not mlx_whisper's default False) suppresses its own prints,
    # which would otherwise corrupt the live progress bar; see
    # docs/design/transcription-pipeline.md.
    if not speech_regions:
        return [], None

    audio = _decode_audio_mono16k(audio_path).numpy()

    segments: list[tuple[float, float, str, float]] = []
    prompt = initial_prompt
    with _redirect_mlx_whisper_progress(progress):
        for start, end in zip(speech_regions[::2], speech_regions[1::2]):
            region_audio = audio[round(start * VAD_SAMPLE_RATE):round(end * VAD_SAMPLE_RATE)]
            result = mlx_whisper.transcribe(
                region_audio,
                path_or_hf_repo=str(model_path),
                verbose=None,
                initial_prompt=prompt,
                language=language,
                compression_ratio_threshold=compression_ratio_threshold,
                best_of=best_of,
            )
            if language is None:
                language = result.get("language")
            region_text = "".join(segment["text"] for segment in result["segments"])
            prompt = ((initial_prompt or "") + " " + _tail_words(region_text)).strip()
            segments.extend(
                (
                    segment["start"] + start,
                    segment["end"] + start,
                    segment["text"],
                    segment["compression_ratio"],
                )
                for segment in result["segments"]
            )
    return segments, language


class VocabularyParseError(Exception):
    """Raised for a structural or content error in a vocabulary.md file."""


def parse_vocabulary(path: Path) -> dict[str, dict]:
    """Parse a `vocabulary.md` file into `{term: {misheard_as, fields}}`.

    See docs/plans/completed/9-add-campaign-vocabulary-file.md for the format.
    """
    vocabulary: dict[str, dict] = {}
    current_term: str | None = None

    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("### "):
            name = line[len("### "):].strip()
            if name in vocabulary:
                raise VocabularyParseError(
                    f"{path}:{lineno}: duplicate term '{name}'"
                )
            vocabulary[name] = {"misheard_as": [], "fields": {}}
            current_term = name
            continue
        if line.startswith("# ") or line.startswith("## "):
            continue
        if line.startswith("- "):
            if current_term is None:
                raise VocabularyParseError(
                    f"{path}:{lineno}: bullet found outside any term section: {raw_line!r}"
                )
            body = line[len("- "):]
            if ":" not in body:
                raise VocabularyParseError(
                    f"{path}:{lineno}: malformed bullet (expected 'Label: value'): {raw_line!r}"
                )
            label, value = body.split(":", 1)
            label = label.strip()
            value = value.strip()
            if label.lower() == "misheard as":
                vocabulary[current_term]["misheard_as"] = [
                    v.strip() for v in value.split(",")
                ]
            else:
                vocabulary[current_term]["fields"][label] = value
            continue
        raise VocabularyParseError(f"{path}:{lineno}: malformed line: {raw_line!r}")

    return vocabulary


_SENTENCE_START = re.compile(r"^\s*$|[.!?]\s*$")


def apply_vocabulary(text: str, vocabulary: dict) -> str:
    """Case-insensitively replace known `misheard_as` variants with the canonical term.

    Capitalizes only the replacement's first letter, regardless of the
    term's own casing, when the match starts `text` or follows `.`/`!`/`?`
    plus whitespace; otherwise uses the term's exact canonical casing.
    """
    for term, data in vocabulary.items():
        for variant in data["misheard_as"]:
            pattern = re.compile(re.escape(variant), re.IGNORECASE)

            def _replace(match: re.Match, term: str = term) -> str:
                prefix = text[: match.start()]
                if _SENTENCE_START.search(prefix):
                    return term[:1].upper() + term[1:]
                return term

            text = pattern.sub(_replace, text)
    return text


def build_initial_prompt(vocabulary: dict, enrich_fields: set[str] | None) -> str:
    """Build Whisper's `initial_prompt` from vocabulary term names.

    `enrich_fields is None` -> bare term names only. `enrich_fields == set()`
    -> append every field found per term ("all"). Non-empty `enrich_fields`
    -> append only fields whose label case-insensitively matches an entry.
    """
    parts = []
    for term, data in vocabulary.items():
        fields = data["fields"]
        if enrich_fields is None or not fields:
            parts.append(term)
            continue
        if enrich_fields:
            wanted_lower = {f.lower() for f in enrich_fields}
            values = [v for label, v in fields.items() if label.lower() in wanted_lower]
        else:
            values = list(fields.values())
        if values:
            parts.append(f"{term} ({'; '.join(values)})")
        else:
            parts.append(term)
    return ", ".join(parts)


def resolve_speaker(username: str, speaker_map: dict[str, str], warned: set[str]) -> str:
    """Map a Discord username to its configured display name.

    Warns at most once per username, since a missing speaker_map entry
    applies to every segment from that track.
    """
    if username in speaker_map:
        return speaker_map[username]
    if username not in warned:
        print(f"!! warning: no speaker_map entry for {username}, using raw Discord name")
        warned.add(username)
    return username


_TRAILING_PUNCTUATION = re.compile(r"[.,!?;:\s]+$")


def _normalize(text: str) -> str:
    """Lowercase and strip trailing punctuation/whitespace, for repeat/filler comparisons."""
    return _TRAILING_PUNCTUATION.sub("", text.lower())


# Genuine emphasis ("no no no") repeats a handful of times at most; a
# repetition-hallucination loop runs for dozens or hundreds of words (see
# docs/analysis/best-of-decoding-analysis.md and
# docs/analysis/model-weight-analysis.md for real examples). Set well above
# normal speech's repeat count so collapse_repetition() only fires on loops.
REPETITION_MIN_REPEATS = 6
REPETITION_MAX_PHRASE_WORDS = 6

# A hallucination loop can also come out as one run-on token with no spaces
# (e.g. a single character repeated hundreds of times, tokenized as one
# "word"). Only worth checking on words far longer than any real one, so
# ordinary long words/interjections are never touched.
REPETITION_MIN_WORD_LENGTH_FOR_CHAR_COLLAPSE = 20
REPETITION_MAX_CHAR_PERIOD = 4


def _collapse_runs(tokens: list[str], keys: list[str], min_repeats: int, max_period: int) -> list[str]:
    """Collapse a run of `min_repeats`+ immediately-repeated `keys` down to one.

    Generic over what a "token" is (a word or a single character) — the
    caller picks `tokens` (kept verbatim in the output) and `keys` (used for
    comparison, e.g. normalized/lowercased). At each position, tries period
    lengths from 1 up to `max_period`, smallest first, so a 1-token period
    ("H H H H") is preferred over a spuriously-matching longer one.
    """
    n = len(tokens)
    collapsed: list[str] = []
    i = 0
    while i < n:
        for period in range(1, max_period + 1):
            if i + period > n:
                continue
            phrase = keys[i : i + period]
            if any(key == "" for key in phrase):
                continue
            repeats = 1
            j = i + period
            while j + period <= n and keys[j : j + period] == phrase:
                repeats += 1
                j += period
            if repeats >= min_repeats:
                collapsed.extend(tokens[i : i + period])
                i = j
                break
        else:
            collapsed.append(tokens[i])
            i += 1
    return collapsed


def collapse_char_repetition(
    word: str,
    min_repeats: int = REPETITION_MIN_REPEATS,
    max_period: int = REPETITION_MAX_CHAR_PERIOD,
    min_word_length: int = REPETITION_MIN_WORD_LENGTH_FOR_CHAR_COLLAPSE,
) -> str:
    """Collapse a run-on repeated-character/substring hallucination in one word.

    Whisper occasionally hallucinates a single character (or short substring)
    repeated hundreds of times with no whitespace at all — e.g. "řřřřř...ř" —
    which `collapse_repetition()`'s word-splitting can't see since it's one
    token. Only applies to words at least `min_word_length` long, since no
    real word or interjection gets anywhere close to that; short ones (e.g.
    "hahaha") are left alone. Returns `word` unchanged if nothing collapses.
    """
    if len(word) < min_word_length:
        return word
    chars = list(word)
    collapsed = "".join(_collapse_runs(chars, [c.lower() for c in chars], min_repeats, max_period))
    return collapsed if len(collapsed) < len(word) else word


def collapse_repetition(
    text: str,
    min_repeats: int = REPETITION_MIN_REPEATS,
    max_phrase_words: int = REPETITION_MAX_PHRASE_WORDS,
) -> str:
    """Collapse an immediately-repeated word/phrase run down to one occurrence.

    A repetition-hallucination loop repeats the same short (1-6 word) phrase
    dozens or hundreds of times in a row — e.g. "H H H H...", "listen listen
    listen...", "they all work there they all work there...". Collapses a
    run to its first occurrence once it repeats at least `min_repeats` times
    immediately in a row; see `_collapse_runs()`. Comparison is done on
    `_normalize()`d words so trailing punctuation ("Sh," vs "Sh") and case
    don't block a match; the original word tokens (with punctuation/case)
    are kept in the output.

    Each resulting word is then also passed through
    `collapse_char_repetition()`, to catch a hallucination that came out as
    one long run-on token instead of repeated whitespace-separated words.

    Returns the text unchanged if no qualifying run is found.
    """
    words = text.split()
    normalized = [_normalize(word) for word in words]
    collapsed_words = _collapse_runs(words, normalized, min_repeats, max_phrase_words)
    return " ".join(collapse_char_repetition(word) for word in collapsed_words)


def merge_tracks(
    session_start: datetime,
    transcripts: dict[str, list[tuple[float, float, str, float]]],
    merge_gap_seconds: float,
    filler_words: list[str],
    compression_ratio_threshold: float = COMPRESSION_RATIO_THRESHOLD,
) -> tuple[list[tuple[datetime, str, str]], dict[str, int]]:
    """Interleave per-track segments into one chronological transcript.

    Consecutive segments from the same speaker (per-track, ignoring what
    other speakers' tracks were doing at the same time) are merged into one
    line when the silence between them (`next.start - prev.end`) is at most
    `merge_gap_seconds`; see docs/design/transcription-pipeline.md.

    A segment is dropped entirely if its text, normalized (lowercased,
    trailing punctuation stripped), exactly matches an entry in
    `filler_words` (whole-line match only) or matches the previous surviving
    segment from the same speaker (repeat dedup, also normalized so
    "Okay."/"okay"/"Okay," collapse). The original, non-normalized text is
    kept for whichever occurrence is emitted.

    Segments whose compression ratio is still above
    `compression_ratio_threshold` after mlx_whisper's own temperature
    fallback are flagged with a "!! " prefix rather than silently emitted —
    that ratio means mlx_whisper itself treated the decode as a repetition
    failure and had nothing better to fall back to. This must be the same
    threshold passed to `transcribe_track`'s `compression_ratio_threshold`,
    since a segment's compression ratio is only meaningful relative to the
    threshold that drove its own retries.

    Every segment's text is also passed through `collapse_repetition()`
    before anything else, regardless of its compression ratio — a
    repetition-hallucination loop can be diluted below the ratio threshold
    if it's only part of a longer segment. A segment collapse_repetition()
    actually changes is flagged too, using the collapsed text, since a
    changed segment means a real hallucination loop was caught: unedited
    repeated-word walls made the exact case compression_ratio_threshold is
    meant to catch unreadable even after flagging.

    A flagged segment (either way) is exempt from filler-word dropping and
    repeat dedup, and is never merged with a neighbor in either direction —
    it's always emitted as its own line for human review.

    Also returns a stats dict counting what was pruned along the way
    (`segments_total`, `empty_dropped`, `repetition_collapsed`,
    `filler_dropped`, `repeat_deduped`, `flagged_for_review`), for the
    caller to report and to record in the transcript's frontmatter.
    """
    merged: list[tuple[datetime, str, str]] = []
    normalized_filler_words = {_normalize(word) for word in filler_words}
    stats = {
        "segments_total": 0,
        "empty_dropped": 0,
        "repetition_collapsed": 0,
        "filler_dropped": 0,
        "repeat_deduped": 0,
        "flagged_for_review": 0,
    }

    for username, segments in transcripts.items():
        previous_normalized: str | None = None
        buffer_start: float | None = None
        buffer_end: float | None = None
        buffer_text: str | None = None

        for start, end, text, compression_ratio in segments:
            stats["segments_total"] += 1
            stripped = text.strip()
            if stripped in ("", "..."):
                stats["empty_dropped"] += 1
                continue
            collapsed_text = collapse_repetition(stripped)
            repetition_collapsed = collapsed_text != stripped
            if repetition_collapsed:
                stats["repetition_collapsed"] += 1
            stripped = collapsed_text
            flagged = compression_ratio > compression_ratio_threshold or repetition_collapsed
            normalized = _normalize(stripped)
            if not flagged and normalized in normalized_filler_words:
                stats["filler_dropped"] += 1
                continue
            if not flagged and normalized == previous_normalized:
                stats["repeat_deduped"] += 1
                continue
            previous_normalized = normalized

            if flagged:
                stats["flagged_for_review"] += 1
                if buffer_text is not None:
                    timestamp = session_start + timedelta(seconds=buffer_start)
                    merged.append((timestamp, username, buffer_text))
                    buffer_start = buffer_end = buffer_text = None
                timestamp = session_start + timedelta(seconds=start)
                merged.append((timestamp, username, f"!! {stripped}"))
                continue

            if buffer_text is not None and start - buffer_end <= merge_gap_seconds:
                buffer_text = f"{buffer_text} {stripped}"
                buffer_end = end
            else:
                if buffer_text is not None:
                    timestamp = session_start + timedelta(seconds=buffer_start)
                    merged.append((timestamp, username, buffer_text))
                buffer_start, buffer_end, buffer_text = start, end, stripped

        if buffer_text is not None:
            timestamp = session_start + timedelta(seconds=buffer_start)
            merged.append((timestamp, username, buffer_text))

    merged.sort(key=lambda entry: entry[0])
    return merged, stats


def write_transcript(
    merged: list[tuple[datetime, str, str]],
    session_title: str,
    output_dir: Path,
    timestamp_format: str,
    line_format: str,
    merge_stats: dict[str, int],
    filename_suffix: str = "",
) -> Path:
    """Write the merged transcript as a Markdown file with YAML frontmatter.

    `line_format` is a str.format() template with `{timestamp}`, `{speaker}`,
    and `{text}` placeholders, letting the config control per-line Markdown
    (e.g. `**{speaker}**` to bold speaker names).

    `merge_stats` (as returned by `merge_tracks()`) is recorded in the
    frontmatter under `pruning`, so what was dropped/collapsed/flagged stays
    attached to the transcript file itself, not just the console output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(session_title)
    output_path = output_dir / f"{slug}{filename_suffix}.md"

    speakers = list(dict.fromkeys(name for _, name, _ in merged))
    date = merged[0][0].date().isoformat() if merged else ""

    frontmatter = yaml.safe_dump(
        {"date": date, "title": session_title, "speakers": speakers, "pruning": merge_stats},
        sort_keys=False,
    )

    try:
        lines = [
            line_format.format(timestamp=timestamp.strftime(timestamp_format), speaker=name, text=text)
            for timestamp, name, text in merged
        ]
    except KeyError as e:
        raise ValueError(
            f"config `line_format` uses unknown placeholder {e}; "
            "only {timestamp}, {speaker}, {text} are supported"
        ) from e

    output_path.write_text(f"---\n{frontmatter}---\n\n" + "\n".join(lines) + "\n")
    return output_path


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _rule(char: str = "=") -> str:
    width = min(shutil.get_terminal_size((80, 20)).columns, 80)
    return char * width


def main() -> None:
    start_time = time.monotonic()

    print(_rule())
    print("rpg-scribe".center(len(_rule())))
    print(_rule())
    print()

    parser = argparse.ArgumentParser(
        description="Transcribe a Craig-recorded Discord TTRPG session into Markdown."
    )
    parser.add_argument(
        "craig_folder",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the Craig export folder or .zip file. If omitted, the oldest "
        "*.zip in input_dir is picked up automatically.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml"
    )
    parser.add_argument("--model", help="Override the configured whisper_model")
    parser.add_argument(
        "--language",
        help="Force this language code (e.g. 'en') for every track instead of "
        "auto-detecting per track. Overrides the config `language` key.",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=None,
        help="Gzip compression ratio above which mlx_whisper retries a decode "
        "at a higher temperature, and above which a segment still failing "
        "after retries is flagged for review. Lower catches more repetitive/"
        "garbled decodes at the cost of more retries. Defaults to 2.4 if "
        "omitted. Overrides the config `compression_ratio_threshold` key.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=None,
        help="Sample this many independent decodes and keep the best-ranked one, "
        "but only for segments already being retried at a higher temperature "
        "(mlx_whisper disables best_of on its initial greedy decode). "
        "mlx_whisper doesn't implement beam search. Omit to leave mlx_whisper's "
        "default (a single resample per retry). Overrides the config `best_of` key.",
    )
    parser.add_argument("--title", help="Override the session title")
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=None,
        help="Path to vocabulary.md. Overrides the config `vocabulary_file` key "
        "and the default ./vocabulary.md lookup.",
    )
    parser.add_argument(
        "--enrich-vocabulary",
        nargs="?",
        const="",
        default=None,
        help="Seed Whisper's initial_prompt with vocabulary enrichment fields "
        "in addition to term names. Bare flag includes every field found; "
        "a comma-separated list (e.g. Tagline,Description) includes only those.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any saved partial output for this session and retranscribe "
        "every track from scratch.",
    )
    parser.add_argument(
        "--test",
        nargs="?",
        type=int,
        const=120,
        default=None,
        metavar="SECONDS",
        help="Rapid-iteration mode: only VAD and transcribe a shared window of "
        "SECONDS (default 120) across all tracks, picked randomly unless "
        "--test-start is given. Writes to <slug>-test.md and never touches "
        "partial output or the input zip.",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default=None,
        metavar="HH:MM:SS",
        help="Pin the --test window's start instead of picking randomly. "
        "Only meaningful alongside --test.",
    )
    args = parser.parse_args()

    if args.test_start is not None and args.test is None:
        sys.exit("error: --test-start requires --test")

    if not args.config.exists():
        sys.exit(
            f"error: config file not found at {args.config}. "
            "Run `cp config.example.yaml config.yaml` and fill in your values."
        )
    config = yaml.safe_load(args.config.read_text())

    model_repo = args.model or config["whisper_model"]
    language = args.language or config.get("language")
    if language is not None and language not in LANGUAGES:
        sys.exit(f"error: unknown language code '{language}'")
    compression_ratio_threshold = (
        args.compression_ratio_threshold
        if args.compression_ratio_threshold is not None
        else config.get("compression_ratio_threshold", COMPRESSION_RATIO_THRESHOLD)
    )
    best_of = args.best_of if args.best_of is not None else config.get("best_of")
    input_dir = Path(config.get("input_dir", "./input"))
    vad_min_silence_ms = config.get("vad_min_silence_ms", 800)
    vad_min_speech_ms = config.get("vad_min_speech_ms", 250)
    merge_gap_ms = config.get("merge_gap_ms", 2000)
    timestamp_format = config.get("timestamp_format", "%H:%M:%S")
    line_format = config.get("line_format", "[{timestamp}] {speaker}: {text}")
    filler_words = config.get("filler_words", ["so", "you", "um", "uh", "hm", "hmm"])

    default_vocab_path = Path("./vocabulary.md")
    if args.vocabulary is not None:
        vocabulary_path = args.vocabulary
    elif config.get("vocabulary_file") and Path(config["vocabulary_file"]).exists():
        vocabulary_path = Path(config["vocabulary_file"])
    elif default_vocab_path.exists():
        vocabulary_path = default_vocab_path
    else:
        vocabulary_path = None

    if vocabulary_path is None:
        print("No vocabulary file configured or found; continuing without vocabulary.")
        vocabulary: dict = {}
    else:
        try:
            vocabulary = parse_vocabulary(vocabulary_path)
        except VocabularyParseError as e:
            sys.exit(f"error: {e}")
        field_counts: dict[str, tuple[str, int]] = {}
        for term_data in vocabulary.values():
            for label in term_data["fields"]:
                key = label.lower()
                display, count = field_counts.get(key, (label, 0))
                field_counts[key] = (display, count + 1)
        fields_summary = ", ".join(
            f"{display} ×{count}" for display, count in field_counts.values()
        )
        fields_note = f" (fields found: {fields_summary})" if fields_summary else ""
        print(
            f"Loaded vocabulary: {len(vocabulary)} terms from {vocabulary_path.name}"
            f"{fields_note}"
        )

    if args.enrich_vocabulary is None:
        enrich_fields = None
    elif args.enrich_vocabulary == "":
        enrich_fields = set()
    else:
        enrich_fields = {f.strip() for f in args.enrich_vocabulary.split(",")}
    initial_prompt = build_initial_prompt(vocabulary, enrich_fields)

    if args.craig_folder is None:
        found_zip = find_default_zip(input_dir)
        if found_zip is None:
            sys.exit(
                f"error: no .zip files found in {input_dir}. "
                "Drop a Craig export there, or pass a path explicitly."
            )
        resolved_input_path = found_zip
    else:
        resolved_input_path = args.craig_folder

    print(f"Input: {resolved_input_path.name}")

    is_zip = resolved_input_path.suffix == ".zip"
    session_title = args.title or (
        resolved_input_path.resolve().stem if is_zip else resolved_input_path.resolve().name
    )

    slug = _slugify(session_title)
    partial_dir = partial_dir_for(Path(config["output_dir"]), slug)

    extraction_ctx = tempfile.TemporaryDirectory() if is_zip else contextlib.nullcontext()
    with extraction_ctx as tmp_dir, contextlib.ExitStack() as exit_stack:
        if is_zip:
            print("Extracting zip...")
            craig_folder = extract_zip(resolved_input_path, Path(tmp_dir))
        else:
            craig_folder = resolved_input_path
        print()

        session_start, usernames = parse_manifest(craig_folder)
        print(f"Parsed manifest: {len(usernames)} speakers, session started {session_start}")
        matched = match_tracks(craig_folder, usernames)
        print()

        window_start = 0.0
        if args.test is not None:
            durations = {username: _probe_duration(path) for username, path in matched.items()}
            shortest_username = min(durations, key=durations.get)
            min_duration = durations[shortest_username]
            if args.test_start is not None:
                try:
                    window_start = _parse_hms(args.test_start)
                except ValueError as e:
                    sys.exit(f"error: invalid --test-start: {e}")
            else:
                window_start = random.uniform(0, max(0.0, min_duration - args.test))
            if window_start + args.test > min_duration:
                sys.exit(
                    f"error: requested --test window "
                    f"({timedelta(seconds=round(window_start))}-"
                    f"{timedelta(seconds=round(window_start + args.test))}) exceeds the "
                    f"shortest matched track's duration ({shortest_username}: "
                    f"{timedelta(seconds=round(min_duration))})"
                )
            crop_dir = Path(exit_stack.enter_context(tempfile.TemporaryDirectory()))
            matched = {
                username: crop_track(path, window_start, args.test, crop_dir, username)
                for username, path in matched.items()
            }
            print(
                f"Test mode: using window {timedelta(seconds=round(window_start))}-"
                f"{timedelta(seconds=round(window_start + args.test))} ({args.test}s)"
            )
            print()

        print(f"Analyzing speech regions (VAD) for {len(matched)} tracks...")
        track_regions: dict[str, list[float]] = {}
        total_raw_s = 0.0
        total_speech_s = 0.0
        vad_total = len(matched)
        with tqdm.tqdm(total=100, unit="%", leave=False) as vad_progress:

            def _on_vad_progress(pct: float) -> None:
                vad_progress.n = round(pct)
                vad_progress.refresh()

            for i, (username, audio_path) in enumerate(matched.items(), start=1):
                vad_progress.reset(total=100)
                vad_progress.set_description(f"VAD {i}/{vad_total}: {username}")
                regions, duration = detect_speech_regions(
                    audio_path,
                    vad_min_silence_ms,
                    vad_min_speech_ms,
                    _on_vad_progress,
                )
                track_regions[username] = regions
                track_speech_s = speech_seconds(regions)
                total_raw_s += duration
                total_speech_s += track_speech_s
                vad_progress.write(
                    f"  -> {_format_duration(track_speech_s)} speech / "
                    f"{_format_duration(duration)} raw ({track_speech_s / duration * 100:.0f}%)"
                )
        speech_pct = (total_speech_s / total_raw_s * 100) if total_raw_s else 0.0
        print(
            f"  VAD: {_format_duration(total_speech_s)} speech / "
            f"{_format_duration(total_raw_s)} raw audio ({speech_pct:.0f}%) to transcribe"
        )
        print()

        model_dir = Path(config["model_dir"])
        target_model_dir = model_dir / model_repo.split("/")[-1]
        print(f"Resolving model {model_repo}...")
        if not target_model_dir.exists() or not any(target_model_dir.iterdir()):
            print("  Downloading model (first run)...")
        model_path = resolve_model_path(model_repo, model_dir)
        print()

        total_frames = sum(speech_frames(regions) for regions in track_regions.values())
        transcripts = {}
        total = len(matched)

        if args.test is None:
            if args.fresh and partial_dir.exists():
                shutil.rmtree(partial_dir)
                print("Ignoring saved partial output (--fresh).")
            elif partial_dir.exists():
                meta = load_partial_meta(partial_dir)
                if not meta_matches(
                    meta,
                    model_repo,
                    vad_min_silence_ms,
                    initial_prompt,
                    language,
                    compression_ratio_threshold,
                    best_of,
                ):
                    print(
                        "!! warning: saved partial output for this session doesn't match the "
                        "current model/VAD/vocabulary/language/compression-ratio/best-of config, "
                        "discarding and retranscribing all tracks"
                    )
                    shutil.rmtree(partial_dir)
            write_partial_meta(
                partial_dir,
                model_repo,
                vad_min_silence_ms,
                initial_prompt,
                language,
                compression_ratio_threshold,
                best_of,
            )

        with tqdm.tqdm(
            total=total_frames,
            unit="frames",
            desc="Transcribing",
            disable=total_frames == 0,
            leave=False,
        ) as progress:
            for i, (username, audio_path) in enumerate(matched.items(), start=1):
                partial = (
                    None
                    if args.test is not None
                    else load_track_partial(partial_dir, username, audio_path, progress)
                )
                if partial is not None:
                    segments, detected_language = partial
                    progress.write(
                        f"Resuming track {i}/{total}: {username} (from saved partial)..."
                    )
                    progress.update(speech_frames(track_regions[username]))
                else:
                    progress.write(f"Transcribing track {i}/{total}: {username}...")
                    segments, detected_language = transcribe_track(
                        audio_path,
                        model_path,
                        track_regions[username],
                        progress,
                        initial_prompt,
                        language,
                        compression_ratio_threshold,
                        best_of,
                    )
                    if args.test is None:
                        write_track_partial(
                            partial_dir, username, audio_path, segments, detected_language
                        )
                if detected_language:
                    progress.write(
                        f"  Detected language: "
                        f"{LANGUAGES.get(detected_language, detected_language).title()}"
                    )
                progress.write(f"  -> {len(segments)} segments")
                transcripts[username] = [
                    (
                        offset + window_start,
                        end + window_start,
                        apply_vocabulary(text, vocabulary),
                        compression_ratio,
                    )
                    for offset, end, text, compression_ratio in segments
                ]
        print()

        print("Merging tracks...")
        merged, merge_stats = merge_tracks(
            session_start,
            transcripts,
            merge_gap_ms / 1000,
            filler_words,
            compression_ratio_threshold,
        )
        print(
            f"  {merge_stats['repetition_collapsed']} repetition hallucination(s) collapsed, "
            f"{merge_stats['filler_dropped']} filler word(s) dropped, "
            f"{merge_stats['repeat_deduped']} repeat(s) deduped, "
            f"{merge_stats['empty_dropped']} empty segment(s) dropped, "
            f"{merge_stats['flagged_for_review']} line(s) flagged for review "
            f"({merge_stats['segments_total']} segments in)"
        )
        warned_speakers: set[str] = set()
        resolved = [
            (timestamp, resolve_speaker(username, config["speaker_map"], warned_speakers), text)
            for timestamp, username, text in merged
        ]
        print()

        print("Writing transcript...")
        output_path = write_transcript(
            resolved,
            session_title,
            Path(config["output_dir"]),
            timestamp_format,
            line_format,
            merge_stats,
            filename_suffix="-test" if args.test is not None else "",
        )
        if args.test is None:
            shutil.rmtree(partial_dir, ignore_errors=True)

        if is_zip and args.test is None:
            mark_zip_processed(resolved_input_path, input_dir)

    elapsed = time.monotonic() - start_time
    print(f"Wrote transcript to {output_path} ({_format_duration(elapsed)})")
    print()
    print(_rule())


if __name__ == "__main__":
    main()
