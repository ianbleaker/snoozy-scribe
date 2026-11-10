"""Transcribe a Craig-recorded Discord TTRPG session into a Markdown file."""

from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import importlib
import json
import random
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import unicodedata
import zipfile
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import mlx.core as mx
import mlx_whisper
import numpy as np
import tqdm
import yaml
from mlx_whisper.audio import HOP_LENGTH
from mlx_whisper.tokenizer import LANGUAGES

from snoozy_scribe import identity, paths, silero_onnx
from snoozy_scribe.config import check_legacy_layout, load_config
from snoozy_scribe.models import resolve_model_path
from snoozy_scribe.text_filters import (
    DEFAULT_HALLUCINATION_PHRASES,
    compile_hallucination_patterns,
    strip_hallucination_phrases,
)
from snoozy_scribe.vocabulary import (
    VocabularyParseError,
    apply_vocabulary,
    build_initial_prompt,
    campaign_path,
    list_campaigns,
    parse_vocabulary,
)

VAD_SAMPLE_RATE = 16000

AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".aac"}

# Arbitrary fixed epoch used as merge_tracks()'s timestamp base in "elapsed"
# timestamp mode, so the first segment lands at 00:00:00 instead of the real
# session wall-clock time. Sessions run well under 24h, so this never wraps.
_ELAPSED_TIMESTAMP_EPOCH = datetime(2000, 1, 1)

# Bump when a change alters what a .partial/ directory's saved segments mean
# (e.g. #20: per-region transcription calls instead of one batched call per
# track) so stale partials from before the change are discarded rather than
# resumed as-is.
PARTIAL_SCHEMA_VERSION = 3

# Matches mlx_whisper.transcribe's own default: a segment above this gzip
# compression ratio is repetitive enough that mlx_whisper already retried it
# at higher temperatures internally. If it's still above threshold here,
# every retry failed and mlx_whisper fell back to the last (still bad)
# decode rather than dropping it.
COMPRESSION_RATIO_THRESHOLD = 2.4

# Matches mlx_whisper.transcribe's own default: a segment whose average
# per-token log-probability falls below this is a low-confidence decode —
# mlx_whisper already retried it at higher temperatures internally, so a
# segment still below threshold here means every retry failed. Catches
# garbled/wrong-language hallucinations that compression_ratio_threshold
# misses, since made-up words are high-entropy (they don't compress well)
# even though the model had no real signal for them.
LOGPROB_THRESHOLD = -1.0

_vad_model = None


class TranscriptionCancelled(Exception):
    """Raised once the user confirms cancellation via the Escape-then-y prompt."""


class _CancelWatcher:
    """Watches stdin in a background thread for a bare Escape keypress
    followed by 'y', setting `cancelled` so the main loop can stop between
    work units (tracks, VAD passes, speech regions).

    Runs stdin in cbreak mode so a keypress is seen immediately without
    waiting for Enter. The confirmation prompt itself is shown inside
    `tqdm.external_write_mode()`, which grabs tqdm's own write lock and
    clears any active bar first — without that, an active bar's own
    refreshes (driven by the main thread transcribing in parallel with
    this watcher thread) redraw over the prompt line as fast as several
    times a second, so the prompt flickers away before it can be read or
    answered. Disabled entirely when stdin isn't a real terminal (piped
    input, CI), since raw mode there would just consume whatever's on the
    other end of the pipe for no benefit.
    """

    def __init__(self) -> None:
        self.cancelled = threading.Event()
        self._enabled = sys.stdin.isatty()
        self._old_settings = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __enter__(self) -> "_CancelWatcher":
        if self._enabled:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self._stop.set()
        if self._enabled and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
        return False

    def _watch(self) -> None:
        while not self._stop.is_set():
            if not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
            if sys.stdin.read(1) != "\x1b":
                continue
            # An escape *sequence* (arrow keys etc.) sends more bytes right
            # behind the ESC byte; a bare Escape keypress doesn't. Swallow
            # the rest of a sequence instead of treating it as a cancel.
            if select.select([sys.stdin], [], [], 0.05)[0]:
                sys.stdin.read(1)
                continue
            with tqdm.tqdm.external_write_mode(file=sys.stderr):
                sys.stderr.write("Cancel transcription? [y/N] ")
                sys.stderr.flush()
                confirm = sys.stdin.read(1) if select.select([sys.stdin], [], [], 10.0)[0] else ""
                sys.stderr.write("\n")
            if confirm.lower() == "y":
                self.cancelled.set()
                return


PROGRESS_EVENT_PREFIX = "@@PROGRESS@@ "


class _ProgressEvents:
    """Emits single-line, prefixed JSON progress events on stdout for a GUI
    driving this script as a subprocess (`--progress-events`). A no-op unless
    `enabled`, so default CLI output is unchanged. Per-phase progress is
    throttled to one event per whole-percent change, since the VAD callback
    and the transcription bar both tick far more often than that.
    """

    def __init__(self) -> None:
        self.enabled = False
        self._last_pct: dict[str, int] = {}

    def emit(self, **event: object) -> None:
        if self.enabled:
            print(PROGRESS_EVENT_PREFIX + json.dumps(event), flush=True)

    def progress(self, phase: str, n: float, total: float, track: str | None) -> None:
        if not self.enabled or not total:
            return
        pct = min(n / total * 100, 100.0)
        if self._last_pct.get(phase) == int(pct):
            return
        self._last_pct[phase] = int(pct)
        self.emit(phase=phase, pct=round(pct, 1), track=track)

    def step(self, label: str) -> None:
        """Announce a stretch of work that has no percentage of its own, so the
        GUI's status line doesn't sit on its last label while it runs. Resets
        the throttle so the next progress event replaces the step label at once."""
        self._last_pct.clear()
        self.emit(phase="step", label=label)


_progress_events = _ProgressEvents()

# Whether a cancelled run leaves resumable partial output behind. main() clears
# it for windowed runs (--start/--end/--test), which never write partials, so the cancel message in the
# __main__ block doesn't promise a resume that won't happen.
_saves_partials = True


def _raise_cancelled(signum: int, frame: object) -> None:
    raise TranscriptionCancelled()


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
    vad_min_speech_ms: int,
    initial_prompt: str,
    language: str | None,
    compression_ratio_threshold: float,
    logprob_threshold: float,
    best_of: int | None,
) -> None:
    partial_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": PARTIAL_SCHEMA_VERSION,
        "model_repo": model_repo,
        "vad_min_silence_ms": vad_min_silence_ms,
        "vad_min_speech_ms": vad_min_speech_ms,
        "initial_prompt_hash": hashlib.sha256(initial_prompt.encode()).hexdigest(),
        "language": language,
        "compression_ratio_threshold": compression_ratio_threshold,
        "logprob_threshold": logprob_threshold,
        "best_of": best_of,
    }
    (partial_dir / "_meta.json").write_text(json.dumps(meta))


def prune_partials(output_dir: Path, max_age_days: float, keep: str | None = None) -> list[str]:
    """Delete `.partial/<slug>/` dirs unused for at least `max_age_days`.

    Age comes from `_meta.json`'s mtime, since every full run rewrites it
    (falling back to the directory's own mtime if it's missing).
    `max_age_days=0` removes every dir. `keep` names a slug never to remove
    (the current run's). Returns the removed slugs.
    """
    root = output_dir / ".partial"
    if not root.is_dir():
        return []
    cutoff = time.time() - max_age_days * 86400
    removed = []
    for slug_dir in sorted(root.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name == keep:
            continue
        meta_path = slug_dir / "_meta.json"
        last_used = (meta_path if meta_path.exists() else slug_dir).stat().st_mtime
        if max_age_days == 0 or last_used < cutoff:
            shutil.rmtree(slug_dir)
            removed.append(slug_dir.name)
    return removed


def load_partial_meta(partial_dir: Path) -> dict | None:
    try:
        return json.loads((partial_dir / "_meta.json").read_text())
    except (json.JSONDecodeError, OSError):
        return None


def meta_mismatches(
    meta: dict | None,
    model_repo: str,
    vad_min_silence_ms: int,
    vad_min_speech_ms: int,
    initial_prompt: str,
    language: str | None,
    compression_ratio_threshold: float,
    logprob_threshold: float,
    best_of: int | None,
) -> list[str]:
    """Describe each way a saved `_meta.json` differs from this run's config.

    Returns one human-readable line per differing key (empty when it matches),
    so the caller can say exactly why the cache can't be reused.
    """
    if meta is None:
        return ["_meta.json is missing or unreadable"]
    if meta.get("schema_version") != PARTIAL_SCHEMA_VERSION:
        return [
            f"cache format: saved v{meta.get('schema_version')}, this version of "
            f"transcribe.py uses v{PARTIAL_SCHEMA_VERSION}"
        ]
    current = {
        "model_repo": model_repo,
        "vad_min_silence_ms": vad_min_silence_ms,
        "vad_min_speech_ms": vad_min_speech_ms,
        "language": language,
        "compression_ratio_threshold": compression_ratio_threshold,
        "logprob_threshold": logprob_threshold,
        "best_of": best_of,
    }
    diffs = [
        f"{key}: saved {meta.get(key)!r}, current {value!r}"
        for key, value in current.items()
        if meta.get(key) != value
    ]
    if meta.get("initial_prompt_hash") != hashlib.sha256(initial_prompt.encode()).hexdigest():
        diffs.append("initial prompt: vocabulary or enrichment fields changed since the cache was built")
    return diffs


Segment = tuple[float, float, str, float, float]


class TrackPartial(NamedTuple):
    """A track's saved progress: its segments so far and where to pick up.

    `regions_done` counts the VAD regions (pairs in the flat region list)
    already transcribed; it equals the track's region count once the track
    is complete. `last_region_text` is the last finished region's decoded
    text, which seeds the next region's rolling prompt on resume.
    """

    segments: list[Segment]
    language: str | None
    regions_done: int
    last_region_text: str


def write_track_partial(
    partial_dir: Path,
    username: str,
    audio_path: Path,
    progress: TrackPartial,
    speech_regions: list[float],
) -> None:
    """Save a track's progress, written after every region so a crash or
    cancel mid-track loses at most the region in flight.

    `speech_regions` is the track's full flat region list; the regions
    already done are saved with it so a resume can check that this run's
    VAD still produced the same leading regions. Written via a temp file
    and rename, so a crash mid-write never leaves a truncated file behind.
    """
    stat = audio_path.stat()
    data = {
        "audio_size": stat.st_size,
        "language": progress.language,
        "segments": progress.segments,
        "regions": speech_regions[: progress.regions_done * 2],
        "last_region_text": progress.last_region_text,
    }
    partial_path = partial_dir / f"{_safe_track_filename(username)}.json"
    tmp_path = partial_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data))
    tmp_path.replace(partial_path)


def load_track_partial(
    partial_dir: Path,
    username: str,
    audio_path: Path,
    speech_regions: list[float],
    progress: tqdm.tqdm | None = None,
) -> TrackPartial | None:
    """Load a track's saved progress, or None if there's nothing usable.

    A file without a `regions` key predates per-region checkpoints and was
    only ever written once the whole track finished, so it counts as
    complete. Otherwise the saved regions must be a prefix of this run's
    `speech_regions`, or the saved segments no longer line up and the track
    is retranscribed.
    """
    partial_path = partial_dir / f"{_safe_track_filename(username)}.json"
    try:
        data = json.loads(partial_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    def warn(reason: str) -> None:
        message = f"!! warning: partial for {username} is {reason}, retranscribing"
        progress.write(message) if progress is not None else print(message)

    stat = audio_path.stat()
    try:
        if data["audio_size"] != stat.st_size:
            warn("stale (audio file changed)")
            return None

        segments = [
            (start, end, text, compression_ratio, avg_logprob)
            for start, end, text, compression_ratio, avg_logprob in data["segments"]
        ]
        if "regions" not in data:
            return TrackPartial(segments, data["language"], len(speech_regions) // 2, "")
        saved_regions = [float(bound) for bound in data["regions"]]
        if len(saved_regions) % 2 or saved_regions != speech_regions[: len(saved_regions)]:
            warn("stale (speech regions changed)")
            return None
        return TrackPartial(
            segments, data["language"], len(saved_regions) // 2, str(data["last_region_text"])
        )
    except (KeyError, TypeError, ValueError):
        warn("malformed")
        return None


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
            if re.fullmatch(rf"\d+-{re.escape(username)}", path.stem):
                matched[username] = path
                break
        else:
            print(f"!! warning: no audio file found for {username}, skipping")

    return matched


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
    """Move `zip_path` into `<input_dir>/processed/`, creating it if needed.

    No-op when the zip is already there (a rerun of a processed zip, which
    reuses the kept segment cache)."""
    processed_dir = input_dir / "processed"
    if zip_path.resolve().parent == processed_dir.resolve():
        return zip_path
    processed_dir.mkdir(parents=True, exist_ok=True)
    new_path = processed_dir / zip_path.name
    shutil.move(str(zip_path), str(new_path))
    return new_path


def _decode_audio_mono16k(audio_path: Path) -> np.ndarray:
    """Load audio as a mono float32 array resampled to `VAD_SAMPLE_RATE`.

    Shared decode helper for both VAD and transcription (Whisper itself
    expects 16kHz mono input, same as VAD)."""
    # Shells out to ffmpeg rather than soundfile/torchaudio.
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
    return samples


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


def parse_shared_mic(value: str) -> tuple[str, float, float, str]:
    """Parse `USERNAME@START[-END]=NAME1,NAME2[,...]` from `--shared-mic`.

    Returns `(username, start_s, end_s, label)`, where `end_s` is infinite
    when END is omitted and `label` is the names joined with `/`.
    """
    if "@" not in value or "=" not in value:
        sys.exit(f"error: --shared-mic {value!r} must look like USERNAME@START[-END]=NAME1,NAME2")
    username, _, rest = value.partition("@")
    span, _, names_part = rest.partition("=")
    names = [name.strip() for name in names_part.split(",")]
    if not username.strip() or len(names) < 2 or not all(names):
        sys.exit(f"error: --shared-mic {value!r} needs a track name and at least two non-empty names")
    start_text, _, end_text = span.partition("-")
    try:
        start_s = _parse_hms(start_text)
        end_s = _parse_hms(end_text) if end_text else float("inf")
    except ValueError as e:
        sys.exit(f"error: invalid --shared-mic time in {value!r}: {e}")
    if end_s <= start_s:
        sys.exit(f"error: --shared-mic {value!r} ends before it starts")
    return username.strip(), start_s, end_s, "/".join(names)


def _window_stamp(seconds: float) -> str:
    """Format a window bound for a filename, e.g. `01h30m` or `01h30m15s`."""
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}h{minutes:02d}m" + (f"{secs:02d}s" if secs else "")


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


# 10ms frames at VAD_SAMPLE_RATE; dedup_bleed() converts seconds to frame
# indices as `int(seconds * 100)`.
ENVELOPE_FRAME_SAMPLES = 160


def compute_envelope(wav: np.ndarray) -> np.ndarray:
    """Linear RMS per 10ms frame of `wav`, as float32; trailing partial frame dropped."""
    frame_count = len(wav) // ENVELOPE_FRAME_SAMPLES
    frames = wav[: frame_count * ENVELOPE_FRAME_SAMPLES].reshape(frame_count, ENVELOPE_FRAME_SAMPLES)
    # einsum avoids materialising frames**2, which would double peak memory
    # on a multi-hour track.
    mean_square = np.einsum("ij,ij->i", frames, frames) / ENVELOPE_FRAME_SAMPLES
    return np.sqrt(mean_square).astype(np.float32)


def detect_speech_regions(
    audio_path: Path,
    min_silence_ms: int,
    min_speech_ms: int = 250,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[list[float], float, np.ndarray]:
    """Return (flat [start1, end1, start2, end2, ...] speech regions, raw duration, envelope).

    Regions and duration are in seconds; `envelope` is `compute_envelope()`
    of the same decoded audio, used by `dedup_bleed()`.

    `min_silence_ms` sets how long a pause must be before it splits speech
    into separate regions; the default (800ms) overrides Silero's own 100ms.

    `min_speech_ms` drops detected speech regions shorter than this (coughs,
    breath, mic pops) before they reach Whisper; defaults to Silero's own
    250ms default.
    """
    global _vad_model
    if _vad_model is None:
        _vad_model = silero_onnx.SileroVADSequence()

    wav = _decode_audio_mono16k(audio_path)
    timestamps = silero_onnx.get_speech_timestamps(
        wav,
        _vad_model,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        progress_callback=progress_callback,
    )

    regions: list[float] = []
    for entry in timestamps:
        regions.append(entry["start"])
        regions.append(entry["end"])
    duration = len(wav) / VAD_SAMPLE_RATE
    return regions, duration, compute_envelope(wav)


def speech_seconds(regions: list[float]) -> float:
    """Sum of region durations in a flat [start1, end1, start2, end2, ...] list."""
    return sum(end - start for start, end in zip(regions[::2], regions[1::2]))


def speech_frames(regions: list[float]) -> int:
    """Whisper mel-frame count Whisper will actually decode for these regions.

    Mirrors `transcribe_track`'s own sample-based slicing (seconds -> samples
    -> mel frames) so this estimate and mlx_whisper's reported frame count
    for each region agree exactly.
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
    """Replaces the `tqdm` name mlx_whisper.transcribe's module sees."""

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
    logprob_threshold: float = LOGPROB_THRESHOLD,
    best_of: int | None = None,
    cancelled: threading.Event | None = None,
    resume: TrackPartial | None = None,
    checkpoint: Callable[[TrackPartial], None] | None = None,
) -> tuple[list[Segment], str | None]:
    """Transcribe one audio track into ((segment_start_offset, segment_end_offset,
    text, compression_ratio, avg_logprob) tuples, detected language).

    Issues one mlx_whisper.transcribe() call per VAD speech region (not one
    batched call for the whole track): mlx_whisper's own `clip_timestamps`
    handling never resets its internal `seek` pointer between clips, so a
    single batched call decodes straight through the silence between
    regions. A single-clip-per-call shape sidesteps that entirely.

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
    after retrying means every retry failed.

    `logprob_threshold` is the same idea for a different failure mode: a
    segment whose average per-token log-probability falls below it is a
    low-confidence decode and mlx_whisper retries it the same way. This
    catches garbled/wrong-language hallucinations that a high compression
    ratio misses — made-up words are high-entropy text, so they don't trip
    the ratio check even though the model had no real signal for them (e.g.
    forcing `language="en"` sets the decode's language token but doesn't
    constrain which tokens can come out, so a low-confidence region can
    still decode as other-script gibberish). `merge_tracks` flags on this
    the same way, for the same reason.

    `best_of` only affects those same retries: mlx_whisper's first decode
    attempt is always greedy (temperature 0), and `best_of` is silently
    disabled at temperature 0, so it has no effect unless a segment already
    failed and is being retried at a higher temperature. When it does apply,
    mlx_whisper samples `best_of` independent decodes at that temperature and
    keeps the one it ranks highest, instead of a single resample — mlx_whisper
    doesn't implement beam search (`beam_size` raises `NotImplementedError`),
    so this is the available way to spend extra compute on a hard segment.

    Each region's raw result is passed through `collapse_duplicate_segments()`
    before it's used for anything, since mlx_whisper can hallucinate a loop
    as several whole segments with identical text rather than one segment
    with repeated words inside it — the latter is what `collapse_repetition()`
    (used downstream in `merge_tracks()`) catches; this catches the former.

    `resume`, a track's saved progress from `load_track_partial()`, skips the
    regions it already covers, carrying over its segments, detected language
    and rolling prompt; the caller advances `progress` past those regions.
    `checkpoint`, if given, is called with the track's progress after every
    region, so the caller can save it and a crash or cancel mid-track only
    loses the region in flight.
    """
    # verbose=None (not mlx_whisper's default False) suppresses its own prints,
    # which would otherwise corrupt the live progress bar.
    if not speech_regions:
        return [], None

    segments: list[Segment] = []
    prompt = initial_prompt
    regions_done = 0
    if resume is not None:
        segments = list(resume.segments)
        language = language or resume.language
        regions_done = resume.regions_done
        if regions_done:
            prompt = ((initial_prompt or "") + " " + _tail_words(resume.last_region_text)).strip()

    audio = _decode_audio_mono16k(audio_path)

    with _redirect_mlx_whisper_progress(progress):
        regions = list(zip(speech_regions[::2], speech_regions[1::2]))
        for start, end in regions[regions_done:]:
            if cancelled is not None and cancelled.is_set():
                raise TranscriptionCancelled()
            region_audio = audio[round(start * VAD_SAMPLE_RATE):round(end * VAD_SAMPLE_RATE)]
            result = mlx_whisper.transcribe(
                region_audio,
                path_or_hf_repo=str(model_path),
                verbose=None,
                initial_prompt=prompt,
                language=language,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                best_of=best_of,
            )
            if language is None:
                language = result.get("language")
            region_segments = collapse_duplicate_segments(result["segments"])
            region_text = "".join(segment["text"] for segment in region_segments)
            prompt = ((initial_prompt or "") + " " + _tail_words(region_text)).strip()
            segments.extend(
                (
                    segment["start"] + start,
                    segment["end"] + start,
                    segment["text"],
                    segment["compression_ratio"],
                    segment["avg_logprob"],
                )
                for segment in region_segments
            )
            regions_done += 1
            if checkpoint is not None:
                checkpoint(TrackPartial(segments, language, regions_done, region_text))
    return segments, language


def resolve_speaker(username: str, speaker_map: dict[str, str], warned: set[str]) -> str:
    """Map a Discord username to its configured display name.

    Warns at most once per username, since a missing speaker_map entry
    applies to every segment from that track.
    """
    if username in speaker_map:
        return speaker_map[username]
    if username not in warned:
        close = difflib.get_close_matches(username, speaker_map.keys(), n=1, cutoff=0.6)
        hint = f" (did you mean '{close[0]}'?)" if close else ""
        print(f"!! warning: no speaker_map entry for {username}, using raw Discord name{hint}")
        warned.add(username)
    return username


_TRAILING_PUNCTUATION = re.compile(r"[.,!?;:\s]+$")


def _normalize(text: str) -> str:
    """Lowercase and strip trailing punctuation/whitespace, for repeat/filler comparisons."""
    return _TRAILING_PUNCTUATION.sub("", text.lower())


_NON_PHRASE_CHARS = re.compile(r"[^a-z0-9'\- ]")


def _normalize_phrase(text: str) -> str:
    """Normalize `text` for `backchannel_phrases` matching: lowercase, keep
    only letters/digits/apostrophes/hyphens/spaces, collapse whitespace, then
    collapse immediately repeated words ("Yeah, yeah, yeah." -> "yeah").
    """
    words = _NON_PHRASE_CHARS.sub("", " ".join(text.lower().split())).split()
    deduped = [word for index, word in enumerate(words) if index == 0 or word != words[index - 1]]
    return " ".join(deduped)


# Genuine emphasis ("no no no") repeats a handful of times at most; a
# repetition-hallucination loop runs for dozens or hundreds of words. Set well above
# normal speech's repeat count so collapse_repetition() only fires on loops.
REPETITION_MIN_REPEATS = 6
REPETITION_MAX_PHRASE_WORDS = 6

# A hallucination loop can also come out as one run-on token with no spaces
# (e.g. a single character repeated hundreds of times, tokenized as one
# "word"). Only worth checking on words far longer than any real one, so
# ordinary long words/interjections are never touched.
REPETITION_MIN_WORD_LENGTH_FOR_CHAR_COLLAPSE = 20
REPETITION_MAX_CHAR_PERIOD = 4

# A hallucination loop can also come out as several whole mlx_whisper
# segments in a row with identical text, from one region's decode call
# (e.g. eight separate "! I have." segments) — collapse_repetition() can't
# see this since it only looks within one segment's own text. Unlike
# REPETITION_MIN_REPEATS (6, tuned above normal speech's within-segment
# repeat count), there's no legitimate reason for mlx_whisper to emit two
# whole segments with verbatim-identical text back to back in one call, so
# this threshold is much lower.
SEGMENT_REPETITION_MIN_REPEATS = 2


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


def collapse_duplicate_segments(
    segments: list[dict], min_repeats: int = SEGMENT_REPETITION_MIN_REPEATS
) -> list[dict]:
    """Collapse immediately-adjacent mlx_whisper segments with identical text.

    Unlike `_collapse_runs()` (used elsewhere for words/chars), this keeps
    the *run's* end time, not just its first segment's — `merge_tracks()`
    merges on `next.start - prev.end`, so a collapsed segment reporting only
    its first duplicate's (earlier, shorter) end would understate the run's
    true duration and could wrongly suppress a legitimate merge with
    whatever comes after. Comparison key is `_normalize()`d text. Called
    once per VAD region's `mlx_whisper.transcribe()` result, before that
    result's segments are used for anything else (the rolling
    `initial_prompt` or the caller's segment list) — see `transcribe_track()`.
    """
    collapsed: list[dict] = []
    i = 0
    n = len(segments)
    while i < n:
        key = _normalize(segments[i]["text"])
        j = i + 1
        while j < n and _normalize(segments[j]["text"]) == key:
            j += 1
        run = segments[i:j]
        if len(run) >= min_repeats:
            kept = dict(run[0])
            kept["end"] = run[-1]["end"]
            collapsed.append(kept)
        else:
            collapsed.extend(run)
        i = j
    return collapsed


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


# A Unicode replacement character means mlx_whisper's decode produced a byte
# sequence that isn't even valid text in the language it guessed — never a
# legitimate transcription.
_REPLACEMENT_CHAR = "�"

# Script families distinct enough that seeing two-plus of them in one short
# segment (alongside Latin, which speaker names/loanwords/URLs can
# legitimately mix in) is a strong wrong-language hallucination signal, not
# real code-switching — real code-switching stays within one language's
# script(s) for a whole utterance, not several unrelated ones in one line.
# CJK/HIRAGANA/KATAKANA are one family, not three: a single real Japanese
# sentence routinely mixes kanji and kana.
_SCRIPT_FAMILIES = {
    "CYRILLIC": "CYRILLIC",
    "CJK": "CJK",
    "HIRAGANA": "CJK",
    "KATAKANA": "CJK",
    "HANGUL": "HANGUL",
    "ARABIC": "ARABIC",
    "HEBREW": "HEBREW",
    "GREEK": "GREEK",
    "THAI": "THAI",
    "DEVANAGARI": "DEVANAGARI",
}


def _script_of(char: str) -> str | None:
    """Return a coarse non-Latin script family name for `char`, or None if it's
    Latin, a digit, punctuation, or whitespace (all of which are ignored)."""
    try:
        name = unicodedata.name(char)
    except ValueError:
        return None
    for marker, family in _SCRIPT_FAMILIES.items():
        if marker in name:
            return family
    return None


def looks_garbled(text: str) -> bool:
    """Return True if `text` contains a Unicode replacement character, a mix
    of two or more unrelated non-Latin scripts (e.g. Cyrillic and CJK in the
    same line), or a single word mixing Latin letters with a non-Latin
    script ("abra제") — all signal a wrong-language/garbled decode that
    compression_ratio and avg_logprob can miss, since a short, varied
    hallucinated line doesn't always compress poorly or read as
    low-confidence to Whisper itself. A real name or loanword in another
    script stands as its own word, so one non-Latin script alongside Latin
    words is still allowed.
    """
    if _REPLACEMENT_CHAR in text:
        return True
    scripts = {script for script in (_script_of(char) for char in text) if script is not None}
    if len(scripts) > 1:
        return True
    return bool(scripts) and any(
        any(_script_of(char) for char in word) and any(char.isascii() and char.isalpha() for char in word)
        for word in text.split()
    )


# Speaker/text of the single line a multi-speaker laughter run collapses to;
# keeps the `[ts] Speaker: text` grammar downstream parsers rely on.
LAUGHTER_SPEAKER = "Table"
LAUGHTER_MARKER = "[laughter]"

# `backchannel_phrases` default: whole-line reactions that add nothing when
# said over another speaker (see merge_tracks()).
DEFAULT_BACKCHANNEL_PHRASES = [
    "yeah", "yep", "yup", "yes", "okay", "ok", "alright", "all right", "right", "nice", "cool",
    "sure", "wow", "ooh", "oh", "oh yeah", "oh okay", "oh nice", "oh wow", "oh no", "oh my god",
    "hell yeah", "uh-huh", "mm-hmm", "mhm", "true", "gotcha", "got it", "damn",
]

_LAUGH_ANNOTATION = re.compile(
    r"^[(\[*]?\s*(?:laugh(?:s|ing|ter)?|chuckles?|giggles?)\s*[)\]*]?[.!]*$", re.IGNORECASE
)
_LAUGH_WORD = re.compile(r"^(?:a?h+[ae]+)+h*$")
_LAUGH_ABBREVIATIONS = {"lol", "lmao", "lmfao"}
_NON_LETTERS = re.compile(r"[^a-z]")


def is_laughter(text: str) -> bool:
    """Return True if `text` is laughter and nothing else: a laugh annotation
    ("(laughs)", "[Laughter]", "*chuckles*") or a line whose every word is a
    laugh ("Ha ha ha.", "hehehehe", "lol"). Matched by pattern rather than
    via `filler_words`, so `merge_tracks()` can find multi-speaker laughter
    runs before anything drops them.
    """
    stripped = text.strip()
    if _LAUGH_ANNOTATION.match(stripped):
        return True
    words = [_NON_LETTERS.sub("", word) for word in stripped.lower().split()]
    words = [word for word in words if word]
    if not words:
        return False
    return all(_LAUGH_WORD.match(word) or word in _LAUGH_ABBREVIATIONS for word in words)


# 10ms envelope frames per second, see ENVELOPE_FRAME_SAMPLES.
_ENVELOPE_FRAMES_PER_SECOND = 100
# A pair's span is widened to at least this long so the level/correlation
# checks have enough frames to mean anything on a one-word segment.
BLEED_MIN_SPAN_S = 1.0
# The quieter track counts as "active" (carrying sound, not silence) over a
# span if at least this fraction of its frames are above BLEED_ACTIVE_DB.
BLEED_ACTIVE_DB = -60.0
BLEED_ACTIVE_FRACTION = 0.2
# Envelope correlation is the max over lags of +/- this many frames
# (150ms), covering the 50-110ms network offset measured between tracks.
BLEED_MAX_LAG_FRAMES = 15
# The shorter text needs this many words before word containment counts
# towards text similarity, so "Okay." never "contains" a longer line.
BLEED_MIN_CONTAINMENT_WORDS = 3


def _bleed_text_similarity(text_a: str, text_b: str) -> float:
    """Max of SequenceMatcher ratio and word containment of the normalized texts.

    Returns 0 when either text normalizes to empty."""
    normalized_a, normalized_b = _normalize(text_a), _normalize(text_b)
    if not normalized_a or not normalized_b:
        return 0.0
    score = difflib.SequenceMatcher(None, normalized_a, normalized_b).ratio()
    words_a = {_normalize(word) for word in text_a.split()} - {""}
    words_b = {_normalize(word) for word in text_b.split()} - {""}
    shorter = min(len(words_a), len(words_b))
    if shorter >= BLEED_MIN_CONTAINMENT_WORDS:
        score = max(score, len(words_a & words_b) / shorter)
    return score


def _envelope_db(rms: np.ndarray) -> np.ndarray:
    return 20 * np.log10(rms.astype(np.float64) + 1e-10)


def _max_lagged_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    """Max Pearson correlation of `x` and `y` over lags -max_lag..+max_lag.

    A lag whose overlap is under half the span, or where either slice has
    zero variance, contributes 0."""
    n = len(x)
    best = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = x[lag:], y[: n - lag]
        else:
            a, b = x[: n + lag], y[-lag:]
        if len(a) < max(2, n // 2):
            continue
        if a.std() == 0 or b.std() == 0:
            continue
        best = max(best, float(np.corrcoef(a, b)[0, 1]))
    return best


def dedup_bleed(
    transcripts: dict[str, list[tuple[float, float, str, float, float]]],
    envelopes: dict[str, np.ndarray],
    window_start: float,
    speaker_names: dict[str, str],
    level_gap_db: float,
    min_correlation: float,
    text_similarity: float,
    max_offset_s: float,
    show_markers: bool,
    verbose: bool,
) -> tuple[dict[str, list[tuple[float, float, str, float, float]]], dict[str, int]]:
    """Drop cross-track duplicate segments caused by co-located mic bleed.

    A pair of segments on different tracks, starting within `max_offset_s`
    with matching text, is bleed only if the audio agrees: the quieter
    track is active over the span (not silent — a silent quieter track
    means someone repeated the line), its loudness envelope correlates with
    the louder track's, and the level gap is large enough. Bleed drops the
    quieter segment; a matching pair with an active quieter track that
    misses the gap/correlation thresholds is kept and counted as ambiguous,
    optionally marked.

    `envelopes` are `compute_envelope()` output per track, with index 0 at
    `window_start`. Runs before `merge_tracks()`, while segments still carry
    their own start/end times.
    """
    stats = {"bleed_deduped": 0, "bleed_ambiguous": 0}
    flat = sorted(
        (segment[0], username, index)
        for username, segments in transcripts.items()
        for index, segment in enumerate(segments)
    )
    dropped: set[tuple[str, int]] = set()
    markers: dict[tuple[str, int], str] = {}

    def frame(seconds: float) -> int:
        return int((seconds - window_start) * _ENVELOPE_FRAMES_PER_SECOND)

    for i, (start_a, user_a, index_a) in enumerate(flat):
        for start_b, user_b, index_b in flat[i + 1 :]:
            if start_b - start_a > max_offset_s:
                break
            if (user_a, index_a) in dropped:
                break
            if user_b == user_a or (user_b, index_b) in dropped:
                continue
            segment_a = transcripts[user_a][index_a]
            segment_b = transcripts[user_b][index_b]
            if _bleed_text_similarity(segment_a[2], segment_b[2]) < text_similarity:
                continue

            span_start = min(segment_a[0], segment_b[0])
            span_end = max(segment_a[1], segment_b[1], span_start + BLEED_MIN_SPAN_S)
            first, last = max(0, frame(span_start)), frame(span_end)
            if user_a not in envelopes or user_b not in envelopes:
                continue
            rms_a = envelopes[user_a][first:last]
            rms_b = envelopes[user_b][first:last]
            length = min(len(rms_a), len(rms_b))
            if length == 0:
                continue
            rms_a, rms_b = rms_a[:length], rms_b[:length]

            level_a = 20 * np.log10(np.sqrt(np.mean(rms_a.astype(np.float64) ** 2)) + 1e-10)
            level_b = 20 * np.log10(np.sqrt(np.mean(rms_b.astype(np.float64) ** 2)) + 1e-10)
            if level_a >= level_b:
                louder, quieter, quieter_index, quieter_rms = user_a, user_b, index_b, rms_b
            else:
                louder, quieter, quieter_index, quieter_rms = user_b, user_a, index_a, rms_a
            gap = abs(level_a - level_b)

            if np.mean(_envelope_db(quieter_rms) > BLEED_ACTIVE_DB) < BLEED_ACTIVE_FRACTION:
                continue

            correlation = _max_lagged_correlation(
                _envelope_db(rms_a), _envelope_db(rms_b), BLEED_MAX_LAG_FRAMES
            )
            if gap >= level_gap_db and correlation >= min_correlation:
                dropped.add((quieter, quieter_index))
                markers.pop((quieter, quieter_index), None)
                stats["bleed_deduped"] += 1
                decision = f"dropped {quieter}"
            else:
                markers.setdefault((quieter, quieter_index), speaker_names.get(louder, louder))
                stats["bleed_ambiguous"] += 1
                decision = "ambiguous"
            if verbose:
                hours, rest = divmod(int(span_start), 3600)
                print(
                    f"  bleed? {hours:02d}:{rest // 60:02d}:{rest % 60:02d} "
                    f"{user_a}/{user_b} gap={gap:.1f}dB corr={correlation:.2f} -> {decision}"
                )

    deduped: dict[str, list[tuple[float, float, str, float, float]]] = {}
    for username, segments in transcripts.items():
        kept = []
        for index, segment in enumerate(segments):
            if (username, index) in dropped:
                continue
            louder_name = markers.get((username, index))
            if show_markers and louder_name is not None:
                start, end, text, compression_ratio, avg_logprob = segment
                segment = (
                    start, end, f"[possible bleed: {louder_name}] {text}", compression_ratio, avg_logprob
                )
            kept.append(segment)
        deduped[username] = kept
    return deduped, stats


def merge_tracks(
    timestamp_base: datetime,
    transcripts: dict[str, list[tuple[float, float, str, float, float]]],
    merge_gap_seconds: float,
    filler_words: list[str],
    compression_ratio_threshold: float = COMPRESSION_RATIO_THRESHOLD,
    logprob_threshold: float = LOGPROB_THRESHOLD,
    include_inaudible_placeholders: bool = False,
    hallucination_patterns: list[re.Pattern] | None = None,
    shared_spans: dict[str, list[tuple[float, float, str]]] | None = None,
    backchannel_phrases: list[str] | None = None,
    collapse_laughter: bool = True,
    laughter_run_gap_seconds: float = 5.0,
) -> tuple[
    list[tuple[datetime, str, str]],
    list[tuple[datetime, str, str]],
    dict[str, int],
    set[str],
    list[tuple[datetime, str, str]],
]:
    """Interleave per-track segments into one chronological transcript.

    Consecutive segments from the same speaker (per-track, ignoring what
    other speakers' tracks were doing at the same time) are merged into one
    line when the silence between them (`next.start - prev.end`) is at most
    `merge_gap_seconds`.

    A segment is dropped entirely if its text, normalized (lowercased,
    trailing punctuation stripped), is empty — this catches not just a
    literal empty string but punctuation-only noise like "." or "..." that
    Whisper emits for near-silence — or exactly matches an entry in
    `filler_words` (whole-line match only), or matches the previous surviving
    segment from the same speaker with the same flagged/unflagged status
    (repeat dedup, also normalized so "Okay."/"okay"/"Okay," collapse). The
    original, non-normalized text is kept for whichever occurrence is
    emitted.

    Segments whose compression ratio is still above
    `compression_ratio_threshold`, or whose average log-probability is still
    below `logprob_threshold`, after mlx_whisper's own temperature fallback
    are flagged with a "!! " prefix rather than silently emitted — either
    means mlx_whisper itself treated the decode as a failure (repetition or
    low-confidence, respectively) and had nothing better to fall back to.
    These must be the same thresholds passed to `transcribe_track`'s
    `compression_ratio_threshold`/`logprob_threshold`, since a segment's
    ratio and log-probability are only meaningful relative to the thresholds
    that drove its own retries. A segment is also flagged, regardless of
    either threshold, when `looks_garbled()` finds a Unicode replacement
    character, a mix of unrelated non-Latin scripts, or a word mixing Latin
    with a non-Latin script — all catch
    wrong-language decodes that stay short/varied enough to read as
    high-confidence to mlx_whisper itself.

    Before anything else, every `hallucination_patterns` match (Whisper's
    stock silence phrases, e.g. "Thanks for watching!") is removed from the
    segment's text as a substring, so a phrase never gets glued onto real
    speech by the same-speaker merge. A segment left with nothing word-like
    is dropped; either way it counts toward `hallucination_stripped`, and
    the removed text is not reported in `flagged_lines` — it's known junk.

    Every segment's text is then passed through `collapse_repetition()`
    ahead of flagging and merging, regardless of its compression ratio — a
    repetition-hallucination loop can be diluted below the ratio threshold
    if it's only part of a longer segment. The collapsed text is what gets
    flagged/merged downstream, but a collapse alone doesn't flag the
    segment: genuine speech (emphatic "no, no, no...", onomatopoeia,
    laughter) repeats past the run threshold with clean ratio/logprob, while
    a real loop also trips one of the two checks above.

    A flagged segment (either way) is exempt from filler-word dropping, and
    is never merged with a neighbor in either direction — it's always
    emitted as its own line for human review. It's still subject to repeat
    dedup against an immediately preceding *also-flagged* segment with the
    same text, though: `transcribe_track()`'s `collapse_duplicate_segments()`
    already collapses this within a single region's decode call, but a
    hallucination loop split across region or track-merge boundaries could
    still produce several adjacent identical flagged segments here, and
    there's no legitimate reason to review the same flagged line twice. An
    unflagged segment never dedups against a flagged one (or vice versa)
    even with identical text, so a distinct flagged line is never silently
    absorbed into unrelated surrounding speech.

    A flagged line's original garbled text is always returned in
    `flagged_lines` (same shape as `merged`) for the caller to write to a
    companion errors file. Whether it also gets a `!! [inaudible]`
    placeholder line in `merged` is controlled by
    `include_inaudible_placeholders`: by default (False) a flagged line is
    dropped from `merged` entirely, so the final transcript reads as
    continuous speech with review-worthy gaps only visible in the errors
    file; set True to keep an inline marker at each flagged line's position.
    Either way, flagged text never merges into a neighboring buffered line,
    and repeat dedup against an adjacent flagged segment (above) always
    compares the original text, not the placeholder.

    `shared_spans` maps a username to `(start, end, label)` spans where that
    track carries several players (`--shared-mic`). A segment whose start
    falls in `[start, end)` is emitted under `label` instead of the
    username, flagged or not, and always as its own line: it never merges
    with a neighbor in either direction, so each line can still be
    attributed to one player by hand. Every label used is returned in
    `shared_labels`, so the caller can skip `speaker_map` for it.

    An unflagged segment whose `_normalize_phrase()` text is in
    `backchannel_phrases` ("yeah", "uh-huh", ...) is dropped as crosstalk
    when its midpoint falls inside a segment on another track that is itself
    real speech (not a backchannel or laughter), unless it ends in "?". A
    backchannel nobody else is talking over is kept. An empty or None list
    disables this rule.

    With `collapse_laughter`, every unflagged `is_laughter()` segment is
    pulled out of the per-track flow. Laughs from 2+ distinct tracks, each
    starting within `laughter_run_gap_seconds` of the latest end so far, are
    collapsed into one `LAUGHTER_SPEAKER: LAUGHTER_MARKER` line at the run's
    first start (`LAUGHTER_SPEAKER` is returned in `shared_labels`); any other
    laugh is dropped.

    Every line removed by filler, crosstalk or laughter is returned in
    `pruned_lines` (same shape as `merged`, text prefixed with `(filler)`,
    `(crosstalk)`, `(laughter)` or `(laughter-run)`) for the caller to write
    to a companion audit file.

    Also returns a stats dict counting what was pruned along the way
    (`segments_total`, `empty_dropped`, `hallucination_stripped`,
    `repetition_collapsed`, `filler_dropped`, `crosstalk_dropped`,
    `laughter_dropped`, `laughter_runs_collapsed`, `repeat_deduped`,
    `flagged_for_review`), for the caller to report and to record in the
    transcript's frontmatter.
    """
    merged: list[tuple[datetime, str, str]] = []
    flagged_lines: list[tuple[datetime, str, str]] = []
    pruned_lines: list[tuple[datetime, str, str]] = []
    shared_labels: set[str] = set()
    normalized_filler_words = {_normalize(word) for word in filler_words}
    normalized_backchannels = {_normalize_phrase(phrase) for phrase in backchannel_phrases or []}
    normalized_backchannels.discard("")
    stats = {
        "segments_total": 0,
        "empty_dropped": 0,
        "hallucination_stripped": 0,
        "repetition_collapsed": 0,
        "filler_dropped": 0,
        "crosstalk_dropped": 0,
        "laughter_dropped": 0,
        "laughter_runs_collapsed": 0,
        "repeat_deduped": 0,
        "flagged_for_review": 0,
    }

    # Per-track spans of real speech that a backchannel on another track can
    # be "said over". Taken before the same-speaker merge, which loses end times.
    host_intervals: dict[str, list[tuple[float, float]]] = {}
    if normalized_backchannels:
        for username, segments in transcripts.items():
            host_intervals[username] = [
                (start, end)
                for start, end, text, _, _ in segments
                if _normalize_phrase(text) not in normalized_backchannels
                and _normalize_phrase(text) != ""
                and not is_laughter(text)
            ]
    laugh_segments: list[tuple[float, float, str, str]] = []

    for username, segments in transcripts.items():
        spans = (shared_spans or {}).get(username, [])
        previous_normalized: str | None = None
        previous_flagged = False
        buffer_start: float | None = None
        buffer_end: float | None = None
        buffer_text: str | None = None

        for start, end, text, compression_ratio, avg_logprob in segments:
            stats["segments_total"] += 1
            stripped = text.strip()
            if hallucination_patterns:
                cleaned, hits = strip_hallucination_phrases(stripped, hallucination_patterns)
                if hits:
                    stats["hallucination_stripped"] += 1
                    if _normalize(cleaned) == "":
                        continue
                    stripped = cleaned
            if _normalize(stripped) == "":
                stats["empty_dropped"] += 1
                continue
            collapsed_text = collapse_repetition(stripped)
            # collapse_repetition() re-joins words with single spaces, so compare
            # against the same whitespace-normalized text; otherwise a stray
            # double space would count as a repetition collapse and get flagged.
            repetition_collapsed = collapsed_text != " ".join(stripped.split())
            if repetition_collapsed:
                stats["repetition_collapsed"] += 1
            stripped = collapsed_text
            # A collapse alone doesn't flag: a real loop also trips the ratio or
            # logprob check, and genuine speech ("no, no, no...", onomatopoeia,
            # laughter) can repeat past REPETITION_MIN_REPEATS with clean scores.
            flagged = (
                compression_ratio > compression_ratio_threshold
                or avg_logprob < logprob_threshold
                or looks_garbled(stripped)
            )
            if collapse_laughter and not flagged and is_laughter(stripped):
                laugh_segments.append((start, end, username, stripped))
                continue
            if (
                not flagged
                and not stripped.endswith("?")
                and _normalize_phrase(stripped) in normalized_backchannels
            ):
                midpoint = (start + end) / 2
                if any(
                    host_start <= midpoint <= host_end
                    for other, intervals in host_intervals.items()
                    if other != username
                    for host_start, host_end in intervals
                ):
                    stats["crosstalk_dropped"] += 1
                    timestamp = timestamp_base + timedelta(seconds=start)
                    pruned_lines.append((timestamp, username, f"(crosstalk) {stripped}"))
                    continue
            normalized = _normalize(stripped)
            if not flagged and normalized in normalized_filler_words:
                stats["filler_dropped"] += 1
                timestamp = timestamp_base + timedelta(seconds=start)
                pruned_lines.append((timestamp, username, f"(filler) {stripped}"))
                continue
            if normalized == previous_normalized and flagged == previous_flagged:
                stats["repeat_deduped"] += 1
                continue
            previous_normalized = normalized
            previous_flagged = flagged
            shared_label = next(
                (label for span_start, span_end, label in spans if span_start <= start < span_end),
                None,
            )
            speaker = shared_label or username
            if shared_label is not None:
                shared_labels.add(shared_label)

            if flagged or shared_label is not None:
                if buffer_text is not None:
                    timestamp = timestamp_base + timedelta(seconds=buffer_start)
                    merged.append((timestamp, username, buffer_text))
                    buffer_start = buffer_end = buffer_text = None
                timestamp = timestamp_base + timedelta(seconds=start)
                if not flagged:
                    merged.append((timestamp, speaker, stripped))
                    continue
                stats["flagged_for_review"] += 1
                flagged_lines.append((timestamp, speaker, stripped))
                if include_inaudible_placeholders:
                    merged.append((timestamp, speaker, "!! [inaudible]"))
                continue

            if buffer_text is not None and start - buffer_end <= merge_gap_seconds:
                buffer_text = f"{buffer_text} {stripped}"
                buffer_end = end
            else:
                if buffer_text is not None:
                    timestamp = timestamp_base + timedelta(seconds=buffer_start)
                    merged.append((timestamp, username, buffer_text))
                buffer_start, buffer_end, buffer_text = start, end, stripped

        if buffer_text is not None:
            timestamp = timestamp_base + timedelta(seconds=buffer_start)
            merged.append((timestamp, username, buffer_text))

    laugh_segments.sort(key=lambda entry: entry[0])
    clusters: list[list[tuple[float, float, str, str]]] = []
    cluster_end = 0.0
    for laugh in laugh_segments:
        if clusters and laugh[0] <= cluster_end + laughter_run_gap_seconds:
            clusters[-1].append(laugh)
            cluster_end = max(cluster_end, laugh[1])
        else:
            clusters.append([laugh])
            cluster_end = laugh[1]
    for cluster in clusters:
        is_run = len({username for _, _, username, _ in cluster}) >= 2
        if is_run:
            merged.append(
                (timestamp_base + timedelta(seconds=cluster[0][0]), LAUGHTER_SPEAKER, LAUGHTER_MARKER)
            )
            shared_labels.add(LAUGHTER_SPEAKER)
            stats["laughter_runs_collapsed"] += 1
        reason = "laughter-run" if is_run else "laughter"
        for start, _, username, text in cluster:
            stats["laughter_dropped"] += 1
            pruned_lines.append((timestamp_base + timedelta(seconds=start), username, f"({reason}) {text}"))

    merged.sort(key=lambda entry: entry[0])
    flagged_lines.sort(key=lambda entry: entry[0])
    pruned_lines.sort(key=lambda entry: entry[0])
    return merged, flagged_lines, stats, shared_labels, pruned_lines


def _format_lines(
    entries: list[tuple[datetime, str, str]], timestamp_format: str, line_format: str
) -> list[str]:
    try:
        return [
            line_format.format(timestamp=timestamp.strftime(timestamp_format), speaker=name, text=text)
            for timestamp, name, text in entries
        ]
    except KeyError as e:
        raise ValueError(
            f"config `line_format` uses unknown placeholder {e}; "
            "only {timestamp}, {speaker}, {text} are supported"
        ) from e


def write_transcript(
    merged: list[tuple[datetime, str, str]],
    flagged_lines: list[tuple[datetime, str, str]],
    pruned_lines: list[tuple[datetime, str, str]],
    session_title: str,
    output_dir: Path,
    timestamp_format: str,
    line_format: str,
    merge_stats: dict[str, int],
    session_start: datetime,
    filename_suffix: str = "",
    output_name: str | None = None,
) -> Path:
    """Write the merged transcript as a Markdown file with YAML frontmatter.

    The frontmatter `date` comes from `session_start` (the Craig manifest's
    start time), not from the line timestamps: in elapsed mode those are
    offsets from a placeholder epoch, so their date is meaningless.

    `line_format` is a str.format() template with `{timestamp}`, `{speaker}`,
    and `{text}` placeholders, letting the config control per-line Markdown
    (e.g. `**{speaker}**` to bold speaker names).

    `merge_stats` (as returned by `merge_tracks()`) is recorded in the
    frontmatter under `pruning`, so what was dropped/collapsed/flagged stays
    attached to the transcript file itself, not just the console output.

    `flagged_lines` holds the original (pre-placeholder) text of each
    flagged line in `merged`. When non-empty, it's written to a companion
    `<slug>{filename_suffix}.errors.md` file next to the main transcript,
    using the same `timestamp_format`/`line_format`, with no YAML
    frontmatter — a disposable side file, not a document. When empty, any
    stale `.errors.md` from a previous run of the same slug is removed.

    `pruned_lines` (deliberate filler/crosstalk/laughter drops, see
    `merge_tracks()`) goes to `<slug>{filename_suffix}.pruned.md` the same
    way, kept apart from `.errors.md` so review-worthy lines aren't buried.

    `output_name`, if given, overrides the filename slug (via `--output-name`)
    without affecting `session_title`, which is still recorded as-is in the
    frontmatter `title` field.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(output_name) if output_name else _slugify(session_title)
    output_path = output_dir / f"{slug}{filename_suffix}.md"
    errors_path = output_dir / f"{slug}{filename_suffix}.errors.md"
    pruned_path = output_dir / f"{slug}{filename_suffix}.pruned.md"

    speakers = list(dict.fromkeys(name for _, name, _ in merged))
    date = session_start.date().isoformat()

    frontmatter = yaml.safe_dump(
        {"date": date, "title": session_title, "speakers": speakers, "pruning": merge_stats},
        sort_keys=False,
    )

    lines = _format_lines(merged, timestamp_format, line_format)
    output_path.write_text(f"---\n{frontmatter}---\n\n" + "\n".join(lines) + "\n")

    if flagged_lines:
        error_lines = _format_lines(flagged_lines, timestamp_format, line_format)
        errors_path.write_text("\n".join(error_lines) + "\n")
    elif errors_path.exists():
        errors_path.unlink()

    if pruned_lines:
        pruned_text_lines = _format_lines(pruned_lines, timestamp_format, line_format)
        pruned_path.write_text("\n".join(pruned_text_lines) + "\n")
    elif pruned_path.exists():
        pruned_path.unlink()

    return output_path


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _rule(char: str = "=") -> str:
    width = min(shutil.get_terminal_size((80, 20)).columns, 80)
    return char * width


def _config_number(config: dict, key: str, default: float) -> float:
    """Read a numeric config key, exiting with a clear error if it isn't a number."""
    value = config.get(key, default)
    # bool is an int subclass, but `merge_gap_ms: true` is a typo, not 1.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        sys.exit(f"error: config `{key}` must be a number, got {value!r}")
    return value


def main() -> None:
    start_time = time.monotonic()

    # mlx_whisper's temperature-retry decodes (anything above the initial
    # greedy temperature-0 attempt) sample via mx.random.categorical, and
    # mlx_whisper never seeds MLX's global RNG itself — left alone, every
    # run draws different retry tokens, so a segment that needed a retry
    # (often the exact segments worth scrutinizing) reads differently from
    # one run to the next on identical audio. Seeding here doesn't make
    # decoding fully bit-identical across runs (GPU float reduction order
    # can still vary), but it removes this specific, confirmed source of
    # run-to-run variance.
    mx.random.seed(0)

    print(_rule())
    print(identity.APP_NAME.center(len(_rule())))
    print(_rule())
    print()

    legacy = check_legacy_layout()
    if legacy:
        print(*legacy, sep="\n")
        sys.exit(1)

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
        "--config",
        type=Path,
        default=paths.DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: settings/config.yaml)",
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
        "--logprob-threshold",
        type=float,
        default=None,
        help="Average per-token log-probability below which mlx_whisper retries a "
        "decode at a higher temperature, and below which a segment still failing "
        "after retries is flagged for review. Catches low-confidence garbled/"
        "wrong-language decodes that compression-ratio-threshold misses. Higher "
        "(closer to 0) catches more. Defaults to -1.0 if omitted. Overrides the "
        "config `logprob_threshold` key.",
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
        "--output-name",
        help="Override the transcript filename stem (e.g. `--output-name foo` "
        "writes foo.md and, if any lines are flagged, foo.errors.md), without "
        "changing the session `title` recorded in the frontmatter. Defaults to "
        "a slugified version of the session title.",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=None,
        help="Path to a vocabulary file. Overrides --campaign and the config "
        "`campaign` key.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Use settings/vocabulary/<CAMPAIGN>.md as the vocabulary. "
        "Overrides the config `campaign` key.",
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
    inaudible_group = parser.add_mutually_exclusive_group()
    inaudible_group.add_argument(
        "--show-inaudible",
        dest="show_inaudible",
        action="store_true",
        default=None,
        help="Keep an inline '!! [inaudible]' placeholder in the main transcript "
        "for each flagged/low-confidence line, instead of omitting it (the "
        "original garbled text is always written to the companion .errors.md "
        "file either way). Overrides the config `show_inaudible_placeholders` "
        "key, which defaults to false.",
    )
    inaudible_group.add_argument(
        "--hide-inaudible",
        dest="show_inaudible",
        action="store_false",
        default=None,
        help="Omit flagged/low-confidence lines from the main transcript (they "
        "still go to the companion .errors.md file). Overrides the config "
        "`show_inaudible_placeholders` key; only needed when that is set to true.",
    )
    bleed_marker_group = parser.add_mutually_exclusive_group()
    bleed_marker_group.add_argument(
        "--show-bleed-markers",
        dest="show_bleed_markers",
        action="store_true",
        default=None,
        help="Prefix the quieter copy of an ambiguous possible mic-bleed pair "
        "with '[possible bleed: <louder speaker>]' for manual review (both "
        "copies are kept either way). Overrides the config `show_bleed_markers` "
        "key, which defaults to false.",
    )
    bleed_marker_group.add_argument(
        "--hide-bleed-markers",
        dest="show_bleed_markers",
        action="store_false",
        default=None,
        help="Don't mark ambiguous possible mic-bleed pairs. Overrides the config "
        "`show_bleed_markers` key; only needed when that is set to true.",
    )
    timestamp_mode_group = parser.add_mutually_exclusive_group()
    timestamp_mode_group.add_argument(
        "--elapsed-timestamps",
        dest="elapsed_timestamps",
        action="store_true",
        default=None,
        help="Stamp each line with time elapsed since the start of the "
        "recording (00:00:00 at the first segment) instead of the real "
        "wall-clock time it was spoken. Overrides the config `timestamp_mode` "
        "key, which defaults to 'elapsed'. This is the default, so the flag "
        "only matters to override a config set to 'clock'.",
    )
    timestamp_mode_group.add_argument(
        "--clock-timestamps",
        dest="elapsed_timestamps",
        action="store_false",
        default=None,
        help="Stamp each line with the real wall-clock time it was spoken, "
        "instead of time elapsed since the start of the recording. Overrides "
        "the config `timestamp_mode` key, which defaults to 'elapsed'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per possible mic-bleed pair (time, tracks, level "
        "gap, envelope correlation, decision), for tuning the `bleed_*` config "
        "thresholds.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any saved partial output for this session and retranscribe "
        "every track from scratch.",
    )
    parser.add_argument(
        "--shared-mic",
        action="append",
        default=None,
        metavar="USER@START[-END]=NAME,NAME",
        help="Declare that a track carries several players from START (to END, "
        "or the end of the recording) in recording time, e.g. "
        "bob42@01:44:00=Bob,Carol. Lines in the span are labelled "
        "Bob/Carol and never merged. Repeatable.",
    )
    parser.add_argument(
        "--prune-cache",
        nargs="?",
        type=float,
        const=0,
        default=None,
        metavar="DAYS",
        help="Delete cached per-track segments under <output_dir>/.partial/ "
        "and exit without transcribing. With DAYS, only delete caches unused "
        "for at least that many days; without it, delete all of them.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="HH:MM:SS",
        help="Only transcribe from this point in the recording (default: the "
        "beginning). A windowed run (--start and/or --end) writes to "
        "<slug>-<start>-<end>.md, and never touches partial output or the "
        "input zip.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        metavar="HH:MM:SS",
        help="Only transcribe up to this point in the recording (default: the "
        "end of the longest track). See --start.",
    )
    parser.add_argument(
        "--test",
        nargs="?",
        type=int,
        const=120,
        default=None,
        metavar="SECONDS",
        help="Rapid-iteration mode: a windowed run of SECONDS (default 120), "
        "starting at --start if given or at a random point otherwise. Writes "
        "to <slug>-<start>-<end>.md like --start/--end.",
    )
    parser.add_argument(
        "--progress-events",
        action="store_true",
        help="Also print machine-readable '@@PROGRESS@@ {json}' progress lines "
        "to stdout. Used by gui.py; not meant for interactive use.",
    )
    args = parser.parse_args()
    _progress_events.enabled = args.progress_events
    if args.progress_events:
        # gui.py cancels by terminating this process. Turning SIGTERM into the
        # same exception the Escape-then-y cancel raises lets the `with` blocks
        # below unwind, so temp dirs (zip extraction, --test crops) are cleaned
        # up instead of leaking a whole extracted session per cancel.
        signal.signal(signal.SIGTERM, _raise_cancelled)

    if args.test is not None and args.end is not None:
        sys.exit("error: --end can't be combined with --test, which sets the window length")
    try:
        window_start_arg = _parse_hms(args.start) if args.start is not None else None
        window_end_arg = _parse_hms(args.end) if args.end is not None else None
    except ValueError as e:
        sys.exit(f"error: invalid --start/--end: {e}")
    windowed = args.test is not None or args.start is not None or args.end is not None
    shared_mic_specs = [parse_shared_mic(value) for value in args.shared_mic or []]
    if args.prune_cache is not None:
        if args.prune_cache < 0:
            sys.exit("error: --prune-cache DAYS must be 0 or more")
        if args.craig_folder is not None or windowed or args.fresh or args.shared_mic:
            sys.exit("error: --prune-cache only prunes and exits; drop the other arguments")
    if windowed and args.fresh:
        sys.exit("error: --fresh doesn't apply to windowed runs, which never use saved partial output")
    global _saves_partials
    _saves_partials = not windowed

    config = load_config(args.config)
    if config.get("speaker_map") is not None and not isinstance(config["speaker_map"], dict):
        sys.exit(f"error: speaker_map in {args.config} must be a mapping of Discord name to speaker name")
    if args.prune_cache is not None:
        prune_root = paths.resolve(config.get("output_dir", "./data/transcripts"))
        removed = prune_partials(prune_root, args.prune_cache)
        for slug in removed:
            print(f"  removed {prune_root / '.partial' / slug}")
        print(f"Pruned {len(removed)} cached session(s).")
        return
    if not args.model and "whisper_model" not in config:
        sys.exit(f"error: whisper_model is missing from {args.config} and --model wasn't passed")

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"error: {tool} not found on PATH. Install it with `brew install ffmpeg`.")

    model_repo = args.model or config.get("whisper_model")
    language = args.language or config.get("language")
    if language is not None and language not in LANGUAGES:
        sys.exit(f"error: unknown language code '{language}'")
    compression_ratio_threshold = (
        args.compression_ratio_threshold
        if args.compression_ratio_threshold is not None
        else config.get("compression_ratio_threshold", COMPRESSION_RATIO_THRESHOLD)
    )
    logprob_threshold = (
        args.logprob_threshold
        if args.logprob_threshold is not None
        else config.get("logprob_threshold", LOGPROB_THRESHOLD)
    )
    best_of = args.best_of if args.best_of is not None else config.get("best_of")
    input_dir = paths.resolve(config.get("input_dir", "./data/input"))
    output_dir = paths.resolve(config.get("output_dir", "./data/transcripts"))
    partial_max_age_days = _config_number(config, "partial_max_age_days", 30)
    if partial_max_age_days < 0:
        sys.exit(f"error: config `partial_max_age_days` must be 0 or more, got {partial_max_age_days!r}")
    model_dir = paths.resolve(config.get("model_dir", "./data/models"))
    speaker_map = config.get("speaker_map") or {}
    vad_min_silence_ms = _config_number(config, "vad_min_silence_ms", 800)
    vad_min_speech_ms = _config_number(config, "vad_min_speech_ms", 250)
    merge_gap_ms = _config_number(config, "merge_gap_ms", 2000)
    timestamp_format = config.get("timestamp_format", "%H:%M:%S")
    line_format = config.get("line_format", "[{timestamp}] {speaker}: {text}")
    filler_words = config.get(
        "filler_words", ["so", "you", "um", "uh", "hm", "hmm", "mm", "ah", "eh", "er", "thank you"]
    )
    backchannel_phrases = config.get("backchannel_phrases", DEFAULT_BACKCHANNEL_PHRASES)
    if not isinstance(backchannel_phrases, list) or not all(
        isinstance(phrase, str) for phrase in backchannel_phrases
    ):
        sys.exit(f"error: config `backchannel_phrases` must be a list of strings, got {backchannel_phrases!r}")
    collapse_laughter = bool(config.get("collapse_laughter", True))
    laughter_run_gap_ms = _config_number(config, "laughter_run_gap_ms", 5000)
    hallucination_phrases = config.get("hallucination_phrases", DEFAULT_HALLUCINATION_PHRASES)
    if not isinstance(hallucination_phrases, list) or not all(
        isinstance(phrase, str) for phrase in hallucination_phrases
    ):
        sys.exit(f"error: config `hallucination_phrases` must be a list of strings, got {hallucination_phrases!r}")
    try:
        hallucination_patterns = compile_hallucination_patterns(hallucination_phrases)
    except ValueError as err:
        sys.exit(f"error: {err}")
    show_inaudible_placeholders = (
        args.show_inaudible
        if args.show_inaudible is not None
        else config.get("show_inaudible_placeholders", False)
    )
    show_bleed_markers = (
        args.show_bleed_markers
        if args.show_bleed_markers is not None
        else config.get("show_bleed_markers", False)
    )
    bleed_thresholds = {
        key: _config_number(config, key, default)
        for key, default in (
            ("bleed_level_gap_db", 4),
            ("bleed_min_correlation", 0.3),
            ("bleed_text_similarity", 0.6),
            ("bleed_max_offset_s", 2),
        )
    }
    timestamp_mode = config.get("timestamp_mode", "elapsed")
    if timestamp_mode not in ("clock", "elapsed"):
        sys.exit(f"error: config `timestamp_mode` must be 'clock' or 'elapsed', got {timestamp_mode!r}")
    if args.elapsed_timestamps is not None:
        timestamp_mode = "elapsed" if args.elapsed_timestamps else "clock"

    campaign = args.campaign or config.get("campaign")
    if args.vocabulary is not None:
        vocabulary_path = args.vocabulary
    elif campaign:
        vocabulary_path = campaign_path(campaign)
        if not vocabulary_path.exists():
            available = ", ".join(list_campaigns()) or "none"
            sys.exit(
                f"error: no vocabulary file for campaign '{campaign}' at {vocabulary_path}. "
                f"Available campaigns: {available}"
            )
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
    partial_dir = partial_dir_for(output_dir, slug)
    # Checked before extraction and VAD so a mismatch costs nothing. Never
    # discard the cache here: rebuilding it can take an hour, and the usual
    # cause is a forgotten flag (e.g. the cache was built with --model).
    if not windowed and not args.fresh and partial_dir.exists():
        mismatches = meta_mismatches(
            load_partial_meta(partial_dir),
            model_repo,
            vad_min_silence_ms,
            vad_min_speech_ms,
            initial_prompt,
            language,
            compression_ratio_threshold,
            logprob_threshold,
            best_of,
        )
        if mismatches:
            details = "\n".join(f"  - {line}" for line in mismatches)
            sys.exit(
                f"error: saved partial output in {partial_dir} doesn't match this run's config:\n"
                f"{details}\n"
                "Rerun with the saved settings (e.g. --model <saved model_repo>) to reuse it, "
                "or pass --fresh (or delete that directory) to discard it and retranscribe every track."
            )
    if not windowed and partial_max_age_days > 0:
        pruned = prune_partials(output_dir, partial_max_age_days, keep=slug)
        if pruned:
            print(
                f"Pruned {len(pruned)} cached session(s) unused for "
                f"{partial_max_age_days:g}+ days: {', '.join(pruned)}"
            )

    extraction_ctx = tempfile.TemporaryDirectory() if is_zip else contextlib.nullcontext()
    with (
        extraction_ctx as tmp_dir,
        contextlib.ExitStack() as exit_stack,
        _CancelWatcher() as cancel_watcher,
    ):
        if cancel_watcher.enabled:
            print("(press Escape, then y, to cancel — progress on completed tracks is saved)")
            print()
        if is_zip:
            print("Extracting zip...")
            _progress_events.step("Extracting zip")
            craig_folder = extract_zip(resolved_input_path, Path(tmp_dir))
        else:
            craig_folder = resolved_input_path
        print()

        print("PARSING MANIFEST...")
        _progress_events.step("Reading manifest")
        session_start, usernames = parse_manifest(craig_folder)
        print(f"  Parsed manifest: {len(usernames)} speakers, session started {session_start}")
        shared_spans: dict[str, list[tuple[float, float, str]]] = {}
        for username, span_start, span_end, label in shared_mic_specs:
            if username not in usernames:
                sys.exit(
                    f"error: --shared-mic track {username!r} not in this recording "
                    f"(tracks: {', '.join(usernames)})"
                )
            for other_start, other_end, _ in shared_spans.get(username, []):
                if span_start < other_end and other_start < span_end:
                    sys.exit(f"error: --shared-mic spans on track {username!r} overlap")
            shared_spans.setdefault(username, []).append((span_start, span_end, label))
        matched = match_tracks(craig_folder, usernames)
        warned_speakers: set[str] = set()
        for username in matched:
            resolve_speaker(username, speaker_map, warned_speakers)
        print()

        window_start = 0.0
        filename_suffix = ""
        if windowed:
            # Craig tracks end when that speaker's recording stops, so they
            # can differ by hours. The window may run to the end of the
            # longest one; shorter tracks are cropped to whatever part of the
            # window they cover, and dropped if they end before it starts.
            durations = {username: _probe_duration(path) for username, path in matched.items()}
            session_duration = max(durations.values())
            if args.test is not None:
                if window_start_arg is not None:
                    window_start = window_start_arg
                else:
                    # A random test window stays within the shortest track so
                    # every speaker has audio in it.
                    window_start = random.uniform(0, max(0.0, min(durations.values()) - args.test))
                window_end = window_start + args.test
            else:
                window_start = window_start_arg if window_start_arg is not None else 0.0
                window_end = window_end_arg if window_end_arg is not None else session_duration
            window_label = (
                f"{timedelta(seconds=round(window_start))}-{timedelta(seconds=round(window_end))}"
            )
            if window_end <= window_start:
                sys.exit(f"error: requested window ({window_label}) ends before it starts")
            if window_end > session_duration:
                sys.exit(
                    f"error: requested window ({window_label}) runs past the end of the "
                    f"recording ({timedelta(seconds=round(session_duration))})"
                )
            _progress_events.step("Cropping tracks to the window")
            crop_dir = Path(exit_stack.enter_context(tempfile.TemporaryDirectory()))
            dropped = [username for username in matched if durations[username] <= window_start]
            matched = {
                username: crop_track(path, window_start, window_end - window_start, crop_dir, username)
                for username, path in matched.items()
                if username not in dropped
            }
            for username in dropped:
                print(
                    f"  Skipping {username}: track ends at "
                    f"{timedelta(seconds=round(durations[username]))}, before the window starts"
                )
            filename_suffix = f"-{_window_stamp(window_start)}-{_window_stamp(window_end)}"
            if args.test is not None:
                print(f"Test mode: using window {window_label} ({args.test}s)")
            else:
                print(f"Windowed run: transcribing {window_label} only")
            print()

        print("ANALYZING SPEECH (VAD)...")
        track_regions: dict[str, list[float]] = {}
        # Index 0 is window_start in windowed runs (tracks are cropped), else 0.
        track_envelopes: dict[str, np.ndarray] = {}
        total_raw_s = 0.0
        total_speech_s = 0.0
        vad_total = len(matched)
        track_durations = {username: _probe_duration(path) for username, path in matched.items()}
        vad_start = time.monotonic()
        with tqdm.tqdm(
            total=round(sum(track_durations.values())), unit="s", leave=False, desc="Analyzing speech"
        ) as vad_progress:
            completed_s = 0.0
            for i, (username, audio_path) in enumerate(matched.items(), start=1):
                if cancel_watcher.cancelled.is_set():
                    raise TranscriptionCancelled()
                vad_progress.write(f"  VAD track {i}/{vad_total} ({username})...")
                # Each track is decoded in full before VAD reports any progress.
                _progress_events.step(f"Analyzing speech — decoding audio ({username})")
                track_duration = track_durations[username]

                def _on_vad_progress(
                    pct: float, _base=completed_s, _dur=track_duration, _track=username
                ) -> None:
                    vad_progress.n = round(_base + _dur * pct / 100)
                    vad_progress.refresh()
                    _progress_events.progress("vad", vad_progress.n, vad_progress.total, _track)

                track_start = time.monotonic()
                regions, duration, envelope = detect_speech_regions(
                    audio_path,
                    vad_min_silence_ms,
                    vad_min_speech_ms,
                    _on_vad_progress,
                )
                track_elapsed = time.monotonic() - track_start
                track_regions[username] = regions
                track_envelopes[username] = envelope
                track_speech_s = speech_seconds(regions)
                total_raw_s += duration
                total_speech_s += track_speech_s
                completed_s += track_duration
                vad_progress.n = round(completed_s)
                vad_progress.refresh()
                _progress_events.progress("vad", vad_progress.n, vad_progress.total, username)
                vad_progress.write(
                    f"    -> {_format_duration(track_speech_s)} speech / "
                    f"{_format_duration(duration)} raw ({track_speech_s / duration * 100:.0f}%) "
                    f"in {_format_duration(track_elapsed)}"
                )
        vad_elapsed = time.monotonic() - vad_start
        speech_pct = (total_speech_s / total_raw_s * 100) if total_raw_s else 0.0
        print(
            f"VAD completed: {_format_duration(total_speech_s)} speech / "
            f"{_format_duration(total_raw_s)} raw audio ({speech_pct:.0f}%) to transcribe "
            f"in {_format_duration(vad_elapsed)}"
        )
        print()

        print("RESOLVING MODEL...")
        print(f"  {model_repo}")

        def announce_download() -> None:
            print("  Downloading model (first run)...")
            _progress_events.step("Downloading Whisper model (first run)")

        model_path = resolve_model_path(model_repo, model_dir, on_download=announce_download)
        print()

        print("TRANSCRIBING TRACKS...")
        total_frames = sum(speech_frames(regions) for regions in track_regions.values())
        transcripts = {}
        total = len(matched)
        total_segments = 0

        if not windowed:
            if args.fresh and partial_dir.exists():
                shutil.rmtree(partial_dir)
                print("  Ignoring saved partial output (--fresh).")
            write_partial_meta(
                partial_dir,
                model_repo,
                vad_min_silence_ms,
                vad_min_speech_ms,
                initial_prompt,
                language,
                compression_ratio_threshold,
                logprob_threshold,
                best_of,
            )

        transcribe_start = time.monotonic()
        with tqdm.tqdm(
            total=total_frames,
            unit="frames",
            desc="Transcribing",
            disable=total_frames == 0,
            leave=False,
        ) as progress:
            if _progress_events.enabled:
                # Every transcription progress tick (mlx_whisper's own, via
                # _ForwardingPbar, and the resume path's) lands in
                # progress.update(), so hooking it here covers them all.
                # `username` is read at call time, i.e. the loop's current track.
                _tqdm_update = progress.update

                def _update_and_emit(n: float = 1) -> bool | None:
                    result = _tqdm_update(n)
                    _progress_events.progress("transcribe", progress.n, progress.total, username)
                    return result

                progress.update = _update_and_emit
            for i, (username, audio_path) in enumerate(matched.items(), start=1):
                if cancel_watcher.cancelled.is_set():
                    raise TranscriptionCancelled()
                track_start = time.monotonic()
                regions = track_regions[username]
                region_count = len(regions) // 2
                partial = (
                    None
                    if windowed
                    else load_track_partial(partial_dir, username, audio_path, regions, progress)
                )
                if partial is not None and partial.regions_done >= region_count:
                    segments, detected_language = partial.segments, partial.language
                    progress.write(
                        f"  Resuming track {i}/{total} ({username}) (from saved partial)..."
                    )
                    progress.update(speech_frames(regions))
                else:
                    if partial is not None:
                        progress.write(
                            f"  Resuming track {i}/{total} ({username}) at region "
                            f"{partial.regions_done + 1}/{region_count} (from saved partial)..."
                        )
                        progress.update(speech_frames(regions[: partial.regions_done * 2]))
                    else:
                        progress.write(f"  Transcribing track {i}/{total} ({username})...")
                    _progress_events.step(f"Transcribing — decoding audio ({username})")

                    def save_progress(
                        track_progress: TrackPartial,
                        username: str = username,
                        audio_path: Path = audio_path,
                        regions: list[float] = regions,
                    ) -> None:
                        write_track_partial(
                            partial_dir, username, audio_path, track_progress, regions
                        )

                    segments, detected_language = transcribe_track(
                        audio_path,
                        model_path,
                        regions,
                        progress,
                        initial_prompt,
                        language,
                        compression_ratio_threshold,
                        logprob_threshold,
                        best_of,
                        cancel_watcher.cancelled,
                        resume=partial,
                        checkpoint=None if windowed else save_progress,
                    )
                    if not windowed:
                        save_progress(
                            TrackPartial(segments, detected_language, region_count, "")
                        )
                track_elapsed = time.monotonic() - track_start
                if detected_language:
                    progress.write(
                        f"    Detected language: "
                        f"{LANGUAGES.get(detected_language, detected_language).title()}"
                    )
                progress.write(
                    f"    -> {len(segments)} segments in {_format_duration(track_elapsed)}"
                )
                total_segments += len(segments)
                transcripts[username] = [
                    (
                        offset + window_start,
                        end + window_start,
                        apply_vocabulary(text, vocabulary),
                        compression_ratio,
                        avg_logprob,
                    )
                    for offset, end, text, compression_ratio, avg_logprob in segments
                ]
        transcribe_elapsed = time.monotonic() - transcribe_start
        print(
            f"Transcription completed: {total_segments} segments "
            f"in {_format_duration(transcribe_elapsed)}"
        )
        print()

        print("MERGING TRACKS...")
        transcripts, bleed_stats = dedup_bleed(
            transcripts,
            track_envelopes,
            window_start,
            {u: resolve_speaker(u, speaker_map, warned_speakers) for u in transcripts},
            bleed_thresholds["bleed_level_gap_db"],
            bleed_thresholds["bleed_min_correlation"],
            bleed_thresholds["bleed_text_similarity"],
            bleed_thresholds["bleed_max_offset_s"],
            show_bleed_markers,
            args.verbose,
        )
        timestamp_base = _ELAPSED_TIMESTAMP_EPOCH if timestamp_mode == "elapsed" else session_start
        merged, flagged_lines, merge_stats, shared_labels, pruned_lines = merge_tracks(
            timestamp_base,
            transcripts,
            merge_gap_ms / 1000,
            filler_words,
            compression_ratio_threshold,
            logprob_threshold,
            show_inaudible_placeholders,
            hallucination_patterns,
            shared_spans,
            backchannel_phrases,
            collapse_laughter,
            laughter_run_gap_ms / 1000,
        )
        merge_stats.update(bleed_stats)
        print(f"  {merge_stats['segments_total']} segments")
        print(f"  {merge_stats['bleed_deduped']} bleed duplicate(s) dropped")
        print(f"  {merge_stats['bleed_ambiguous']} possible bleed pair(s) kept")
        print(f"  {merge_stats['hallucination_stripped']} hallucination phrase(s) stripped")
        print(f"  {merge_stats['repetition_collapsed']} repetition hallucination(s) collapsed")
        print(f"  {merge_stats['filler_dropped']} filler word(s) dropped")
        print(f"  {merge_stats['crosstalk_dropped']} crosstalk backchannel(s) dropped")
        print(f"  {merge_stats['laughter_dropped']} laugh line(s) dropped")
        print(f"  {merge_stats['laughter_runs_collapsed']} laughter run(s) collapsed")
        print(f"  {merge_stats['repeat_deduped']} repeat(s) deduped")
        print(f"  {merge_stats['empty_dropped']} empty segment(s) dropped")
        print(f"  {merge_stats['flagged_for_review']} line(s) flagged for review")
        print("Merging completed")
        # A --shared-mic label is already a display name, not a Discord username.
        resolved = [
            (
                timestamp,
                speaker if speaker in shared_labels else resolve_speaker(speaker, speaker_map, warned_speakers),
                text,
            )
            for timestamp, speaker, text in merged
        ]
        resolved_flagged = [
            (
                timestamp,
                speaker if speaker in shared_labels else resolve_speaker(speaker, speaker_map, warned_speakers),
                text,
            )
            for timestamp, speaker, text in flagged_lines
        ]
        resolved_pruned = [
            (
                timestamp,
                speaker if speaker in shared_labels else resolve_speaker(speaker, speaker_map, warned_speakers),
                text,
            )
            for timestamp, speaker, text in pruned_lines
        ]
        print()

        print("WRITING TRANSCRIPT...")
        output_path = write_transcript(
            resolved,
            resolved_flagged,
            resolved_pruned,
            session_title,
            output_dir,
            timestamp_format,
            line_format,
            merge_stats,
            session_start,
            filename_suffix=filename_suffix,
            output_name=args.output_name,
        )
        if is_zip and not windowed:
            mark_zip_processed(resolved_input_path, input_dir)

    elapsed = time.monotonic() - start_time
    print(f"Wrote transcript to {output_path} ({_format_duration(elapsed)})")
    print()
    _progress_events.emit(phase="done", output_path=str(output_path.resolve()))
    print(_rule())


if __name__ == "__main__":
    try:
        main()
    except TranscriptionCancelled:
        message = (
            "Cancelled. Progress so far was saved; rerun to resume."
            if _saves_partials
            else "Cancelled."
        )
        print(message)
        _progress_events.emit(phase="error", message=message)
        sys.exit(130)
    except SystemExit as e:
        # sys.exit("error: ...") messages go to stderr, which a GUI parent
        # doesn't read — surface them as an error event too.
        if isinstance(e.code, str):
            _progress_events.emit(phase="error", message=e.code)
        raise
    except Exception as e:
        _progress_events.emit(phase="error", message=f"{type(e).__name__}: {e}")
        raise
