"""Transcribe a Craig-recorded Discord TTRPG session into a Markdown file."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import re
import shutil
import sys
import tempfile
import time
import zipfile
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
from mlx_whisper.audio import FRAMES_PER_SECOND

VAD_SAMPLE_RATE = 16000

AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".aac"}

_vad_model = None


def _strip_discriminator(username: str) -> str:
    return username.split("#", 1)[0]


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
            print(f"warning: no audio file found for {username}, skipping")

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


def _read_audio_for_vad(audio_path: Path) -> torch.Tensor:
    """Load audio as a mono float32 tensor resampled to `VAD_SAMPLE_RATE`.

    Shells out to ffmpeg (mirroring `mlx_whisper.audio.load_audio`) rather than
    using `soundfile`/`torchaudio`, since Craig tracks are commonly AAC and
    libsndfile cannot decode AAC. This also bypasses `silero_vad.read_audio()`,
    which fails on this environment: `torchaudio` >=2.9 (the only versions
    with Python 3.14 wheels) requires `torchcodec` for audio I/O, and the
    available `torchcodec` build has an ABI mismatch with `torch` 2.9.1 (see
    plan doc Escalation section).
    """
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


def detect_speech_regions(
    audio_path: Path, min_silence_ms: int
) -> tuple[list[float], float]:
    """Return (flat [start1, end1, start2, end2, ...] speech regions, raw duration), in seconds.

    `min_silence_ms` sets how long a pause must be before it splits speech
    into separate regions. Silero's own default (100ms) fragments speech on
    tiny pauses/breaths, forcing Whisper to decode across many small seek
    boundaries. That fragmentation reliably triggers Whisper's low-confidence
    temperature-fallback decoding at region edges, which can degrade into
    garbled/wrong-language text. Wider gaps merge natural pauses back into
    one clip, eliminating the trigger (verified locally: 8/8 clean runs at
    800ms vs. 8/8 degraded runs at the 100ms default, same test audio).
    """
    global _vad_model
    if _vad_model is None:
        _vad_model = silero_vad.load_silero_vad()

    wav = _read_audio_for_vad(audio_path)
    timestamps = silero_vad.get_speech_timestamps(
        wav,
        _vad_model,
        sampling_rate=VAD_SAMPLE_RATE,
        return_seconds=True,
        min_silence_duration_ms=min_silence_ms,
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

    Mirrors mlx_whisper's own `seek_points = round(ts * FRAMES_PER_SECOND)`
    rounding so this matches, frame for frame, what its internal progress
    bar advances by.
    """
    frames = [round(ts * FRAMES_PER_SECOND) for ts in regions]
    return sum(end - start for start, end in zip(frames[::2], frames[1::2]))


class _ForwardingPbar:
    """Stand-in for `tqdm.tqdm` that forwards `.update()` calls to a shared bar."""

    def __init__(self, master: tqdm.tqdm):
        self._master = master

    def update(self, n: int = 1) -> None:
        self._master.update(n)

    def __enter__(self) -> "_ForwardingPbar":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _TqdmShim:
    """Replaces the `tqdm` name mlx_whisper.transcribe's module sees.

    mlx_whisper's internal progress bar is keyed to the full audio file's
    frame count, not to the speech-only regions we hand it via
    `clip_timestamps` — so it stalls well short of 100% on any track with
    real silence. Its `pbar.update(n)` calls only ever fire for frames it
    actually decodes though, so redirecting them into one master bar sized
    to the true speech-frame total (from VAD) gives an accurate, live,
    whole-job progress percentage instead.
    """

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


def transcribe_track(
    audio_path: Path,
    model_path: Path,
    speech_regions: list[float],
    progress: tqdm.tqdm,
) -> list[tuple[float, str]]:
    """Transcribe one audio track into (segment_start_offset, text) tuples."""
    if not speech_regions:
        return []

    with _redirect_mlx_whisper_progress(progress):
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=str(model_path),
            clip_timestamps=speech_regions,
            verbose=False,
        )
    return [(segment["start"], segment["text"]) for segment in result["segments"]]


def apply_vocabulary(text: str, vocabulary: dict[str, list[str]]) -> str:
    """Case-insensitively replace known wrong variants with the correct term."""
    for correct_term, variants in vocabulary.items():
        for variant in variants:
            text = re.sub(re.escape(variant), correct_term, text, flags=re.IGNORECASE)
    return text


def resolve_speaker(username: str, speaker_map: dict[str, str], warned: set[str]) -> str:
    """Map a Discord username to its configured display name.

    Warns at most once per username, since a missing speaker_map entry
    applies to every segment from that track.
    """
    if username in speaker_map:
        return speaker_map[username]
    if username not in warned:
        print(f"warning: no speaker_map entry for {username}, using raw Discord name")
        warned.add(username)
    return username


def merge_tracks(
    session_start: datetime,
    transcripts: dict[str, list[tuple[float, str]]],
) -> list[tuple[datetime, str, str]]:
    """Interleave per-track segments into one chronological transcript."""
    merged: list[tuple[datetime, str, str]] = []

    for username, segments in transcripts.items():
        previous_text: str | None = None
        for offset, text in segments:
            stripped = text.strip()
            if stripped in ("", "..."):
                continue
            if stripped == previous_text:
                continue
            previous_text = stripped
            timestamp = session_start + timedelta(seconds=offset)
            merged.append((timestamp, username, stripped))

    merged.sort(key=lambda entry: entry[0])
    return merged


def write_transcript(
    merged: list[tuple[datetime, str, str]],
    session_title: str,
    output_dir: Path,
    timestamp_format: str,
    line_format: str,
) -> Path:
    """Write the merged transcript as a Markdown file with YAML frontmatter.

    `line_format` is a str.format() template with `{timestamp}`, `{speaker}`,
    and `{text}` placeholders, letting the config control per-line Markdown
    (e.g. `**{speaker}**` to bold speaker names).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "-", session_title.lower()).strip("-")
    output_path = output_dir / f"{slug}.md"

    speakers = list(dict.fromkeys(name for _, name, _ in merged))
    date = merged[0][0].date().isoformat() if merged else ""

    frontmatter = yaml.safe_dump(
        {"date": date, "title": session_title, "speakers": speakers},
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


def main() -> None:
    start_time = time.monotonic()

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
    parser.add_argument("--title", help="Override the session title")
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit(
            f"error: config file not found at {args.config}. "
            "Run `cp config.example.yaml config.yaml` and fill in your values."
        )
    config = yaml.safe_load(args.config.read_text())

    model_repo = args.model or config["whisper_model"]
    input_dir = Path(config.get("input_dir", "./input"))
    vad_min_silence_ms = config.get("vad_min_silence_ms", 800)
    timestamp_format = config.get("timestamp_format", "%H:%M:%S")
    line_format = config.get("line_format", "[{timestamp}] {speaker}: {text}")

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

    extraction_ctx = tempfile.TemporaryDirectory() if is_zip else contextlib.nullcontext()
    with extraction_ctx as tmp_dir:
        if is_zip:
            print("Extracting zip...")
            craig_folder = extract_zip(resolved_input_path, Path(tmp_dir))
        else:
            craig_folder = resolved_input_path

        session_start, usernames = parse_manifest(craig_folder)
        print(f"Parsed manifest: {len(usernames)} speakers, session started {session_start}")
        matched = match_tracks(craig_folder, usernames)

        print(f"Analyzing speech regions (VAD) for {len(matched)} tracks...")
        track_regions: dict[str, list[float]] = {}
        total_raw_s = 0.0
        total_speech_s = 0.0
        for username, audio_path in matched.items():
            regions, duration = detect_speech_regions(audio_path, vad_min_silence_ms)
            track_regions[username] = regions
            total_raw_s += duration
            total_speech_s += speech_seconds(regions)
        speech_pct = (total_speech_s / total_raw_s * 100) if total_raw_s else 0.0
        print(
            f"VAD: {_format_duration(total_speech_s)} speech / "
            f"{_format_duration(total_raw_s)} raw audio ({speech_pct:.0f}%) to transcribe"
        )

        model_dir = Path(config["model_dir"])
        target_model_dir = model_dir / model_repo.split("/")[-1]
        print(f"Resolving model {model_repo}...")
        if not target_model_dir.exists() or not any(target_model_dir.iterdir()):
            print("Downloading model (first run)...")
        model_path = resolve_model_path(model_repo, model_dir)

        total_frames = sum(speech_frames(regions) for regions in track_regions.values())
        transcripts = {}
        total = len(matched)
        with tqdm.tqdm(
            total=total_frames, unit="frames", desc="Transcribing", disable=total_frames == 0
        ) as progress:
            for i, (username, audio_path) in enumerate(matched.items(), start=1):
                print(f"Transcribing track {i}/{total}: {username}...")
                segments = transcribe_track(
                    audio_path, model_path, track_regions[username], progress
                )
                print(f"  -> {len(segments)} segments")
                transcripts[username] = [
                    (offset, apply_vocabulary(text, config["vocabulary"]))
                    for offset, text in segments
                ]

        print("Merging tracks...")
        merged = merge_tracks(session_start, transcripts)
        warned_speakers: set[str] = set()
        resolved = [
            (timestamp, resolve_speaker(username, config["speaker_map"], warned_speakers), text)
            for timestamp, username, text in merged
        ]

        print("Writing transcript...")
        output_path = write_transcript(
            resolved, session_title, Path(config["output_dir"]), timestamp_format, line_format
        )

        if is_zip:
            mark_zip_processed(resolved_input_path, input_dir)

    elapsed = time.monotonic() - start_time
    print(f"Wrote transcript to {output_path} ({_format_duration(elapsed)})")


if __name__ == "__main__":
    main()
