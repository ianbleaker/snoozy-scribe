"""Transcribe a Craig-recorded Discord TTRPG session into a Markdown file."""

from __future__ import annotations

import argparse
import contextlib
import re
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import mlx_whisper
import silero_vad
import soundfile
import torch
import torchaudio.functional as AF
import yaml
from huggingface_hub import snapshot_download

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

    Bypasses `silero_vad.read_audio()`, which fails on this environment:
    `torchaudio` >=2.9 (the only versions with Python 3.14 wheels) requires
    `torchcodec` for audio I/O, and the available `torchcodec` build has an
    ABI mismatch with `torch` 2.9.1 (see plan doc Escalation section).
    """
    data, sample_rate = soundfile.read(str(audio_path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)
    if sample_rate != VAD_SAMPLE_RATE:
        wav = AF.resample(wav, sample_rate, VAD_SAMPLE_RATE)
    return wav


def detect_speech_regions(audio_path: Path) -> list[float]:
    """Return flat [start1, end1, start2, end2, ...] speech regions in seconds."""
    global _vad_model
    if _vad_model is None:
        _vad_model = silero_vad.load_silero_vad()

    wav = _read_audio_for_vad(audio_path)
    timestamps = silero_vad.get_speech_timestamps(
        wav, _vad_model, sampling_rate=VAD_SAMPLE_RATE, return_seconds=True
    )

    regions: list[float] = []
    for entry in timestamps:
        regions.append(entry["start"])
        regions.append(entry["end"])
    return regions


def transcribe_track(audio_path: Path, model_path: Path) -> list[tuple[float, str]]:
    """Transcribe one audio track into (segment_start_offset, text) tuples."""
    speech_regions = detect_speech_regions(audio_path)
    if not speech_regions:
        return []

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=str(model_path),
        clip_timestamps=speech_regions,
    )
    return [(segment["start"], segment["text"]) for segment in result["segments"]]


def apply_vocabulary(text: str, vocabulary: dict[str, list[str]]) -> str:
    """Case-insensitively replace known wrong variants with the correct term."""
    for correct_term, variants in vocabulary.items():
        for variant in variants:
            text = re.sub(re.escape(variant), correct_term, text, flags=re.IGNORECASE)
    return text


def resolve_speaker(username: str, speaker_map: dict[str, str]) -> str:
    """Map a Discord username to its configured display name."""
    if username in speaker_map:
        return speaker_map[username]
    print(f"warning: no speaker_map entry for {username}, using raw Discord name")
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
) -> Path:
    """Write the merged transcript as a Markdown file with YAML frontmatter."""
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "-", session_title.lower()).strip("-")
    output_path = output_dir / f"{slug}.md"

    speakers = list(dict.fromkeys(name for _, name, _ in merged))
    date = merged[0][0].date().isoformat() if merged else ""

    frontmatter = yaml.safe_dump(
        {"date": date, "title": session_title, "speakers": speakers},
        sort_keys=False,
    )

    lines = [
        f"[{timestamp.strftime('%H:%M:%S')}] {name}: {text}"
        for timestamp, name, text in merged
    ]

    output_path.write_text(f"---\n{frontmatter}---\n\n" + "\n".join(lines) + "\n")
    return output_path


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

        model_dir = Path(config["model_dir"])
        target_model_dir = model_dir / model_repo.split("/")[-1]
        print(f"Resolving model {model_repo}...")
        if not target_model_dir.exists() or not any(target_model_dir.iterdir()):
            print("Downloading model (first run)...")
        model_path = resolve_model_path(model_repo, model_dir)

        transcripts = {}
        total = len(matched)
        for i, (username, audio_path) in enumerate(matched.items(), start=1):
            print(f"Transcribing track {i}/{total}: {username}...")
            segments = transcribe_track(audio_path, model_path)
            print(f"  -> {len(segments)} segments")
            transcripts[username] = [
                (offset, apply_vocabulary(text, config["vocabulary"]))
                for offset, text in segments
            ]

        print("Merging tracks...")
        merged = merge_tracks(session_start, transcripts)
        resolved = [
            (timestamp, resolve_speaker(username, config["speaker_map"]), text)
            for timestamp, username, text in merged
        ]

        print("Writing transcript...")
        output_path = write_transcript(resolved, session_title, Path(config["output_dir"]))

        if is_zip:
            mark_zip_processed(resolved_input_path, input_dir)

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    elapsed_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    print(f"Wrote transcript to {output_path} ({elapsed_str})")


if __name__ == "__main__":
    main()
