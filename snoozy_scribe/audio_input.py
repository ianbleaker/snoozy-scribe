"""Classify a transcription input path: Craig export, audio file, or folder.

Stdlib only, so the transcription window can import it without pulling in
mlx or numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".aac"}


class InputKind(Enum):
    CRAIG_ZIP = "craig-zip"
    CRAIG_FOLDER = "craig-folder"
    SINGLE_FILE = "single-file"
    LOOSE_FOLDER = "loose-folder"


class InputError(ValueError):
    """The path isn't something transcribe.py can take as input."""


@dataclass
class DetectedInput:
    kind: InputKind
    path: Path
    # Only set for LOOSE_FOLDER (every file) and SINGLE_FILE (the one file).
    audio_files: list[Path] = field(default_factory=list)


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def _extensions_text() -> str:
    return ", ".join(sorted(AUDIO_EXTENSIONS))


def detect_input(path: Path) -> DetectedInput:
    """Classify `path`, or raise `InputError`.

    A `.zip` is always taken as a Craig export. A folder holding `info.txt`
    is a Craig export; any other folder is scanned at the top level only
    for audio files, and one holding exactly one counts as that file.
    """
    if path.suffix.lower() == ".zip":
        return DetectedInput(InputKind.CRAIG_ZIP, path)
    if path.is_dir():
        if (path / "info.txt").is_file():
            return DetectedInput(InputKind.CRAIG_FOLDER, path)
        audio_files = sorted(
            (
                child
                for child in path.iterdir()
                if child.is_file() and not child.name.startswith(".") and is_audio_file(child)
            ),
            key=lambda child: child.name,
        )
        if not audio_files:
            raise InputError(f"no audio files in {path}")
        if len(audio_files) == 1:
            return DetectedInput(InputKind.SINGLE_FILE, audio_files[0], audio_files)
        return DetectedInput(InputKind.LOOSE_FOLDER, path, audio_files)
    if path.is_file() and is_audio_file(path):
        return DetectedInput(InputKind.SINGLE_FILE, path, [path])
    raise InputError(
        f"unsupported input {path}: expected a Craig .zip/folder, an audio file "
        f"({_extensions_text()}), or a folder of audio files"
    )
