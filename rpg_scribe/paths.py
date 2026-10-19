"""Repo-relative locations of settings and data.

Relative paths in config resolve against the repo root, not the cwd: the
menu-bar app launched at login doesn't start in the repo folder.
"""

from __future__ import annotations

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
SETTINGS_DIR = REPO_DIR / "settings"
VOCABULARY_DIR = SETTINGS_DIR / "vocabulary"
DATA_DIR = REPO_DIR / "data"
DEFAULT_CONFIG_PATH = SETTINGS_DIR / "config.yaml"


def resolve(path: str | Path) -> Path:
    """Return `path` unchanged if absolute, else joined onto REPO_DIR."""
    path = Path(path)
    return path if path.is_absolute() else REPO_DIR / path
