"""Where the code, its built-in defaults and the user's settings and data live.

User state lives outside the code, in one folder shared by every build:
`~/Library/Application Support/snoozy-scribe/` (override it with the
`SNOOZY_SCRIBE_HOME` environment variable, e.g. to test a fresh install).
Its layout:

    settings/config.yaml
    settings/vocabulary/<name>.md
    settings/prompts/<name>.md      (optional overrides of the built-in prompts)
    data/models/  data/input/  data/cache/partial/  data/dictation-state.json

Relative paths in config resolve against that folder, not the cwd.
`ensure_user_dirs()` creates it and seeds the config and the example
vocabulary on first run; every entry point calls it.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from snoozy_scribe import identity

CODE_DIR = Path(__file__).resolve().parent.parent
DEFAULTS_DIR = Path(__file__).resolve().parent / "defaults"
# Temporary alias, removed once no caller uses it.
REPO_DIR = CODE_DIR

HOME_ENV = "SNOOZY_SCRIBE_HOME"
USER_DIR = (
    Path(os.environ[HOME_ENV]).expanduser()
    if os.environ.get(HOME_ENV)
    else Path.home() / "Library/Application Support" / identity.SUPPORT_DIR_NAME
)
SETTINGS_DIR = USER_DIR / "settings"
VOCABULARY_DIR = SETTINGS_DIR / "vocabulary"
# Only user overrides; the defaults are in DEFAULTS_DIR / "prompts".
PROMPTS_DIR = SETTINGS_DIR / "prompts"
DATA_DIR = USER_DIR / "data"
PARTIAL_CACHE_DIR = DATA_DIR / "cache" / "partial"
DEFAULT_CONFIG_PATH = SETTINGS_DIR / "config.yaml"

DEFAULT_MODEL_DIR = "./data/models"
DEFAULT_INPUT_DIR = "./data/input"
DEFAULT_OUTPUT_DIR = "~/Documents/snoozy-scribe/transcripts"


def resolve(path: str | Path) -> Path:
    """Expand `~`, then return `path` unchanged if absolute, else joined onto USER_DIR."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else USER_DIR / path


def ensure_user_dirs() -> None:
    """Create the user folders and seed the config and example vocabulary.

    Copies a default only when its destination is missing; never overwrites.
    Prompts aren't seeded, so new built-in prompts keep reaching the user.
    """
    for directory in (VOCABULARY_DIR, PROMPTS_DIR, DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    seeds = (
        (DEFAULTS_DIR / "config.example.yaml", DEFAULT_CONFIG_PATH),
        (DEFAULTS_DIR / "vocabulary.example.md", VOCABULARY_DIR / "vocabulary.example.md"),
    )
    for source, dest in seeds:
        if not dest.exists():
            shutil.copyfile(source, dest)


def prompt_path(name: str) -> Path:
    """The user's override of prompt `name` if there is one, else the built-in default."""
    override = PROMPTS_DIR / f"{name}.md"
    return override if override.exists() else DEFAULTS_DIR / "prompts" / f"{name}.md"
