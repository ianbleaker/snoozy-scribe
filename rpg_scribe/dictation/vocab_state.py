"""Dictation's vocabulary: base.md plus the active campaign, reloaded on change.

The active campaign is chosen from the menu bar and persisted in
data/dictation-state.json, so it survives restarts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from rpg_scribe import paths
from rpg_scribe.vocabulary import VocabularyParseError, campaign_path, parse_vocabulary

STATE_PATH = paths.DATA_DIR / "dictation-state.json"
BASE_PATH = paths.VOCABULARY_DIR / "base.md"


def get_active_campaign() -> str | None:
    """The persisted campaign name, or None if unset, missing or corrupt."""
    try:
        campaign = json.loads(STATE_PATH.read_text()).get("campaign")
    except (OSError, ValueError, AttributeError):
        return None
    return campaign if isinstance(campaign, str) and campaign else None


def set_active_campaign(name: str | None) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({"campaign": name}) + "\n")


class VocabState:
    """Parses vocabulary files lazily and re-parses one only when its mtime changes.

    A file that fails to parse keeps its last good parse. `notify` is called
    once per bad version of the file with a `file:line: problem` message.
    """

    def __init__(self, notify: Callable[[str], None]) -> None:
        self._notify = notify
        # path -> (mtime of the last good parse, parsed vocabulary)
        self._good: dict[Path, tuple[float, dict]] = {}
        self._bad_mtimes: dict[Path, float] = {}

    def _load(self, path: Path) -> dict:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return {}
        cached = self._good.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        if self._bad_mtimes.get(path) == mtime:
            return cached[1] if cached else {}
        try:
            vocab = parse_vocabulary(path)
        except VocabularyParseError as e:
            self._bad_mtimes[path] = mtime
            self._notify(f"Vocabulary not reloaded: {e}")
            return cached[1] if cached else {}
        self._good[path] = (mtime, vocab)
        return vocab

    def current_vocab(self) -> dict:
        """`{"base": ..., "campaign": ...}`, kept separate for prompt ordering."""
        campaign = get_active_campaign()
        return {
            "base": self._load(BASE_PATH),
            "campaign": self._load(campaign_path(campaign)) if campaign else {},
        }
