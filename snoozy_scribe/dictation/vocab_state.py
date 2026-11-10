"""Dictation's vocabulary: the selected vocabulary files, reloaded on change.

Any number of vocabularies can be ticked in the menu bar. The selection is
persisted as `vocabularies` in data/dictation-state.json, so it survives
restarts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from snoozy_scribe.dictation.state import load_state, update_state
from snoozy_scribe.vocabulary import (
    VocabularyParseError,
    list_vocabularies,
    parse_vocabulary,
    vocabulary_path,
)


def get_selected_vocabularies() -> list[str]:
    """The persisted selection, sorted. Corrupt entries are dropped.

    Migrates the old single-choice `campaign` key on first read: dictation
    used to apply base.md (when present) plus that one file.
    """
    state = load_state()
    if "vocabularies" not in state:
        legacy = state.get("campaign")
        if isinstance(legacy, str) and legacy:
            names = {legacy}
            if vocabulary_path("base").exists():
                names.add("base")
            selected = sorted(names)
            update_state(vocabularies=selected, campaign=None)
            return selected
    names = state.get("vocabularies")
    if not isinstance(names, list):
        return []
    return sorted({n for n in names if isinstance(n, str) and n})


def set_vocabulary_selected(name: str, selected: bool) -> None:
    names = set(get_selected_vocabularies())
    if selected:
        names.add(name)
    else:
        names.discard(name)
    update_state(vocabularies=sorted(names))


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

    def current_vocabs(self) -> list[dict]:
        """Parsed selected vocabularies whose file exists, alphabetically."""
        available = set(list_vocabularies())
        return [
            self._load(vocabulary_path(name))
            for name in get_selected_vocabularies()
            if name in available
        ]
