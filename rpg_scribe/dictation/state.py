"""Menu-bar choices that survive restarts, kept in data/dictation-state.json.

Several features share the file (active campaign, hotkey overrides), so
writes merge into what's there rather than replacing it.
"""

from __future__ import annotations

import json

from rpg_scribe import paths

STATE_PATH = paths.DATA_DIR / "dictation-state.json"


def load_state() -> dict:
    """The whole state file, or {} if missing or corrupt."""
    try:
        state = json.loads(STATE_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def update_state(**changes) -> None:
    """Merge `changes` into the state file; a value of None removes that key."""
    state = load_state()
    for key, value in changes.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")
