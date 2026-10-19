"""Config loading and the old-layout startup check."""

from __future__ import annotations

from pathlib import Path

import yaml

from rpg_scribe.identity import APP_NAME
from rpg_scribe.paths import DEFAULT_CONFIG_PATH, REPO_DIR

# Old root-level location -> new location, for files and folders moved by
# the settings/ + data/ layout. Nothing is moved automatically.
_LEGACY_MOVES = [
    ("config.yaml", "settings/config.yaml"),
    ("vocabulary.md", "settings/vocabulary/<campaign>.md"),
    ("models", "data/models"),
    ("input", "data/input"),
    ("transcripts", "data/transcripts"),
]

# Directory keys whose old defaults now live under ./data/.
_LEGACY_DIR_VALUES = {
    "model_dir": "./models",
    "input_dir": "./input",
    "output_dir": "./transcripts",
}


def load_config(path: Path) -> dict:
    """Load a YAML config mapping. An empty file is `{}`.

    Exits with a message if the file is missing, isn't valid YAML, or isn't
    a mapping.
    """
    if not path.exists():
        raise SystemExit(
            f"error: config file not found at {path}. "
            "Run `cp settings/config.example.yaml settings/config.yaml` and fill in your values."
        )
    try:
        config = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise SystemExit(f"error: {path} is not valid YAML: {e}")
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise SystemExit(
            f"error: {path} must be a mapping of config keys. "
            "Copy settings/config.example.yaml to see the expected layout."
        )
    return config


def _read_mapping(path: Path) -> dict:
    try:
        config = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return config if isinstance(config, dict) else {}


def check_legacy_layout(config_path: Path = DEFAULT_CONFIG_PATH) -> list[str]:
    """Return instructions for migrating an old-layout checkout, or `[]` if clean.

    Checks for root-level files and folders that moved under settings/ or
    data/, and for old keys or directory values in the config at
    `config_path` (and a still-unmoved root config.yaml).
    """
    commands = []
    for old, new in _LEGACY_MOVES:
        if (REPO_DIR / old).exists():
            commands.append(f"  mv {old} {new}")
    config_notes = []
    config_paths = [config_path, REPO_DIR / "config.yaml"]
    for path in dict.fromkeys(p.resolve() for p in config_paths if p.exists()):
        config = _read_mapping(path)
        # A root config.yaml gets fixed after it's moved into settings/.
        label = "settings/config.yaml" if path == (REPO_DIR / "config.yaml").resolve() else str(path)
        if "vocabulary_file" in config:
            config_notes.append(
                f"  {label}: remove `vocabulary_file` (removed) and set `campaign: <campaign>`"
            )
        for key, old_value in _LEGACY_DIR_VALUES.items():
            value = config.get(key)
            if isinstance(value, str) and Path(value) == Path(old_value):
                config_notes.append(
                    f"  {label}: `{key}: {value}` is the old layout; remove the key "
                    f"or point it at ./data/{Path(old_value).name}"
                )
    if not commands and not config_notes:
        return []
    lines = [
        f"{APP_NAME} now keeps settings in settings/ and data in data/, and this "
        "checkout still uses the old layout. Nothing is moved automatically.",
    ]
    if commands:
        lines += [
            "",
            f"From {REPO_DIR}, run:",
            "  mkdir -p settings/vocabulary data",
            *commands,
        ]
        if any("<campaign>" in c for c in commands):
            lines.append(
                "Replace <campaign> with a short campaign name, and set "
                "`campaign: <campaign>` in settings/config.yaml."
            )
    if config_notes:
        lines += ["", "Then fix these config keys:", *config_notes]
    return lines
