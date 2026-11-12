"""Config loading."""

from __future__ import annotations

from pathlib import Path

import yaml


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
