"""Local model directory resolution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from huggingface_hub import snapshot_download


def resolve_model_path(
    model_repo: str, model_dir: Path, on_download: Callable[[], None] | None = None
) -> Path:
    """Resolve (downloading if needed) the local directory for a model repo.

    `on_download` is called just before a download starts, so callers can
    announce it.
    """
    target_dir = model_dir / model_repo.split("/")[-1]
    if not target_dir.exists() or not any(target_dir.iterdir()):
        if on_download is not None:
            on_download()
        snapshot_download(repo_id=model_repo, local_dir=str(target_dir))
    return target_dir
