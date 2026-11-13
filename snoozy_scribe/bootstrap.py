"""Release build entry point: install the libraries, then run dictate.py.

The release launcher (snoozy_scribe/launcher/snoozy-scribe.c, RELEASE_BUILD)
runs this file on the bundled Python, with only the stdlib, the app code
(Contents/Resources/app) and the bootstrap's PyObjC
(Contents/Resources/bootstrap) on sys.path. The libraries aren't shipped in
the bundle: on first launch, and whenever an update changes the lock, the
bundled `uv` syncs the hashed lock into
`<Application Support>/data/runtime/<lock sha256>/`.

A plain `--target` folder rather than a venv: a venv's pyvenv.cfg pins the
interpreter's absolute path, which would break when the user moves the app.
The install goes to `<hash>.partial/` and is renamed into place only when it
completes, so an interrupted install never becomes the live folder.

Imports are stdlib plus `snoozy_scribe.paths` (which imports only
`identity`, also stdlib): nothing else exists yet on first launch.
"""

from __future__ import annotations

import hashlib
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

from snoozy_scribe import paths

APP_DIR = Path(__file__).resolve().parent.parent  # Contents/Resources/app
RESOURCES_DIR = APP_DIR.parent
BOOTSTRAP_DIR = RESOURCES_DIR / "bootstrap"
PYTHON = RESOURCES_DIR / "python" / "bin" / "python3"
UV = RESOURCES_DIR / "uv"
LOCK = APP_DIR / "requirements.runtime.lock"
RUNTIME_ROOT = paths.DATA_DIR / "runtime"
UV_CACHE_DIR = paths.DATA_DIR / "cache" / "uv"


def log(message: str) -> None:
    # stdout is the app log (the launcher redirects it).
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} bootstrap: {message}", flush=True)


def lock_hash() -> str:
    return hashlib.sha256(LOCK.read_bytes()).hexdigest()


def runtime_dir() -> Path:
    return RUNTIME_ROOT / lock_hash()


def install_runtime(target: Path) -> None:
    """Sync the lock into `target` via `<target>.partial`; raise on failure."""
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        log(f"removing an interrupted install at {partial}")
        shutil.rmtree(partial)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, UV_CACHE_DIR=str(UV_CACHE_DIR))
    cmd = [
        str(UV), "pip", "sync",
        "--require-hashes",
        "--python", str(PYTHON),
        "--target", str(partial),
        "--no-progress",
        str(LOCK),
    ]
    log(f"installing libraries into {target}")
    started = time.monotonic()
    # Output goes straight to the app log (fds 1 and 2).
    result = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"uv pip sync failed with exit code {result.returncode}")
    os.replace(partial, target)
    log(f"libraries installed in {time.monotonic() - started:.0f}s")


def activate_runtime(runtime: Path) -> None:
    """Put the runtime on this process's path and on every child's."""
    sys.path.insert(0, str(runtime))
    # The transcription window runs transcribe.py on the bundled python3 as
    # a child process; it needs the same code, bootstrap and runtime folders.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        str(p) for p in (APP_DIR, BOOTSTRAP_DIR, runtime)
    )
    os.environ["PYTHONNOUSERSITE"] = "1"


def main() -> None:
    runtime = runtime_dir()
    if not runtime.is_dir():
        try:
            install_runtime(runtime)
        except Exception as e:
            log(f"error: {e}")
            sys.exit(f"snoozy-scribe couldn't install its libraries: {e}")
    activate_runtime(runtime)

    dictate = APP_DIR / "dictate.py"
    sys.argv = [str(dictate), *sys.argv[1:]]
    runpy.run_path(str(dictate), run_name="__main__")


if __name__ == "__main__":
    main()
