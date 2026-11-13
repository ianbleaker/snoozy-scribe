"""The app's name, bundle ID, log path and signing identity: the one place
to change them.

Two builds share this module. The dev build (`make-app.sh` from the repo)
is "snoozy-scribe Dev"; the release build (`packaging/build-release-app.sh`)
is "snoozy-scribe". They have separate bundle IDs and certificates, so each
keeps its own permission grants. At runtime the release build is detected
from the enclosing bundle's Info.plist: its code sits in
`<app>/Contents/Resources/app/snoozy_scribe/`, so the transcribe.py child
gets the same answer as the app.

make-app.sh and setup.sh read these through
`.venv/bin/python -m snoozy_scribe.identity --shell` (`--shell --release` for
the release values), so this module imports only the standard library.
"""

import plistlib
import shlex
import sys
from pathlib import Path

# Release build. RELEASE_BUNDLE_ID keys every user's permission grants for
# good: never change it after the first release (docs/releasing.md).
RELEASE_APP_NAME = "snoozy-scribe"
RELEASE_BUNDLE_ID = "io.github.ianbleaker.snoozy-scribe"
RELEASE_SIGNING_IDENTITY = "snoozy-scribe Release"

# Dev build. The bundle ID is unchanged from before the split, so existing
# grants survive.
DEV_APP_NAME = "snoozy-scribe Dev"
DEV_BUNDLE_ID = "local.snoozy-scribe.gui"
# Common name of the self-signed certificate setup.sh creates in the login
# keychain; make-app.sh signs with it so permission grants survive rebuilds.
# A literal, not derived from the app name, so existing certificates match.
DEV_SIGNING_IDENTITY = "snoozy-scribe Code Signing"

# The Application Support folder for settings and data. Fixed: dev and release
# builds share it (one copy of the models, one set of vocabularies), so it
# doesn't follow APP_NAME.
SUPPORT_DIR_NAME = "snoozy-scribe"


def _running_release_build() -> bool:
    """True when this code sits inside a release bundle's Contents/Resources/app."""
    info = Path(__file__).resolve().parents[3] / "Info.plist"
    try:
        with info.open("rb") as f:
            return plistlib.load(f).get("CFBundleIdentifier") == RELEASE_BUNDLE_ID
    except (OSError, plistlib.InvalidFileException):
        return False


IS_RELEASE = _running_release_build()

APP_NAME = RELEASE_APP_NAME if IS_RELEASE else DEV_APP_NAME
BUNDLE_ID = RELEASE_BUNDLE_ID if IS_RELEASE else DEV_BUNDLE_ID
SIGNING_IDENTITY = RELEASE_SIGNING_IDENTITY if IS_RELEASE else DEV_SIGNING_IDENTITY
# Both bundle IDs, for checks that only care about running as an app bundle.
BUNDLE_IDS = (DEV_BUNDLE_ID, RELEASE_BUNDLE_ID)


def log_path_suffix(app_name: str) -> str:
    """The log path relative to $HOME; the C launcher appends it at runtime."""
    return f"/Library/Logs/{app_name}.log"


LOG_PATH_SUFFIX = log_path_suffix(APP_NAME)
LOG_PATH = Path.home() / LOG_PATH_SUFFIX.lstrip("/")


def _shell_vars(release: bool) -> dict[str, str]:
    name = RELEASE_APP_NAME if release else DEV_APP_NAME
    return {
        "APP_NAME": name,
        "BUNDLE_ID": RELEASE_BUNDLE_ID if release else DEV_BUNDLE_ID,
        "LOG_PATH_SUFFIX": log_path_suffix(name),
        "SIGNING_IDENTITY": RELEASE_SIGNING_IDENTITY if release else DEV_SIGNING_IDENTITY,
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    if args not in (["--shell"], ["--shell", "--release"]):
        sys.exit("usage: python -m snoozy_scribe.identity --shell [--release]")
    for key, value in _shell_vars(release=args == ["--shell", "--release"]).items():
        print(f"{key}={shlex.quote(value)}")
