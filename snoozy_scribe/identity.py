"""The app's name, bundle ID, log path and signing identity: the one place
to change them.

make-app.sh and setup.sh read these through
`.venv/bin/python -m snoozy_scribe.identity --shell`, so this module imports
only the standard library.
"""

import shlex
import sys
from pathlib import Path

APP_NAME = "snoozy-scribe"
BUNDLE_ID = "local.snoozy-scribe.gui"
# Launch at login used to be a hand-written LaunchAgent with this label; the
# app now registers itself with SMAppService and removes the old agent.
LAUNCH_AGENT_LABEL = "local.snoozy-scribe.dictate"
# The identity before the rename, so the app and setup.sh can clean up an
# install of the old name.
LEGACY_APP_NAME = "rpg-scribe"
LEGACY_BUNDLE_ID = "local.rpg-scribe.gui"
LEGACY_LAUNCH_AGENT_LABEL = "local.rpg-scribe.dictate"
# Common name of the self-signed certificate setup.sh creates in the login
# keychain; make-app.sh signs with it so permission grants survive rebuilds.
SIGNING_IDENTITY = f"{APP_NAME} Code Signing"
LOG_PATH = Path.home() / "Library/Logs" / f"{APP_NAME}.log"
# The C launcher appends this to $HOME at runtime.
LOG_PATH_SUFFIX = f"/Library/Logs/{APP_NAME}.log"

_SHELL_VARS = (
    "APP_NAME",
    "BUNDLE_ID",
    "LEGACY_APP_NAME",
    "LEGACY_BUNDLE_ID",
    "LOG_PATH_SUFFIX",
    "SIGNING_IDENTITY",
)


if __name__ == "__main__":
    if sys.argv[1:] != ["--shell"]:
        sys.exit("usage: python -m snoozy_scribe.identity --shell")
    for name in _SHELL_VARS:
        print(f"{name}={shlex.quote(globals()[name])}")
