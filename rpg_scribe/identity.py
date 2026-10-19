"""The app's name, bundle ID and log path: the one place to change them.

make-app.sh and setup.sh read these through
`.venv/bin/python -m rpg_scribe.identity --shell`, so this module imports
only the standard library.
"""

import shlex
import sys
from pathlib import Path

APP_NAME = "rpg-scribe"
BUNDLE_ID = "local.rpg-scribe.gui"
LAUNCH_AGENT_LABEL = "local.rpg-scribe.dictate"
LOG_PATH = Path.home() / "Library/Logs" / f"{APP_NAME}.log"
# The C launcher appends this to $HOME at runtime.
LOG_PATH_SUFFIX = f"/Library/Logs/{APP_NAME}.log"

_SHELL_VARS = ("APP_NAME", "BUNDLE_ID", "LAUNCH_AGENT_LABEL", "LOG_PATH_SUFFIX")


if __name__ == "__main__":
    if sys.argv[1:] != ["--shell"]:
        sys.exit("usage: python -m rpg_scribe.identity --shell")
    for name in _SHELL_VARS:
        print(f"{name}={shlex.quote(globals()[name])}")
