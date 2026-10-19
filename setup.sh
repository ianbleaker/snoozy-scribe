#!/bin/sh
# Install or repair: builds .venv from requirements.lock, creates
# settings/config.yaml if missing, and builds and installs the app to
# ~/Applications. Safe to rerun, e.g. after `git pull` or a Homebrew Python
# upgrade. See README.md ("Install") and docs/development.md.
#
# The app is rebuilt only when it has to be (installed copy missing, venv
# recreated, or --rebuild-app), because every rebuild changes the ad-hoc
# signature and macOS then asks for the permissions again.
set -eu

usage() {
    echo "usage: ./setup.sh [--rebuild-app]" >&2
    exit 2
}

REBUILD=0
for arg in "$@"; do
    case "$arg" in
        --rebuild-app) REBUILD=1 ;;
        *) usage ;;
    esac
done

REPO_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$REPO_DIR"

die() {
    echo "error: $1" >&2
    exit 1
}

# 1-3. Platform and toolchain.
[ "$(uname -m)" = arm64 ] || die "this needs a Mac with Apple Silicon (MLX runs on its GPU)."
command -v brew >/dev/null 2>&1 || die "Homebrew not found. Install it from https://brew.sh, then rerun ./setup.sh."
xcode-select -p >/dev/null 2>&1 || die "Xcode command-line tools not found. Run: xcode-select --install"

# 4. Homebrew's Python: make-app.sh needs python3-config --embed from a
# framework build, which Apple's /usr/bin/python3 doesn't provide.
PY="$(brew --prefix)/bin/python3"
[ -x "$PY" ] || die "Homebrew Python not found. Run: brew install python"

# 5. The venv. A Homebrew Python upgrade can leave it pointing at a removed
# interpreter, so recreate it when it can't start.
VENV_RECREATED=0
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import sys' >/dev/null 2>&1; then
    echo "Creating .venv with $PY"
    rm -rf .venv
    "$PY" -m venv .venv
    VENV_RECREATED=1
fi

# 6. Dependencies. --no-deps skips torch, which mlx-whisper declares but
# doesn't need (docs/development.md).
echo "Installing dependencies from requirements.lock"
.venv/bin/pip install --quiet --disable-pip-version-check --no-deps -r requirements.lock

# 7. Config: never overwrite an existing one.
if [ ! -f settings/config.yaml ]; then
    cp settings/config.example.yaml settings/config.yaml
    echo "Created settings/config.yaml from settings/config.example.yaml"
fi

# 8. ffmpeg is only needed for session transcription.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "warning: ffmpeg not found; needed only for session transcription: brew install ffmpeg" >&2
fi

# 9. App name, bundle ID and log path (-m finds the package from the repo).
eval "$(.venv/bin/python -m rpg_scribe.identity --shell)"
INSTALLED="$HOME/Applications/$APP_NAME.app"

# 10-11. Build and install the app when needed.
if [ ! -d "$INSTALLED" ] || [ "$VENV_RECREATED" = 1 ] || [ "$REBUILD" = 1 ]; then
    if [ "$(osascript -e "application id \"$BUNDLE_ID\" is running" 2>/dev/null)" = true ]; then
        die "$APP_NAME is running. Quit $APP_NAME from its menu first, then rerun ./setup.sh."
    fi
    ./make-app.sh
    mkdir -p "$HOME/Applications"
    rm -rf "$INSTALLED"
    ditto "$APP_NAME.app" "$INSTALLED"
    # Ad-hoc grants are keyed to the old build's code hash. Left in place they
    # still show as switched on in System Settings but no longer apply, so
    # clear them and let macOS ask again.
    for service in Microphone ListenEvent Accessibility PostEvent; do
        tccutil reset "$service" "$BUNDLE_ID" >/dev/null 2>&1 || true
    done
    echo
    echo "Installed $INSTALLED"
    echo "On first launch, grant $APP_NAME these permissions (System Settings > Privacy & Security):"
    echo "  - Microphone, to record"
    echo "  - Input Monitoring, for the global hotkeys"
    echo "  - Accessibility, to paste and read the focused window title"
    echo "Every rebuild changes the app's signature, so the old grants were cleared; grant them again."
else
    echo "App up to date (pass --rebuild-app to rebuild)."
fi

# 12. A copy dragged to /Applications by the old install goes stale.
if [ -d "/Applications/$APP_NAME.app" ]; then
    echo
    echo "warning: an old copy is at /Applications/$APP_NAME.app. Delete it and use $INSTALLED." >&2
    if [ -f "$HOME/Library/LaunchAgents/$LAUNCH_AGENT_LABEL.plist" ]; then
        echo "Launch at login is on: toggle it off and on in the menu so it opens the ~/Applications copy." >&2
    fi
fi

# 13.
echo
echo "Open $APP_NAME from Spotlight, or run: open -a \"$INSTALLED\""
