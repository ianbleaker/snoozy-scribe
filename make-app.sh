#!/bin/sh
# Build the app bundle, a double-clickable macOS launcher for the dictate.py
# menu-bar app (its menu opens the gui.py session transcriber). The name,
# bundle ID and log path come from rpg_scribe/identity.py.
#
# The bundle's executable is a small compiled launcher with this repo's
# absolute path baked in, so the .app can be moved anywhere (Applications, Dock,
# Desktop). Rerun this script if the repo moves. setup.sh runs it and installs
# the result; see docs/development.md.
set -eu

REPO_DIR=$(cd "$(dirname "$0")" && pwd)

if [ ! -x "$REPO_DIR/.venv/bin/python" ]; then
    echo "error: $REPO_DIR/.venv not found. Run ./setup.sh first." >&2
    exit 1
fi

# -m finds the package from the current directory, hence the cd.
eval "$(cd "$REPO_DIR" && .venv/bin/python -m rpg_scribe.identity --shell)"
APP="$REPO_DIR/$APP_NAME.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>$APP_NAME records your voice for dictation.</string>
</dict>
</plist>
EOF

# The executable embeds Python rather than exec'ing .venv/bin/python, so
# macOS credits permissions to the app and not to Python. See
# rpg_scribe/launcher/rpg-scribe.c. It links the venv's base Python, so
# rerun ./setup.sh after a Homebrew Python upgrade (and re-grant the
# permissions afterwards, since the ad-hoc signature changes).
PY_CONFIG="$("$REPO_DIR/.venv/bin/python" -c 'import sys; print(sys._base_executable)')-config"
# shellcheck disable=SC2046
clang -o "$APP/Contents/MacOS/$APP_NAME" "$REPO_DIR/rpg_scribe/launcher/rpg-scribe.c" \
    $("$PY_CONFIG" --cflags) $("$PY_CONFIG" --ldflags --embed) \
    -DREPO_DIR="\"$REPO_DIR\"" -DLOG_PATH_SUFFIX="\"$LOG_PATH_SUFFIX\""

# Ad-hoc sign so macOS can attach Microphone, Input Monitoring and
# Accessibility permissions to a stable identity.
codesign --force --deep -s - "$APP"

echo "Built $APP"
echo "Run ./setup.sh to install it to ~/Applications, or drag it there to launch $APP_NAME with a click. It runs in the menu bar."
