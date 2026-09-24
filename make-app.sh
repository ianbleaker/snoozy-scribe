#!/bin/sh
# Build rpg-scribe.app, a double-clickable macOS launcher for gui.py.
#
# The bundle's executable is a tiny shell script with this repo's absolute
# path baked in, so the .app can be moved anywhere (Applications, Dock,
# Desktop). Rerun this script if the repo moves. See README.md.
set -eu

REPO_DIR=$(cd "$(dirname "$0")" && pwd)
APP="$REPO_DIR/rpg-scribe.app"

if [ ! -x "$REPO_DIR/.venv/bin/python" ]; then
    echo "error: $REPO_DIR/.venv not found. Set up the venv first (see README.md)." >&2
    exit 1
fi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"

cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>rpg-scribe</string>
    <key>CFBundleIdentifier</key>
    <string>local.rpg-scribe.gui</string>
    <key>CFBundleExecutable</key>
    <string>rpg-scribe</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
</dict>
</plist>
EOF

# Apps launched from Finder get a bare PATH (/usr/bin:/bin:...), which would
# hide Homebrew's ffmpeg from transcribe.py. Output goes to a log file since
# there's no terminal to show a startup crash.
cat > "$APP/Contents/MacOS/rpg-scribe" <<EOF
#!/bin/sh
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"
cd "$REPO_DIR"
exec "$REPO_DIR/.venv/bin/python" "$REPO_DIR/gui.py" >>"\$HOME/Library/Logs/rpg-scribe.log" 2>&1
EOF
chmod +x "$APP/Contents/MacOS/rpg-scribe"

echo "Built $APP"
echo "Drag it to the Dock or Applications to launch rpg-scribe with a click."
