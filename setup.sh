#!/bin/sh
# Install or repair: builds .venv from requirements.lock, and builds and
# installs the app to ~/Applications. Settings and data live in
# ~/Library/Application Support/snoozy-scribe/, created on first run. Safe to rerun, e.g. after `git pull` or a Homebrew Python
# upgrade. See README.md ("Install") and docs/development.md.
#
# The app is signed with a self-signed certificate this script creates in the
# login keychain, so permission grants survive rebuilds. It is rebuilt only
# when it has to be: installed copy missing or ad-hoc signed while the
# certificate exists, venv recreated, or --rebuild-app.
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
# interpreter, so recreate it when it can't start. Its scripts (pip
# included) hold the venv's absolute path, so also recreate it when the repo
# has moved. The comparison ignores case, like the default macOS filesystem.
venv_moved() {
    [ -f .venv/bin/pip ] || return 1
    expected=$(printf '#!%s/.venv/bin/' "$PWD" | tr '[:upper:]' '[:lower:]')
    case "$(head -n 1 .venv/bin/pip | tr '[:upper:]' '[:lower:]')" in
        "$expected"*) return 1 ;;
    esac
    return 0
}
VENV_RECREATED=0
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import sys' >/dev/null 2>&1 || venv_moved; then
    echo "Creating .venv with $PY"
    rm -rf .venv
    "$PY" -m venv .venv
    VENV_RECREATED=1
fi

# 6. Dependencies. --no-deps skips torch, which mlx-whisper declares but
# doesn't need (docs/development.md).
echo "Installing dependencies from requirements.lock"
.venv/bin/pip install --quiet --disable-pip-version-check --no-deps -r requirements.lock

# 7. ffmpeg is only needed for session transcription.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "warning: ffmpeg not found; needed only for session transcription: brew install ffmpeg" >&2
fi

# 8. App name, bundle ID and log path (-m finds the package from the repo).
eval "$(.venv/bin/python -m snoozy_scribe.identity --shell)"
INSTALLED="$HOME/Applications/$APP_NAME.app"
# Before the dev/release split the dev app was "snoozy-scribe.app". Same
# bundle ID and certificate, so its grants carry over to the renamed copy. A
# release build installed under that name has another bundle ID; leave it.
LEGACY="$HOME/Applications/snoozy-scribe.app"
legacy_is_dev() {
    [ -d "$LEGACY" ] && [ "$LEGACY" != "$INSTALLED" ] \
        && [ "$(defaults read "$LEGACY/Contents/Info" CFBundleIdentifier 2>/dev/null)" = "$BUNDLE_ID" ]
}

# 9. The code-signing certificate. macOS keys permission grants to the
# signature's designated requirement: for a certificate-signed app that names
# the certificate, so grants survive rebuilds; for an ad-hoc one it's the
# build's code hash. Created once; never regenerated, since a new certificate
# would lose the grants again. Without it make-app.sh signs ad-hoc.
have_signing_identity() {
    security find-identity -v -p codesigning 2>/dev/null | grep -qF "\"$SIGNING_IDENTITY\""
}

create_signing_identity() {
    keychain="$HOME/Library/Keychains/login.keychain-db"
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    if security find-certificate -c "$SIGNING_IDENTITY" "$keychain" >/dev/null 2>&1; then
        # Imported by an earlier run whose trust step didn't finish.
        security find-certificate -c "$SIGNING_IDENTITY" -p "$keychain" >"$tmp/cert.pem" || return 1
    else
        echo "Creating the \"$SIGNING_IDENTITY\" certificate in your login keychain"
        # Apple's LibreSSL: Homebrew's OpenSSL 3 writes .p12 files that
        # `security import` can't read.
        /usr/bin/openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
            -subj "/CN=$SIGNING_IDENTITY" -keyout "$tmp/key.pem" -out "$tmp/cert.pem" \
            -addext "basicConstraints=critical,CA:false" \
            -addext "keyUsage=critical,digitalSignature" \
            -addext "extendedKeyUsage=critical,codeSigning" >/dev/null 2>&1 || return 1
        /usr/bin/openssl pkcs12 -export -inkey "$tmp/key.pem" -in "$tmp/cert.pem" \
            -out "$tmp/identity.p12" -passout pass:snoozy-scribe || return 1
        # -T lets codesign use the key without a keychain prompt.
        security import "$tmp/identity.p12" -k "$keychain" -P snoozy-scribe \
            -T /usr/bin/codesign >/dev/null || return 1
    fi
    echo "Trusting it for code signing; macOS asks for your password once"
    security add-trusted-cert -r trustRoot -p codeSign -k "$keychain" "$tmp/cert.pem" || return 1
    have_signing_identity
}

if have_signing_identity || create_signing_identity; then
    HAVE_CERT=1
else
    HAVE_CERT=0
    echo "warning: couldn't set up the \"$SIGNING_IDENTITY\" certificate; the app will be signed ad-hoc," >&2
    echo "so every rebuild means granting its permissions again. See docs/development.md." >&2
fi

is_adhoc() {
    codesign -dv "$1" 2>&1 | grep -q '^Signature=adhoc'
}

designated_requirement() {
    codesign -d -r- "$1" 2>/dev/null | grep 'designated =>' || true
}

# 10-11. Build and install the app when needed.
if [ ! -d "$INSTALLED" ] || [ "$VENV_RECREATED" = 1 ] || [ "$REBUILD" = 1 ] \
    || { [ "$HAVE_CERT" = 1 ] && is_adhoc "$INSTALLED"; }; then
    if [ "$(osascript -e "application id \"$BUNDLE_ID\" is running" 2>/dev/null)" = true ]; then
        die "$APP_NAME is running. Quit $APP_NAME from its menu first, then rerun ./setup.sh."
    fi
    OLD_DR=""
    if [ -d "$INSTALLED" ]; then
        OLD_DR=$(designated_requirement "$INSTALLED")
    elif legacy_is_dev; then
        OLD_DR=$(designated_requirement "$LEGACY")
    fi
    ./make-app.sh
    mkdir -p "$HOME/Applications"
    rm -rf "$INSTALLED"
    ditto "$APP_NAME.app" "$INSTALLED"
    if legacy_is_dev; then
        rm -rf "$LEGACY"
        echo "Removed the old $LEGACY (renamed to $INSTALLED)."
        echo "If Launch at login was on, toggle it off and on from the menu once."
    fi
    # Grants made for a different designated requirement (an ad-hoc build's
    # code hash, or the first switch to the certificate) still show as
    # switched on in System Settings but no longer apply, so clear them and
    # let macOS ask again. Certificate rebuilds keep the same requirement.
    if [ "$OLD_DR" != "$(designated_requirement "$INSTALLED")" ] || is_adhoc "$INSTALLED"; then
        GRANTS_RESET=1
        for service in Microphone ListenEvent Accessibility PostEvent; do
            tccutil reset "$service" "$BUNDLE_ID" >/dev/null 2>&1 || true
        done
    else
        GRANTS_RESET=0
    fi
    echo
    echo "Installed $INSTALLED"
    if [ "$GRANTS_RESET" = 1 ]; then
        echo "On first launch, $APP_NAME opens a permissions window. Grant:"
        echo "  - Microphone, to record"
        echo "  - Accessibility, for the global hotkeys, pasting and the focused window title"
        if is_adhoc "$INSTALLED"; then
            echo "It's signed ad-hoc, so every rebuild clears these grants."
        else
            echo "It's signed with \"$SIGNING_IDENTITY\", so later rebuilds keep them."
        fi
    else
        echo "Same signing certificate as before, so the permission grants carry over."
    fi
else
    echo "App up to date (pass --rebuild-app to rebuild)."
fi

# 12. A copy dragged to /Applications by the old install goes stale.
if [ -d "/Applications/$APP_NAME.app" ]; then
    echo
    echo "warning: an old copy is at /Applications/$APP_NAME.app. Delete it and use $INSTALLED." >&2
    echo "If you turned Launch at login on from that copy, toggle it off and on from $INSTALLED." >&2
fi

# 13.
echo
echo "Open $APP_NAME from Spotlight, or run: open -a \"$INSTALLED\""
