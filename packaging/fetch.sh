# Sourced, not run, with PACKAGING_DIR set. Downloads the pinned build inputs
# in packaging/versions.env into a cache and verifies their sha256.

. "$PACKAGING_DIR/versions.env"
BUILD_CACHE="${SNOOZY_SCRIBE_BUILD_CACHE:-$HOME/Library/Caches/snoozy-scribe-build}"

# fetch_input <PYTHON|UV|SPARKLE>: prints the path of the verified download.
# A cached file whose hash doesn't match is downloaded again; a fresh
# download that doesn't match is an error.
fetch_input() {
    case "$1" in
        PYTHON) url=$PYTHON_URL want=$PYTHON_SHA256 ;;
        UV) url=$UV_URL want=$UV_SHA256 ;;
        SPARKLE) url=$SPARKLE_URL want=$SPARKLE_SHA256 ;;
        *) echo "error: unknown input $1" >&2; return 1 ;;
    esac
    file="$BUILD_CACHE/$(basename "$url")"
    mkdir -p "$BUILD_CACHE"
    if [ ! -f "$file" ] || ! echo "$want  $file" | shasum -a 256 -c - >/dev/null 2>&1; then
        echo "Downloading $url" >&2
        curl -sSfL -o "$file.part" "$url" || { echo "error: download failed: $url" >&2; return 1; }
        if ! echo "$want  $file.part" | shasum -a 256 -c - >/dev/null 2>&1; then
            echo "error: sha256 mismatch for $url (expected $want); refusing it" >&2
            mv -f "$file.part" "$file.bad"
            return 1
        fi
        mv -f "$file.part" "$file"
    fi
    echo "$file"
}
