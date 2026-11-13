#!/bin/sh
# One-time setup of the two release secrets. Run it yourself, once, on your
# Mac; never in CI. See docs/releasing.md.
#
# 1. The "snoozy-scribe Release" self-signed code-signing certificate, created
#    in the login keychain (never trusted, here or anywhere) and exported as
#    a password-protected .p12 for the CI secret. It is the app's identity:
#    macOS keys every user's permission grants to it. Losing it means every
#    user grants permissions again.
# 2. The Sparkle EdDSA key pair, stored in the login keychain by Sparkle's
#    generate_keys, with the private key exported for the CI secret. Losing
#    it is worse: installed copies reject every future update.
#
# The public Sparkle key is written to packaging/sparkle-public-key.txt
# (committed). Usage: packaging/make-release-keys.sh [output dir]
set -eu

PACKAGING_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(dirname "$PACKAGING_DIR")
. "$PACKAGING_DIR/codesign-cert.sh"
. "$PACKAGING_DIR/fetch.sh"

OUT="${1:-$HOME/snoozy-scribe-release-keys}"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"
SPARKLE_ACCOUNT=snoozy-scribe

eval "$(cd "$REPO_DIR" && python3 -m snoozy_scribe.identity --shell --release)"

if security find-certificate -c "$SIGNING_IDENTITY" "$KEYCHAIN" >/dev/null 2>&1; then
    echo "error: a \"$SIGNING_IDENTITY\" certificate already exists in your login keychain." >&2
    echo "Never replace it: every user would have to grant permissions again." >&2
    echo "To export it, use Keychain Access → My Certificates → Export." >&2
    exit 1
fi
if [ -e "$OUT" ]; then
    echo "error: $OUT already exists; pass another output folder." >&2
    exit 1
fi

printf 'Password for the .p12 (store it in your password manager): '
stty -echo
read -r P12_PASSWORD
stty echo
printf '\nAgain: '
stty -echo
read -r P12_CONFIRM
stty echo
echo
[ -n "$P12_PASSWORD" ] || { echo "error: empty password" >&2; exit 1; }
[ "$P12_PASSWORD" = "$P12_CONFIRM" ] || { echo "error: passwords differ" >&2; exit 1; }

mkdir -m 700 "$OUT"

# 1. Certificate. Imported so local release builds can sign; not trusted:
# nothing needs it trusted, and build-release-app.sh finds it untrusted.
echo "Creating the \"$SIGNING_IDENTITY\" certificate"
make_codesign_identity "$SIGNING_IDENTITY" "$OUT" "$P12_PASSWORD"
mv "$OUT/identity.p12" "$OUT/snoozy-scribe-release.p12"
rm -f "$OUT/cert.pem"
security import "$OUT/snoozy-scribe-release.p12" -k "$KEYCHAIN" -P "$P12_PASSWORD" \
    -T /usr/bin/codesign >/dev/null
echo "Imported it into your login keychain (untrusted)"

# 2. Sparkle keys. generate_keys keeps an existing key for the account rather
# than replacing it; macOS may ask to let it use the keychain.
SPARKLE_DIR="$BUILD_CACHE/sparkle-$SPARKLE_VERSION"
if [ ! -x "$SPARKLE_DIR/bin/generate_keys" ]; then
    archive=$(fetch_input SPARKLE)
    mkdir -p "$SPARKLE_DIR"
    tar -xJf "$archive" -C "$SPARKLE_DIR"
fi
echo "Creating the Sparkle EdDSA key (account \"$SPARKLE_ACCOUNT\")"
"$SPARKLE_DIR/bin/generate_keys" --account "$SPARKLE_ACCOUNT" >/dev/null
"$SPARKLE_DIR/bin/generate_keys" --account "$SPARKLE_ACCOUNT" -x "$OUT/sparkle-private-key.txt"
"$SPARKLE_DIR/bin/generate_keys" --account "$SPARKLE_ACCOUNT" -p > "$PACKAGING_DIR/sparkle-public-key.txt"
chmod 600 "$OUT"/*

cat <<EOF

Done. Back up now, before anything else (a password manager entry each):
  1. $OUT/snoozy-scribe-release.p12  (attach the file)
  2. The .p12 password you just typed
  3. $OUT/sparkle-private-key.txt  (its contents)

Then add the GitHub Actions secrets (docs/releasing.md):
  RELEASE_CERT_P12_BASE64   base64 -i "$OUT/snoozy-scribe-release.p12" | pbcopy
  RELEASE_CERT_PASSWORD     the .p12 password
  SPARKLE_PRIVATE_KEY       pbcopy < "$OUT/sparkle-private-key.txt"

Commit packaging/sparkle-public-key.txt. Once the backups are confirmed,
delete $OUT: both secrets also stay in your login keychain.
EOF
