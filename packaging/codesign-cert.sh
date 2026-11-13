# Sourced, not run. Shared by setup.sh (the dev certificate) and
# packaging/make-release-keys.sh (the release certificate).

# make_codesign_identity <common name> <dir> <p12 password>
# Writes a self-signed code-signing certificate and its key to <dir>/cert.pem
# and <dir>/identity.p12 (protected by <p12 password>). Uses Apple's
# /usr/bin/openssl (LibreSSL): Homebrew's OpenSSL 3 writes .p12 files that
# `security import` can't read.
make_codesign_identity() {
    /usr/bin/openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
        -subj "/CN=$1" -keyout "$2/key.pem" -out "$2/cert.pem" \
        -addext "basicConstraints=critical,CA:false" \
        -addext "keyUsage=critical,digitalSignature" \
        -addext "extendedKeyUsage=critical,codeSigning" >/dev/null 2>&1 || return 1
    /usr/bin/openssl pkcs12 -export -inkey "$2/key.pem" -in "$2/cert.pem" \
        -out "$2/identity.p12" -passout "pass:$3" || return 1
    rm -f "$2/key.pem"
}
