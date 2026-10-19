#!/bin/sh
# Install the rc image compressor from GitHub Releases.
#
#   curl -fsSL https://raw.githubusercontent.com/coder-11125/rust-compress/main/install.sh | sh
#
# Environment variables:
#   RC_VERSION      Release tag to install, e.g. v0.1.0 (default: latest)
#   RC_INSTALL_DIR  Directory to install the binary into
#                   (default: /usr/local/bin when writable, otherwise ~/.local/bin)
set -eu

REPO="coder-11125/rust-compress"
BIN="rc"

info() { printf '%s\n' "$*" >&2; }
err() { printf 'error: %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || err "required command not found: $1"; }
need curl
need tar
need uname

case $(uname -s) in
    Linux) os=unknown-linux-gnu ;;
    Darwin) os=apple-darwin ;;
    *)
        err "unsupported operating system: $(uname -s) (Windows users: download the .zip asset from the releases page)"
        ;;
esac

case $(uname -m) in
    x86_64 | amd64) arch=x86_64 ;;
    arm64 | aarch64) arch=aarch64 ;;
    *) err "unsupported architecture: $(uname -m)" ;;
esac

target="${arch}-${os}"
version=${RC_VERSION:-latest}

if [ "$version" = latest ]; then
    # The "latest" URL redirects to /releases/tag/<tag>; take the last path component.
    url=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest") ||
        err "could not reach GitHub to resolve the latest release"
    version=${url##*/}
    [ -n "$version" ] && [ "$version" != latest ] ||
        err "could not determine the latest release tag"
fi

asset="rust-compress-${version}-${target}.tar.gz"
download="https://github.com/$REPO/releases/download/$version/$asset"

if [ -n "${RC_INSTALL_DIR:-}" ]; then
    dir=$RC_INSTALL_DIR
elif [ -d /usr/local/bin ] && [ -w /usr/local/bin ]; then
    dir=/usr/local/bin
else
    dir="$HOME/.local/bin"
fi

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d ' ' -f 1
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    else
        err "no SHA-256 tool found (need sha256sum, shasum or openssl)"
    fi
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

info "Downloading $asset ..."
curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 "$download" -o "$tmp/$asset" ||
    err "download failed: $download"

if curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 "$download.sha256" -o "$tmp/$asset.sha256" 2>/dev/null; then
    expected=$(cut -d ' ' -f 1 "$tmp/$asset.sha256")
    actual=$(sha256 "$tmp/$asset")
    [ "$expected" = "$actual" ] || err "checksum mismatch for $asset"
else
    info "warning: no checksum published for $asset; skipping verification"
fi

mkdir -p "$dir" || err "could not create $dir"
tar -xzf "$tmp/$asset" -C "$tmp"
src="$tmp/${asset%.tar.gz}/$BIN"
[ -f "$src" ] || src=$(find "$tmp" -type f -name "$BIN" -print | head -n 1)
[ -n "$src" ] && [ -f "$src" ] || err "archive did not contain the $BIN binary"

if ! cp "$src" "$dir/$BIN" 2>/dev/null; then
    err "could not write to $dir (set RC_INSTALL_DIR to a writable directory)"
fi
chmod 755 "$dir/$BIN"

info "Installed $BIN $version to $dir/$BIN"
case ":${PATH:-}:" in
    *":$dir:"*) ;;
    *) info "note: $dir is not on your PATH" ;;
esac
