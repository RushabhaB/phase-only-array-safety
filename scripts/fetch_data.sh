#!/usr/bin/env bash
# Download the auxiliary data files that are NOT shipped inside this Git
# repository.
#
# Usage (from the release/ folder):
#
#     bash scripts/fetch_data.sh
#
# What this fetches:
#
#   1. Smat_36x36_90MHz.mat   (~517 MB)  -- required by antenna_array.Array
#   2. S_data_cube_Vivaldi36.h5 (~16 GB) -- required by scripts/run_imm.py
#
# Both are hosted as assets of a GitHub Release of this repository. The
# URL and checksum below need to be set after the release is published.

set -euo pipefail

REPO_RELEASE_URL="https://github.com/RushabhaB/phase-only-array-safety/releases/download/v0.1.0"
# expected SHA256 checksums
SMAT_SHA256="4704b49dec5d7e5735e73b5477ac4945c2ffc8d29a814b3ea40f7042191ae69b"
CUBE_SHA256=""   # fill in once the 16 GB cube is uploaded

DEST="$(dirname "$0")/../data/Data"
mkdir -p "$DEST"

fetch() {
    local name="$1"; local url="$2"; local sha="$3"; local out="$DEST/$name"
    if [[ -f "$out" ]]; then
        echo "[fetch] $name already present at $out -- skipping"
        return
    fi
    echo "[fetch] downloading $name from $url ..."
    curl -fL --progress-bar -o "$out" "$url"
    if [[ -n "$sha" ]]; then
        echo "[fetch] verifying SHA256 ..."
        actual=$(sha256sum "$out" | awk '{print $1}')
        if [[ "$actual" != "$sha" ]]; then
            echo "[fetch] FAIL: checksum mismatch"
            echo "  expected $sha"
            echo "  got      $actual"
            exit 1
        fi
        echo "[fetch] checksum OK"
    fi
}

fetch "Smat_36x36_90MHz.mat" \
      "$REPO_RELEASE_URL/Smat_36x36_90MHz.mat" \
      "$SMAT_SHA256"

if [[ -n "$CUBE_SHA256" ]]; then
    fetch "../S_data_cube_Vivaldi36.h5" \
          "$REPO_RELEASE_URL/S_data_cube_Vivaldi36.h5" \
          "$CUBE_SHA256"
else
    echo "[fetch] S_data_cube_Vivaldi36.h5 not configured -- skipping"
    echo "        (only needed for scripts/run_imm.py)"
fi

echo "[fetch] done"
