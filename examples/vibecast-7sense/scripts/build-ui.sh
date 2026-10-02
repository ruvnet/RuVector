#!/usr/bin/env bash
#
# Builds the self-contained acoustic manifold page.
#
# The page runs under a strict CSP that blocks every external request, so the
# compiled WebAssembly module cannot be fetched at runtime -- it is base64'd
# and inlined into the HTML instead. This script performs that substitution,
# keeping the checked-in source free of a 400 KB blob.
#
# Usage: scripts/build-ui.sh [output-path]

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

source_html="ui/index.html"
output="${1:-ui/dist/index.html}"
wasm="target/wasm32-unknown-unknown/release/sevensense_wasm.wasm"

if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
  echo "error: the wasm32-unknown-unknown target is not installed." >&2
  echo "       run: rustup target add wasm32-unknown-unknown" >&2
  exit 1
fi

echo "building sevensense-wasm for wasm32-unknown-unknown..."
cargo build -p sevensense-wasm --release --target wasm32-unknown-unknown

if [[ ! -f "$wasm" ]]; then
  echo "error: expected module at $wasm but it was not produced." >&2
  exit 1
fi

mkdir -p "$(dirname "$output")"

# Python rather than sed: the base64 payload is far past sed's line-length
# comfort zone, and a literal replacement avoids escaping the whole blob.
python3 - "$source_html" "$wasm" "$output" <<'PY'
import base64, sys

source, module, destination = sys.argv[1], sys.argv[2], sys.argv[3]

html = open(source, encoding="utf-8").read()
payload = base64.b64encode(open(module, "rb").read()).decode("ascii")

token = "%%WASM_BASE64%%"
if token not in html:
    raise SystemExit(f"error: {source} has no {token} placeholder")

open(destination, "w", encoding="utf-8").write(html.replace(token, payload))

raw_kb = len(payload) * 3 / 4 / 1024
print(f"  module   {raw_kb:>8.0f} KB")
print(f"  encoded  {len(payload) / 1024:>8.0f} KB")
PY

size_kb=$(( $(wc -c < "$output") / 1024 ))
printf "  page     %8d KB  ->  %s\n" "$size_kb" "$output"

# The artifact host rejects anything past 16 MB, so fail here rather than there.
if (( size_kb > 16000 )); then
  echo "error: page exceeds the 16 MB artifact limit." >&2
  exit 1
fi
