#!/usr/bin/env bash
# Release build of the wasm fallback, used by .github/workflows/build-kge.yml.
#
#   npm/packages/kge/wasm/          (wasm-bindgen --target nodejs, loaded by index.js)
#   npm/packages/kge/wasm/bundler/  (wasm-bindgen --target bundler, for webpack/vite)
#
# Differs from build-wasm.sh (the local-dev path) in two ways:
#   1. It calls cargo + wasm-bindgen-cli directly instead of wasm-pack, so CI
#      needs no curl|sh installer. The CLI version MUST equal the wasm-bindgen
#      crate version in Cargo.lock, or wasm-bindgen refuses the module; the
#      script checks this up front and fails loudly.
#   2. It writes no nested .gitignore (wasm-pack drops a `*` .gitignore into
#      its out-dir, which npm honours inside a "files" directory and would ship
#      an empty wasm/).
# No wasm-opt pass: the module is functional, only larger than a wasm-pack -Oz
# build.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"          # npm/packages/kge
root="$(cd "$here/../../.." && pwd)"              # repo root
target_dir="${CARGO_TARGET_DIR:-$root/target}"
wasm_in="$target_dir/wasm32-unknown-unknown/release/ruvector_kge_wasm.wasm"

lock_ver="$(awk '/^name = "wasm-bindgen"$/{getline; gsub(/version = |"/,""); print; exit}' "$root/Cargo.lock")"
if [ -z "$lock_ver" ]; then
  echo "could not read the wasm-bindgen version from Cargo.lock" >&2
  exit 1
fi
if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "wasm-bindgen not found; install with: cargo install wasm-bindgen-cli --version $lock_ver --locked" >&2
  exit 1
fi
cli_ver="$(wasm-bindgen --version | awk '{print $2}')"
if [ "$cli_ver" != "$lock_ver" ]; then
  echo "wasm-bindgen-cli $cli_ver != wasm-bindgen $lock_ver in Cargo.lock" >&2
  exit 1
fi

# Global RUSTFLAGS on some hosts force -fuse-ld=<linker>, which rust-lld (the
# wasm linker) rejects. Scope rustflags for wasm32 to empty.
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS=""
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS

cargo build -p ruvector-kge-wasm --release --target wasm32-unknown-unknown \
  --manifest-path "$root/Cargo.toml"

rm -rf "$here/wasm"
mkdir -p "$here/wasm/bundler"
wasm-bindgen --target nodejs  --out-dir "$here/wasm"         "$wasm_in"
wasm-bindgen --target bundler --out-dir "$here/wasm/bundler" "$wasm_in"

# ADR-005 (c): no WASI / fs / net / fetch imports.
node "$here/scripts/check-wasm-imports.mjs" "$here/wasm/ruvector_kge_wasm_bg.wasm"

# Smoke: the nodejs glue loads and answers version().
node -e "const w=require(process.argv[1]); if(typeof w.version!=='function'||typeof w.Model!=='function'){console.error('wasm glue exports: '+Object.keys(w).join(','));process.exit(1)}; console.log('wasm loads OK: '+w.version())" \
  "$here/wasm/ruvector_kge_wasm.js"

echo "wasm release build complete -> $here/wasm"
