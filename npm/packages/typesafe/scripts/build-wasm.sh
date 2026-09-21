#!/usr/bin/env bash
# Build the wasm-bindgen fallback for both the nodejs and bundler targets:
#   npm/packages/typesafe/wasm/          (--target nodejs, loaded by index.js)
#   npm/packages/typesafe/wasm/bundler/  (--target bundler, for webpack/vite)
#
# The crate's [profile.release] (opt-level "z", lto, panic=abort) is used.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"          # npm/packages/typesafe
root="$(cd "$here/../../.." && pwd)"              # repo root
crate_dir="$root/crates/ruvector-typesafe-wasm"

# This host's global RUSTFLAGS forces `-C link-arg=-fuse-ld=mold`, which the
# wasm linker (rust-lld) rejects. mold is a host-native linker, irrelevant to
# wasm. Cargo reads RUSTFLAGS *before* any target-scoped override, so the mold
# flag must be stripped from RUSTFLAGS (and its encoded form) itself — only the
# mold arg is removed, anything else is preserved. Harmless on CI where the var
# is unset.
strip_mold() {
  printf '%s' "$1" | sed -E 's/-C[[:space:]]+link-arg=-fuse-ld=mold//g; s/-Clink-arg=-fuse-ld=mold//g'
}
if [ -n "${RUSTFLAGS:-}" ]; then
  export RUSTFLAGS="$(strip_mold "$RUSTFLAGS")"
fi
if [ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]; then
  # Encoded form is \x1f-separated; drop any unit that is the mold flag.
  export CARGO_ENCODED_RUSTFLAGS="$(
    printf '%s' "$CARGO_ENCODED_RUSTFLAGS" | tr '\037' '\n' \
      | grep -vxF -- '-fuse-ld=mold' | grep -vxF -- '-Clink-arg=-fuse-ld=mold' \
      | paste -sd '\037' -
  )"
fi

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack not found on PATH" >&2
  exit 1
fi

echo "building ruvector-typesafe-wasm (nodejs target) -> $here/wasm"
wasm-pack build "$crate_dir" --release --target nodejs --out-dir "$here/wasm"

echo "building ruvector-typesafe-wasm (bundler target) -> $here/wasm/bundler"
wasm-pack build "$crate_dir" --release --target bundler --out-dir "$here/wasm/bundler"

echo "wasm build complete"
