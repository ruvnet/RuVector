#!/usr/bin/env bash
# Build the wasm-bindgen fallback for both the nodejs and bundler targets:
#   npm/packages/kge/wasm/          (--target nodejs, loaded by index.js)
#   npm/packages/kge/wasm/bundler/  (--target bundler, for webpack/vite)
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"          # npm/packages/kge
root="$(cd "$here/../../.." && pwd)"              # repo root
crate_dir="$root/crates/ruvector-kge-wasm"

# This host's global RUSTFLAGS forces a native linker (`-fuse-ld=mold` or
# `-fuse-ld=lld`), which rust-lld (the wasm linker) rejects — it is a C-driver
# flag, irrelevant to wasm. Cargo reads RUSTFLAGS *before* any target-scoped
# override, so the flag must be stripped from RUSTFLAGS itself. Only the
# `-fuse-ld=<linker>` arg is removed; anything else is preserved. Harmless on
# CI where the var is unset.
strip_fuse_ld() {
  printf '%s' "$1" | sed -E 's/-C[[:space:]]+link-arg=-fuse-ld=[A-Za-z0-9._-]+//g; s/-Clink-arg=-fuse-ld=[A-Za-z0-9._-]+//g'
}
if [ -n "${RUSTFLAGS:-}" ]; then
  export RUSTFLAGS="$(strip_fuse_ld "$RUSTFLAGS")"
fi
if [ -n "${CARGO_ENCODED_RUSTFLAGS:-}" ]; then
  export CARGO_ENCODED_RUSTFLAGS="$(
    printf '%s' "$CARGO_ENCODED_RUSTFLAGS" | tr '\037' '\n' \
      | grep -vE -- '-fuse-ld=[A-Za-z0-9._-]+' \
      | paste -sd '\037' -
  )"
fi
# Belt-and-braces: also clear the target-scoped wasm rustflags (the lead's env
# sets this empty already; keep it empty here so nothing leaks in).
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="${CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS:-}"

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack not found on PATH" >&2
  exit 1
fi

echo "building ruvector-kge-wasm (nodejs target) -> $here/wasm"
wasm-pack build "$crate_dir" --release --target nodejs --out-dir "$here/wasm"

echo "building ruvector-kge-wasm (bundler target) -> $here/wasm/bundler"
wasm-pack build "$crate_dir" --release --target bundler --out-dir "$here/wasm/bundler"

# ADR-005 (c) gate: fail the build if the module imports any WASI/fs/net symbol.
echo "checking wasm imports (ADR-005)"
node "$here/scripts/check-wasm-imports.mjs" "$here/wasm/ruvector_kge_wasm_bg.wasm"

echo "wasm build complete"
