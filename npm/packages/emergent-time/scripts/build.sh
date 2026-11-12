#!/usr/bin/env bash
# Build pipeline for @ruvector/emergent-time.
#
# wasm-pack's bundled `wasm-opt -O` rejects the toolchain's default bulk-memory /
# nontrapping-float-to-int opcodes, so we drive the three stages manually:
#   1. cargo build  (1.89 toolchain — the one with wasm32-unknown-unknown std)
#   2. wasm-bindgen (--target web)
#   3. wasm-opt -Oz (with the feature flags the toolchain emits)
# then copy the optimized artifacts into pkg/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$HERE/../pkg"
CRATE_DIR="$HERE/../../../../crates/emergent-time-wasm"
CRATE_DIR="$(cd "$CRATE_DIR" && pwd)"

# Use the caller's toolchain (it must have the wasm32-unknown-unknown target).
# Set RUSTUP_TOOLCHAIN to override; the old default was a Windows-only triple.
RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-}"
[ -n "$RUSTUP_TOOLCHAIN" ] && export RUSTUP_TOOLCHAIN
# Host linker flags (e.g. -fuse-ld=mold) do not apply to wasm32.
export RUSTFLAGS="${WASM_RUSTFLAGS:-}"

# Prefer a wasm-opt on PATH; otherwise fetch binaryen's through npx.
if command -v wasm-opt >/dev/null 2>&1; then WASM_OPT=(wasm-opt); else
  WASM_OPT=(npx -y -p binaryen wasm-opt); fi

WASM_FEATURES="--enable-bulk-memory --enable-bulk-memory-opt \
  --enable-nontrapping-float-to-int --enable-sign-ext --enable-mutable-globals \
  --enable-multivalue --enable-reference-types"

echo "[1/3] cargo build (RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-default})"
cargo build --release --target wasm32-unknown-unknown \
  --manifest-path "$CRATE_DIR/Cargo.toml"

RAW_WASM="$CRATE_DIR/target/wasm32-unknown-unknown/release/emergent_time_wasm.wasm"
# Some setups place target/ at the crate; others at the workspace. Resolve it.
if [ ! -f "$RAW_WASM" ]; then
  RAW_WASM="$(find "$CRATE_DIR" -path '*/wasm32-unknown-unknown/release/emergent_time_wasm.wasm' | head -1)"
fi

echo "[2/3] wasm-bindgen --target web"
mkdir -p "$PKG_DIR"
wasm-bindgen --target web --out-dir "$PKG_DIR" "$RAW_WASM"

echo "[3/3] wasm-opt -Oz"
RAW_BYTES=$(stat -c%s "$PKG_DIR/emergent_time_wasm_bg.wasm")
# shellcheck disable=SC2086
"${WASM_OPT[@]}" -Oz $WASM_FEATURES \
  "$PKG_DIR/emergent_time_wasm_bg.wasm" \
  -o "$PKG_DIR/emergent_time_wasm_bg.opt.wasm"
mv "$PKG_DIR/emergent_time_wasm_bg.opt.wasm" "$PKG_DIR/emergent_time_wasm_bg.wasm"
OPT_BYTES=$(stat -c%s "$PKG_DIR/emergent_time_wasm_bg.wasm")

echo "done: raw=${RAW_BYTES}B opt=${OPT_BYTES}B"
