#!/usr/bin/env bash
# Verifies that ruvector-wasm's `lattice-simd` feature actually vectorizes on
# wasm32: builds the crate with and without `-C target-feature=+simd128` and
# counts SIMD128 opcodes in each emitted .wasm artifact. Fails closed on any
# missing prerequisite, missing/empty artifact, build failure, grep failure,
# or unmet opcode condition.
set -euo pipefail

PACKAGE="ruvector-wasm"
RUST_TARGET="wasm32-unknown-unknown"
FEATURE="lattice-simd"
ARTIFACT="ruvector_wasm.wasm"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_TARGET_DIR="${CARGO_TARGET_DIR:-target}/wasm-simd-check"
SIMD_TARGET_DIR="$BASE_TARGET_DIR/simd128"
CONTROL_TARGET_DIR="$BASE_TARGET_DIR/control"

echo "== Prerequisites =="

if ! rustup target list --installed 2>/dev/null | grep -qx "$RUST_TARGET"; then
  echo "FAIL: rust target '$RUST_TARGET' is not installed." >&2
  echo "      Install it with: rustup target add $RUST_TARGET" >&2
  exit 1
fi
echo "OK: $RUST_TARGET target installed"

if ! command -v wasm-objdump >/dev/null 2>&1; then
  echo "FAIL: wasm-objdump not found on PATH (ships with WABT)." >&2
  echo "      Install it with: brew install wabt (macOS) or apt-get install wabt (Linux)." >&2
  exit 1
fi
echo "OK: wasm-objdump found at $(command -v wasm-objdump)"

count_simd128_opcodes() {
  local wasm_file="$1"
  if [[ ! -s "$wasm_file" ]]; then
    echo "FAIL: expected wasm artifact at '$wasm_file' but it is missing or empty." >&2
    exit 1
  fi

  local disasm
  if ! disasm="$(wasm-objdump -d "$wasm_file")"; then
    echo "FAIL: wasm-objdump could not disassemble '$wasm_file'." >&2
    exit 1
  fi

  # grep -c exits 1 on zero matches (a legitimate count: the control arm is
  # expected to land here) and 0 on a match; capture the real status
  # ourselves rather than let `|| true` fold every other status (2+: a real
  # grep failure) into the same "zero" bucket a broken grep invocation would
  # otherwise be indistinguishable from a genuine zero-opcode build.
  local count grep_status
  count="$(grep -Ec '\b(v128|i8x16|i16x8|i32x4|i64x2|f32x4|f64x2)\.[a-z_0-9]+' <<<"$disasm")" && grep_status=0 || grep_status=$?
  if (( grep_status != 0 && grep_status != 1 )); then
    echo "FAIL: grep exited $grep_status while counting SIMD128 opcodes for '$wasm_file'." >&2
    exit 1
  fi
  if ! [[ "$count" =~ ^[0-9]+$ ]]; then
    echo "FAIL: grep produced a non-numeric SIMD128 opcode count ('$count') for '$wasm_file'." >&2
    exit 1
  fi

  echo "$count"
}

build_artifact() {
  local target_dir="$1"
  local rustflags="$2"
  local artifact_path="$target_dir/$RUST_TARGET/release/$ARTIFACT"

  # Delete any artifact left over from a previous run of this script before
  # building: target dirs persist across invocations, so a failed build must
  # not be able to fall through to a stale non-empty artifact satisfying the
  # downstream "exists and is non-empty" check.
  if ! rm -f "$artifact_path"; then
    echo "FAIL: could not remove stale artifact '$artifact_path'." >&2
    return 1
  fi

  # Check the build's exit status explicitly instead of leaning on `set -e`:
  # this call sits inside a function invoked via command substitution
  # (`X="$(build_artifact ...)"`), and a failing non-final command in that
  # position does not reliably abort the script under `errexit` — only the
  # function's own final exit status, captured here, does.
  if ! CARGO_TARGET_DIR="$target_dir" RUSTFLAGS="$rustflags" \
      cargo build --release -p "$PACKAGE" --target "$RUST_TARGET" --features "$FEATURE" 1>&2; then
    echo "FAIL: cargo build failed (CARGO_TARGET_DIR=$target_dir, RUSTFLAGS='$rustflags')." >&2
    return 1
  fi

  echo "$artifact_path"
}

echo ""
echo "== Arm A: RUSTFLAGS='-C target-feature=+simd128', --features $FEATURE =="
if ! SIMD_WASM="$(build_artifact "$SIMD_TARGET_DIR" "-C target-feature=+simd128")"; then
  echo "FAIL: build_artifact failed for the +simd128 arm." >&2
  exit 1
fi
SIMD_COUNT="$(count_simd128_opcodes "$SIMD_WASM")"
if ! [[ "$SIMD_COUNT" =~ ^[0-9]+$ ]]; then
  echo "FAIL: non-numeric SIMD128 opcode count for the +simd128 arm: '$SIMD_COUNT'." >&2
  exit 1
fi
echo "SIMD128 opcode count: $SIMD_COUNT"
if (( SIMD_COUNT <= 0 )); then
  echo "FAIL: expected > 0 SIMD128 opcodes with +simd128 and --features $FEATURE, got $SIMD_COUNT." >&2
  exit 1
fi

echo ""
echo "== Arm B (control): no target-feature flag, --features $FEATURE =="
if ! CONTROL_WASM="$(build_artifact "$CONTROL_TARGET_DIR" "")"; then
  echo "FAIL: build_artifact failed for the control arm." >&2
  exit 1
fi
CONTROL_COUNT="$(count_simd128_opcodes "$CONTROL_WASM")"
if ! [[ "$CONTROL_COUNT" =~ ^[0-9]+$ ]]; then
  echo "FAIL: non-numeric SIMD128 opcode count for the control arm: '$CONTROL_COUNT'." >&2
  exit 1
fi
echo "SIMD128 opcode count: $CONTROL_COUNT"

# Without the target-feature flag, lattice-embed's wasm32 kernels take their
# scalar fallback (crates/ruvector-core/src/distance.rs), so this arm
# currently measures 0 SIMD128 opcodes. A future dependency could
# legitimately contribute some vector code even without the flag, so this
# asserts the delta direction (control strictly below the +simd128 build)
# rather than hard-coding zero.
if (( CONTROL_COUNT >= SIMD_COUNT )); then
  echo "FAIL: expected the control build to carry fewer SIMD128 opcodes than the +simd128 build (control=$CONTROL_COUNT, simd128=$SIMD_COUNT)." >&2
  exit 1
fi

echo ""
echo "== PASS =="
echo "+simd128 build: $SIMD_COUNT SIMD128 opcodes"
echo "control build:  $CONTROL_COUNT SIMD128 opcodes"
