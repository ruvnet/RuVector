#!/usr/bin/env bash
# Build the native N-API binding and place it where index.js loads it:
#   npm/packages/typesafe/native/typesafe.<triple>.node
#
# We compile the cdylib with cargo and copy it, rather than invoking the napi
# CLI, on purpose: a napi-rs cdylib *is* the Node addon (the `.node` extension
# is just a rename), and copying it ourselves keeps the artifact name
# deterministic instead of depending on package.json's napi config (which is
# owned by another agent). The hash embedder is the default feature, so no
# extra flags are needed.
#
#   scripts/build-native.sh            # hash-only (default, shippable artifact)
#   scripts/build-native.sh --onnx     # + native ONNX (bge/MiniLM) for the bench
#
# The --onnx build statically links ONNX Runtime, whose operator-schema docs
# embed documentation URLs; check-security allowlists exactly those hosts
# (ADR-005), so this script runs check-security on whichever binary it built.
set -euo pipefail

features=""
label="hash-only"
for arg in "$@"; do
  case "$arg" in
    --onnx) features="--features native-onnx"; label="native-onnx" ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

here="$(cd "$(dirname "$0")/.." && pwd)"          # npm/packages/typesafe
root="$(cd "$here/../../.." && pwd)"              # repo root
crate="ruvector-typesafe-ffi"

# Map node's platform/arch to the napi triple used in index.js's platformMap.
# (console.log adds the trailing newline `read` needs to return success under set -e.)
read -r plat arch < <(node -e "console.log(process.platform, process.arch)")
case "$plat-$arch" in
  linux-x64)    triple="linux-x64-gnu";  libname="libruvector_typesafe_ffi.so" ;;
  linux-arm64)  triple="linux-arm64-gnu"; libname="libruvector_typesafe_ffi.so" ;;
  darwin-x64)   triple="darwin-x64";     libname="libruvector_typesafe_ffi.dylib" ;;
  darwin-arm64) triple="darwin-arm64";   libname="libruvector_typesafe_ffi.dylib" ;;
  win32-x64)    triple="win32-x64-msvc"; libname="ruvector_typesafe_ffi.dll" ;;
  *) echo "unsupported platform: $plat-$arch" >&2; exit 1 ;;
esac

echo "building $crate ($label, release) for $plat-$arch -> typesafe.$triple.node"
# shellcheck disable=SC2086
cargo build -p "$crate" --release $features --manifest-path "$root/Cargo.toml"

mkdir -p "$here/native"
src="$root/target/release/$libname"
dst="$here/native/typesafe.$triple.node"
cp "$src" "$dst"
echo "wrote $dst"

echo "running check-security on the built binary"
node "$here/scripts/check-security.mjs" --native "$dst"
