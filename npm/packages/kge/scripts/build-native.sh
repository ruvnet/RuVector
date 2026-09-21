#!/usr/bin/env bash
# Build the native N-API binding and place it where index.js loads it:
#   npm/packages/kge/native/kge.<triple>.node
#
# A napi-rs cdylib *is* the Node addon (the `.node` extension is a rename), so
# we compile with cargo and copy it — keeping the artifact name deterministic
# rather than depending on package.json's napi config (owned by another agent).
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"          # npm/packages/kge
root="$(cd "$here/../../.." && pwd)"              # repo root
crate="ruvector-kge-ffi"

# Honour CARGO_TARGET_DIR (this worktree points it at the shared target dir);
# fall back to the repo-root target/ when unset.
target_dir="${CARGO_TARGET_DIR:-$root/target}"

# Map node's platform/arch to the napi triple used in index.js's platformMap.
read -r plat arch < <(node -e "console.log(process.platform, process.arch)")
case "$plat-$arch" in
  linux-x64)    triple="linux-x64-gnu";   libname="libruvector_kge_ffi.so" ;;
  linux-arm64)  triple="linux-arm64-gnu"; libname="libruvector_kge_ffi.so" ;;
  darwin-x64)   triple="darwin-x64";      libname="libruvector_kge_ffi.dylib" ;;
  darwin-arm64) triple="darwin-arm64";    libname="libruvector_kge_ffi.dylib" ;;
  win32-x64)    triple="win32-x64-msvc";  libname="ruvector_kge_ffi.dll" ;;
  *) echo "unsupported platform: $plat-$arch" >&2; exit 1 ;;
esac

echo "building $crate (release) for $plat-$arch -> kge.$triple.node"
cargo build -p "$crate" --release --manifest-path "$root/Cargo.toml"

mkdir -p "$here/native"
src="$target_dir/release/$libname"
dst="$here/native/kge.$triple.node"
cp "$src" "$dst"
echo "wrote $dst"
