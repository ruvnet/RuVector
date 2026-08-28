#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <header> <static-library> <abi-symbol-manifest>" >&2
  exit 64
fi

header=$1
library=$2
manifest=$3
for input in "$header" "$library" "$manifest"; do
  if [ ! -f "$input" ]; then
    echo "required ABI input is not a regular file: $input" >&2
    exit 66
  fi
done

llvm_nm=${LLVM_NM:-}
if [ -z "$llvm_nm" ]; then
  rust_sysroot=$(rustc --print sysroot)
  llvm_nm=$(find "$rust_sysroot" -type f -name llvm-nm -print -quit)
fi
if [ -z "$llvm_nm" ] || [ ! -x "$llvm_nm" ]; then
  echo "llvm-nm is unavailable; install the rustup llvm-tools component" >&2
  exit 69
fi

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

LC_ALL=C grep -Eo \
  'ruvector_apple_core_[a-z0-9_]+[[:space:]]*\(' "$header" \
  | LC_ALL=C sed -E 's/[[:space:]]*\($//' \
  | LC_ALL=C sort -u > "$work_dir/header.symbols"
LC_ALL=C sed -E '/^[[:space:]]*(#|$)/d; s/[[:space:]]+$//' "$manifest" \
  | LC_ALL=C sort -u > "$work_dir/manifest.symbols"
"$llvm_nm" --defined-only --extern-only --just-symbol-name "$library" \
  2>/dev/null \
  | LC_ALL=C sed 's/^_//' \
  | LC_ALL=C awk '/^ruvector_apple_core_[a-z0-9_]+$/' \
  | LC_ALL=C sort -u > "$work_dir/library.symbols"

if ! cmp -s "$work_dir/header.symbols" "$work_dir/manifest.symbols"; then
  echo "C header and frozen ABI manifest differ:" >&2
  diff -u "$work_dir/manifest.symbols" "$work_dir/header.symbols" >&2 || true
  exit 1
fi
if ! cmp -s "$work_dir/library.symbols" "$work_dir/manifest.symbols"; then
  echo "static-library exports and frozen ABI manifest differ:" >&2
  diff -u "$work_dir/manifest.symbols" "$work_dir/library.symbols" >&2 || true
  exit 1
fi

echo "verified $(wc -l < "$work_dir/manifest.symbols" | tr -d ' ') Apple core ABI symbols"
