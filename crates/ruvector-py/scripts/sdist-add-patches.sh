#!/usr/bin/env bash
# Add the vendored `patches/hnsw_rs` crate to a `maturin sdist` tarball.
#
# The workspace root Cargo.toml has `[patch.crates-io] hnsw_rs = { path =
# "./patches/hnsw_rs" }`. maturin copies path *dependencies* into the sdist
# but not `[patch]` paths, and its `include` globs may not contain `..`, so a
# plain `maturin sdist` produces a tarball that cannot resolve its dependency
# graph ("failed to read patches/hnsw_rs/Cargo.toml"; reproduced with a
# clean-venv `pip install ruvector-0.1.0.tar.gz`). This script repairs the
# tarball in place. It also puts LICENSE at the sdist root, where PKG-INFO's
# `License-File: LICENSE` points (maturin only ships it under crates/ruvector-py/).
#
# usage: sdist-add-patches.sh <sdist.tar.gz> [workspace-root]
set -euo pipefail

sdist=$(realpath "$1")
root=$(realpath "${2:-$(git rev-parse --show-toplevel)}")
top=$(tar tzf "$sdist" | awk -F/ 'NR==1{print $1}')
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

tar xzf "$sdist" -C "$work"
mkdir -p "$work/$top/patches"
cp -r "$root/patches/hnsw_rs" "$work/$top/patches/hnsw_rs"
rm -rf "$work/$top/patches/hnsw_rs/target"
cp "$root/crates/ruvector-py/LICENSE" "$work/$top/LICENSE"
tar czf "$sdist" -C "$work" "$top"
echo "patched $sdist: added $top/patches/hnsw_rs and $top/LICENSE"
