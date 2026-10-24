#!/usr/bin/env bash
# Add every workspace path patch to a `maturin sdist` tarball.
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
# setup-python pins Python 3.12 in the sdist job (tomllib requires >=3.11).
python3 - "$root" "$work/$top" <<'PATCH_PATHS'
import pathlib
import shutil
import sys
import tomllib

root = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
manifest = tomllib.loads((root / "Cargo.toml").read_text())
for spec in manifest.get("patch", {}).get("crates-io", {}).values():
    if not isinstance(spec, dict) or "path" not in spec:
        continue
    relative = pathlib.Path(spec["path"])
    source = (root / relative).resolve()
    if relative.is_absolute() or not source.is_relative_to(root):
        raise ValueError(f"Patch path must stay inside the source workspace: {relative}")
    if not (source / "Cargo.toml").is_file() or not list(source.glob("LICENSE*")):
        raise ValueError(f"Patch manifest or retained license missing: {relative}")
    shutil.copytree(source, destination / relative, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("target", ".git"))
    print(f"vendored patch: {relative}")
PATCH_PATHS
cp "$root/crates/ruvector-py/LICENSE" "$work/$top/LICENSE"
# Maturin trims the workspace. Prune its copied lock to that exact source
# graph, then reject any registry version/source/checksum change.
cargo metadata --offline --format-version 1 --manifest-path "$work/$top/Cargo.toml" > "$work/metadata.json"
python3 - "$root/Cargo.lock" "$work/$top/Cargo.lock" <<'LOCK_CLOSURE'
import pathlib
import sys
import tomllib

def registry_packages(path):
    lock = tomllib.loads(pathlib.Path(path).read_text())
    return {(p["name"], p["version"], p["source"], p.get("checksum"))
            for p in lock["package"] if p.get("source", "").startswith("registry+")}

before, after = map(registry_packages, sys.argv[1:])
if not after.issubset(before):
    raise ValueError(f"sdist resolution changed locked registry packages: {sorted(after - before)}")
print(f"sdist lock closure: {len(after)} registry package tuples verified against workspace lock (subset)")
LOCK_CLOSURE
tar czf "$sdist" -C "$work" "$top"
echo "patched $sdist: added workspace path patches and $top/LICENSE"
