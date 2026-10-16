#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

for manifest in crates/ruvector-language/Cargo.toml crates/ruvector-language-native/Cargo.toml; do
  cargo +1.90.0 fmt --manifest-path "$manifest" -- --check
  cargo +1.90.0 clippy --manifest-path "$manifest" --locked --all-targets --no-deps -- -D warnings
  cargo +1.90.0 test --manifest-path "$manifest" --locked
done
cargo +1.90.0 build --manifest-path crates/ruvector-language-native/Cargo.toml --locked --bins
export RUVECTOR_LANGUAGE_BIN="${CARGO_TARGET_DIR:-$project_root/crates/ruvector-language-native/target}/debug/rvg"
node --test npm/packages/graph-language/test/*.test.mjs
PYTHONPATH="$project_root/python/ruvector-language${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m unittest discover -s python/ruvector-language/tests -v
node scripts/adr-index.mjs --check
