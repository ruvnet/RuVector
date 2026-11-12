#!/usr/bin/env bash
set -euo pipefail

umask 077
ulimit -c 0
ulimit -f 1048576
ulimit -n 1024
ulimit -u 512

export CARGO_HOME=/cargo-home
export CARGO_TARGET_DIR=/target
export CARGO_NET_OFFLINE=true
export RUST_BACKTRACE=0
export RUSTFLAGS="-Dwarnings"

cd /workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --all-targets --locked --offline

if [[ ! -f research/run-candidate.sh ]]; then
  echo "candidate must provide research/run-candidate.sh" >&2
  exit 2
fi
bash research/run-candidate.sh /output

test -f /output/raw-results.json
test -f /output/report.json
