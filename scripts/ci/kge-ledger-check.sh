#!/usr/bin/env bash
# CI gate for the ADR-007 test-scoring ledger (plan M3). Run on the protected
# `kge-final-ledger` branch (and on PRs into it). Fails on duplicate
# (dataset, seed) keys, seeds outside {0..4}, results without intents, more
# than one config_hash per dataset or one differing from selection.json, mixed
# tag SHAs, any modified/deleted pre-existing line, and selection.json changing
# after the first intent. The rules live in Rust:
# crates/ruvector-kge-bench/src/ledger.rs (binary `kge-ledger-check`).
#
# Usage: scripts/ci/kge-ledger-check.sh [BASE_REF]
#   BASE_REF: the commit to compare against for append-only (default HEAD~1
#   when it exists; pass the PR base SHA in CI).
set -euo pipefail

DIR=npm/packages/kge/bench/results/final
LEDGER="$DIR/ledger.jsonl"
SELECTION="$DIR/selection.json"
BASE="${1:-}"
if [ -z "$BASE" ] && git rev-parse -q --verify HEAD~1 >/dev/null; then BASE=HEAD~1; fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
ARGS=("$LEDGER")
[ -f "$SELECTION" ] && ARGS+=(--selection "$SELECTION")
if [ -n "$BASE" ] && git cat-file -e "$BASE:$LEDGER" 2>/dev/null; then
  git show "$BASE:$LEDGER" > "$TMP/prev-ledger.jsonl"
  ARGS+=(--previous "$TMP/prev-ledger.jsonl")
  if git cat-file -e "$BASE:$SELECTION" 2>/dev/null; then
    git show "$BASE:$SELECTION" > "$TMP/prev-selection.json"
    ARGS+=(--previous-selection "$TMP/prev-selection.json")
  fi
fi
if git rev-parse -q --verify "refs/tags/kge-prereg-v1" >/dev/null; then
  ARGS+=(--tag-sha "$(git rev-parse refs/tags/kge-prereg-v1)")
fi
exec cargo run -q --release -p ruvector-kge-bench --bin kge-ledger-check -- "${ARGS[@]}"
