#!/usr/bin/env bash
# ruvllm-cluster-smoke — first multi-Pi cluster benchmark (ADR-179 iter 15+).
#
# Sends a single completion to each of N workers in parallel, reports
# per-worker tok/s and aggregate cluster tok/s. Mirrors the spirit of
# ruvector-hailo-cluster-bench but for the iter-9 newline-delimited-JSON
# transport (gRPC lands later).
#
# Until iter 11+ has a real `ruvllm-cluster-bench` Rust bin with P2C+EWMA,
# this gives us numbers fast.
#
# Usage:
#   bash ruvllm-cluster-smoke.sh [WORKERS_CSV] [PROMPT] [MAX_TOKENS]
#
# Defaults:
#   WORKERS_CSV   cognitum-cluster-1:50053,cognitum-cluster-2:50053,cognitum-cluster-3:50053
#   PROMPT        "The capital of France is"
#   MAX_TOKENS    16

set -uo pipefail

WORKERS=${1:-cognitum-cluster-1:50053,cognitum-cluster-2:50053,cognitum-cluster-3:50053}
PROMPT=${2:-"The capital of France is"}
MAX_TOKENS=${3:-16}

REQ=$(printf '{"prompt":"%s","max_tokens":%d}' "$PROMPT" "$MAX_TOKENS")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "=== ruvllm cluster smoke =="
echo "workers:   $WORKERS"
echo "prompt:    $PROMPT"
echo "max_tokens:$MAX_TOKENS"
echo

# fan out
START_NS=$(date +%s%N)
i=0
IFS=','
for w in $WORKERS; do
  (
    t0=$(date +%s%N)
    out=$(echo "$REQ" | timeout 600 nc -q 1 ${w%:*} ${w#*:} 2>&1 | head -1)
    t1=$(date +%s%N)
    echo "$w" > "$TMP/$i.host"
    echo "$out" > "$TMP/$i.body"
    echo "$(( (t1 - t0) / 1000000 ))" > "$TMP/$i.ms"
  ) &
  i=$((i + 1))
done
wait
END_NS=$(date +%s%N)
unset IFS

WALL_MS=$(( (END_NS - START_NS) / 1000000 ))

echo "=== per-worker results ==="
TOTAL_TOKS=0
for f in "$TMP"/*.body; do
  i=$(basename "$f" .body)
  host=$(cat "$TMP/$i.host")
  body=$(cat "$f")
  ms=$(cat "$TMP/$i.ms")
  toks=$(echo "$body" | jq -r '.tokens // "?"' 2>/dev/null || echo "?")
  text=$(echo "$body" | jq -r '.text // .error // ""' 2>/dev/null || echo "<unparsable>")
  if [[ "$toks" =~ ^[0-9]+$ ]]; then
    rate=$(awk "BEGIN {printf \"%.2f\", $toks / ($ms / 1000.0)}")
    TOTAL_TOKS=$((TOTAL_TOKS + toks))
  else
    rate="?"
  fi
  printf "%-32s ms=%-7s toks=%-5s tok/s=%-7s text=%s\n" "$host" "$ms" "$toks" "$rate" "$text"
done

echo
echo "=== cluster aggregate ==="
echo "wall_ms:        $WALL_MS"
echo "total_tokens:   $TOTAL_TOKS"
if [[ "$TOTAL_TOKS" -gt 0 ]]; then
  AGG=$(awk "BEGIN {printf \"%.2f\", $TOTAL_TOKS / ($WALL_MS / 1000.0)}")
  echo "aggregate tok/s:$AGG"
fi
