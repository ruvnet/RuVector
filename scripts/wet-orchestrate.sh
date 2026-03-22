#!/bin/bash
# Orchestrate WET processing across multiple segments
# Usage: ./wet-orchestrate.sh [CRAWL_INDEX] [START_SEGMENT] [NUM_SEGMENTS]
set -euo pipefail

CRAWL_INDEX="${1:-CC-MAIN-2026-08}"
START="${2:-0}"
COUNT="${3:-5}"  # Process 5 segments by default
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BRAIN_URL="${BRAIN_URL:-https://pi.ruv.io}"

echo "=== WET Orchestrator ==="
echo "Crawl: $CRAWL_INDEX"
echo "Segments: $START to $((START + COUNT - 1))"
echo ""

# Record starting state
BEFORE=$(curl -s "$BRAIN_URL/v1/status" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('total_memories', 0))" 2>/dev/null || echo "0")
echo "Brain memories before: $BEFORE"
echo ""

for i in $(seq "$START" "$((START + COUNT - 1))"); do
  echo "=== Segment $i ==="
  bash "$SCRIPT_DIR/wet-processor.sh" "$CRAWL_INDEX" "$i" 2>&1 || {
    echo "Segment $i failed, continuing..."
  }

  # Brief pause between segments
  sleep 5

  # Check brain growth
  CURRENT=$(curl -s "$BRAIN_URL/v1/status" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('total_memories', 0))" 2>/dev/null || echo "0")
  echo "Brain memories: $CURRENT (+$((CURRENT - BEFORE)) total)"
  echo ""
done

# Final report
echo "--- Final Report ---"
curl -s "$BRAIN_URL/v1/status" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'Final state:')
    print(f'  Memories: {d.get(\"total_memories\", \"N/A\")}')
    print(f'  Graph: {d.get(\"graph_edges\", \"N/A\")} edges')
    print(f'  Sparsifier: {d.get(\"sparsifier_compression\", 0):.1f}x')
except Exception as e:
    print(f'Could not fetch final status: {e}')
" 2>/dev/null || echo "Could not fetch final brain status"
echo ""
echo "=== Orchestration Complete ==="
