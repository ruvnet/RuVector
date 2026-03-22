#!/bin/bash
# Full 6-year medical + CS import via WET processing
# Processes quarterly Common Crawl snapshots from 2020-2026
set -euo pipefail

PROJECT="${1:-ruv-dev}"
SEGS_PER_CRAWL="${2:-100}"  # segments per crawl to process

# Quarterly crawl indices (2020-2026)
CRAWLS=(
  "CC-MAIN-2020-16"
  "CC-MAIN-2020-50"
  "CC-MAIN-2021-17"
  "CC-MAIN-2021-43"
  "CC-MAIN-2022-05"
  "CC-MAIN-2022-33"
  "CC-MAIN-2023-06"
  "CC-MAIN-2023-40"
  "CC-MAIN-2024-10"
  "CC-MAIN-2024-42"
  "CC-MAIN-2025-13"
  "CC-MAIN-2025-40"
  "CC-MAIN-2026-06"
  "CC-MAIN-2026-08"
)

BRAIN_URL="https://pi.ruv.io"

echo "=== Full 6-Year Medical + CS Import ==="
echo "Crawls: ${#CRAWLS[@]}"
echo "Segments per crawl: $SEGS_PER_CRAWL"
echo "Total segments: $((${#CRAWLS[@]} * SEGS_PER_CRAWL))"
echo ""

BEFORE=$(curl -s "$BRAIN_URL/v1/status" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['total_memories'])" 2>/dev/null || echo "0")
echo "Brain memories before: $BEFORE"
echo ""

for crawl in "${CRAWLS[@]}"; do
  echo "=== Deploying job for $crawl ==="
  bash scripts/deploy-wet-job.sh "$PROJECT" "$crawl" 0 "$SEGS_PER_CRAWL"

  # Execute the job
  JOB_NAME="wet-import-$(echo $crawl | tr '[:upper:]' '[:lower:]' | tr -d '-' | tail -c 8)"
  gcloud run jobs execute "$JOB_NAME" --project="$PROJECT" --region=us-central1 --async 2>&1

  echo "Job $JOB_NAME submitted (async)"
  echo ""

  # Don't flood -- wait 30s between job submissions
  sleep 30
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with: gcloud run jobs executions list --project=$PROJECT --region=us-central1"
echo ""
echo "Check brain growth:"
echo "  curl -s $BRAIN_URL/v1/status | python3 -c \"import sys,json; d=json.load(sys.stdin); print(f'Memories: {d[\\\"total_memories\\\"]}')\""
