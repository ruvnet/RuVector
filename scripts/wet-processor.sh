#!/bin/bash
# Common Crawl WET Processor -- Medical + CS Corpus Import
# Processes pre-extracted text (no HTML parsing needed)
# Usage: ./wet-processor.sh [CRAWL_INDEX] [SEGMENT_NUM]
set -euo pipefail

CRAWL_INDEX="${1:-CC-MAIN-2026-08}"
SEGMENT_NUM="${2:-0}"
BRAIN_URL="${BRAIN_URL:-https://pi.ruv.io}"
AUTH="Authorization: Bearer ruvector-crawl-2026"
WORK_DIR="/tmp/wet-processing"
BATCH_SIZE=10  # items per batch inject call

# Medical + CS domains to filter for
DOMAINS=(
  "pubmed.ncbi.nlm.nih.gov"
  "ncbi.nlm.nih.gov"
  "who.int"
  "cancer.org"
  "aad.org"
  "skincancer.org"
  "dermnetnz.org"
  "melanoma.org"
  "mayoclinic.org"
  "clevelandclinic.org"
  "medlineplus.gov"
  "cdc.gov"
  "nih.gov"
  "nejm.org"
  "thelancet.com"
  "bmj.com"
  "nature.com/articles"
  "sciencedirect.com"
  "arxiv.org"
  "acm.org"
  "ieee.org"
  "dl.acm.org"
  "proceedings.mlr.press"
  "openreview.net"
  "paperswithcode.com"
  "github.com"
  "stackoverflow.com"
  "medium.com"
  "towardsdatascience.com"
  "distill.pub"
)

mkdir -p "$WORK_DIR"

echo "=== WET Processor ==="
echo "Crawl: $CRAWL_INDEX"
echo "Domains: ${#DOMAINS[@]}"
echo ""

# Step 1: Get WET file list for this crawl
echo "--- Fetching WET file paths ---"
PATHS_URL="https://data.commoncrawl.org/crawl-data/${CRAWL_INDEX}/wet.paths.gz"
curl -sL "$PATHS_URL" | gunzip > "$WORK_DIR/wet-paths.txt" 2>/dev/null || {
  echo "ERROR: Could not fetch WET paths for $CRAWL_INDEX"
  exit 1
}

TOTAL_SEGMENTS=$(wc -l < "$WORK_DIR/wet-paths.txt")
echo "Total WET segments: $TOTAL_SEGMENTS"

# Select segment to process
WET_PATH=$(sed -n "$((SEGMENT_NUM + 1))p" "$WORK_DIR/wet-paths.txt")
if [ -z "$WET_PATH" ]; then
  echo "ERROR: Segment $SEGMENT_NUM not found"
  exit 1
fi

echo "Processing segment $SEGMENT_NUM: $WET_PATH"

# Step 2: Download and decompress WET file
echo "--- Downloading WET segment ---"
WET_FILE="$WORK_DIR/segment.wet.gz"
curl -sL "https://data.commoncrawl.org/$WET_PATH" -o "$WET_FILE" --max-time 300 || {
  echo "ERROR: Download failed"
  exit 1
}

echo "Downloaded: $(du -h "$WET_FILE" | cut -f1)"

# Step 3: Extract, filter by domain, and inject
echo "--- Processing and filtering ---"
gunzip -c "$WET_FILE" | node "$(dirname "$0")/wet-filter-inject.js" \
  --brain-url "$BRAIN_URL" \
  --auth "$AUTH" \
  --batch-size "$BATCH_SIZE" \
  --domains "$(IFS=,; echo "${DOMAINS[*]}")" \
  --crawl-index "$CRAWL_INDEX"

# Cleanup
rm -f "$WET_FILE"
echo ""
echo "=== Done ==="
