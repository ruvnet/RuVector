#!/bin/bash
# Deploy WET processor as Cloud Run Job for large-scale Common Crawl import
# Usage: ./deploy-wet-job.sh [PROJECT] [CRAWL_INDEX] [START_SEGMENT] [NUM_SEGMENTS]
set -euo pipefail

PROJECT="${1:-ruv-dev}"
CRAWL_INDEX="${2:-CC-MAIN-2026-08}"
START_SEG="${3:-0}"
NUM_SEGS="${4:-100}"
REGION="us-central1"
JOB_NAME="wet-import-$(echo "$CRAWL_INDEX" | tr '[:upper:]' '[:lower:]' | tr -d '-' | tail -c 8)"

echo "=== WET Cloud Run Job Deployment ==="
echo "Project: $PROJECT"
echo "Crawl: $CRAWL_INDEX"
echo "Segments: $START_SEG to $((START_SEG + NUM_SEGS - 1))"
echo "Job name: $JOB_NAME"
echo ""

# Get the WET paths file
echo "--- Fetching WET paths ---"
PATHS_URL="https://data.commoncrawl.org/crawl-data/${CRAWL_INDEX}/wet.paths.gz"
curl -sL "$PATHS_URL" | gunzip | sed -n "$((START_SEG + 1)),$((START_SEG + NUM_SEGS))p" > /tmp/wet-paths-batch.txt
ACTUAL_SEGS=$(wc -l < /tmp/wet-paths-batch.txt)
echo "Segments to process: $ACTUAL_SEGS"

# Build the domain list (passed via env var to avoid comma-splitting in --args)
DOMAIN_LIST="pubmed.ncbi.nlm.nih.gov,ncbi.nlm.nih.gov,who.int,cancer.org,aad.org,dermnetnz.org,melanoma.org,arxiv.org,acm.org,ieee.org,nature.com,nejm.org,bmj.com,mayoclinic.org,clevelandclinic.org,medlineplus.gov,cdc.gov,nih.gov,thelancet.com,sciencedirect.com,webmd.com,healthline.com,medscape.com,jamanetwork.com,frontiersin.org,plos.org,biomedcentral.com,cell.com,springer.com,cochrane.org,clinicaltrials.gov,fda.gov,mskcc.org,mdanderson.org,nccn.org,dl.acm.org,ieeexplore.ieee.org,proceedings.neurips.cc,huggingface.co,pytorch.org,tensorflow.org,cs.stanford.edu,deepmind.google,research.google,microsoft.com/research,openreview.net,paperswithcode.com,asco.org,esmo.org,dana-farber.org,cancer.net,uptodate.com,wiley.com,elsevier.com,mdpi.com,aaai.org,usenix.org,jmlr.org,aclanthology.org"

# Create a temporary build context with all files baked into the image
BUILD_DIR=$(mktemp -d)
cp scripts/wet-filter-inject.js "$BUILD_DIR/filter.js"
cp /tmp/wet-paths-batch.txt "$BUILD_DIR/paths.txt"

cat > "$BUILD_DIR/Dockerfile" <<'DOCKERFILE'
FROM node:20-alpine
RUN apk add --no-cache curl bash
COPY filter.js /app/filter.js
COPY paths.txt /app/paths.txt
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
DOCKERFILE

cat > "$BUILD_DIR/entrypoint.sh" <<'ENTRYPOINT'
#!/bin/bash
set -euo pipefail

# Get the WET path for this task index from baked-in paths file
TASK_IDX="${CLOUD_RUN_TASK_INDEX:-0}"
WET_PATH=$(sed -n "$((TASK_IDX + 1))p" /app/paths.txt | head -1)

if [ -z "$WET_PATH" ]; then
  echo "No WET path for task index $TASK_IDX"
  exit 0
fi

echo "Task $TASK_IDX processing: $WET_PATH"
echo "Brain URL: $BRAIN_URL"
echo "Crawl index: $CRAWL_INDEX"

curl -sL "https://data.commoncrawl.org/$WET_PATH" \
  | gunzip \
  | node /app/filter.js \
    --brain-url "$BRAIN_URL" \
    --auth "$AUTH_HEADER" \
    --batch-size "$BATCH_SIZE" \
    --crawl-index "$CRAWL_INDEX" \
    --domains "$DOMAINS"

echo "Task $TASK_IDX complete"
ENTRYPOINT

echo "--- Building and deploying Cloud Run Job ---"

# Write env vars file (avoids comma-parsing issues with --set-env-vars)
cat > "$BUILD_DIR/env.yaml" <<ENVYAML
CRAWL_INDEX: "$CRAWL_INDEX"
BRAIN_URL: "https://pi.ruv.io"
AUTH_HEADER: "Authorization: Bearer ruvector-crawl-2026"
BATCH_SIZE: "10"
DOMAINS: "$DOMAIN_LIST"
ENVYAML

# Deploy from source (builds container automatically)
gcloud run jobs deploy "$JOB_NAME" \
  --project="$PROJECT" \
  --region="$REGION" \
  --source="$BUILD_DIR" \
  --tasks="$ACTUAL_SEGS" \
  --parallelism=10 \
  --max-retries=1 \
  --cpu=1 \
  --memory=1Gi \
  --task-timeout=3600s \
  --env-vars-file="$BUILD_DIR/env.yaml" \
  2>&1

# Clean up build dir
rm -rf "$BUILD_DIR"

echo ""
echo "--- Job deployed. To execute: ---"
echo "gcloud run jobs execute $JOB_NAME --project=$PROJECT --region=$REGION"
echo ""
echo "--- To monitor: ---"
echo "gcloud run jobs executions list --job=$JOB_NAME --project=$PROJECT --region=$REGION"
