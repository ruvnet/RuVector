#!/bin/bash
# Deploy WET processor as Cloud Run Job for large-scale Common Crawl import
# Usage: ./deploy-wet-job.sh [PROJECT] [CRAWL_INDEX] [START_SEGMENT] [NUM_SEGMENTS]
set -euo pipefail

PROJECT="${1:-ruv-dev}"
CRAWL_INDEX="${2:-CC-MAIN-2026-08}"
START_SEG="${3:-0}"
NUM_SEGS="${4:-100}"
REGION="us-central1"
JOB_NAME="wet-import-$(echo $CRAWL_INDEX | tr '[:upper:]' '[:lower:]' | tr -d '-' | tail -c 8)"

echo "=== WET Cloud Run Job Deployment ==="
echo "Project: $PROJECT"
echo "Crawl: $CRAWL_INDEX"
echo "Segments: $START_SEG to $((START_SEG + NUM_SEGS - 1))"
echo "Job name: $JOB_NAME"
echo ""

# First, upload the filter script to GCS so the job can access it
echo "--- Uploading filter script to GCS ---"
gsutil cp scripts/wet-filter-inject.js gs://ruvector-brain-dev/scripts/wet-filter-inject.js 2>&1

# Get the WET paths file
echo "--- Fetching WET paths ---"
PATHS_URL="https://data.commoncrawl.org/crawl-data/${CRAWL_INDEX}/wet.paths.gz"
curl -sL "$PATHS_URL" | gunzip | sed -n "$((START_SEG + 1)),$((START_SEG + NUM_SEGS))p" > /tmp/wet-paths-batch.txt
ACTUAL_SEGS=$(wc -l < /tmp/wet-paths-batch.txt)
echo "Segments to process: $ACTUAL_SEGS"

# Upload paths file
gsutil cp /tmp/wet-paths-batch.txt gs://ruvector-brain-dev/scripts/wet-paths-batch.txt 2>&1

# Build the domain list for the job command
DOMAIN_LIST="pubmed.ncbi.nlm.nih.gov,ncbi.nlm.nih.gov,who.int,cancer.org,aad.org,dermnetnz.org,melanoma.org,arxiv.org,acm.org,ieee.org,nature.com,nejm.org,bmj.com,mayoclinic.org,clevelandclinic.org,medlineplus.gov,cdc.gov,nih.gov,thelancet.com,sciencedirect.com,webmd.com,healthline.com,medscape.com,jamanetwork.com,frontiersin.org,plos.org,biomedcentral.com,cell.com,springer.com,cochrane.org,clinicaltrials.gov,fda.gov,mskcc.org,mdanderson.org,nccn.org,dl.acm.org,ieeexplore.ieee.org,proceedings.neurips.cc,huggingface.co,pytorch.org,tensorflow.org,cs.stanford.edu,deepmind.google,research.google,microsoft.com/research,openreview.net,paperswithcode.com,asco.org,esmo.org,dana-farber.org,cancer.net,uptodate.com,wiley.com,elsevier.com,mdpi.com,plos.org,aaai.org,usenix.org,jmlr.org,aclanthology.org"

# Create/update the Cloud Run Job
echo "--- Creating Cloud Run Job ---"
gcloud run jobs create "$JOB_NAME" \
  --project="$PROJECT" \
  --region="$REGION" \
  --image="node:20-alpine" \
  --command="/bin/sh" \
  --args="-c,apk add --no-cache curl bash > /dev/null 2>&1 && gsutil cp gs://ruvector-brain-dev/scripts/wet-filter-inject.js /tmp/filter.js 2>/dev/null && WET_PATH=\$(gsutil cat gs://ruvector-brain-dev/scripts/wet-paths-batch.txt 2>/dev/null | sed -n \"\${CLOUD_RUN_TASK_INDEX:-0}p\" | head -1) && echo \"Processing: \$WET_PATH\" && curl -sL \"https://data.commoncrawl.org/\$WET_PATH\" | gunzip | node /tmp/filter.js --brain-url https://pi.ruv.io --auth 'Authorization: Bearer ruvector-crawl-2026' --batch-size 10 --crawl-index $CRAWL_INDEX --domains '$DOMAIN_LIST'" \
  --task-count="$ACTUAL_SEGS" \
  --parallelism=10 \
  --max-retries=1 \
  --cpu=1 \
  --memory=1Gi \
  --task-timeout=3600s \
  --set-env-vars="CRAWL_INDEX=$CRAWL_INDEX" \
  2>&1 || \
gcloud run jobs update "$JOB_NAME" \
  --project="$PROJECT" \
  --region="$REGION" \
  --task-count="$ACTUAL_SEGS" \
  --parallelism=10 \
  2>&1

echo ""
echo "--- Job created. To execute: ---"
echo "gcloud run jobs execute $JOB_NAME --project=$PROJECT --region=$REGION"
echo ""
echo "--- To monitor: ---"
echo "gcloud run jobs executions list --job=$JOB_NAME --project=$PROJECT --region=$REGION"
