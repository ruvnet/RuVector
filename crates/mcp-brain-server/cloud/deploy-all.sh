#!/usr/bin/env bash
# Full deployment: build + deploy Cloud Run + setup Pub/Sub + deploy Scheduler
# Usage: ./deploy-all.sh [PROJECT_ID]

set -euo pipefail

PROJECT_ID="${1:-ruv-dev}"
REGION="us-central1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== RuVector Brain Full Deployment ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Step 1: Build container
echo "--- Step 1: Building container ---"
cd "$ROOT_DIR"
gcloud builds submit \
  --config=crates/mcp-brain-server/cloudbuild.yaml \
  --project="$PROJECT_ID" .

# Step 2: Deploy to Cloud Run
echo "--- Step 2: Deploying to Cloud Run ---"
gcloud run deploy ruvbrain \
  --image="gcr.io/${PROJECT_ID}/ruvbrain:latest" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --platform=managed \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=10 \
  --timeout=300 \
  --concurrency=80 \
  --set-env-vars="RUST_LOG=info,GWT_ENABLED=true,TEMPORAL_ENABLED=true,META_LEARNING_ENABLED=true,SONA_ENABLED=true" \
  --allow-unauthenticated

# Step 3: Setup Pub/Sub
echo "--- Step 3: Setting up Pub/Sub ---"
bash "$SCRIPT_DIR/setup-pubsub.sh" "$PROJECT_ID"

# Step 4: Deploy Scheduler
echo "--- Step 4: Deploying Scheduler ---"
bash "$SCRIPT_DIR/deploy-scheduler.sh" "$PROJECT_ID"

echo ""
echo "=== Deployment Complete ==="
echo "Service URL: https://pi.ruv.io"
echo "Health: curl https://pi.ruv.io/v1/health"
echo "Status: curl -H 'Authorization: Bearer ruvector-swarm' https://pi.ruv.io/v1/status"
echo "Pipeline: curl -H 'Authorization: Bearer ruvector-swarm' https://pi.ruv.io/v1/pipeline/metrics"
