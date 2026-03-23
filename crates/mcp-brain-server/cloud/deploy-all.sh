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
# IMPORTANT: Use --update-env-vars (NOT --set-env-vars) to preserve existing env vars
# like FIRESTORE_URL, GEMINI_API_KEY, and feature flags that were set manually.
echo "--- Step 2: Deploying to Cloud Run ---"

# Fetch secrets from Google Secrets Manager
GEMINI_KEY=$(gcloud secrets versions access latest --secret=GOOGLE_AI_API_KEY --project="$PROJECT_ID" 2>/dev/null || echo "")

gcloud run deploy ruvbrain \
  --image="gcr.io/${PROJECT_ID}/ruvbrain:latest" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --platform=managed \
  --memory=4Gi \
  --cpu=4 \
  --min-instances=1 \
  --max-instances=20 \
  --timeout=300 \
  --concurrency=80 \
  --session-affinity \
  --update-env-vars="\
RUST_LOG=info,\
FIRESTORE_URL=https://firestore.googleapis.com/v1/projects/${PROJECT_ID}/databases/(default)/documents,\
GEMINI_API_KEY=${GEMINI_KEY},\
GEMINI_MODEL=gemini-2.5-flash,\
GEMINI_GROUNDING=true,\
GWT_ENABLED=true,\
TEMPORAL_ENABLED=true,\
META_LEARNING_ENABLED=true,\
SONA_ENABLED=true,\
MIDSTREAM_ATTRACTOR=true,\
MIDSTREAM_SOLVER=true,\
MIDSTREAM_STRANGE_LOOP=true,\
MIDSTREAM_SCHEDULER=true,\
COGNITIVE_HOPFIELD=true,\
COGNITIVE_HDC=true,\
COGNITIVE_DENTATE=true,\
SPARSIFIER_ENABLED=true,\
GRAPH_AUTO_REBUILD=true,\
QUANTIZATION_ENABLED=true,\
LORA_FEDERATION=true,\
DOMAIN_EXPANSION=true,\
RVF_PII_STRIP=true,\
RVF_DP_ENABLED=true" \
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
