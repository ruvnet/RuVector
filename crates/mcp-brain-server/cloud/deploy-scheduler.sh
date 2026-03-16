#!/usr/bin/env bash
# Deploy Cloud Scheduler jobs for brain optimization
# Usage: ./deploy-scheduler.sh [PROJECT_ID]

set -euo pipefail

PROJECT_ID="${1:-ruv-dev}"
REGION="us-central1"
SERVICE_URL="https://pi.ruv.io"
SA_EMAIL="ruvbrain-scheduler@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Deploying Cloud Scheduler jobs for brain optimization"

deploy_job() {
  local name="$1"
  local schedule="$2"
  local description="$3"
  local body="$4"

  echo "  Deploying: $name ($schedule)"

  gcloud scheduler jobs delete "$name" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --quiet 2>/dev/null || true

  gcloud scheduler jobs create http "$name" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --schedule="$schedule" \
    --time-zone="UTC" \
    --description="$description" \
    --uri="${SERVICE_URL}/v1/pipeline/optimize" \
    --http-method=POST \
    --message-body="$body" \
    --headers="Content-Type=application/json,Authorization=Bearer ruvector-swarm" \
    --oidc-service-account-email="$SA_EMAIL" \
    --max-retry-attempts=3 \
    --min-backoff="5s" \
    --max-backoff="60s"
}

# Training - every 5 minutes
deploy_job "brain-train" \
  "*/5 * * * *" \
  "SONA training cycle" \
  '{"actions":["train"]}'

# Drift monitoring - every 15 minutes
deploy_job "brain-drift" \
  "*/15 * * * *" \
  "Embedding drift check" \
  '{"actions":["drift_check"]}'

# Domain transfer - every 30 minutes
deploy_job "brain-transfer" \
  "*/30 * * * *" \
  "Cross-domain knowledge transfer" \
  '{"actions":["transfer_all"]}'

# Graph rebalance - hourly
deploy_job "brain-graph" \
  "0 * * * *" \
  "Graph CSR + MinCut rebuild" \
  '{"actions":["rebuild_graph"]}'

# Attractor analysis - every 20 minutes
deploy_job "brain-attractor" \
  "*/20 * * * *" \
  "Lyapunov attractor analysis" \
  '{"actions":["attractor_analysis"]}'

# Full sweep - daily 3 AM UTC
deploy_job "brain-full-optimize" \
  "0 3 * * *" \
  "Complete daily optimization" \
  '{"actions":["train","drift_check","transfer_all","rebuild_graph","cleanup","attractor_analysis"]}'

# Cleanup - daily 4 AM UTC
deploy_job "brain-cleanup" \
  "0 4 * * *" \
  "Low-quality memory pruning" \
  '{"actions":["cleanup"]}'

echo ""
echo "=== Scheduler Jobs Deployed ==="
gcloud scheduler jobs list --project="$PROJECT_ID" --location="$REGION" --filter="name:brain-"
echo ""
echo "Manual trigger: gcloud scheduler jobs run brain-train --project=$PROJECT_ID --location=$REGION"
