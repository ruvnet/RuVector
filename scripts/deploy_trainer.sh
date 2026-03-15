#!/usr/bin/env bash
# Deploy the Daily Discovery Brain Trainer to Cloud Run
# Created by rUv — altruistic knowledge enrichment for π.ruv.io
#
# Usage: ./scripts/deploy_trainer.sh [--schedule "CRON_EXPR"]
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-ruv-dev}"
REGION="${GCP_REGION:-us-central1}"
JOB_NAME="ruvbrain-trainer"
IMAGE="gcr.io/${PROJECT_ID}/${JOB_NAME}:latest"
SCHEDULE="${1:-0 2 * * *}"  # Default: daily at 02:00 UTC

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     π.ruv.io Daily Discovery Brain Trainer — Deployment      ║"
echo "║              Altruistic Knowledge Enrichment                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Build the container image
echo "▸ Building trainer image..."
gcloud builds submit \
    --config=crates/mcp-brain-server/cloudbuild-trainer.yaml \
    --project="${PROJECT_ID}" \
    .

# Step 2: Create or update the Cloud Run Job
echo "▸ Deploying Cloud Run Job: ${JOB_NAME}..."
gcloud run jobs create "${JOB_NAME}" \
    --image="${IMAGE}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --memory=512Mi \
    --cpu=1 \
    --max-retries=2 \
    --task-timeout=1800s \
    --set-env-vars="RUST_LOG=info,BRAIN_URL=https://pi.ruv.io" \
    2>/dev/null || \
gcloud run jobs update "${JOB_NAME}" \
    --image="${IMAGE}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --memory=512Mi \
    --cpu=1 \
    --max-retries=2 \
    --task-timeout=1800s \
    --set-env-vars="RUST_LOG=info,BRAIN_URL=https://pi.ruv.io"

# Step 3: Create or update the Cloud Scheduler trigger
echo "▸ Setting schedule: ${SCHEDULE}..."
SCHEDULER_NAME="${JOB_NAME}-schedule"
gcloud scheduler jobs create http "${SCHEDULER_NAME}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --schedule="${SCHEDULE}" \
    --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
    --http-method=POST \
    --oauth-service-account-email="${PROJECT_ID}@appspot.gserviceaccount.com" \
    --description="Daily discovery & brain training for π.ruv.io" \
    2>/dev/null || \
gcloud scheduler jobs update http "${SCHEDULER_NAME}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --schedule="${SCHEDULE}"

echo ""
echo "✓ Trainer deployed successfully!"
echo "  Job:      ${JOB_NAME}"
echo "  Region:   ${REGION}"
echo "  Schedule: ${SCHEDULE}"
echo "  Image:    ${IMAGE}"
echo ""
echo "  Manual run: gcloud run jobs execute ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID}"
echo "  View logs:  gcloud run jobs executions list --job=${JOB_NAME} --region=${REGION} --project=${PROJECT_ID}"
