#!/usr/bin/env bash
#
# deploy-dragnes.sh — Deploy DrAgnes to Google Cloud Run
#
# Usage:
#   ./scripts/deploy-dragnes.sh [--rollback]
#
# Prerequisites:
#   - gcloud CLI authenticated with project ruv-dev
#   - Docker installed and configured for GCR
#   - Google Secrets Manager entries:
#       OPENROUTER_API_KEY
#

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-ruv-dev}"
REGION="us-central1"
SERVICE_NAME="dragnes"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
TAG="${DRAGNES_TAG:-latest}"
FULL_IMAGE="${IMAGE}:${TAG}"

RUVOCAL_DIR="$(cd "$(dirname "$0")/../ui/ruvocal" && pwd)"
DOCKERFILE="${RUVOCAL_DIR}/Dockerfile.dragnes"

# ---------- Helpers -----------------------------------------------------------

log()  { printf '\033[1;35m[DrAgnes]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- Rollback ---------------------------------------------------------

if [[ "${1:-}" == "--rollback" ]]; then
  log "Rolling back to previous revision..."
  PREV_REVISION=$(gcloud run revisions list \
    --service="${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --sort-by="~creationTimestamp" \
    --limit=2 \
    --format="value(metadata.name)" | tail -1)

  if [[ -z "${PREV_REVISION}" ]]; then
    err "No previous revision found for rollback."
  fi

  gcloud run services update-traffic "${SERVICE_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --to-revisions="${PREV_REVISION}=100"

  log "Rolled back to revision: ${PREV_REVISION}"
  exit 0
fi

# ---------- Build -------------------------------------------------------------

log "Building DrAgnes image: ${FULL_IMAGE}"

cd "${RUVOCAL_DIR}"

# Install dependencies and build SvelteKit
log "Installing dependencies..."
npm ci --ignore-scripts 2>/dev/null || npm install

log "Building SvelteKit application..."
npm run build

log "Building Docker image..."
docker build -f "${DOCKERFILE}" -t "${FULL_IMAGE}" .

log "Pushing image to GCR..."
docker push "${FULL_IMAGE}"

# ---------- Deploy ------------------------------------------------------------

log "Deploying to Cloud Run (${REGION})..."

gcloud run deploy "${SERVICE_NAME}" \
  --image="${FULL_IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --allow-unauthenticated \
  --cpu=2 \
  --memory=2Gi \
  --min-instances=1 \
  --max-instances=10 \
  --concurrency=80 \
  --timeout=300 \
  --port=3000 \
  --set-env-vars="NODE_ENV=production" \
  --set-env-vars="OPENAI_BASE_URL=https://openrouter.ai/api/v1" \
  --set-env-vars="DRAGNES_ENABLED=true" \
  --set-env-vars="DRAGNES_BRAIN_URL=https://pi.ruv.io" \
  --set-env-vars="DRAGNES_MODEL_VERSION=0.1.0" \
  --update-secrets="OPENAI_API_KEY=OPENROUTER_API_KEY:latest" \
  --set-env-vars='MCP_SERVERS=[{"name":"pi-brain","url":"https://mcp.pi.ruv.io"}]'

# ---------- CDN for WASM assets -----------------------------------------------

log "Configuring Cloud CDN for WASM assets..."
gcloud run services update "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --session-affinity 2>/dev/null || true

# ---------- Health check ------------------------------------------------------

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")

log "Verifying health check..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health" || echo "000")

if [[ "${HTTP_STATUS}" == "200" ]]; then
  log "Health check passed."
else
  log "Warning: Health check returned ${HTTP_STATUS}. Service may still be starting."
fi

# ---------- Done --------------------------------------------------------------

log "Deployment complete."
log "Service URL: ${SERVICE_URL}"
log "DrAgnes URL: ${SERVICE_URL}/dragnes"
