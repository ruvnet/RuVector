#!/usr/bin/env bash
set -euo pipefail

# RuVector Shared Brain - Google Cloud Deployment
# Project: ruv-dev
# Region: us-central1
# Domain: brain.ruv.io (added later via Cloudflare)
#
# Two deployment paths:
#   Path A (dev):  Cloud Run + direct URL (no LB/Cloud Armor/CDN)
#   Path B (prod): External HTTPS LB + serverless NEG + Cloud Armor + CDN + brain.ruv.io
#
# Usage:
#   ./deploy.sh          # Path A (default — dev)
#   DEPLOY_PATH=B ./deploy.sh  # Path B (prod with LB)

PROJECT_ID="${GCP_PROJECT_ID:-ruv-dev}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="ruvbrain"
BUCKET_NAME="ruvector-brain-${REGION}"
DB_NAME="(default)"
DEPLOY_PATH="${DEPLOY_PATH:-A}"

echo "=== RuVector Shared Brain Deployment ==="
echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo "Bucket:  ${BUCKET_NAME}"
echo "DB:      ${DB_NAME}"
echo "Path:    ${DEPLOY_PATH}"
echo ""

# 1. Enable required APIs
echo "=== Step 1: Enabling APIs ==="
gcloud services enable \
    firestore.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com \
    storage.googleapis.com \
    --project="${PROJECT_ID}" --quiet

# 2. Create GCS bucket for RVF containers
echo "=== Step 2: GCS Bucket ==="
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud storage buckets create "gs://${BUCKET_NAME}" \
        --project="${PROJECT_ID}" \
        --location="${REGION}" \
        --uniform-bucket-level-access \
        --public-access-prevention
    echo "Created bucket: ${BUCKET_NAME}"
else
    echo "Bucket already exists: ${BUCKET_NAME}"
fi

# Lifecycle: archive after 90 days, delete after 365 days
gcloud storage buckets update "gs://${BUCKET_NAME}" \
    --lifecycle-file=<(cat <<'LIFECYCLE'
{
  "rule": [
    {"action": {"type": "SetStorageClass", "storageClass": "NEARLINE"}, "condition": {"age": 90}},
    {"action": {"type": "Delete"}, "condition": {"age": 365}}
  ]
}
LIFECYCLE
)

# 3. Firestore — use the existing (default) database (free tier)
echo "=== Step 3: Firestore ==="
echo "Using existing (default) Firestore database"

# 4. Secret Manager — brain-specific secrets
echo "=== Step 4: Secrets ==="
if ! gcloud secrets describe brain-api-key --project="${PROJECT_ID}" &>/dev/null; then
    openssl rand -hex 32 | gcloud secrets create brain-api-key \
        --data-file=- --project="${PROJECT_ID}" --replication-policy="automatic"
    echo "Created secret: brain-api-key"
else
    echo "Secret already exists: brain-api-key"
fi

if ! gcloud secrets describe brain-signing-key --project="${PROJECT_ID}" &>/dev/null; then
    openssl rand -base64 64 | gcloud secrets create brain-signing-key \
        --data-file=- --project="${PROJECT_ID}" --replication-policy="automatic"
    echo "Created secret: brain-signing-key"
else
    echo "Secret already exists: brain-signing-key"
fi

# 5. Service account with minimal permissions
echo "=== Step 5: IAM ==="
SA_NAME="mcp-brain-server"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "${SA}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud iam service-accounts create "${SA_NAME}" \
        --project="${PROJECT_ID}" \
        --display-name="MCP Brain Server"
    echo "Created service account: ${SA}"
else
    echo "Service account already exists: ${SA}"
fi

# Grant permissions
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA}" --role="roles/datastore.user" --quiet
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
    --member="serviceAccount:${SA}" --role="roles/storage.objectAdmin" --quiet
for SECRET in brain-api-key brain-signing-key; do
    gcloud secrets add-iam-policy-binding "${SECRET}" \
        --project="${PROJECT_ID}" \
        --member="serviceAccount:${SA}" \
        --role="roles/secretmanager.secretAccessor" --quiet
done

# 6. Build container image and push to GCR
echo "=== Step 6: Build ==="
FIRESTORE_URL="https://firestore.googleapis.com/v1/projects/${PROJECT_ID}/databases/(default)/documents"

echo "Building release binary..."
cargo build --release -p mcp-brain-server

echo "Building Docker image..."
cp target/release/mcp-brain-server /tmp/mcp-brain-server
docker build -t "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest" \
    -f crates/mcp-brain-server/Dockerfile.runtime /tmp/

echo "Pushing to GCR..."
gcloud auth configure-docker gcr.io --quiet
docker push "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

# 7. Deploy Cloud Run
echo "=== Step 7: Cloud Run Deploy ==="
if [ "${DEPLOY_PATH}" = "A" ]; then
    AUTH_FLAG="--allow-unauthenticated"
else
    AUTH_FLAG="--no-allow-unauthenticated"
fi

gcloud run deploy "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest" \
    --service-account="${SA}" \
    --cpu=2 --memory=2Gi \
    --min-instances=0 --max-instances=10 \
    --concurrency=80 --port=8080 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GCS_BUCKET=${BUCKET_NAME},FIRESTORE_URL=${FIRESTORE_URL},RUST_LOG=info" \
    --set-secrets="BRAIN_API_KEY=brain-api-key:latest,BRAIN_SIGNING_KEY=brain-signing-key:latest" \
    ${AUTH_FLAG} --quiet

# Get the Cloud Run URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" --region="${REGION}" \
    --format='value(status.url)')
echo ""
echo "Cloud Run URL: ${SERVICE_URL}"

# 8. Domain setup (deferred — Cloudflare will handle brain.ruv.io later)
if [ "${DEPLOY_PATH}" = "B" ]; then
    echo "=== Step 8: External HTTPS LB + Serverless NEG (Path B) ==="

    # Serverless NEG → Cloud Run
    gcloud compute network-endpoint-groups create brain-neg \
        --project="${PROJECT_ID}" --region="${REGION}" \
        --network-endpoint-type=serverless \
        --cloud-run-service="${SERVICE_NAME}" 2>/dev/null || true

    # Backend service
    gcloud compute backend-services create brain-backend \
        --project="${PROJECT_ID}" --global --protocol=HTTPS --port-name=http 2>/dev/null || true
    gcloud compute backend-services add-backend brain-backend \
        --project="${PROJECT_ID}" --global \
        --network-endpoint-group=brain-neg \
        --network-endpoint-group-region="${REGION}" 2>/dev/null || true

    # URL map
    gcloud compute url-maps create brain-lb \
        --project="${PROJECT_ID}" --default-service=brain-backend 2>/dev/null || true

    # Static IP + managed SSL cert
    gcloud compute addresses create brain-ip \
        --project="${PROJECT_ID}" --global 2>/dev/null || true
    gcloud compute ssl-certificates create brain-cert \
        --project="${PROJECT_ID}" --domains="brain.ruv.io" --global 2>/dev/null || true

    # HTTPS proxy + forwarding rule
    gcloud compute target-https-proxies create brain-https-proxy \
        --project="${PROJECT_ID}" --url-map=brain-lb \
        --ssl-certificates=brain-cert 2>/dev/null || true
    BRAIN_IP=$(gcloud compute addresses describe brain-ip --project="${PROJECT_ID}" --global --format='value(address)')
    gcloud compute forwarding-rules create brain-https-rule \
        --project="${PROJECT_ID}" --global \
        --target-https-proxy=brain-https-proxy \
        --ports=443 --address="${BRAIN_IP}" 2>/dev/null || true

    # Cloud Armor WAF
    echo "=== Step 9: Cloud Armor ==="
    gcloud compute security-policies create brain-waf-policy \
        --project="${PROJECT_ID}" --description="WAF for brain.ruv.io" 2>/dev/null || true
    gcloud compute security-policies rules create 1000 \
        --project="${PROJECT_ID}" --security-policy="brain-waf-policy" \
        --expression="true" --action="rate-based-ban" \
        --rate-limit-threshold-count=1000 --rate-limit-threshold-interval-sec=60 \
        --ban-duration-sec=300 2>/dev/null || true
    gcloud compute security-policies rules create 2000 \
        --project="${PROJECT_ID}" --security-policy="brain-waf-policy" \
        --expression="evaluatePreconfiguredExpr('sqli-v33-stable')" \
        --action="deny-403" 2>/dev/null || true
    gcloud compute security-policies rules create 2001 \
        --project="${PROJECT_ID}" --security-policy="brain-waf-policy" \
        --expression="evaluatePreconfiguredExpr('xss-v33-stable')" \
        --action="deny-403" 2>/dev/null || true
    gcloud compute backend-services update brain-backend \
        --project="${PROJECT_ID}" --global --security-policy=brain-waf-policy

    # Cloud CDN
    gcloud compute backend-services update brain-backend \
        --project="${PROJECT_ID}" --global --enable-cdn

    echo "DNS: A record brain.ruv.io → ${BRAIN_IP}"
else
    echo ""
    echo "=== Path A: Direct Cloud Run URL ==="
    echo "Domain mapping deferred — Cloudflare DNS will be configured separately."
fi

echo ""
echo "=== Deployment Complete ==="
echo "Service URL: ${SERVICE_URL}"
echo "Health:      ${SERVICE_URL}/v1/health"
echo "Status:      ${SERVICE_URL}/v1/status"
