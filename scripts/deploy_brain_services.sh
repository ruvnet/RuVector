#!/usr/bin/env bash
set -euo pipefail

# Deploy ADR-130 brain service split
# Usage: ./scripts/deploy_brain_services.sh [api|sse|worker|all]

PROJECT=ruv-dev
REGION=us-central1

deploy_api() {
    echo "=== Deploying ruvbrain-api ==="
    gcloud builds submit --config=crates/mcp-brain-server/cloudbuild.yaml --project=$PROJECT
    gcloud run deploy ruvbrain \
        --image=gcr.io/$PROJECT/ruvbrain:latest \
        --region=$REGION \
        --memory=4Gi --cpu=2 \
        --concurrency=80 --max-instances=15 --min-instances=1 \
        --timeout=300 --session-affinity \
        --allow-unauthenticated
}

deploy_sse() {
    echo "=== Deploying ruvbrain-sse ==="
    # Build SSE image
    gcloud builds submit \
        --config=crates/mcp-brain-server/cloudbuild-sse.yaml \
        --project=$PROJECT
    # Deploy SSE service
    gcloud run deploy ruvbrain-sse \
        --image=gcr.io/$PROJECT/ruvbrain-sse:latest \
        --region=$REGION \
        --memory=512Mi --cpu=1 \
        --concurrency=500 --max-instances=10 --min-instances=0 \
        --timeout=3600 --session-affinity \
        --allow-unauthenticated \
        --set-env-vars="BRAIN_API_URL=https://ruvbrain-HASH.us-central1.run.app,RUST_LOG=info"
    # Note: BRAIN_API_URL needs the actual Cloud Run URL of ruvbrain-api
}

deploy_worker() {
    echo "=== Deploying ruvbrain-worker ==="
    # Build worker image
    gcloud builds submit \
        --config=crates/mcp-brain-server/cloudbuild-worker.yaml \
        --project=$PROJECT
    # Create Cloud Run Job (not a service)
    gcloud run jobs create ruvbrain-worker \
        --image=gcr.io/$PROJECT/ruvbrain-worker:latest \
        --region=$REGION \
        --memory=4Gi --cpu=2 \
        --max-retries=1 --task-timeout=3600s \
        --set-env-vars="RUST_LOG=info" \
        --set-secrets=BRAIN_API_KEY=brain-api-key:latest,BRAIN_SIGNING_KEY=brain-signing-key:latest \
        2>/dev/null || \
    gcloud run jobs update ruvbrain-worker \
        --image=gcr.io/$PROJECT/ruvbrain-worker:latest \
        --region=$REGION \
        --memory=4Gi --cpu=2
}

# Parse argument
case "${1:-all}" in
    api) deploy_api ;;
    sse) deploy_sse ;;
    worker) deploy_worker ;;
    all) deploy_api && deploy_sse && deploy_worker ;;
    *) echo "Usage: $0 [api|sse|worker|all]"; exit 1 ;;
esac

echo "=== Done ==="
