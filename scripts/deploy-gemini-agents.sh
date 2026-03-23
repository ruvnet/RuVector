#!/bin/bash
# Deploy Gemini grounding agents as Cloud Run Job (ADR-122)
# Usage: bash scripts/deploy-gemini-agents.sh [PROJECT]
set -euo pipefail

PROJECT="${1:-ruv-dev}"
REGION="us-central1"

echo "=== Deploying Gemini Grounding Agents ==="
echo "Project: $PROJECT, Region: $REGION"

# Fetch Gemini API key from Secret Manager
GEMINI_KEY=$(gcloud secrets versions access latest --secret=GOOGLE_AI_API_KEY --project="$PROJECT")
if [ -z "$GEMINI_KEY" ]; then
  echo "ERROR: Could not fetch GOOGLE_AI_API_KEY from Secret Manager"
  exit 1
fi
echo "Gemini API key retrieved from Secret Manager"

# Create temporary build directory
BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT

cp scripts/gemini-agents.js "$BUILD_DIR/agents.js"

cat > "$BUILD_DIR/Dockerfile" <<'DEOF'
FROM node:20-alpine
COPY agents.js /app/agents.js
WORKDIR /app
ENTRYPOINT ["node", "agents.js"]
DEOF

cat > "$BUILD_DIR/env.yaml" <<ENVEOF
BRAIN_URL: "https://pi.ruv.io"
BRAIN_AUTH: "Bearer ruvector-crawl-2026"
GEMINI_API_KEY: "$GEMINI_KEY"
GEMINI_MODEL: "gemini-2.5-flash"
MAX_MEMORIES: "10"
ENVEOF

echo ""
echo "Building and deploying gemini-agents Cloud Run Job..."
gcloud run jobs create gemini-agents \
  --project="$PROJECT" --region="$REGION" \
  --source="$BUILD_DIR" \
  --args="--phase=all" \
  --task-count=1 \
  --max-retries=1 \
  --cpu=1 --memory=512Mi \
  --task-timeout=600s \
  --env-vars-file="$BUILD_DIR/env.yaml" \
  2>/dev/null || \
gcloud run jobs update gemini-agents \
  --project="$PROJECT" --region="$REGION" \
  --source="$BUILD_DIR" \
  --args="--phase=all" \
  --env-vars-file="$BUILD_DIR/env.yaml"

echo ""
echo "Job deployed. To execute manually:"
echo "  gcloud run jobs execute gemini-agents --project=$PROJECT --region=$REGION"

# Create Cloud Scheduler jobs for each phase
echo ""
echo "Creating Cloud Scheduler jobs..."

SA_EMAIL="ruvbrain-scheduler@${PROJECT}.iam.gserviceaccount.com"
JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/gemini-agents:run"

for phase_config in \
  "gemini-fact-verify|0 */6 * * *|fact-verify|Fact verification every 6h" \
  "gemini-relate|30 3 * * *|relate|Relation generation daily 3:30AM" \
  "gemini-cross-domain|0 4 * * *|cross-domain|Cross-domain discovery daily 4AM" \
  "gemini-research|0 */12 * * *|research|Research director every 12h"; do

  IFS='|' read -r name schedule phase desc <<< "$phase_config"
  echo "  Creating scheduler: $name ($schedule)"

  gcloud scheduler jobs create http "$name" \
    --project="$PROJECT" --location="$REGION" \
    --schedule="$schedule" \
    --uri="$JOB_URI" \
    --http-method=POST \
    --oauth-service-account-email="$SA_EMAIL" \
    --description="$desc (ADR-122)" \
    --attempt-deadline=600s \
    2>/dev/null || echo "    (already exists, skipping)"
done

echo ""
echo "=== Deployment Complete ==="
echo "Cloud Run Job: gemini-agents"
echo "Scheduler jobs: gemini-fact-verify, gemini-relate, gemini-cross-domain, gemini-research"
echo ""
echo "To test locally first:"
echo "  GEMINI_API_KEY=\$(gcloud secrets versions access latest --secret=GOOGLE_AI_API_KEY) \\"
echo "    node scripts/gemini-agents.js --phase fact-verify"
