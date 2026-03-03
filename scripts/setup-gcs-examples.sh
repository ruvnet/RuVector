#!/bin/bash
# Setup GCS bucket for RVF example hosting
# Run once to create the infrastructure
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - GCP project set: gcloud config set project YOUR_PROJECT
#
# Usage: bash scripts/setup-gcs-examples.sh

set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
BUCKET_NAME="ruvector-examples"
REGION="us-central1"
SA_NAME="rvf-examples-sync"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== RVF Examples GCS Setup ==="
echo "Project: $PROJECT_ID"
echo "Bucket:  gs://$BUCKET_NAME"
echo "Region:  $REGION"
echo ""

# 1. Create bucket
echo "Creating GCS bucket..."
gcloud storage buckets create "gs://$BUCKET_NAME" \
  --location="$REGION" \
  --uniform-bucket-level-access \
  --public-access-prevention=inherited \
  2>/dev/null || echo "  Bucket already exists"

# 2. Enable public read access
echo "Setting public read access..."
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET_NAME" \
  --member=allUsers \
  --role=roles/storage.objectViewer \
  2>/dev/null || echo "  Already set"

# 3. Set CORS policy for browser access
echo "Setting CORS policy..."
cat > /tmp/cors.json << 'CORS'
[
  {
    "origin": ["*"],
    "method": ["GET", "HEAD"],
    "responseHeader": ["Content-Type", "Content-Length", "Content-Disposition"],
    "maxAgeSeconds": 3600
  }
]
CORS
gsutil cors set /tmp/cors.json "gs://$BUCKET_NAME"
rm /tmp/cors.json

# 4. Set lifecycle policy (archive old versions after 90 days)
echo "Setting lifecycle policy..."
cat > /tmp/lifecycle.json << 'LIFECYCLE'
{
  "rule": [
    {
      "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
      "condition": {"age": 90, "matchesPrefix": ["v"]}
    },
    {
      "action": {"type": "Delete"},
      "condition": {"age": 365, "matchesPrefix": ["v"]}
    }
  ]
}
LIFECYCLE
gsutil lifecycle set /tmp/lifecycle.json "gs://$BUCKET_NAME"
rm /tmp/lifecycle.json

# 5. Create service account for CI/CD
echo "Creating service account..."
gcloud iam service-accounts create "$SA_NAME" \
  --display-name="RVF Examples Sync" \
  2>/dev/null || echo "  Already exists"

# 6. Grant write access to service account
echo "Granting write access..."
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET_NAME" \
  --member="serviceAccount:$SA_EMAIL" \
  --role=roles/storage.objectAdmin \
  2>/dev/null || echo "  Already set"

# 7. Setup Workload Identity Federation for GitHub Actions
echo "Setting up Workload Identity Federation..."
POOL_NAME="github-pool"
PROVIDER_NAME="github-provider"

gcloud iam workload-identity-pools create "$POOL_NAME" \
  --location=global \
  --display-name="GitHub Actions Pool" \
  2>/dev/null || echo "  Pool already exists"

gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_NAME" \
  --location=global \
  --workload-identity-pool="$POOL_NAME" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == 'ruvnet/ruvector'" \
  2>/dev/null || echo "  Provider already exists"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/$POOL_NAME/attribute.repository/ruvnet/ruvector" \
  2>/dev/null || echo "  Binding already set"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "GitHub Secrets to configure:"
echo "  GCP_WIF_PROVIDER: projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/$POOL_NAME/providers/$PROVIDER_NAME"
echo "  GCS_SERVICE_ACCOUNT: $SA_EMAIL"
echo ""
echo "To upload examples now:"
echo "  python3 scripts/generate-rvf-manifest.py -i examples/rvf/output/ -v \$(jq -r .version npm/packages/ruvector/package.json) -o examples/rvf/manifest.json"
echo "  gsutil -m cp examples/rvf/output/*.rvf gs://$BUCKET_NAME/v\$(jq -r .version npm/packages/ruvector/package.json)/"
echo "  gsutil cp examples/rvf/manifest.json gs://$BUCKET_NAME/manifest.json"
