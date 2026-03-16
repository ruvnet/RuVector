#!/usr/bin/env bash
# Setup Google Cloud Pub/Sub for brain data pipeline
# Usage: ./setup-pubsub.sh [PROJECT_ID]

set -euo pipefail

PROJECT_ID="${1:-ruv-dev}"
REGION="us-central1"
SERVICE_URL="https://pi.ruv.io"

echo "Setting up Pub/Sub for brain pipeline in project: $PROJECT_ID"

# Create topics
gcloud pubsub topics create brain-inject \
  --project="$PROJECT_ID" \
  --message-retention-duration=24h \
  --labels=service=ruvbrain,env=prod 2>/dev/null || echo "Topic brain-inject already exists"

gcloud pubsub topics create brain-events \
  --project="$PROJECT_ID" \
  --message-retention-duration=24h \
  --labels=service=ruvbrain,env=prod 2>/dev/null || echo "Topic brain-events already exists"

gcloud pubsub topics create brain-optimize \
  --project="$PROJECT_ID" \
  --message-retention-duration=1h \
  --labels=service=ruvbrain,env=prod 2>/dev/null || echo "Topic brain-optimize already exists"

# Create push subscription for real-time injection
# Pub/Sub pushes directly to Cloud Run endpoint
gcloud pubsub subscriptions create brain-inject-push \
  --project="$PROJECT_ID" \
  --topic=brain-inject \
  --push-endpoint="${SERVICE_URL}/v1/pipeline/pubsub" \
  --push-auth-service-account="ruvbrain-scheduler@${PROJECT_ID}.iam.gserviceaccount.com" \
  --ack-deadline=60 \
  --min-retry-delay=10s \
  --max-retry-delay=600s \
  --message-retention-duration=24h \
  --expiration-period=never \
  --labels=service=ruvbrain,type=push 2>/dev/null || echo "Subscription brain-inject-push already exists"

# Create pull subscription for batch processing
gcloud pubsub subscriptions create brain-inject-pull \
  --project="$PROJECT_ID" \
  --topic=brain-inject \
  --ack-deadline=120 \
  --message-retention-duration=72h \
  --expiration-period=never \
  --labels=service=ruvbrain,type=pull 2>/dev/null || echo "Subscription brain-inject-pull already exists"

# Create event notification subscription (for monitoring/alerting)
gcloud pubsub subscriptions create brain-events-monitor \
  --project="$PROJECT_ID" \
  --topic=brain-events \
  --ack-deadline=30 \
  --message-retention-duration=24h \
  --labels=service=ruvbrain,type=monitor 2>/dev/null || echo "Subscription brain-events-monitor already exists"

# Create service account for scheduler/pubsub if not exists
gcloud iam service-accounts create ruvbrain-scheduler \
  --project="$PROJECT_ID" \
  --display-name="RuVector Brain Scheduler" \
  --description="Service account for Cloud Scheduler and Pub/Sub push" 2>/dev/null || echo "Service account already exists"

# Grant Cloud Run invoker role
gcloud run services add-iam-policy-binding ruvbrain \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --member="serviceAccount:ruvbrain-scheduler@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.invoker" 2>/dev/null || echo "IAM binding already exists"

# Grant Pub/Sub publisher role (for brain to emit events)
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:ruvbrain-scheduler@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/pubsub.publisher" 2>/dev/null || echo "Publisher IAM already set"

echo ""
echo "=== Setup Complete ==="
echo "Topics: brain-inject, brain-events, brain-optimize"
echo "Push endpoint: ${SERVICE_URL}/v1/pipeline/pubsub"
echo ""
echo "Test injection:"
echo "  gcloud pubsub topics publish brain-inject --project=$PROJECT_ID \\"
echo "    --message='{\"source\":\"test\",\"title\":\"Test\",\"content\":\"Hello brain\",\"tags\":[\"test\"]}'"
echo ""
echo "Monitor:"
echo "  gcloud pubsub subscriptions pull brain-events-monitor --project=$PROJECT_ID --auto-ack"
