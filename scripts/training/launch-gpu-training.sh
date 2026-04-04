#!/bin/bash
# Launch deobfuscation model training on GCloud L4 GPU.
#
# Usage:
#   ./scripts/training/launch-gpu-training.sh --local     # Train on local GPU
#   ./scripts/training/launch-gpu-training.sh --cloud-run  # Cloud Run Job with GPU
#   ./scripts/training/launch-gpu-training.sh --spot       # Spot instance (cheapest)
#
# Estimated cost: ~$1.40 (on-demand) or ~$0.57 (spot) for 2-hour training.

set -euo pipefail

PROJECT="${GCP_PROJECT:-ruv-dev}"
REGION="us-central1"
ZONE="us-central1-a"
BUCKET="gs://${PROJECT}-training"
IMAGE="gcr.io/${PROJECT}/deobfuscator-trainer:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:---local}"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }

check_deps() {
    if [ "$MODE" != "--local" ]; then
        command -v gcloud >/dev/null 2>&1 || { echo "ERROR: gcloud CLI not installed"; exit 1; }
    fi
    command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
}

ensure_data() {
    local data_path="${SCRIPT_DIR}/../../training-data.jsonl"
    if [ ! -f "$data_path" ]; then
        log "Generating training data..."
        node "${SCRIPT_DIR}/generate-deobfuscation-data.mjs" --output "$data_path"
    fi
    echo "$data_path"
}

# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

train_local() {
    log "Training locally..."
    local data_path
    data_path=$(ensure_data)
    local output_dir="${SCRIPT_DIR}/../../model"

    python3 "${SCRIPT_DIR}/train-deobfuscator.py" \
        --data "$data_path" \
        --output "$output_dir" \
        --epochs 30 \
        --batch-size 64 \
        --export-onnx

    log "Exporting to GGUF Q4 + RVF..."
    python3 "${SCRIPT_DIR}/export-to-rvf.py" \
        --checkpoint "${output_dir}/best_model.pt" \
        --output "${output_dir}/deobfuscator" \
        --quantize q4

    log "Done. Model at ${output_dir}/deobfuscator.gguf"
}

# ---------------------------------------------------------------------------
# Cloud Run Job with GPU
# ---------------------------------------------------------------------------

train_cloud_run() {
    log "Launching Cloud Run GPU job..."

    # Upload training data.
    local data_path
    data_path=$(ensure_data)
    gsutil -q cp "$data_path" "${BUCKET}/deobfuscation-data.jsonl"
    log "Uploaded training data to ${BUCKET}/"

    # Build and push container if needed.
    if ! gcloud container images describe "$IMAGE" >/dev/null 2>&1; then
        log "Building container..."
        gcloud builds submit \
            --tag "$IMAGE" \
            --timeout=600 \
            "${SCRIPT_DIR}" \
            -f "${SCRIPT_DIR}/Dockerfile.deobfuscator"
    fi

    # Create or update the job.
    gcloud run jobs create deobfuscator-train \
        --image="$IMAGE" \
        --task-timeout=7200 \
        --max-retries=1 \
        --cpu=4 \
        --memory=16Gi \
        --gpu=1 \
        --gpu-type=nvidia-l4 \
        --region="$REGION" \
        --set-env-vars="DATA_PATH=/tmp/data.jsonl,OUTPUT_DIR=/tmp/model,GCS_BUCKET=${BUCKET}" \
        --quiet 2>/dev/null || \
    gcloud run jobs update deobfuscator-train \
        --image="$IMAGE" \
        --region="$REGION" \
        --quiet

    # Execute the job.
    log "Starting training job..."
    gcloud run jobs execute deobfuscator-train \
        --region="$REGION" \
        --wait

    # Download results.
    log "Downloading trained model..."
    local output_dir="${SCRIPT_DIR}/../../model"
    mkdir -p "$output_dir"
    gsutil -q cp "${BUCKET}/models/deobfuscator/*" "$output_dir/"

    log "Done. Model at ${output_dir}/"
}

# ---------------------------------------------------------------------------
# Spot instance (cheapest)
# ---------------------------------------------------------------------------

train_spot() {
    log "Launching spot instance for training..."

    # Upload training data and scripts.
    local data_path
    data_path=$(ensure_data)
    gsutil -q cp "$data_path" "${BUCKET}/deobfuscation-data.jsonl"
    gsutil -q cp "${SCRIPT_DIR}/train-deobfuscator.py" "${BUCKET}/train-deobfuscator.py"
    gsutil -q cp "${SCRIPT_DIR}/export-to-rvf.py" "${BUCKET}/export-to-rvf.py"
    log "Uploaded data and scripts to ${BUCKET}/"

    # Create spot instance with startup script.
    local instance_name="deobfuscator-trainer-$(date +%s)"

    gcloud compute instances create "$instance_name" \
        --zone="$ZONE" \
        --machine-type=g2-standard-4 \
        --accelerator=type=nvidia-l4,count=1 \
        --maintenance-policy=TERMINATE \
        --provisioning-model=SPOT \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=50GB \
        --scopes=storage-full \
        --metadata=startup-script="$(cat <<'STARTUP_EOF'
#!/bin/bash
set -euo pipefail
export BUCKET=BUCKET_PLACEHOLDER

# Download data and scripts.
gsutil cp ${BUCKET}/deobfuscation-data.jsonl /tmp/data.jsonl
gsutil cp ${BUCKET}/train-deobfuscator.py /tmp/train-deobfuscator.py
gsutil cp ${BUCKET}/export-to-rvf.py /tmp/export-to-rvf.py

# Install dependencies.
pip install torch onnx numpy

# Train.
cd /tmp
python train-deobfuscator.py --data data.jsonl --output /tmp/model --epochs 30 --export-onnx

# Export to GGUF Q4.
python export-to-rvf.py --checkpoint /tmp/model/best_model.pt --output /tmp/model/deobfuscator --quantize q4

# Upload results.
gsutil -m cp /tmp/model/* ${BUCKET}/models/deobfuscator/

# Self-destruct.
gcloud compute instances delete $(hostname) --zone=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/zone -H "Metadata-Flavor: Google" | cut -d'/' -f4) --quiet
STARTUP_EOF
)" \
        --quiet

    # Replace bucket placeholder.
    gcloud compute instances add-metadata "$instance_name" \
        --zone="$ZONE" \
        --metadata=startup-script="$(gcloud compute instances describe "$instance_name" --zone="$ZONE" --format='value(metadata.items[startup-script])' | sed "s|BUCKET_PLACEHOLDER|${BUCKET}|g")" \
        --quiet

    log "Spot instance '$instance_name' launched."
    log "Monitor: gcloud compute instances get-serial-port-output $instance_name --zone=$ZONE"
    log "Results will appear at: ${BUCKET}/models/deobfuscator/"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

check_deps

case "$MODE" in
    --local)
        train_local
        ;;
    --cloud-run)
        train_cloud_run
        ;;
    --spot)
        train_spot
        ;;
    --help|-h)
        echo "Usage: $0 [--local|--cloud-run|--spot]"
        echo ""
        echo "  --local       Train on local machine (GPU or CPU)"
        echo "  --cloud-run   Use Cloud Run Job with L4 GPU (~\$1.40)"
        echo "  --spot        Use Compute Engine spot instance (~\$0.57)"
        exit 0
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [--local|--cloud-run|--spot]"
        exit 1
        ;;
esac
