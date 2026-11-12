#!/usr/bin/env bash
# OpenJev v0 training job (ADR-008, docs/research/openjev/v0-plan.md Step 6).
#
# Runs INSIDE a ruvector-gpu-runner instance (repo checked out at the pinned SHA,
# cwd = repo root, $RVGR_ARTIFACT_DIR set). Also runs locally for a dry pass:
#   RVGR_ARTIFACT_DIR=/tmp/art OPENJEV_BIN=target-cuda/release/openjev \
#     scripts/openjev/vast-train.sh --seed 1 --train-args "--max-steps 50 --val-limit 64"
#
# Launch (coordinator). The image MUST be Ubuntu 24.04-based: ort 2.0.0-rc.13's
# prebuilt static onnxruntime (used by the engine's native path, hence by
# `parity`) needs glibc >= 2.38 and GCC 13 libstdc++; the runner's default
# cuda:12.4.1-devel-ubuntu22.04 (glibc 2.35) fails to link (__isoc23_strtol).
# CUDA 12.6 needs host driver >= 560.
#   IMG=nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a
#   ruvector-gpu-runner launch --dry-run --max-usd 5 --max-hours 3 --image "$IMG" --min-cuda 12.6 \
#     --git-ref feat/openjev-train --git-sha "$(git rev-parse HEAD)" \
#     --cmd 'bash scripts/openjev/vast-train.sh --seed 1'
#
# Stages (each writes into $RVGR_ARTIFACT_DIR as it goes, so a timeout still
# uploads everything finished so far):
#   toolchain → build (--features cuda) → fetch (pinned URLs, sha256) → prep
#   (+ Assertion A) → train → transplant → parity → SHA256SUMS
#
# Artifact layout, per seed: $RVGR_ARTIFACT_DIR/seed-N/
#   encoder.safetensors  heads.safetensors (never shipped)  heads-discarded.txt
#   onnx/model.onnx  tokenizer.json  train-text-hashes.txt     <- a bare bench model dir:
#       node bench/run.mjs --model-dir seed-N --model openjev-small-v0 ...
#   training-run.json  curves.jsonl  leakage-report.json  transplant-report.json
#   parity.json  SHA256SUMS
# plus $RVGR_ARTIFACT_DIR/{data-manifest.json,data-leakage-report.json,env.json,logs/}.
#
# No credentials are needed or used. Nothing is published.
set -euo pipefail

SEEDS=()
CONFIG=crates/ruvector-typesafe-train/configs/openjev-small-v0.toml
TRAIN_ARGS=""
DEVICE=cuda:0
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.98.1}
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEEDS+=("$2"); shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --train-args) TRAIN_ARGS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) sed -n 2,30p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ ${#SEEDS[@]} -gt 0 ] || SEEDS=(1)
for s in "${SEEDS[@]}"; do [[ "$s" =~ ^[0-9]+$ ]] || { echo "bad seed $s" >&2; exit 2; }; done

ART="${RVGR_ARTIFACT_DIR:?RVGR_ARTIFACT_DIR must be set}"
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
WORK="${OPENJEV_WORK:-$REPO/runs}"
CACHE="$WORK/cache"
DATA="$WORK/data"
mkdir -p "$ART/logs" "$CACHE"
log() { echo "[openjev] $(date -u +%FT%TZ) $*"; }
t0=$(date +%s)
stamp() { echo "{\"stage\":\"$1\",\"seconds\":$(( $(date +%s) - t0 ))}" >> "$ART/logs/stages.jsonl"; }

# ---- 0. preflight: glibc >= 2.38 (see header) --------------------------------
GLIBC="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
[ -n "$GLIBC" ] || GLIBC=0
if [ "$(printf '%s\n' 2.38 "$GLIBC" | sort -V | head -1)" != 2.38 ]; then
  echo "glibc $GLIBC < 2.38: ort's prebuilt onnxruntime will not link; use an Ubuntu 24.04 image (see header)" >&2
  exit 2
fi

# ---- 1. toolchain -----------------------------------------------------------
if [ -z "${OPENJEV_BIN:-}" ]; then
  # libssl-dev: ort-sys's build script downloads onnxruntime via ureq/native-tls
  # (openssl-sys); the nvidia/cuda devel image does not ship the headers.
  if ! command -v cc >/dev/null || ! command -v pkg-config >/dev/null || ! pkg-config --exists openssl; then
    log "installing build-essential pkg-config libssl-dev"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      build-essential pkg-config libssl-dev ca-certificates curl git >/dev/null
  fi
  if ! command -v cargo >/dev/null; then
    log "installing rust $RUST_TOOLCHAIN"
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
  fi
  # shellcheck disable=SC1091
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
  stamp toolchain

  # ---- 2. build (candle CUDA kernels for this GPU's compute capability) -----
  if [ "$DEVICE" != cpu ] && [ -z "${CUDA_COMPUTE_CAP:-}" ]; then
    CUDA_COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
    export CUDA_COMPUTE_CAP
  fi
  FEATURES=(); [ "$DEVICE" = cpu ] || FEATURES=(--features cuda)
  log "cargo build (compute cap ${CUDA_COMPUTE_CAP:-n/a}) ${FEATURES[*]}"
  # One retry: rustc has been seen to SIGSEGV transiently on long release
  # builds; cargo resumes from the finished units.
  build() { cargo build --release --locked -p ruvector-typesafe-train "${FEATURES[@]}" 2>&1 | tee -a "$ART/logs/build.log" | tail -3; }
  build || { log "build failed; retrying once"; build; }
  BIN="$REPO/target/release/openjev"
  stamp build
else
  BIN="$OPENJEV_BIN"
fi
[ -x "$BIN" ] || { echo "openjev binary missing: $BIN" >&2; exit 1; }

{
  echo "{"
  echo "  \"git_sha\": \"$(git rev-parse HEAD)\","
  echo "  \"gpu\": \"$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)\","
  echo "  \"cpus\": $(nproc),"
  echo "  \"rustc\": \"$(rustc --version 2>/dev/null || echo n/a)\","
  echo "  \"nvcc\": \"$(nvcc --version 2>/dev/null | tail -1 || echo n/a)\","
  echo "  \"device\": \"$DEVICE\", \"config\": \"$CONFIG\", \"seeds\": \"${SEEDS[*]}\", \"train_args\": \"$TRAIN_ARGS\""
  echo "}"
} > "$ART/env.json"
export OPENJEV_GIT_SHA="$(git rev-parse HEAD)"

# ---- 3. fetch every pinned input (sha256-verified; fails closed) -----------
log "fetch pinned inputs"
"$BIN" fetch --download --cache "$CACHE" 2>&1 | tee "$ART/logs/fetch.log"
stamp fetch

# ---- 4. prep: rows + held-out hashes + Assertion A ------------------------
log "prep"
"$BIN" prep --cache "$CACHE" --out "$DATA" > "$ART/data-leakage-report.json" 2> "$ART/logs/prep.log"
cp "$DATA/data-manifest.json" "$ART/data-manifest.json"
stamp prep

for SEED in "${SEEDS[@]}"; do
  RUN="$ART/seed-$SEED"
  mkdir -p "$RUN/onnx"
  # ---- 5. train (prep-check + Assertion A run first; exit 3 on leakage) ----
  log "train seed $SEED"
  # shellcheck disable=SC2086
  "$BIN" train --cache "$CACHE" --data "$DATA" --config "$CONFIG" --seed "$SEED" \
    --device "$DEVICE" --out "$RUN" $TRAIN_ARGS > /dev/null 2> >(tee "$ART/logs/train-seed-$SEED.log" >&2)
  stamp "train-$SEED"

  # ---- 6. transplant into the pinned ONNX graph ---------------------------
  log "transplant seed $SEED"
  "$BIN" transplant --cache "$CACHE" --weights "$RUN/encoder.safetensors" \
    --out "$RUN/onnx/model.onnx" --report "$RUN/transplant-report.json" 2>&1 | tee "$ART/logs/transplant-seed-$SEED.log"
  cp "$CACHE/tokenizer.json" "$RUN/tokenizer.json"
  stamp "transplant-$SEED"

  # ---- 7. parity: ort(ONNX) vs candle on CPU, cosine >= 0.9999 + decisions --
  log "parity seed $SEED"
  "$BIN" parity --cache "$CACHE" --onnx "$RUN/onnx/model.onnx" --weights "$RUN/encoder.safetensors" \
    --data "$DATA" --probes 512 --device cpu --out "$RUN/parity.json" > /dev/null 2> "$ART/logs/parity-seed-$SEED.log"
  stamp "parity-$SEED"

  (cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  log "seed $SEED done: $(grep -o '"best_val_selection": [0-9.]*' "$RUN/training-run.json")"
done
log "all done in $(( $(date +%s) - t0 ))s"
