# RuvLTRA Training Pipeline
# Supports: LoRA SFT, DPO, calibration, TurboQuant profiling, GGUF conversion
# Target: Cloud Run Jobs with L4 GPU
# Build time: ~5 min (no CUDA source compilation)

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (libgomp needed for llama-cpp-python, git for HF downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Core ML dependencies (single layer for cache efficiency)
RUN pip install \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install \
    transformers>=4.44.0 \
    peft>=0.12.0 \
    trl>=0.9.0 \
    datasets>=2.20.0 \
    huggingface_hub>=0.24.0 \
    accelerate>=0.33.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece protobuf safetensors gguf

# llama-cpp-python with prebuilt CUDA wheel
RUN pip install llama-cpp-python>=0.2.80 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    || pip install llama-cpp-python>=0.2.80 \
    || echo "WARN: llama-cpp-python unavailable, using transformers-only mode"

WORKDIR /app
COPY . /app/

ENTRYPOINT ["python", "-u"]
CMD ["run_calibration.py", "--model-id", "ruv/ruvltra-small"]
