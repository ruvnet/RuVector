# RuvLTRA Training Pipeline
# Supports: LoRA SFT, DPO, imatrix calibration, GGUF conversion
# Target: Cloud Run Jobs with L4 GPU or Vertex AI

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip git curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install Python dependencies (prebuilt wheels, no CUDA compilation needed)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    transformers>=4.44.0 \
    peft>=0.12.0 \
    trl>=0.9.0 \
    datasets>=2.20.0 \
    huggingface_hub>=0.24.0 \
    accelerate>=0.33.0 \
    bitsandbytes>=0.43.0 \
    sentencepiece \
    protobuf \
    safetensors

# Install llama-cpp-python prebuilt CUDA wheel (no compilation)
RUN pip install --no-cache-dir llama-cpp-python>=0.2.80 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    || pip install --no-cache-dir llama-cpp-python>=0.2.80 \
    || echo "WARN: llama-cpp-python install failed, using transformers-only mode"

# Install llama.cpp tools via pip (no source build needed)
RUN pip install --no-cache-dir gguf

WORKDIR /app
COPY . /app/

ENTRYPOINT ["python", "-u"]
CMD ["run_calibration.py"]
