FROM python:3.12.11-slim@sha256:47ae396f09c1303b8653019811a8498470603d7ffefc29cb07c88f1f8cb3d19f
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1 HF_HOME=/opt/model-cache
WORKDIR /opt/runtime
COPY requirements.txt requirements.lock pyproject.toml __init__.py app.py auth.py engine.py store.py limits.py ./
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.lock \
    && pip install --no-cache-dir --no-deps . \
    && python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; from ruvllm_microlora_runtime.engine import MODEL, REVISION; AutoTokenizer.from_pretrained(MODEL, revision=REVISION); AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION)" \
    && useradd --uid 10001 --create-home runtime \
    && mkdir -p /var/lib/microlora/adapters \
    && chmod 0700 /var/lib/microlora/adapters \
    && chown -R runtime:runtime /var/lib/microlora /opt/model-cache
USER runtime
ENV MICROLORA_ADAPTER_ROOT=/var/lib/microlora/adapters HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8080
CMD ["uvicorn", "ruvllm_microlora_runtime.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
