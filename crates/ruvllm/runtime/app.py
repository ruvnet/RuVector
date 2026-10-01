"""Optional runtime companion. Run: uvicorn ruvllm_microlora_runtime.app:create_app --factory."""
import math
from datetime import datetime, timezone
import os
import time
from typing import Literal
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from ruvllm_microlora_runtime.auth import IdentityAuth
from ruvllm_microlora_runtime.limits import RequestBodyLimit
from ruvllm_microlora_runtime.store import AdapterStore
from ruvllm_microlora_runtime.engine import Engine, MODEL, REVISION, TRAIN_STEPS, TRAIN_TOKENS, PROMPT_TOKENS


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal['system', 'user', 'assistant']
    content: str = Field(min_length=1, max_length=4096)


class Evaluation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    messages: list[Message] = Field(min_length=1, max_length=16)
    max_tokens: int = Field(default=256, ge=1, le=512)


class Binding(BaseModel):
    model_config = ConfigDict(extra='forbid')
    account_id: str = Field(min_length=1, max_length=256)
    host: str = Field(min_length=1, max_length=256)
    model: Literal['HuggingFaceTB/SmolLM2-135M']


class Adapt(Binding):
    rank: int = Field(default=2, ge=1, le=4)
    quality: float = Field(gt=0, le=1, allow_inf_nan=False)
    interaction_summaries: list[str] = Field(min_length=1, max_length=8)
    budget_usd: float = Field(ge=0, allow_inf_nan=False)


class Quote(Adapt):
    budget_usd: float = Field(default=0, ge=0, allow_inf_nan=False)
    evaluations: list[Evaluation] = Field(default_factory=list, max_length=64)


class Complete(Binding, Evaluation):
    adapter_handle: str | None = None
    budget_usd: float = Field(ge=0, allow_inf_nan=False)


def create_app(engine=None, auth=None, store=None):
    # Tariff explicitly represents runtime compute accounting, never provider billing.
    rate = float(os.environ.get('MICROLORA_COMPUTE_USD_PER_TOKEN', '0'))
    if not math.isfinite(rate) or not 0 <= rate <= 1_000_000:
        raise ValueError('Invalid compute tariff')
    auth = auth or IdentityAuth()
    store = store or AdapterStore(os.environ['MICROLORA_ADAPTER_ROOT'])
    engine = engine or Engine()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestBodyLimit)

    def authorize(request, authorization):
        claims = auth.verify(authorization)
        if claims['account_id'] != request.account_id:
            raise HTTPException(403, 'Tenant mismatch')
        return claims

    def adaptation_quote(request):
        if any(not text.strip() or len(text) > 4096 for text in request.interaction_summaries):
            raise HTTPException(422, 'Summaries must contain 1–4096 characters')
        return len(request.interaction_summaries) * TRAIN_STEPS * TRAIN_TOKENS * rate

    def completion_quote(request):
        return (PROMPT_TOKENS + request.max_tokens) * rate

    def budget(request, ceiling):
        if request.budget_usd < ceiling:
            raise HTTPException(402, 'Budget below reserved compute ceiling')

    @app.get('/health')
    def health():
        return {'status': 'ready', 'model': MODEL, 'model_revision': REVISION,
                'billing': 'configured runtime compute tariff', 'compute_usd_per_token': rate}

    @app.post('/microlora/quote')
    def quote(request: Quote, authorization: str | None = Header(default=None)):
        authorize(request, authorization)
        adaptation = adaptation_quote(request)
        try:
            for evaluation in request.evaluations:
                engine.validate_evaluation(evaluation)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        costs = [completion_quote(e) for e in request.evaluations]
        return {'model': MODEL, 'model_revision': REVISION, 'adaptation_cost_usd': adaptation,
                'completion_costs_usd': costs, 'cost_ceiling_usd': adaptation + sum(costs),
                'compute_usd_per_token': rate, 'billing': 'configured runtime compute tariff'}

    @app.post('/microlora/adapt')
    def adapt(request: Adapt, authorization: str | None = Header(default=None)):
        claims = authorize(request, authorization)
        budget(request, adaptation_quote(request))
        try:
            handle, directory, metrics = engine.adapt(request, lambda: store.allocate(request.account_id), claims['exp'])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if claims['exp'] <= time.time():
            raise HTTPException(401, 'Credential expired during training')
        metadata = store.publish(directory, {'account_id': request.account_id, 'host': request.host,
            'model': MODEL, 'model_revision': REVISION, 'expires_at': claims['exp'], **metrics})
        return {'source': 'LIVE', 'rank': request.rank,
                'samples_seen': len(request.interaction_summaries), 'quality_signal': request.quality,
                'adapter_handle': handle, 'adapter_fingerprint': metadata['adapter_fingerprint'],
                'expires_at': datetime.fromtimestamp(metadata['expires_at'], timezone.utc).isoformat(), 'cost_usd': metrics['training_tokens'] * rate,
                'billing': 'configured runtime compute tariff', **metrics}

    @app.post('/microlora/complete')
    def complete(request: Complete, authorization: str | None = Header(default=None)):
        claims = authorize(request, authorization)
        budget(request, completion_quote(request))
        artifact, metadata = None, None
        if request.adapter_handle:
            directory, metadata = store.resolve(request.adapter_handle, request.account_id, request.host, request.model)
            artifact = directory / metadata['artifact_path']
        try:
            result = engine.complete(request, artifact, min(claims['exp'], metadata['expires_at'] if metadata else claims['exp']))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if claims['exp'] <= time.time() or (metadata and metadata['expires_at'] <= time.time()):
            raise HTTPException(401, 'Authority expired during inference')
        return {**result, 'cost_usd': sum(result['usage'].values()) * rate,
                'billing': 'configured runtime compute tariff',
                'adapter_handle': request.adapter_handle,
                'adapter_fingerprint': metadata['adapter_fingerprint'] if metadata else None}
    return app
