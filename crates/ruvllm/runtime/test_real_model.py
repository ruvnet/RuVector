"""Opt-in proof using pinned downloaded pretrained weights; never a mock model."""
import os
import time
import pytest
import torch
from fastapi.testclient import TestClient
from ruvllm_microlora_runtime.app import create_app
from ruvllm_microlora_runtime.engine import Engine, MODEL, REVISION
from ruvllm_microlora_runtime.store import AdapterStore
from ruvllm_microlora_runtime.test_runtime import authority, bearer


@pytest.mark.skipif(os.getenv('MICROLORA_REAL_MODEL_TEST') != '1', reason='Downloads pretrained CPU model')
def test_real_adapter_changes_logits_and_restores_baseline(tmp_path):
    engine = Engine()
    auth, private, claims = authority()
    claims['exp'] = int(time.time()) + 299
    store = AdapterStore(tmp_path)
    client = TestClient(create_app(engine, auth, store))
    headers = {'authorization': bearer(private, claims)}
    probe = engine.tokenizer('The capital of France is', return_tensors='pt')
    with torch.no_grad(), engine.model.disable_adapter():
        before = engine.model(**probe).logits.detach().clone()
    payload = {'account_id': 'tenant', 'host': 'owned-cpu-proof', 'model': MODEL, 'rank': 2,
               'quality': 1, 'interaction_summaries': ['The capital of France is Paris. Paris is the capital of France.'],
               'budget_usd': 0}
    trained = client.post('/microlora/adapt', json=payload, headers=headers)
    assert trained.status_code == 200, trained.text
    receipt = trained.json()
    directory, metadata = store.resolve(receipt['adapter_handle'], 'tenant', payload['host'], MODEL)
    engine.model.load_adapter(directory / metadata['artifact_path'], adapter_name='proof')
    engine.model.set_adapter('proof')
    with torch.no_grad():
        after = engine.model(**probe).logits.detach().clone()
    delta = float((after - before).abs().max())
    assert delta > 0
    engine.model.set_adapter('default')
    engine.model.delete_adapter('proof')
    request = {'account_id': 'tenant', 'host': payload['host'], 'model': MODEL,
               'messages': [{'role': 'user', 'content': 'The capital of France is'}],
               'max_tokens': 8, 'budget_usd': 0}
    baseline = client.post('/microlora/complete', json=request, headers=headers)
    assert baseline.status_code == 200, baseline.text
    request['adapter_handle'] = receipt['adapter_handle']
    candidate = client.post('/microlora/complete', json=request, headers=headers)
    assert candidate.status_code == 200, candidate.text
    assert candidate.json()['adapter_fingerprint'] == receipt['adapter_fingerprint']
    with torch.no_grad(), engine.model.disable_adapter():
        restored = engine.model(**probe).logits.detach()
    assert torch.equal(before, restored)
    assert baseline.json()['usage']['prompt_tokens'] == candidate.json()['usage']['prompt_tokens']
    print({'model': MODEL, 'revision': REVISION, 'training_tokens': receipt['training_tokens'],
           'training_loss': receipt['training_loss'], 'max_logit_delta': delta,
           'fingerprint': receipt['adapter_fingerprint'], 'baseline_restored': True,
           'baseline_text': baseline.json()['text'], 'candidate_text': candidate.json()['text']})
