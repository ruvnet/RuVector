import json
import time
from types import SimpleNamespace
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.testclient import TestClient
from ruvllm_microlora_runtime.auth import IdentityAuth
from ruvllm_microlora_runtime.app import create_app
from ruvllm_microlora_runtime.engine import MODEL
from ruvllm_microlora_runtime.store import AdapterStore


def authority():
    private = ec.generate_private_key(ec.SECP256R1())
    auth = IdentityAuth.__new__(IdentityAuth)
    auth.issuer = 'https://identity.example'
    auth.actor = 'mcp@example'
    auth.keys = SimpleNamespace(get_signing_key_from_jwt=lambda _: SimpleNamespace(key=private.public_key()))
    claims = {'iss': auth.issuer, 'aud': 'meta-proxy', 'sub': 'user', 'account_id': 'tenant',
              'typ': 'access', 'scope': 'platform:microlora:run', 'exchanged': True,
              'act': auth.actor, 'iat': int(time.time()), 'exp': int(time.time()) + 240}
    return auth, private, claims


def bearer(private, claims):
    return 'Bearer ' + jwt.encode(claims, private, algorithm='ES256', headers={'kid': 'test'})


@pytest.mark.parametrize('change', [
    {'aud': 'other'}, {'iss': 'wrong'}, {'account_id': ''}, {'sub': ''}, {'scope': 'inference'},
    {'scope': 'platform:microlora:run inference'}, {'act': 'other'}, {'exchanged': False},
    {'setup': True}, {'workload': True}, {'typ': 'refresh'},
    {'exp': int(time.time()) - 1}, {'exp': int(time.time()) + 901}, {'iat': True}])
def test_auth_rejects_invalid_authority(change):
    auth, private, claims = authority()
    claims.update(change)
    with pytest.raises(HTTPException):
        auth.verify(bearer(private, claims))


def test_auth_rejects_algorithm_and_signature():
    auth, private, claims = authority()
    with pytest.raises(HTTPException):
        auth.verify('Bearer ' + jwt.encode(claims, 'untrusted', algorithm='HS256'))
    other = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(HTTPException):
        auth.verify(bearer(other, claims))
    assert auth.verify(bearer(private, claims))['account_id'] == 'tenant'


def test_persisted_handle_binding_expiry_integrity(tmp_path):
    store = AdapterStore(tmp_path)
    handle, directory = store.allocate()
    (directory / 'adapter.safetensors').write_bytes(b'weight fixture')
    store.publish(directory, {'account_id': 'tenant', 'host': 'host', 'model': MODEL,
                             'expires_at': time.time() + 30})
    assert AdapterStore(tmp_path).resolve(handle, 'tenant', 'host', MODEL)[0] == directory
    for args in [(handle, 'other', 'host', MODEL), (handle, 'tenant', 'wrong', MODEL),
                 ('../metadata', 'tenant', 'host', MODEL)]:
        with pytest.raises(HTTPException):
            store.resolve(*args)
    metadata = json.loads((directory / 'metadata.json').read_text())
    metadata['expires_at'] = 1
    (directory / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(HTTPException):
        store.resolve(handle, 'tenant', 'host', MODEL)


def test_quote_auth_tenant_and_budget_before_compute(tmp_path, monkeypatch):
    monkeypatch.setenv('MICROLORA_COMPUTE_USD_PER_TOKEN', '0.001')
    auth, private, claims = authority()
    class NeverRun:
        def validate_evaluation(self, request):
            return None
        def adapt(self, *args):
            pytest.fail('Budget-denied request executed')
    client = TestClient(create_app(NeverRun(), auth, AdapterStore(tmp_path)))
    body = {'account_id': 'tenant', 'host': 'host', 'model': MODEL, 'rank': 2,
            'quality': .8, 'interaction_summaries': ['summary with real tokens'], 'budget_usd': 0,
            'evaluations': [{'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 8}]}
    headers = {'authorization': bearer(private, claims)}
    assert client.post('/microlora/quote', json=body).status_code == 401
    quote = client.post('/microlora/quote', json=body, headers=headers).json()
    assert quote['adaptation_cost_usd'] == .256
    assert quote['completion_costs_usd'] == [.52]
    body.pop('evaluations')
    assert client.post('/microlora/adapt', json=body, headers=headers).status_code == 402
    body['account_id'] = 'other'
    assert client.post('/microlora/adapt', json=body, headers=headers).status_code == 403


def test_expired_execution_does_not_touch_model():
    import threading
    from ruvllm_microlora_runtime.engine import Engine
    engine = Engine.__new__(Engine)
    engine.lock = threading.Lock()
    with pytest.raises(HTTPException):
        with engine.execution(time.time() - 1):
            pytest.fail('Expired operation executed')
    assert not engine.lock.locked()


def test_unknown_kid_refreshes_jwks_once():
    first = ec.generate_private_key(ec.SECP256R1())
    second = ec.generate_private_key(ec.SECP256R1())
    def public(key, kid):
        data = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(key.public_key()))
        return {**data, 'kid': kid, 'alg': 'ES256', 'use': 'sig'}
    client = jwt.PyJWKClient('https://identity.example/.well-known/jwks.json')
    calls = []
    def get_set(refresh=False):
        calls.append(refresh)
        return jwt.PyJWKSet.from_dict({'keys': [public(second if refresh else first, 'new' if refresh else 'old')]})
    client.get_jwk_set = get_set
    key = client.get_signing_key('new')
    assert key.key_id == 'new'
    assert calls == [False, True]


def test_weight_tampering_is_denied(tmp_path):
    store = AdapterStore(tmp_path)
    handle, directory = store.allocate()
    weight = directory / 'adapter.safetensors'
    weight.write_bytes(b'first')
    store.publish(directory, {'account_id': 'tenant', 'host': 'host', 'model': MODEL,
                             'expires_at': time.time() + 30})
    weight.write_bytes(b'tampered')
    with pytest.raises(HTTPException):
        store.resolve(handle, 'tenant', 'host', MODEL)


def test_adapter_capacity_preserves_existing_state(tmp_path):
    store = AdapterStore(tmp_path, max_adapters=1)
    handle, directory = store.allocate()
    with pytest.raises(HTTPException):
        store.allocate()
    assert directory.exists()


def test_chunked_request_body_is_bounded_before_parser():
    import asyncio
    from ruvllm_microlora_runtime.limits import RequestBodyLimit
    called, sent = [], []
    async def forbidden(scope, receive, send):
        called.append(True)
    chunks = iter([{'type': 'http.request', 'body': b'a' * 65536, 'more_body': True},
                   {'type': 'http.request', 'body': b'b' * 65537, 'more_body': False}])
    async def receive():
        return next(chunks)
    async def send(message):
        sent.append(message)
    asyncio.run(RequestBodyLimit(forbidden)({'type': 'http', 'method': 'POST', 'headers': []}, receive, send))
    assert not called
    assert sent[0]['status'] == 413


def test_tenant_cannot_exhaust_other_tenants_adapter_capacity(tmp_path):
    store = AdapterStore(tmp_path)
    for _ in range(4):
        store.allocate('tenant-one')
    with pytest.raises(HTTPException):
        store.allocate('tenant-one')
    assert store.allocate('tenant-two')[1].exists()
