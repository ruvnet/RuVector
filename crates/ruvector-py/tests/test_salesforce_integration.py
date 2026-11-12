"""Tests for ``ruvector.integrations.salesforce`` (ADR-352).

Every test that touches the network uses ``httpx.MockTransport`` via the
module's own `transport=` dependency-injection seam — **no real Salesforce
org is contacted anywhere in this file**, per the task boundary.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")  # the `salesforce` extra's one dependency

from ruvector import Collection  # noqa: E402
from ruvector.integrations import salesforce as sf  # noqa: E402

if TYPE_CHECKING:
    # `pytest.importorskip` above returns `Any` to mypy, so `httpx.Request`/
    # `httpx.Response` wouldn't resolve as real types in annotations below -
    # a normal import under TYPE_CHECKING fixes that for type-checking only
    # (mypy doesn't care whether httpx is actually installed at check time
    # here the same way it is for the real package's lazy-import modules).
    import httpx as httpx_types

_FAKE_INSTANCE = "https://fake-test.my.salesforce.com"


def _config() -> "sf.SalesforceConfig":
    return sf.SalesforceConfig(
        instance_url=_FAKE_INSTANCE,
        client_id="fake-client-id",
        client_secret="fake-client-secret",
    )


# ── SalesforceConfig.from_env ───────────────────────────────────────────────


def test_config_from_env_reads_all_three_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUVECTOR_SALESFORCE_INSTANCE_URL", _FAKE_INSTANCE)
    monkeypatch.setenv("RUVECTOR_SALESFORCE_CLIENT_ID", "cid")
    monkeypatch.setenv("RUVECTOR_SALESFORCE_CLIENT_SECRET", "csecret")
    cfg = sf.SalesforceConfig.from_env()
    assert cfg.instance_url == _FAKE_INSTANCE
    assert cfg.client_id == "cid"
    assert cfg.client_secret == "csecret"
    assert cfg.api_version == "61.0"


def test_config_from_env_reports_every_missing_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUVECTOR_SALESFORCE_INSTANCE_URL", raising=False)
    monkeypatch.delenv("RUVECTOR_SALESFORCE_CLIENT_ID", raising=False)
    monkeypatch.delenv("RUVECTOR_SALESFORCE_CLIENT_SECRET", raising=False)
    with pytest.raises(ValueError, match="RUVECTOR_SALESFORCE_INSTANCE_URL"):
        sf.SalesforceConfig.from_env()


# ── get_oauth_token / fetch_records (mocked transport, no real org) ────────


def test_get_oauth_token_returns_access_token_from_mocked_response() -> None:
    def handler(request: "httpx_types.Request") -> Any:
        assert request.url.path == "/services/oauth2/token"
        assert request.method == "POST"
        return httpx.Response(200, json={"access_token": "MOCK_TOKEN_1", "token_type": "Bearer"})

    transport = httpx.MockTransport(handler)
    token = asyncio.run(sf.get_oauth_token(_config(), transport=transport))
    assert token == "MOCK_TOKEN_1"


def test_get_oauth_token_raises_on_non_200() -> None:
    def handler(request: "httpx_types.Request") -> Any:
        return httpx.Response(400, json={"error": "invalid_client", "error_description": "bad secret"})

    transport = httpx.MockTransport(handler)
    with pytest.raises(RuntimeError, match="invalid_client"):
        asyncio.run(sf.get_oauth_token(_config(), transport=transport))


def test_get_oauth_token_raises_when_response_has_no_access_token() -> None:
    def handler(request: "httpx_types.Request") -> Any:
        return httpx.Response(200, json={"token_type": "Bearer"})

    transport = httpx.MockTransport(handler)
    with pytest.raises(RuntimeError, match="no access_token"):
        asyncio.run(sf.get_oauth_token(_config(), transport=transport))


def test_fetch_records_single_page() -> None:
    def handler(request: "httpx_types.Request") -> Any:
        if request.url.path == "/services/oauth2/token":
            return httpx.Response(200, json={"access_token": "T"})
        assert request.headers["Authorization"] == "Bearer T"
        assert "q=" in str(request.url)
        return httpx.Response(
            200,
            json={
                "done": True,
                "records": [{"Id": "001a", "Name": "Acme"}, {"Id": "001b", "Name": "Globex"}],
            },
        )

    transport = httpx.MockTransport(handler)
    records = asyncio.run(sf.fetch_records(_config(), "SELECT Id, Name FROM Account", transport=transport))
    assert [r["Id"] for r in records] == ["001a", "001b"]


def test_fetch_records_follows_pagination() -> None:
    calls: List[str] = []

    def handler(request: "httpx_types.Request") -> Any:
        if request.url.path == "/services/oauth2/token":
            return httpx.Response(200, json={"access_token": "T"})
        calls.append(str(request.url.path))
        if request.url.path.endswith("/query"):
            return httpx.Response(
                200,
                json={"done": False, "nextRecordsUrl": "/services/data/v61.0/query/01-page2", "records": [{"Id": "1"}]},
            )
        return httpx.Response(200, json={"done": True, "records": [{"Id": "2"}]})

    transport = httpx.MockTransport(handler)
    records = asyncio.run(sf.fetch_records(_config(), "SELECT Id FROM Account", transport=transport))
    assert [r["Id"] for r in records] == ["1", "2"]
    assert len(calls) == 2  # exactly two query pages, not re-fetching the first


def test_fetch_records_reuses_provided_token_without_oauth_call() -> None:
    token_calls = 0

    def handler(request: "httpx_types.Request") -> Any:
        nonlocal token_calls
        if request.url.path == "/services/oauth2/token":
            token_calls += 1
            return httpx.Response(200, json={"access_token": "SHOULD_NOT_BE_CALLED"})
        assert request.headers["Authorization"] == "Bearer PROVIDED"
        return httpx.Response(200, json={"done": True, "records": []})

    transport = httpx.MockTransport(handler)
    asyncio.run(sf.fetch_records(_config(), "SELECT Id FROM Account", token="PROVIDED", transport=transport))
    assert token_calls == 0


# ── sync_records_to_collection (no network at all) ──────────────────────────


def _fake_embed(dim: int = 3) -> "Any":
    def embed(text: str) -> "np.ndarray[Any, Any]":
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        return rng.standard_normal(dim).astype(np.float32)

    return embed


def test_sync_records_embeds_and_inserts() -> None:
    coll = Collection.create(dim=3)
    records: List[Dict[str, Any]] = [
        {"Id": "001a", "Name": "Acme", "Description__c": "widgets"},
        {"Id": "001b", "Name": "Globex", "Description__c": "gadgets"},
    ]
    ids = sf.sync_records_to_collection(
        coll, records, embed_fn=_fake_embed(), text_field="Description__c",
        metadata_fields=["Name"], id_field="Id",
    )
    assert len(ids) == 2
    assert len(coll) == 2
    meta = coll.get_metadata(ids[0])
    assert meta is not None
    assert meta["Name"] == "Acme"
    assert meta["_salesforce_id"] == "001a"


def test_sync_records_skips_rows_missing_text_field() -> None:
    coll = Collection.create(dim=3)
    records: List[Dict[str, Any]] = [
        {"Id": "001a", "Description__c": "has text"},
        {"Id": "001b"},  # missing Description__c
    ]
    ids = sf.sync_records_to_collection(coll, records, embed_fn=_fake_embed(), text_field="Description__c")
    assert len(ids) == 1
    assert len(coll) == 1


def test_sync_records_empty_input_returns_empty_list_without_touching_collection() -> None:
    coll = Collection.create(dim=3)
    ids = sf.sync_records_to_collection(coll, [], embed_fn=_fake_embed(), text_field="x")
    assert ids == []
    assert len(coll) == 0


# ── Agentforce actions (framework-agnostic, no network) ────────────────────


def test_action_search_returns_hits_shape() -> None:
    coll = Collection.create(dim=4)
    coll.insert(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), metadata={"text": "alpha"})
    coll.insert(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), metadata={"text": "beta"})
    result = sf.action_search(coll, [1.0, 0.0, 0.0, 0.0], k=1)
    assert result["hits"][0]["metadata"] == {"text": "alpha"}


def test_action_search_applies_filter() -> None:
    coll = Collection.create(dim=4)
    coll.insert(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), metadata={"cat": "a"})
    coll.insert(np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32), metadata={"cat": "b"})
    result = sf.action_search(coll, [1.0, 0.0, 0.0, 0.0], k=2, filter={"cat": "b"})
    assert all(h["metadata"]["cat"] == "b" for h in result["hits"])


def test_action_upsert_inserts_and_returns_id_and_count() -> None:
    coll = Collection.create(dim=4)
    result = sf.action_upsert(coll, [1.0, 2.0, 3.0, 4.0], metadata={"k": "v"})
    assert result["id"] == 0
    assert result["count"] == 1
    assert coll.get_metadata(0) == {"k": "v"}


def test_action_ground_joins_text_and_returns_sources() -> None:
    coll = Collection.create(dim=4)
    coll.insert(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), metadata={"text": "doc one"})
    coll.insert(np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32), metadata={"text": "doc two"})
    coll.insert(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), metadata={})  # no text field - must be excluded from context
    result = sf.action_ground(coll, [1.0, 0.0, 0.0, 0.0], k=3)
    assert "doc one" in result["context"]
    assert "doc two" in result["context"]
    assert len(result["sources"]) == 3  # sources include every hit, unlike context


# ── generate_openapi_spec ───────────────────────────────────────────────────


def test_generate_openapi_spec_is_valid_openapi_3() -> None:
    openapi_spec_validator = pytest.importorskip("openapi_spec_validator")
    spec = sf.generate_openapi_spec("http://localhost:8420")
    openapi_spec_validator.validate(spec)  # raises on an invalid spec


def test_generate_openapi_spec_has_the_three_actions_with_short_operation_ids() -> None:
    spec = sf.generate_openapi_spec("http://localhost:8420")
    assert set(spec["paths"].keys()) == {"/salesforce/search", "/salesforce/upsert", "/salesforce/ground"}
    for path, methods in spec["paths"].items():
        for operation in methods.values():
            assert len(operation["operationId"]) < 255


def test_generate_openapi_spec_has_no_composition_keywords() -> None:
    """Deliberately flat per the module docstring - oneOf/anyOf/allOf
    should not appear anywhere in the generated document."""
    import json

    spec_text = json.dumps(sf.generate_openapi_spec("http://localhost:8420"))
    for keyword in ("oneOf", "anyOf", "allOf"):
        assert keyword not in spec_text


# ── import boundary ──────────────────────────────────────────────────────────


def test_importing_module_does_not_eagerly_import_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name == "httpx" or name.startswith("httpx."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "ruvector.integrations.salesforce", raising=False)

    import importlib

    importlib.import_module("ruvector.integrations.salesforce")
    assert "httpx" not in sys.modules, "importing the module alone should not pull in httpx"


def test_network_call_without_httpx_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError, match=r"pip install 'ruvector\[salesforce\]'"):
        asyncio.run(sf.get_oauth_token(_config()))


# ── pre-publish hardening: https-only instance_url, secret not in repr ──────


@pytest.mark.parametrize(
    "bad", ["http://x.my.salesforce.com", "ftp://x", "x.my.salesforce.com", "", "https://", "file:///etc/passwd"]
)
def test_config_rejects_non_https_instance_url(bad: str) -> None:
    with pytest.raises(ValueError, match="https"):
        sf.SalesforceConfig(instance_url=bad, client_id="a", client_secret="b")


def test_config_from_env_rejects_http_instance_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUVECTOR_SALESFORCE_INSTANCE_URL", "http://insecure.example.com")
    monkeypatch.setenv("RUVECTOR_SALESFORCE_CLIENT_ID", "cid")
    monkeypatch.setenv("RUVECTOR_SALESFORCE_CLIENT_SECRET", "csecret")
    with pytest.raises(ValueError, match="https"):
        sf.SalesforceConfig.from_env()


def test_config_repr_does_not_leak_secret() -> None:
    cfg = sf.SalesforceConfig(instance_url=_FAKE_INSTANCE, client_id="cid", client_secret="TOP-SECRET-VALUE")
    assert "TOP-SECRET-VALUE" not in repr(cfg)
    assert "TOP-SECRET-VALUE" not in str(cfg)
    assert cfg.client_secret == "TOP-SECRET-VALUE"
