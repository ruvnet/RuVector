"""End-to-end tests for ``ruvector.salesforce_routes`` (ADR-352): the
``custom_route``-mounted Agentforce action endpoints, their manual bearer
auth (since ``MCPServer.custom_route`` does NOT get the MCP-level
``token_verifier`` protection — see that module's docstring), and clean
4xx error mapping (regression test: the first live run of these routes
returned a raw 500 for an unknown collection or a missing field before
this was fixed).

Uses Starlette's ``TestClient`` against the real ASGI app
(``MCPServer.streamable_http_app()``) rather than a live socket + curl —
exercises the actual routing/middleware stack with no real network I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

starlette_testclient = pytest.importorskip("starlette.testclient")


@pytest.fixture
def sf_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("RUVECTOR_MCP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RUVECTOR_SALESFORCE_ACTION_TOKEN", "test-sf-token")
    monkeypatch.delenv("RUVECTOR_MCP_TOKEN", raising=False)

    import importlib

    import ruvector.mcp_server as mcp_server_mod

    importlib.reload(mcp_server_mod)  # fresh `server` with no auth configured for /mcp
    mcp_server_mod._cache.clear()

    # `ruvector.salesforce_routes` does `from .mcp_server import server, ...`
    # at ITS import time - those names are plain rebindings, so reloading
    # `mcp_server` above does NOT update them. Reload this module too so
    # its `server`/`_load`/`_save`/`_lock` re-resolve to the fresh objects;
    # otherwise `maybe_register()` below registers the custom routes onto
    # the now-orphaned *old* server, and every request against the new
    # one's `streamable_http_app()` 404s on paths that look registered but
    # live on a server object nothing is actually serving. Caught this by
    # running the full file and seeing test-order-dependent 404s that
    # vanished when each test ran alone - a real staleness bug in the test
    # fixture, not in `salesforce_routes.py` itself.
    import ruvector.salesforce_routes as sr

    importlib.reload(sr)
    sr.maybe_register()

    app = mcp_server_mod.server.streamable_http_app()
    return starlette_testclient.TestClient(app)


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": "Bearer test-sf-token"}


def test_openapi_route_needs_no_auth(sf_client: Any) -> None:
    resp = sf_client.get("/salesforce/openapi.json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["openapi"] == "3.0.3"
    assert "/salesforce/search" in body["paths"]


def test_upsert_without_token_is_401(sf_client: Any) -> None:
    resp = sf_client.post("/salesforce/upsert", json={"collection": "c", "vector": [1.0, 0.0]})
    assert resp.status_code == 401
    assert resp.json()["error"] == "invalid_token"


def test_upsert_with_wrong_token_is_401(sf_client: Any) -> None:
    resp = sf_client.post(
        "/salesforce/upsert",
        json={"collection": "c", "vector": [1.0, 0.0]},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_upsert_unknown_collection_is_clean_404_not_500(sf_client: Any) -> None:
    """Regression test: this used to be an unhandled CollectionError ->
    raw Starlette 500 before _run_action's error mapping was added."""
    resp = sf_client.post(
        "/salesforce/upsert",
        json={"collection": "does-not-exist", "vector": [1.0, 0.0]},
        headers=_auth_headers(),
    )
    assert resp.status_code == 404
    assert resp.json()["error"] == "CollectionError"


def test_upsert_missing_required_field_is_clean_400_not_500(sf_client: Any) -> None:
    """Regression test for the same 500-leak, triggered by a different
    cause (KeyError from a malformed request body instead of a missing
    collection)."""
    from ruvector import Collection

    import ruvector.mcp_server as m

    coll = Collection.create(dim=4)
    m._save("exists", coll)

    resp = sf_client.post(
        "/salesforce/upsert",
        json={"collection": "exists", "metadata": {"text": "no vector field"}},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "KeyError"


def test_full_upsert_search_ground_round_trip(sf_client: Any) -> None:
    from ruvector import Collection

    import ruvector.mcp_server as m

    m._save("rt", Collection.create(dim=4))

    r1 = sf_client.post(
        "/salesforce/upsert",
        json={"collection": "rt", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {"text": "hello world"}},
        headers=_auth_headers(),
    )
    assert r1.status_code == 200
    assert r1.json() == {"id": 0, "count": 1}

    r2 = sf_client.post(
        "/salesforce/search",
        json={"collection": "rt", "query_vector": [1.0, 0.0, 0.0, 0.0], "k": 1},
        headers=_auth_headers(),
    )
    assert r2.status_code == 200
    assert r2.json()["hits"][0]["metadata"] == {"text": "hello world"}

    r3 = sf_client.post(
        "/salesforce/ground",
        json={"collection": "rt", "query_vector": [1.0, 0.0, 0.0, 0.0], "k": 1},
        headers=_auth_headers(),
    )
    assert r3.status_code == 200
    assert r3.json()["context"] == "hello world"


def test_search_applies_filter(sf_client: Any) -> None:
    from ruvector import Collection

    import ruvector.mcp_server as m

    coll = Collection.create(dim=2)
    coll.insert(np.array([1.0, 0.0], dtype=np.float32), metadata={"cat": "a"})
    coll.insert(np.array([0.9, 0.1], dtype=np.float32), metadata={"cat": "b"})
    m._save("filt", coll)

    resp = sf_client.post(
        "/salesforce/search",
        json={"collection": "filt", "query_vector": [1.0, 0.0], "k": 2, "filter": {"cat": "b"}},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert all(h["metadata"]["cat"] == "b" for h in hits)


def test_maybe_register_is_idempotent() -> None:
    import ruvector.salesforce_routes as sr

    first = sr.maybe_register()
    second = sr.maybe_register()
    assert second is False  # already registered - no double-registration
    assert first in (True, False)  # True the very first time this module is imported in the process


# ── pre-publish hardening ───────────────────────────────────────────────────


def test_requires_json_content_type(sf_client: Any) -> None:
    from ruvector import Collection

    import ruvector.mcp_server as m

    m._save("ct", Collection.create(dim=2))
    body = b'{"collection": "ct", "vector": [1.0, 0.0]}'
    for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
        r = sf_client.post("/salesforce/upsert", content=body, headers={**_auth_headers(), "Content-Type": ctype})
        assert r.status_code == 415, ctype
        assert r.json()["error"] == "unsupported_media_type"
    r = sf_client.post("/salesforce/upsert", content=body, headers=_auth_headers())  # no content type at all
    assert r.status_code == 415
    ok = sf_client.post(
        "/salesforce/upsert", content=body, headers={**_auth_headers(), "Content-Type": "application/json; charset=utf-8"}
    )
    assert ok.status_code == 200


def test_non_ascii_bearer_token_is_401_not_500(sf_client: Any) -> None:
    """hmac.compare_digest(str, str) raises TypeError on non-ASCII input;
    that used to surface as an unhandled 500."""
    r = sf_client.post(
        "/salesforce/upsert",
        json={"collection": "c", "vector": [1.0, 0.0]},
        headers=[(b"authorization", "Bearer tøken-é".encode("latin-1")), (b"content-type", b"application/json")],
    )
    assert r.status_code == 401


def test_bounds_violation_is_4xx_not_500(sf_client: Any) -> None:
    from ruvector import Collection

    import ruvector.mcp_server as m

    coll = Collection.create(dim=2)
    coll.insert(np.array([1.0, 0.0], dtype=np.float32))
    m._save("bd", coll)
    r = sf_client.post(
        "/salesforce/search",
        json={"collection": "bd", "query_vector": [1.0, 0.0], "k": 10**12},
        headers=_auth_headers(),
    )
    assert 400 <= r.status_code < 500


def test_read_only_registration_omits_upsert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUVECTOR_MCP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RUVECTOR_SALESFORCE_ACTION_TOKEN", "t")
    import importlib

    import ruvector.mcp_server as mcp_server_mod

    importlib.reload(mcp_server_mod)
    import ruvector.salesforce_routes as sr

    importlib.reload(sr)
    sr.maybe_register(read_only=True)
    client = starlette_testclient.TestClient(mcp_server_mod.server.streamable_http_app())
    h = {"Authorization": "Bearer t"}
    assert client.post("/salesforce/upsert", json={"vector": [1.0]}, headers=h).status_code == 404
    # search is still mounted: its 404 is the handler's JSON CollectionError, not a missing route
    r = client.post("/salesforce/search", json={"collection": "nope", "query_vector": [1.0]}, headers=h)
    assert r.json()["error"] == "CollectionError"


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_register_refuses_without_a_token(monkeypatch: pytest.MonkeyPatch, blank: Any) -> None:
    import importlib

    for var in ("RUVECTOR_SALESFORCE_ACTION_TOKEN", "RUVECTOR_MCP_TOKEN"):
        if blank is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, blank)
    import ruvector.salesforce_routes as sr

    importlib.reload(sr)
    with pytest.raises(sr.SalesforceAuthNotConfiguredError, match="RUVECTOR_SALESFORCE_ACTION_TOKEN"):
        sr.maybe_register()
    assert sr._registered is False


def test_blank_action_token_falls_back_to_mcp_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import ruvector.salesforce_routes as sr

    monkeypatch.setenv("RUVECTOR_SALESFORCE_ACTION_TOKEN", "  ")
    monkeypatch.setenv("RUVECTOR_MCP_TOKEN", " mcp-tok ")
    assert sr._expected_action_token() == "mcp-tok"
