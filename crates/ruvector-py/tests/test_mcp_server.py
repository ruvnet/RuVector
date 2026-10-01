"""End-to-end tests for the ruvector MCP server (ADR-352): tool round trips,
the ui:// widget's _meta shape, the directory-traversal guard on collection
names, and the bearer-auth policy (ADR-352 "Security boundary #2").
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def server_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("RUVECTOR_MCP_DATA_DIR", str(tmp_path))
    import ruvector.mcp_server as m

    m._cache.clear()
    return m


def _call(server_module: ModuleType, tool_name: str, **kwargs: Any) -> Any:
    result = asyncio.run(server_module.server.call_tool(tool_name, kwargs))
    # CallToolResult.structuredContent carries the dict the tool returned;
    # fall back to parsing the first text content block if structured
    # content isn't populated for some reason.
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    text = result.content[0].text
    return json.loads(text)


def test_create_insert_search_roundtrip(server_module: ModuleType) -> None:
    m = server_module
    _call(m, "vector_create_collection", name="t1", dim=4)
    r = _call(m, "vector_insert", name="t1", vector=[1.0, 0.0, 0.0, 0.0], metadata={"cat": "a"})
    assert r["id"] == 0
    r2 = _call(m, "vector_insert", name="t1", vector=[0.0, 1.0, 0.0, 0.0], metadata={"cat": "b"})
    assert r2["id"] == 1

    hits = _call(m, "vector_search", name="t1", query=[1.0, 0.0, 0.0, 0.0], k=2)["hits"]
    assert hits[0]["id"] == 0
    assert hits[0]["metadata"] == {"cat": "a"}


def test_insert_batch(server_module: ModuleType) -> None:
    m = server_module
    _call(m, "vector_create_collection", name="t2", dim=3)
    r = _call(
        m,
        "vector_insert_batch",
        name="t2",
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        metadatas=[{"i": 0}, {"i": 1}, {"i": 2}],
    )
    assert r["ids"] == [0, 1, 2]
    assert r["count"] == 3


def test_search_filter(server_module: ModuleType) -> None:
    m = server_module
    _call(m, "vector_create_collection", name="t3", dim=2)
    _call(m, "vector_insert_batch", name="t3",
          vectors=[[1.0, 0.0], [1.0, 0.1], [1.0, 0.2]],
          metadatas=[{"c": "a"}, {"c": "b"}, {"c": "a"}])
    hits = _call(m, "vector_search", name="t3", query=[1.0, 0.0], k=3, filter={"c": "a"})["hits"]
    assert {h["id"] for h in hits} == {0, 2}


def test_delete_and_vacuum(server_module: ModuleType) -> None:
    # Default collection backend is hnsw (ADR-352 M2): delete() is a real
    # delete already, so vacuum() correctly reports 0 dropped (nothing was
    # queued to drop) - see Collection.vacuum's docstring and
    # test_collection.py::test_hnsw_delete_is_immediate_no_vacuum_needed
    # for the backend-specific coverage of this.
    m = server_module
    _call(m, "vector_create_collection", name="t4", dim=2)
    _call(m, "vector_insert_batch", name="t4", vectors=[[1.0, 0.0], [0.0, 1.0]])
    r = _call(m, "vector_delete", name="t4", id=0, vacuum=True)
    assert r["count"] == 1
    assert r["vacuumed"] == 0


def test_stats_and_list(server_module: ModuleType) -> None:
    m = server_module
    _call(m, "vector_create_collection", name="t5", dim=2)
    _call(m, "vector_insert", name="t5", vector=[1.0, 2.0])
    stats = _call(m, "vector_stats", name="t5")
    assert stats["count"] == 1
    assert stats["dim"] == 2
    names = _call(m, "vector_list_collections")["collections"]
    assert "t5" in names


def test_create_twice_errors(server_module: ModuleType) -> None:
    m = server_module
    _call(m, "vector_create_collection", name="t6", dim=2)
    with pytest.raises(Exception):
        _call(m, "vector_create_collection", name="t6", dim=2)


def test_collection_name_traversal_rejected(server_module: ModuleType) -> None:
    m = server_module
    for bad in ["../etc/passwd", "/etc/passwd", "a/b", "a\x00b", ""]:
        with pytest.raises(ValueError):
            m._safe_path(bad)


def test_search_unknown_collection_errors(server_module: ModuleType) -> None:
    m = server_module
    with pytest.raises(Exception):
        _call(m, "vector_search", name="does-not-exist", query=[1.0], k=1)


def test_widget_meta_shape_matches_adr(server_module: ModuleType) -> None:
    """Mirrors the live-verified _meta shape in ADR-352 exactly."""
    m = server_module
    tools = asyncio.run(m.server.list_tools())
    explore = next(t for t in tools if t.name == "vector_explore")
    assert explore.meta["openai/outputTemplate"] == "ui://ruvector/explore.html"
    assert explore.meta["openai/widgetAccessible"] is True
    assert explore.meta["ui"]["resourceUri"] == "ui://ruvector/explore.html"
    assert explore.meta["ui/resourceUri"] == "ui://ruvector/explore.html"


def test_widget_resource_readable(server_module: ModuleType) -> None:
    m = server_module
    contents = list(asyncio.run(m.server.read_resource(m._WIDGET_URI)))
    assert len(contents) == 1
    html = contents[0].content
    assert "<html>" in html
    assert "window.openai" in html


def test_explore_tool_returns_search_shape(server_module: ModuleType) -> None:
    m = server_module
    _call(m, "vector_create_collection", name="t7", dim=2)
    _call(m, "vector_insert", name="t7", vector=[1.0, 0.0], metadata={"cat": "x"})
    out = _call(m, "vector_explore", name="t7", query=[1.0, 0.0], k=1)
    assert out["hits"][0]["metadata"] == {"cat": "x"}


def test_concurrent_inserts_do_not_collide(server_module: ModuleType) -> None:
    """Regression test for the concurrency gap flagged in ADR-352: without
    the module-level RLock, two threads racing vector_insert on the same
    collection could both read the same next_id. 16 threads x 10 inserts
    each must produce 160 distinct ids and a collection of length 160 -
    not fewer (which would mean a lost write from an id collision)."""
    import threading

    m = server_module
    _call(m, "vector_create_collection", name="tc", dim=2)

    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()

    def worker() -> None:
        for _ in range(10):
            r = _call(m, "vector_insert", name="tc", vector=[1.0, 2.0])
            with results_lock:
                results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 160
    ids = [r["id"] for r in results]
    assert len(set(ids)) == 160, f"id collision: {len(ids) - len(set(ids))} duplicate(s)"
    stats = _call(m, "vector_stats", name="tc")
    assert stats["count"] == 160


# ── bearer auth (ADR-352 "Security boundary #2") ────────────────────────────


def test_no_token_env_var_means_no_auth_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUVECTOR_MCP_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("RUVECTOR_MCP_TOKEN", raising=False)
    import ruvector.mcp_server as m

    importlib.reload(m)
    verifier, settings = m._build_auth()
    assert verifier is None
    assert settings is None


def test_token_env_var_builds_a_verifier_and_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUVECTOR_MCP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RUVECTOR_MCP_TOKEN", "test-secret")
    import ruvector.mcp_server as m

    importlib.reload(m)
    verifier, settings = m._build_auth()
    assert verifier is not None
    assert settings is not None
    assert settings.validate_token_resource is True


def test_static_token_verifier_accepts_correct_token_only() -> None:
    # No pytest-asyncio dependency in this package - run the coroutine the
    # same way `_call()` above already does for async MCP calls.
    import ruvector.mcp_server as m

    verifier = m.StaticTokenVerifier("correct-token")
    ok = asyncio.run(verifier.verify_token("correct-token"))
    assert ok is not None
    assert set(ok.scopes) == {"read", "write"}

    bad = asyncio.run(verifier.verify_token("wrong-token"))
    assert bad is None


def test_require_write_scope_blocks_read_only_token(server_module: ModuleType) -> None:
    from mcp.server.auth.middleware.auth_context import (  # type: ignore[attr-defined]
        AuthenticatedUser,
        auth_context_var,
    )
    from mcp.server.auth.provider import AccessToken

    m = server_module
    token = AuthenticatedUser(auth_info=AccessToken(token="x", client_id="c", scopes=["read"]))
    reset_token = auth_context_var.set(token)
    try:
        with pytest.raises(ValueError, match="write"):
            m._require_write_scope()
    finally:
        auth_context_var.reset(reset_token)


def test_require_write_scope_allows_write_scoped_token(server_module: ModuleType) -> None:
    from mcp.server.auth.middleware.auth_context import (  # type: ignore[attr-defined]
        AuthenticatedUser,
        auth_context_var,
    )
    from mcp.server.auth.provider import AccessToken

    m = server_module
    token = AuthenticatedUser(auth_info=AccessToken(token="x", client_id="c", scopes=["read", "write"]))
    reset_token = auth_context_var.set(token)
    try:
        m._require_write_scope()  # must not raise
    finally:
        auth_context_var.reset(reset_token)


def test_require_write_scope_is_a_noop_with_no_auth_context(server_module: ModuleType) -> None:
    from mcp.server.auth.middleware.auth_context import auth_context_var

    m = server_module
    reset_token = auth_context_var.set(None)
    try:
        m._require_write_scope()  # must not raise - no auth configured at all
    finally:
        auth_context_var.reset(reset_token)
