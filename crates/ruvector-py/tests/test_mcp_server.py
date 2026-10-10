"""End-to-end tests for the ruvector MCP server (ADR-352): tool round trips,
the ui:// widget's _meta shape, the directory-traversal guard on collection
names, and the bearer-auth policy (ADR-352 "Security boundary #2").
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

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


@contextlib.contextmanager
def _raises_chain(pattern: str) -> Iterator[None]:
    """Like ``pytest.raises(Exception, match=...)`` but matches against the
    whole ``__cause__`` chain: the MCP SDK wraps a tool's exception in a
    generic "Error executing tool X" and keeps the real message as cause."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        texts, cur = [], exc  # type: ignore[var-annotated]
        while cur is not None:
            texts.append(str(cur))
            cur = cur.__cause__ or cur.__context__
        assert re.search(pattern, "\n".join(texts)), f"{pattern!r} not in {texts!r}"
    else:
        pytest.fail(f"expected an exception matching {pattern!r}")


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


# ── pre-publish hardening: resource bounds ──────────────────────────────────


def test_search_k_bound_is_a_clean_error_not_a_process_abort(server_module: ModuleType) -> None:
    """vector_search(k=10**12) on the hnsw backend used to abort the whole
    server process (8 TB allocation in Rust)."""
    m = server_module
    _call(m, "vector_create_collection", name="b1", dim=2)
    _call(m, "vector_insert", name="b1", vector=[1.0, 0.0])
    for bad_k in (10**12, 10_001, 0, -1):
        with _raises_chain("k"):
            _call(m, "vector_search", name="b1", query=[1.0, 0.0], k=bad_k)
    with _raises_chain("k"):
        _call(m, "vector_explore", name="b1", query=[1.0, 0.0], k=10**12)
    assert len(_call(m, "vector_search", name="b1", query=[1.0, 0.0], k=10_000)["hits"]) == 1


def test_search_rerank_factor_bound(server_module: ModuleType) -> None:
    m = server_module
    _call(m, "vector_create_collection", name="b2", dim=2)
    _call(m, "vector_insert", name="b2", vector=[1.0, 0.0])
    with _raises_chain("rerank_factor"):
        _call(m, "vector_search", name="b2", query=[1.0, 0.0], k=1, rerank_factor=10**9)


def test_create_collection_dim_and_rerank_bounds(server_module: ModuleType) -> None:
    m = server_module
    for dim in (0, -3, 8193, 10**9):
        with _raises_chain("dim"):
            _call(m, "vector_create_collection", name="b3", dim=dim)
    for rf in (0, 10_001, 10**9):
        with _raises_chain("rerank_factor"):
            _call(m, "vector_create_collection", name="b3", dim=4, rerank_factor=rf)
    assert "b3" not in _call(m, "vector_list_collections")["collections"]
    _call(m, "vector_create_collection", name="b3", dim=8192)  # the bound itself is allowed


def test_oversized_vector_rejected_before_conversion(server_module: ModuleType) -> None:
    m = server_module
    _call(m, "vector_create_collection", name="b4", dim=2)
    with _raises_chain("8192"):
        _call(m, "vector_insert", name="b4", vector=[0.0] * 8193)
    with _raises_chain("8192"):
        _call(m, "vector_search", name="b4", query=[0.0] * 8193, k=1)


def test_batch_row_and_element_caps(server_module: ModuleType) -> None:
    m = server_module
    _call(m, "vector_create_collection", name="b5", dim=2)
    too_many_rows = [[0.0, 0.0]] * (m.MAX_BATCH_ROWS + 1)
    with _raises_chain("rows"):
        _call(m, "vector_insert_batch", name="b5", vectors=too_many_rows)
    # few rows but each over-long: caught by the per-row length cap
    with _raises_chain("8192"):
        _call(m, "vector_insert_batch", name="b5", vectors=[[0.0] * 8193])
    # total element budget
    _call(m, "vector_create_collection", name="b5big", dim=8192)
    rows = m.MAX_BATCH_ELEMENTS // 8192 + 1
    with _raises_chain("elements"):
        _call(m, "vector_insert_batch", name="b5big", vectors=[[0.0] * 8192] * rows)
    with _raises_chain("metadatas"):
        _call(m, "vector_insert_batch", name="b5", vectors=[[0.0, 0.0]], metadatas=[{}, {}])
    assert _call(m, "vector_stats", name="b5")["count"] == 0


def test_metadata_size_cap(server_module: ModuleType) -> None:
    m = server_module
    _call(m, "vector_create_collection", name="b6", dim=2)
    huge = {"blob": "x" * (m.MAX_METADATA_BYTES + 1)}
    with _raises_chain("metadata"):
        _call(m, "vector_insert", name="b6", vector=[1.0, 0.0], metadata=huge)
    with _raises_chain("metadata"):
        _call(m, "vector_insert_batch", name="b6", vectors=[[1.0, 0.0]], metadatas=[huge])
    assert _call(m, "vector_stats", name="b6")["count"] == 0


def test_non_finite_vectors_rejected_via_tools(server_module: ModuleType) -> None:
    m = server_module
    _call(m, "vector_create_collection", name="b7", dim=2)
    for bad in (float("nan"), float("inf")):
        with _raises_chain("finite"):
            _call(m, "vector_insert", name="b7", vector=[bad, 0.0])
        with _raises_chain("finite"):
            _call(m, "vector_insert_batch", name="b7", vectors=[[1.0, 0.0], [bad, 0.0]])
    assert _call(m, "vector_stats", name="b7")["count"] == 0


# ── pre-publish hardening: stored XSS in the vector_explore widget ──────────

_XSS = "<img src=x onerror=alert(1)>"


def test_widget_html_never_uses_innerhtml_or_string_built_markup(server_module: ModuleType) -> None:
    html = server_module._WIDGET_HTML
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in html, f"widget must not use {forbidden}"
    assert "textContent" in html


def test_widget_escapes_metadata_payload_when_executed(server_module: ModuleType) -> None:
    """Run the widget's real <script> under node against a recording fake
    DOM: the hostile metadata must only ever reach `textContent` (inert
    text), never an HTML-parsing sink, and must not become an element."""
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    html = server_module._WIDGET_HTML
    script = re.search(r"<script>(.*?)</script>", html, re.S)
    assert script is not None
    harness = r"""
const vm = require('vm');
const payload = %s;
const made = [];
class El {
  constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this._text = ''; made.push(this); }
  set innerHTML(v) { throw new Error('innerHTML sink used: ' + v); }
  set outerHTML(v) { throw new Error('outerHTML sink used'); }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text; }
  appendChild(c) { this.children.push(c); return c; }
  append(...cs) { cs.forEach(c => this.children.push(typeof c === 'string' ? Object.assign(new El('#text'), {_text: c}) : c)); }
  setAttribute(k, v) { if (/^on/i.test(k)) throw new Error('event handler attribute ' + k); this[k] = v; }
  set className(v) { this._cls = v; }
}
const byId = {};
const document = {
  getElementById: id => byId[id] || (byId[id] = new El('#' + id)),
  createElement: t => new El(t),
  createTextNode: t => Object.assign(new El('#text'), {_text: String(t)}),
};
const window = { openai: { toolOutput: { hits: [{ id: 1, score: 0.25, metadata: { note: payload } }] } } };
vm.runInNewContext(%s, { document, window, Math, JSON, Number, String });
const tags = made.map(e => e.tag.toLowerCase());
if (tags.includes('img')) throw new Error('payload became an <img> element');
const texts = [];
(function walk(e) { texts.push(e._text); e.children.forEach(walk); })(byId['rows']);
if (!texts.join('\n').includes(payload)) throw new Error('payload text not rendered as text');
console.log('OK');
""" % (json.dumps(_XSS), json.dumps(script.group(1)))
    r = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr + r.stdout


# ── pre-publish hardening: auth / exposure / read-only ──────────────────────


@pytest.mark.parametrize("blank", ["", " ", "   ", "\t\n"])
def test_blank_token_is_treated_as_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    monkeypatch.setenv("RUVECTOR_MCP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RUVECTOR_MCP_TOKEN", blank)
    import ruvector.mcp_server as m

    importlib.reload(m)
    assert m._configured_token() is None
    assert m._build_auth() == (None, None)  # NOT a verifier that accepts the blank string


def test_token_is_stripped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUVECTOR_MCP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RUVECTOR_MCP_TOKEN", "  s3cret \n")
    import ruvector.mcp_server as m

    importlib.reload(m)
    assert m._configured_token() == "s3cret"
    verifier, _ = m._build_auth()
    assert asyncio.run(verifier.verify_token("s3cret")) is not None
    assert asyncio.run(verifier.verify_token("  s3cret \n")) is None


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2", "LOCALHOST"])
def test_loopback_without_token_may_start(server_module: ModuleType, host: str) -> None:
    server_module.check_http_exposure(host, token_configured=False)  # must not raise


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "100.104.125.72", "example.com", ""])
def test_non_loopback_without_token_refuses_to_start(server_module: ModuleType, host: str) -> None:
    with pytest.raises(server_module.UnsafeBindError, match="RUVECTOR_MCP_TOKEN"):
        server_module.check_http_exposure(host, token_configured=False)
    server_module.check_http_exposure(host, token_configured=True)  # token present: allowed


def test_run_http_refuses_before_binding(server_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    m = server_module
    ran: list[object] = []
    monkeypatch.setattr(m.server, "run", lambda *a, **kw: ran.append((a, kw)))
    monkeypatch.setattr(m, "_token_verifier", None)
    with pytest.raises(m.UnsafeBindError):
        m.run_http(host="0.0.0.0", port=8420)
    assert ran == []
    m.run_http(host="127.0.0.1", port=8420)  # loopback + no token: warns but starts
    assert len(ran) == 1


def test_apply_read_only_removes_mutating_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUVECTOR_MCP_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("RUVECTOR_MCP_TOKEN", raising=False)
    import ruvector.mcp_server as m

    importlib.reload(m)
    try:
        before = {t.name for t in asyncio.run(m.server.list_tools())}
        assert {"vector_create_collection", "vector_insert", "vector_insert_batch", "vector_delete"} <= before
        m.apply_read_only()
        after = {t.name for t in asyncio.run(m.server.list_tools())}
        assert after == {"vector_search", "vector_stats", "vector_list_collections", "vector_explore"}
        for gone in ("vector_create_collection", "vector_insert", "vector_insert_batch", "vector_delete"):
            with pytest.raises(Exception):
                _call(m, gone, name="x")
    finally:
        importlib.reload(m)


def test_validation_errors_reach_the_client():
    """mcp 2.x masks non-ToolError exceptions; our validation messages must get through (issue #1134)."""
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    from ruvector import mcp_server as m

    async def call(name, args):
        return await m.server.call_tool(name, args)

    try:
        asyncio.run(call("vector_create_collection", {"name": "../escape", "dim": 4}))
    except ToolError as e:
        assert "invalid collection name" in str(e)
    else:
        raise AssertionError("expected ToolError")
    try:
        asyncio.run(call("vector_search", {"name": "no-such-collection", "query": [1.0, 2.0, 3.0, 4.0], "k": 1}))
    except ToolError as e:
        assert "no such collection" in str(e)
    else:
        raise AssertionError("expected ToolError")
