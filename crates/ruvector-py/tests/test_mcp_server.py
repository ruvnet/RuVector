"""End-to-end tests for the ruvector MCP server (ADR-352 M1.5): tool round
trips, the ui:// widget's _meta shape, and the directory-traversal guard on
collection names.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest


@pytest.fixture
def server_module(tmp_path, monkeypatch):
    monkeypatch.setenv("RUVECTOR_MCP_DATA_DIR", str(tmp_path))
    import ruvector.mcp_server as m

    m._cache.clear()
    return m


def _call(server_module, tool_name, **kwargs):
    result = asyncio.run(server_module.server.call_tool(tool_name, kwargs))
    # CallToolResult.structuredContent carries the dict the tool returned;
    # fall back to parsing the first text content block if structured
    # content isn't populated for some reason.
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    text = result.content[0].text
    return json.loads(text)


def test_create_insert_search_roundtrip(server_module):
    m = server_module
    _call(m, "vector_create_collection", name="t1", dim=4)
    r = _call(m, "vector_insert", name="t1", vector=[1.0, 0.0, 0.0, 0.0], metadata={"cat": "a"})
    assert r["id"] == 0
    r2 = _call(m, "vector_insert", name="t1", vector=[0.0, 1.0, 0.0, 0.0], metadata={"cat": "b"})
    assert r2["id"] == 1

    hits = _call(m, "vector_search", name="t1", query=[1.0, 0.0, 0.0, 0.0], k=2)["hits"]
    assert hits[0]["id"] == 0
    assert hits[0]["metadata"] == {"cat": "a"}


def test_insert_batch(server_module):
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


def test_search_filter(server_module):
    m = server_module
    _call(m, "vector_create_collection", name="t3", dim=2)
    _call(m, "vector_insert_batch", name="t3",
          vectors=[[1.0, 0.0], [1.0, 0.1], [1.0, 0.2]],
          metadatas=[{"c": "a"}, {"c": "b"}, {"c": "a"}])
    hits = _call(m, "vector_search", name="t3", query=[1.0, 0.0], k=3, filter={"c": "a"})["hits"]
    assert {h["id"] for h in hits} == {0, 2}


def test_delete_and_vacuum(server_module):
    m = server_module
    _call(m, "vector_create_collection", name="t4", dim=2)
    _call(m, "vector_insert_batch", name="t4", vectors=[[1.0, 0.0], [0.0, 1.0]])
    r = _call(m, "vector_delete", name="t4", id=0, vacuum=True)
    assert r["count"] == 1
    assert r["vacuumed"] == 1


def test_stats_and_list(server_module):
    m = server_module
    _call(m, "vector_create_collection", name="t5", dim=2)
    _call(m, "vector_insert", name="t5", vector=[1.0, 2.0])
    stats = _call(m, "vector_stats", name="t5")
    assert stats["count"] == 1
    assert stats["dim"] == 2
    names = _call(m, "vector_list_collections")["collections"]
    assert "t5" in names


def test_create_twice_errors(server_module):
    m = server_module
    _call(m, "vector_create_collection", name="t6", dim=2)
    with pytest.raises(Exception):
        _call(m, "vector_create_collection", name="t6", dim=2)


def test_collection_name_traversal_rejected(server_module):
    m = server_module
    for bad in ["../etc/passwd", "/etc/passwd", "a/b", "a\x00b", ""]:
        with pytest.raises(ValueError):
            m._safe_path(bad)


def test_search_unknown_collection_errors(server_module):
    m = server_module
    with pytest.raises(Exception):
        _call(m, "vector_search", name="does-not-exist", query=[1.0], k=1)


def test_widget_meta_shape_matches_adr(server_module):
    """Mirrors the live-verified _meta shape in ADR-352 exactly."""
    m = server_module
    tools = asyncio.run(m.server.list_tools())
    explore = next(t for t in tools if t.name == "vector_explore")
    assert explore.meta["openai/outputTemplate"] == "ui://ruvector/explore.html"
    assert explore.meta["openai/widgetAccessible"] is True
    assert explore.meta["ui"]["resourceUri"] == "ui://ruvector/explore.html"
    assert explore.meta["ui/resourceUri"] == "ui://ruvector/explore.html"


def test_widget_resource_readable(server_module):
    m = server_module
    contents = list(asyncio.run(m.server.read_resource(m._WIDGET_URI)))
    assert len(contents) == 1
    html = contents[0].content
    assert "<html>" in html
    assert "window.openai" in html


def test_explore_tool_returns_search_shape(server_module):
    m = server_module
    _call(m, "vector_create_collection", name="t7", dim=2)
    _call(m, "vector_insert", name="t7", vector=[1.0, 0.0], metadata={"cat": "x"})
    out = _call(m, "vector_explore", name="t7", query=[1.0, 0.0], k=1)
    assert out["hits"][0]["metadata"] == {"cat": "x"}
