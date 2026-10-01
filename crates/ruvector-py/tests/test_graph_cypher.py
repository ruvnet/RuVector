"""Tests for ``ruvector.GraphDB.query_cypher`` (ADR-352 graph slice, M2).

The executor behind this method is a line-for-line port of
``crates/ruvector-graph-node/src/cypher_exec.rs`` (a NAPI binding crate's
`MATCH` executor that turned out to have zero NAPI-specific types), so
these tests are themselves close ports of that module's own Rust unit
tests, re-expressed as the Python-visible contract: ``query_cypher``
returns ``{"nodes": [...], "edges": [...]}`` using the same per-row dict
shape as ``get_node``/``get_edge`` — every node/edge the ``MATCH`` touched,
never a ``RETURN``-projected subset (``RETURN`` is valid syntax but not
applied as a projection; see ``src/graph.rs``'s docstring on
``query_cypher``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import ruvector


def _node_ids(result: Dict[str, Any]) -> List[str]:
    return sorted(n["id"] for n in result["nodes"])


def _edge_ids(result: Dict[str, Any]) -> List[str]:
    return sorted(e["id"] for e in result["edges"])


def _fixture() -> ruvector.GraphDB:
    """Two people and one `knows` edge between them, plus an unrelated company."""
    g = ruvector.GraphDB()
    g.create_node(["Person"], {"name": "alice", "age": 30}, id="n1")
    g.create_node(["Person"], {"name": "bob", "age": 41}, id="n2")
    g.create_node(["Company"], {}, id="c1")
    g.create_edge("n1", "n2", "knows", {"since": 2020}, id="e1")
    return g


def test_label_less_match_returns_every_node() -> None:
    g = _fixture()
    result = g.query_cypher("MATCH (n) RETURN n")
    assert _node_ids(result) == ["c1", "n1", "n2"]
    assert result["edges"] == []


def test_label_scan_returns_only_matching_label() -> None:
    g = _fixture()
    result = g.query_cypher("MATCH (n:Person) RETURN n")
    assert _node_ids(result) == ["n1", "n2"]


def test_where_filters_on_identity() -> None:
    g = _fixture()
    result = g.query_cypher("MATCH (n) WHERE n.id = 'n1' RETURN n")
    assert _node_ids(result) == ["n1"]


def test_inline_property_pattern_filters() -> None:
    g = _fixture()
    result = g.query_cypher("MATCH (n {name: 'alice'}) RETURN n")
    assert _node_ids(result) == ["n1"]
    assert result["nodes"][0]["properties"]["name"] == "alice"


def test_where_numeric_comparison() -> None:
    g = _fixture()
    result = g.query_cypher("MATCH (n:Person) WHERE n.age > 35 RETURN n")
    assert _node_ids(result) == ["n2"]


def test_relationship_pattern_returns_edge_and_both_endpoints() -> None:
    g = _fixture()
    result = g.query_cypher("MATCH (a)-[r:knows]->(b) RETURN a, r, b")
    assert _edge_ids(result) == ["e1"]
    assert _node_ids(result) == ["n1", "n2"]
    edge = result["edges"][0]
    assert edge["from"] == "n1"
    assert edge["to"] == "n2"
    assert edge["type"] == "knows"
    assert edge["properties"] == {"since": 2020}


def test_untyped_relationship_pattern_scans_all_edges() -> None:
    g = _fixture()
    result = g.query_cypher("MATCH (a)-[r]->(b) RETURN r")
    assert _edge_ids(result) == ["e1"]


def test_empty_graph_returns_empty_result() -> None:
    g = ruvector.GraphDB()
    result = g.query_cypher("MATCH (n) RETURN n")
    assert result == {"nodes": [], "edges": []}


def test_variable_length_relationship_raises() -> None:
    g = _fixture()
    with pytest.raises(ruvector.RuVectorError, match="variable-length"):
        g.query_cypher("MATCH (a)-[r*1..2]->(b) RETURN r")


def test_chained_relationship_raises() -> None:
    g = _fixture()
    with pytest.raises(ruvector.RuVectorError, match="chained"):
        g.query_cypher("MATCH (a)-[r:knows]->(b)<-[s:knows]-(c) RETURN r")


def test_create_statement_raises_instead_of_being_silently_dropped() -> None:
    g = _fixture()
    with pytest.raises(ruvector.RuVectorError, match="CREATE"):
        g.query_cypher("CREATE (n)")


def test_unparseable_cypher_raises() -> None:
    g = _fixture()
    with pytest.raises(ruvector.RuVectorError, match="parse error"):
        g.query_cypher("this is not cypher @#$")


def test_return_clause_is_syntactically_accepted_but_not_a_projection() -> None:
    """`RETURN n` and `RETURN a, r, b` both return every matched node/edge,
    not just the names listed in RETURN — the executor never applies a
    projection (see this module's and `graph.rs`'s docstrings)."""
    g = _fixture()
    returning_one_var = g.query_cypher("MATCH (a)-[r:knows]->(b) RETURN a")
    returning_all_vars = g.query_cypher("MATCH (a)-[r:knows]->(b) RETURN a, r, b")
    assert returning_one_var == returning_all_vars
