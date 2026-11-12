"""Tests for ``ruvector.GraphDB`` — raw graph CRUD (ADR-352 graph slice, M1).

Covers: node/edge create+get round-tripping labels and arbitrary
JSON-compatible properties, outgoing-edge traversal, nonexistent lookups
returning ``None`` (matching the real ``GraphDB``'s `Option`-returning
contract, not an exception), and the two real error paths this binding
surfaces: an edge referencing a missing endpoint, and a duplicate explicit
``id`` on ``create_node``/``create_edge`` (a footgun this binding
pre-checks and rejects — see ``src/graph.rs``'s doc comment on why, since
the underlying Rust ``create_node``/``create_edge`` would otherwise
silently overwrite and leave stale index entries).
"""

from __future__ import annotations

import pytest

import ruvector


def test_create_and_get_node_roundtrips_labels_and_properties() -> None:
    g = ruvector.GraphDB()
    node_id = g.create_node(
        ["Person", "Employee"],
        {
            "name": "alice",
            "age": 30,
            "score": 1.5,
            "active": True,
            "nickname": None,
            "tags": ["eng", "lead"],
            "meta": {"level": 2},
        },
    )
    assert isinstance(node_id, str) and node_id

    node = g.get_node(node_id)
    assert node is not None
    assert node["id"] == node_id
    assert sorted(node["labels"]) == ["Employee", "Person"]
    assert node["properties"] == {
        "name": "alice",
        "age": 30,
        "score": 1.5,
        "active": True,
        "nickname": None,
        "tags": ["eng", "lead"],
        "meta": {"level": 2},
    }


def test_create_node_defaults_to_no_labels_and_empty_properties() -> None:
    g = ruvector.GraphDB()
    node_id = g.create_node()
    node = g.get_node(node_id)
    assert node is not None
    assert node["labels"] == []
    assert node["properties"] == {}


def test_create_node_accepts_tuple_labels() -> None:
    g = ruvector.GraphDB()
    node_id = g.create_node(("Person", "Employee"))
    node = g.get_node(node_id)
    assert node is not None
    assert sorted(node["labels"]) == ["Employee", "Person"]


def test_create_node_with_explicit_id() -> None:
    g = ruvector.GraphDB()
    node_id = g.create_node(["Thing"], {}, id="my-custom-id")
    assert node_id == "my-custom-id"
    node = g.get_node("my-custom-id")
    assert node is not None
    assert node["id"] == "my-custom-id"


def test_get_node_missing_returns_none() -> None:
    g = ruvector.GraphDB()
    assert g.get_node("does-not-exist") is None


def test_create_edge_between_two_nodes_and_get_edge() -> None:
    g = ruvector.GraphDB()
    a = g.create_node(["Person"], {"name": "alice"})
    b = g.create_node(["Person"], {"name": "bob"})

    edge_id = g.create_edge(a, b, "KNOWS", {"since": 2020})
    assert isinstance(edge_id, str) and edge_id

    edge = g.get_edge(edge_id)
    assert edge is not None
    assert edge == {
        "id": edge_id,
        "from": a,
        "to": b,
        "type": "KNOWS",
        "properties": {"since": 2020},
    }


def test_create_edge_defaults_properties_to_empty_dict() -> None:
    g = ruvector.GraphDB()
    a = g.create_node()
    b = g.create_node()
    edge_id = g.create_edge(a, b, "LINKS")
    edge = g.get_edge(edge_id)
    assert edge is not None
    assert edge["properties"] == {}


def test_create_edge_with_explicit_id() -> None:
    g = ruvector.GraphDB()
    a = g.create_node()
    b = g.create_node()
    edge_id = g.create_edge(a, b, "LINKS", id="my-edge-id")
    assert edge_id == "my-edge-id"


def test_get_edge_missing_returns_none() -> None:
    g = ruvector.GraphDB()
    assert g.get_edge("does-not-exist") is None


def test_get_outgoing_edges_returns_only_edges_from_that_node() -> None:
    g = ruvector.GraphDB()
    a = g.create_node(["Person"], {"name": "alice"})
    b = g.create_node(["Person"], {"name": "bob"})
    c = g.create_node(["Person"], {"name": "carol"})

    e_ab = g.create_edge(a, b, "KNOWS")
    e_ac = g.create_edge(a, c, "KNOWS")
    g.create_edge(b, c, "KNOWS")  # not from `a` — must not appear below

    outgoing = g.get_outgoing_edges(a)
    ids = sorted(e["id"] for e in outgoing)
    assert ids == sorted([e_ab, e_ac])

    # `b`'s only outgoing edge is the one to `c`.
    outgoing_b = g.get_outgoing_edges(b)
    assert len(outgoing_b) == 1
    assert outgoing_b[0]["from"] == b
    assert outgoing_b[0]["to"] == c


def test_get_outgoing_edges_for_unknown_node_is_empty_not_an_error() -> None:
    g = ruvector.GraphDB()
    assert g.get_outgoing_edges("does-not-exist") == []


def test_len_counts_nodes_not_nodes_plus_edges() -> None:
    g = ruvector.GraphDB()
    assert len(g) == 0
    a = g.create_node()
    b = g.create_node()
    g.create_edge(a, b, "KNOWS")
    assert len(g) == 2


def test_repr_reports_node_and_edge_counts() -> None:
    g = ruvector.GraphDB()
    a = g.create_node()
    b = g.create_node()
    g.create_edge(a, b, "KNOWS")
    assert repr(g) == "GraphDB(nodes=2, edges=1)"


def test_create_edge_with_missing_endpoint_raises() -> None:
    g = ruvector.GraphDB()
    a = g.create_node()
    with pytest.raises(ruvector.RuVectorError):
        g.create_edge(a, "does-not-exist", "KNOWS")
    with pytest.raises(ruvector.RuVectorError):
        g.create_edge("does-not-exist", a, "KNOWS")


def test_create_node_with_duplicate_explicit_id_raises() -> None:
    g = ruvector.GraphDB()
    node_id = g.create_node(["Person"], {}, id="dup")
    with pytest.raises(ruvector.RuVectorError):
        g.create_node(["Other"], {}, id=node_id)


def test_create_edge_with_duplicate_explicit_id_raises() -> None:
    g = ruvector.GraphDB()
    a = g.create_node()
    b = g.create_node()
    g.create_edge(a, b, "KNOWS", id="dup-edge")
    with pytest.raises(ruvector.RuVectorError):
        g.create_edge(a, b, "LIKES", id="dup-edge")


def test_create_node_rejects_bare_string_labels() -> None:
    g = ruvector.GraphDB()
    # mypy accepts `str` here (it structurally satisfies `Sequence[str]`),
    # so no `type: ignore` is needed/valid — the rejection is a runtime-only
    # guard against the iterate-by-character footgun, not a type error.
    with pytest.raises(TypeError):
        g.create_node("Person", {})


def test_create_node_rejects_dict_as_labels() -> None:
    g = ruvector.GraphDB()
    with pytest.raises(TypeError):
        g.create_node({"Person": True}, {})  # type: ignore[arg-type]


def test_create_node_rejects_non_dict_properties() -> None:
    g = ruvector.GraphDB()
    with pytest.raises(TypeError):
        g.create_node(["Person"], ["not", "a", "dict"])  # type: ignore[arg-type]


def test_create_node_rejects_unsupported_property_value_type() -> None:
    g = ruvector.GraphDB()

    class NotJsonCompatible:
        pass

    with pytest.raises(TypeError):
        g.create_node(["Person"], {"bad": NotJsonCompatible()})
