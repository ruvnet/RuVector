"""Tests for ``ruvector.integrations.llamaindex.RuVectorStore``.

LlamaIndex nodes carry their own embedding (no ``Embeddings`` abstraction
on the vector-store side, unlike LangChain), so these tests build
``TextNode`` objects directly over a small fixed orthogonal basis — same
"nearest neighbour is unambiguous" style as ``tests/test_collection.py``
and ``test_langchain_integration.py``.
"""

from __future__ import annotations

import sys

import pytest

from ruvector import Collection

llama_index_core = pytest.importorskip("llama_index.core")  # noqa: F841 - presence check only

from ruvector.integrations.llamaindex import RuVectorStore  # noqa: E402

from llama_index.core.schema import (  # noqa: E402
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (  # noqa: E402
    ExactMatchFilter,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

_BASIS = {
    "apple": [1.0, 0.0, 0.0, 0.0],
    "banana": [0.0, 1.0, 0.0, 0.0],
    "car": [0.0, 0.0, 1.0, 0.0],
    "truck": [0.0, 0.0, 0.0, 1.0],
}


def _node(text: str, *, node_id: str, category: str, ref_doc_id: str | None = None) -> TextNode:
    relationships = {}
    if ref_doc_id is not None:
        relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=ref_doc_id)
    node = TextNode(text=text, id_=node_id, metadata={"category": category}, relationships=relationships)
    node.embedding = list(_BASIS[text])
    return node


@pytest.fixture
def store() -> RuVectorStore:
    coll = Collection.create(dim=4)
    return RuVectorStore(coll)


def _add_fruit_and_vehicle_nodes(store: RuVectorStore) -> None:
    store.add(
        [
            _node("apple", node_id="n-apple", category="fruit", ref_doc_id="doc-fruit"),
            _node("banana", node_id="n-banana", category="fruit", ref_doc_id="doc-fruit"),
            _node("car", node_id="n-car", category="vehicle", ref_doc_id="doc-vehicle"),
            _node("truck", node_id="n-truck", category="vehicle", ref_doc_id="doc-vehicle"),
        ]
    )


def test_add_and_query_returns_nearest_node(store: RuVectorStore) -> None:
    _add_fruit_and_vehicle_nodes(store)
    result = store.query(VectorStoreQuery(query_embedding=_BASIS["apple"], similarity_top_k=1))
    assert result.nodes is not None
    assert len(result.nodes) == 1
    assert result.nodes[0].get_content() == "apple"
    assert result.nodes[0].metadata == {"category": "fruit"}
    assert result.similarities is not None
    assert result.similarities[0] == pytest.approx(0.0, abs=1e-6)
    assert result.ids == ["n-apple"]


def test_add_rejects_duplicate_node_id(store: RuVectorStore) -> None:
    store.add([_node("apple", node_id="dup", category="fruit")])
    with pytest.raises(ValueError):
        store.add([_node("banana", node_id="dup", category="fruit")])


def test_query_with_exact_match_filter(store: RuVectorStore) -> None:
    _add_fruit_and_vehicle_nodes(store)
    filters = MetadataFilters(filters=[ExactMatchFilter(key="category", value="vehicle")])
    result = store.query(VectorStoreQuery(query_embedding=_BASIS["apple"], similarity_top_k=4, filters=filters))
    assert result.nodes is not None
    texts = {n.get_content() for n in result.nodes}
    assert texts == {"car", "truck"}


def test_query_with_operator_filter_and_or_condition(store: RuVectorStore) -> None:
    _add_fruit_and_vehicle_nodes(store)
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="fruit", operator=FilterOperator.EQ),
            MetadataFilter(key="category", value="vehicle", operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.OR,
    )
    result = store.query(VectorStoreQuery(query_embedding=_BASIS["apple"], similarity_top_k=4, filters=filters))
    assert result.nodes is not None
    assert len(result.nodes) == 4  # OR of both categories matches everything


def test_delete_by_ref_doc_id_removes_all_its_nodes(store: RuVectorStore) -> None:
    _add_fruit_and_vehicle_nodes(store)
    store.delete("doc-fruit")
    result = store.query(VectorStoreQuery(query_embedding=_BASIS["apple"], similarity_top_k=4))
    assert result.nodes is not None
    texts = {n.get_content() for n in result.nodes}
    assert texts == {"car", "truck"}


def test_delete_nodes_by_id(store: RuVectorStore) -> None:
    _add_fruit_and_vehicle_nodes(store)
    store.delete_nodes(node_ids=["n-apple"])
    result = store.query(VectorStoreQuery(query_embedding=_BASIS["apple"], similarity_top_k=4))
    assert result.nodes is not None
    texts = {n.get_content() for n in result.nodes}
    assert "apple" not in texts
    assert len(texts) == 3


def test_clear_removes_everything(store: RuVectorStore) -> None:
    _add_fruit_and_vehicle_nodes(store)
    store.clear()
    assert len(store.client) == 0


def test_client_property_exposes_the_collection(store: RuVectorStore) -> None:
    assert isinstance(store.client, Collection)


def test_query_rejects_non_default_mode(store: RuVectorStore) -> None:
    from llama_index.core.vector_stores.types import VectorStoreQueryMode

    _add_fruit_and_vehicle_nodes(store)
    with pytest.raises(NotImplementedError):
        store.query(VectorStoreQuery(query_embedding=_BASIS["apple"], similarity_top_k=1, mode=VectorStoreQueryMode.MMR))


def test_import_error_has_install_hint_when_llama_index_core_missing(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Same sys.modules-sentinel technique as the LangChain adapter's equivalent test.

    See that test's docstring for why the whole ``llama_index*`` family
    (not just ``llama_index.core``) must be evicted first: a
    ``from llama_index.core.schema import X`` resolves straight from an
    already-cached ``sys.modules["llama_index.core.schema"]`` without
    consulting the root sentinel at all.
    """
    for name in list(sys.modules):
        if name == "llama_index" or name.startswith("llama_index."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "llama_index.core", None)
    monkeypatch.delitem(sys.modules, "ruvector.integrations.llamaindex", raising=False)

    import importlib

    with pytest.raises(ImportError, match=r"pip install 'ruvector\[llamaindex\]'"):
        importlib.import_module("ruvector.integrations.llamaindex")
