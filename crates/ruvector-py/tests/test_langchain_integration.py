"""Tests for ``ruvector.integrations.langchain.RuVectorStore``.

Uses a deterministic, offline, hash-free fake ``Embeddings`` (a fixed
lookup table over a small orthogonal basis — same style as
``tests/test_collection.py``'s "search returns self as nearest" tests) so
nearest-neighbor answers are unambiguous and the suite stays fast and
network-free.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

import numpy as np
import pytest

from ruvector import Collection

langchain_core = pytest.importorskip("langchain_core")  # noqa: F841 - presence check only

from ruvector.integrations.langchain import RuVectorStore  # noqa: E402

from langchain_core.embeddings import Embeddings  # noqa: E402

# Fixed 4-dim orthogonal basis: nearest neighbour for any of these query
# texts is unambiguous (exact match, distance 0; every other pair is
# equidistant and non-zero).
_VOCAB: Dict[str, List[float]] = {
    "apple": [1.0, 0.0, 0.0, 0.0],
    "banana": [0.0, 1.0, 0.0, 0.0],
    "car": [0.0, 0.0, 1.0, 0.0],
    "truck": [0.0, 0.0, 0.0, 1.0],
}


class FixedVocabEmbeddings(Embeddings):
    """Deterministic fake embedder: exact lookup over ``_VOCAB``, no model."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(_VOCAB[t]) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return list(_VOCAB[text])


@pytest.fixture
def embedding() -> FixedVocabEmbeddings:
    return FixedVocabEmbeddings()


@pytest.fixture
def store(embedding: FixedVocabEmbeddings) -> RuVectorStore:
    coll = Collection.create(dim=4)
    return RuVectorStore(coll, embedding)


def test_add_texts_and_similarity_search(store: RuVectorStore) -> None:
    ids = store.add_texts(["apple", "banana", "car", "truck"])
    assert len(ids) == 4
    assert len(set(ids)) == 4  # uuid4 ids, all distinct

    docs = store.similarity_search("apple", k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "apple"


def test_similarity_search_with_score_is_zero_distance_for_exact_match(store: RuVectorStore) -> None:
    store.add_texts(["apple", "banana", "car", "truck"])
    results = store.similarity_search_with_score("banana", k=1)
    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "banana"
    assert score == pytest.approx(0.0, abs=1e-6)


def test_metadata_is_merged_and_text_key_not_leaked_into_metadata(store: RuVectorStore) -> None:
    store.add_texts(["apple", "car"], metadatas=[{"category": "fruit"}, {"category": "vehicle"}])
    docs = store.similarity_search("apple", k=1)
    assert docs[0].metadata == {"category": "fruit"}
    assert "text" not in docs[0].metadata


def test_add_texts_with_explicit_ids_round_trip(store: RuVectorStore) -> None:
    ids = store.add_texts(["apple", "banana"], ids=["id-apple", "id-banana"])
    assert ids == ["id-apple", "id-banana"]
    docs = store.similarity_search("apple", k=1)
    assert docs[0].id == "id-apple"


def test_add_texts_rejects_duplicate_ids(store: RuVectorStore) -> None:
    store.add_texts(["apple"], ids=["dup"])
    with pytest.raises(ValueError):
        store.add_texts(["banana"], ids=["dup"])


def test_delete_removes_from_search_results(store: RuVectorStore) -> None:
    ids = store.add_texts(["apple", "banana", "car", "truck"])
    ok = store.delete([ids[0]])  # delete "apple"
    assert ok is True
    assert len(store._collection) == 3

    # "apple"'s own nearest neighbour was itself; with it gone, a k=4
    # search must not resurrect it.
    docs = store.similarity_search("apple", k=4)
    assert all(d.page_content != "apple" for d in docs)


def test_delete_unknown_id_returns_false(store: RuVectorStore) -> None:
    store.add_texts(["apple"])
    assert store.delete(["not-a-real-id"]) is False


def test_delete_all_when_ids_is_none(store: RuVectorStore) -> None:
    store.add_texts(["apple", "banana"])
    assert store.delete(None) is True
    assert len(store._collection) == 0


def test_from_texts_classmethod_builds_fresh_collection(embedding: FixedVocabEmbeddings) -> None:
    store2 = RuVectorStore.from_texts(["apple", "banana", "car"], embedding)
    assert isinstance(store2, RuVectorStore)
    assert store2._collection.stats().dim == 4
    docs = store2.similarity_search("car", k=1)
    assert docs[0].page_content == "car"


def test_from_texts_requires_at_least_one_text(embedding: FixedVocabEmbeddings) -> None:
    with pytest.raises(ValueError):
        RuVectorStore.from_texts([], embedding)


def test_embeddings_property_returns_the_embedder(store: RuVectorStore, embedding: FixedVocabEmbeddings) -> None:
    assert store.embeddings is embedding


def test_similarity_search_respects_filter(store: RuVectorStore) -> None:
    store.add_texts(
        ["apple", "banana", "car", "truck"],
        metadatas=[
            {"category": "fruit"},
            {"category": "fruit"},
            {"category": "vehicle"},
            {"category": "vehicle"},
        ],
    )
    docs = store.similarity_search("apple", k=4, filter={"category": "vehicle"})
    assert len(docs) == 2
    assert {d.page_content for d in docs} == {"car", "truck"}


def test_import_error_has_install_hint_when_langchain_core_missing(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Simulate 'langchain-core not installed' via the sys.modules sentinel pattern.

    Setting ``sys.modules["langchain_core"] = None`` makes a *fresh*
    ``import langchain_core...`` statement raise ``ImportError`` immediately
    (a documented CPython import-system behavior for blocking optional
    imports) — but only if ``langchain_core``'s submodules aren't already
    cached: ``from langchain_core.documents import Document`` resolves
    straight from ``sys.modules["langchain_core.documents"]`` without even
    consulting the ``langchain_core`` root once that submodule is already
    loaded, so the root-only sentinel is a no-op after this test file's own
    top-level ``from langchain_core... import`` statements already warmed
    the cache (verified empirically while writing this test — the
    root-only version silently didn't raise). Evicting the whole
    ``langchain_core*`` family first, then sentinel-ing the root, closes
    that gap. ``monkeypatch`` undoes both the sentinel and every eviction
    at teardown, and nothing here touches the real installed package.
    """
    for name in list(sys.modules):
        if name == "langchain_core" or name.startswith("langchain_core."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.delitem(sys.modules, "ruvector.integrations.langchain", raising=False)

    import importlib

    with pytest.raises(ImportError, match=r"pip install 'ruvector\[langchain\]'"):
        importlib.import_module("ruvector.integrations.langchain")
