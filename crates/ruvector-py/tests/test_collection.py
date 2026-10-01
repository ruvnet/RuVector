"""Tests for ruvector.Collection (ADR-352 M1.5): ids, metadata, filtering,
soft delete, vacuum, and persistence over the M1 RabitqIndex.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ruvector import Collection, CollectionError


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def vectors(rng):
    return rng.standard_normal((64, 8)).astype(np.float32)


def test_from_vectors_basic(vectors):
    coll = Collection.from_vectors(vectors, rerank_factor=10)
    assert len(coll) == 64
    assert coll.stats().dim == 8
    assert coll.stats().tombstoned == 0


def test_from_vectors_with_metadata(vectors):
    metas = [{"cat": "a" if i % 2 == 0 else "b"} for i in range(len(vectors))]
    coll = Collection.from_vectors(vectors, metadatas=metas, rerank_factor=10)
    hits = coll.search(vectors[0], 5)
    assert hits[0].id == 0
    assert hits[0].metadata == {"cat": "a"}


def test_search_returns_self_as_nearest(vectors):
    coll = Collection.from_vectors(vectors, rerank_factor=10)
    hits = coll.search(vectors[5], 1)
    assert hits[0].id == 5
    assert hits[0].score == pytest.approx(0.0, abs=1e-4)


def test_search_dict_filter(vectors):
    metas = [{"cat": "a" if i % 2 == 0 else "b"} for i in range(len(vectors))]
    coll = Collection.from_vectors(vectors, metadatas=metas, rerank_factor=10)
    hits = coll.search(vectors[0], 5, filter={"cat": "a"})
    assert len(hits) == 5
    assert all(h.metadata["cat"] == "a" for h in hits)


def test_search_callable_filter(vectors):
    metas = [{"score_tier": i} for i in range(len(vectors))]
    coll = Collection.from_vectors(vectors, metadatas=metas, rerank_factor=10)
    hits = coll.search(vectors[0], 5, filter=lambda m: m.get("score_tier", -1) >= 30)
    assert all(h.metadata["score_tier"] >= 30 for h in hits)


def test_delete_excludes_from_search(vectors):
    coll = Collection.from_vectors(vectors, rerank_factor=10)
    hits = coll.search(vectors[2], 1)
    victim = hits[0].id
    coll.delete(victim)
    hits2 = coll.search(vectors[2], 5)
    assert victim not in [h.id for h in hits2]
    assert len(coll) == len(vectors) - 1


def test_vacuum_physically_removes(vectors):
    coll = Collection.from_vectors(vectors, rerank_factor=10)
    coll.delete(0)
    coll.delete(1)
    dropped = coll.vacuum()
    assert dropped == 2
    assert coll.stats().tombstoned == 0
    assert len(coll) == len(vectors) - 2


def test_insert_and_insert_batch(vectors, rng):
    coll = Collection.from_vectors(vectors, rerank_factor=10)
    nid = coll.insert(rng.standard_normal(8).astype(np.float32), metadata={"x": 1})
    assert nid == len(vectors)
    assert coll.get_metadata(nid) == {"x": 1}
    ids = coll.insert_batch(rng.standard_normal((3, 8)).astype(np.float32))
    assert ids == [nid + 1, nid + 2, nid + 3]
    assert len(coll) == len(vectors) + 4


def test_empty_collection_lazy_build(rng):
    coll = Collection.create(dim=8)
    assert len(coll) == 0
    assert coll.search(rng.standard_normal(8).astype(np.float32), 5) == []
    fid = coll.insert(rng.standard_normal(8).astype(np.float32))
    assert fid == 0
    assert len(coll) == 1


def test_dim_mismatch_raises(vectors, rng):
    coll = Collection.from_vectors(vectors, rerank_factor=10)
    with pytest.raises(CollectionError):
        coll.insert(rng.standard_normal(9).astype(np.float32))
    with pytest.raises(CollectionError):
        coll.search(rng.standard_normal(9).astype(np.float32), 1)


def test_duplicate_ids_rejected(vectors):
    with pytest.raises(CollectionError):
        Collection.from_vectors(vectors, ids=[0] * len(vectors))


def test_save_load_roundtrip(vectors):
    metas = [{"cat": "a" if i % 2 == 0 else "b"} for i in range(len(vectors))]
    coll = Collection.from_vectors(vectors, metadatas=metas, rerank_factor=10)
    coll.delete(3)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.rbpx"
        coll.save(path)
        assert path.exists()
        assert path.with_suffix(path.suffix + ".meta.json").exists()
        coll2 = Collection.load(path)
    assert len(coll2) == len(coll)
    assert coll2.stats().dim == coll.stats().dim
    h1 = coll.search(vectors[0], 5)
    h2 = coll2.search(vectors[0], 5)
    assert [h.id for h in h1] == [h.id for h in h2]
    assert coll2.get_metadata(0) == {"cat": "a"}


def test_save_load_empty_collection():
    coll = Collection.create(dim=4)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "empty.rbpx"
        coll.save(path)
        coll2 = Collection.load(path)
    assert len(coll2) == 0


def test_load_missing_sidecar_raises():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nope.rbpx"
        path.write_bytes(b"not a real index")
        with pytest.raises(CollectionError):
            Collection.load(path)
