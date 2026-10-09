"""Tests for ruvector.Collection (ADR-352): ids, metadata, filtering,
delete, vacuum, and persistence — parametrized across both backends
(``hnsw``, the default since ADR-352's M2 slice, and ``rabitq``, the
original M1.5 backend) wherever the behavior is meant to be identical.
Backend-specific semantics (rabitq's tombstone+vacuum vs hnsw's real
delete; rabitq's u32 id ceiling) get their own, explicitly-pinned tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ruvector import Collection, CollectionError

Vectors = NDArray[np.float32]

BACKENDS = ["hnsw", "rabitq"]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def vectors(rng: np.random.Generator) -> Vectors:
    return rng.standard_normal((64, 8)).astype(np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
def test_from_vectors_basic(vectors: Vectors, backend: str) -> None:
    coll = Collection.from_vectors(vectors, backend=backend, rerank_factor=10)
    assert len(coll) == 64
    assert coll.stats().dim == 8
    assert coll.stats().tombstoned == 0
    assert coll.stats().backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
def test_from_vectors_with_metadata(vectors: Vectors, backend: str) -> None:
    metas = [{"cat": "a" if i % 2 == 0 else "b"} for i in range(len(vectors))]
    coll = Collection.from_vectors(vectors, metadatas=metas, backend=backend, rerank_factor=10)
    hits = coll.search(vectors[0], 5)
    assert hits[0].id == 0
    assert hits[0].metadata == {"cat": "a"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_search_returns_self_as_nearest(vectors: Vectors, backend: str) -> None:
    coll = Collection.from_vectors(vectors, backend=backend, rerank_factor=10)
    hits = coll.search(vectors[5], 1)
    assert hits[0].id == 5
    assert hits[0].score == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_search_dict_filter(vectors: Vectors, backend: str) -> None:
    metas = [{"cat": "a" if i % 2 == 0 else "b"} for i in range(len(vectors))]
    coll = Collection.from_vectors(vectors, metadatas=metas, backend=backend, rerank_factor=10)
    hits = coll.search(vectors[0], 5, filter={"cat": "a"})
    assert len(hits) == 5
    assert all((h.metadata or {}).get("cat") == "a" for h in hits)


@pytest.mark.parametrize("backend", BACKENDS)
def test_search_callable_filter(vectors: Vectors, backend: str) -> None:
    metas = [{"score_tier": i} for i in range(len(vectors))]
    coll = Collection.from_vectors(vectors, metadatas=metas, backend=backend, rerank_factor=10)
    hits = coll.search(vectors[0], 5, filter=lambda m: m.get("score_tier", -1) >= 30)
    assert all((h.metadata or {}).get("score_tier", -1) >= 30 for h in hits)


@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_excludes_from_search(vectors: Vectors, backend: str) -> None:
    coll = Collection.from_vectors(vectors, backend=backend, rerank_factor=10)
    hits = coll.search(vectors[2], 1)
    victim = hits[0].id
    coll.delete(victim)
    hits2 = coll.search(vectors[2], 5)
    assert victim not in [h.id for h in hits2]
    assert len(coll) == len(vectors) - 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_clears_metadata(vectors: Vectors, backend: str) -> None:
    coll = Collection.from_vectors(vectors, metadatas=[{"i": i} for i in range(len(vectors))], backend=backend, rerank_factor=10)
    coll.delete(5)
    assert coll.get_metadata(5) is None


def test_vacuum_physically_removes_rabitq_tombstones(vectors: Vectors) -> None:
    """rabitq-only: delete() is a tombstone, vacuum() reclaims it. (hnsw's
    delete is already real — see test_hnsw_delete_is_immediate_no_vacuum_needed.)"""
    coll = Collection.from_vectors(vectors, backend="rabitq", rerank_factor=10)
    coll.delete(0)
    coll.delete(1)
    dropped = coll.vacuum()
    assert dropped == 2
    assert coll.stats().tombstoned == 0
    assert len(coll) == len(vectors) - 2


def test_hnsw_delete_is_immediate_no_vacuum_needed(vectors: Vectors) -> None:
    coll = Collection.from_vectors(vectors, backend="hnsw")
    coll.delete(0)
    coll.delete(1)
    # already gone, no vacuum() call needed:
    assert len(coll) == len(vectors) - 2
    assert coll.stats().tombstoned == 0
    # vacuum() is a documented no-op for this backend - real delete already happened.
    assert coll.vacuum() == 0
    assert len(coll) == len(vectors) - 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_insert_and_insert_batch(vectors: Vectors, rng: np.random.Generator, backend: str) -> None:
    coll = Collection.from_vectors(vectors, backend=backend, rerank_factor=10)
    nid = coll.insert(rng.standard_normal(8).astype(np.float32), metadata={"x": 1})
    assert nid == len(vectors)
    assert coll.get_metadata(nid) == {"x": 1}
    ids = coll.insert_batch(rng.standard_normal((3, 8)).astype(np.float32))
    assert ids == [nid + 1, nid + 2, nid + 3]
    assert len(coll) == len(vectors) + 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_collection_build(rng: np.random.Generator, backend: str) -> None:
    coll = Collection.create(dim=8, backend=backend)
    assert len(coll) == 0
    assert coll.search(rng.standard_normal(8).astype(np.float32), 5) == []
    fid = coll.insert(rng.standard_normal(8).astype(np.float32))
    assert fid == 0
    assert len(coll) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_dim_mismatch_raises(vectors: Vectors, rng: np.random.Generator, backend: str) -> None:
    coll = Collection.from_vectors(vectors, backend=backend, rerank_factor=10)
    with pytest.raises(CollectionError):
        coll.insert(rng.standard_normal(9).astype(np.float32))
    with pytest.raises(CollectionError):
        coll.search(rng.standard_normal(9).astype(np.float32), 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_duplicate_ids_rejected(vectors: Vectors, backend: str) -> None:
    with pytest.raises(CollectionError):
        Collection.from_vectors(vectors, ids=[0] * len(vectors), backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_custom_ids_round_trip_in_search(vectors: Vectors, backend: str) -> None:
    """Regression test: an earlier cut of from_vectors accepted a
    non-identity `ids=` kwarg but silently returned row indices from
    search() instead of the caller's own ids."""
    custom_ids = [1000 + i * 7 for i in range(len(vectors))]
    coll = Collection.from_vectors(vectors, ids=custom_ids, backend=backend, rerank_factor=10)
    hits = coll.search(vectors[5], 1)
    assert hits[0].id == custom_ids[5]
    assert hits[0].id not in (5,)

    # ids allocated after a custom-id bulk build must not collide with them.
    new_id = coll.insert(vectors[0])
    assert new_id not in custom_ids
    assert new_id == max(custom_ids) + 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_custom_ids_with_metadata(vectors: Vectors, backend: str) -> None:
    custom_ids = [100 + i for i in range(len(vectors))]
    metas = [{"tag": f"item-{i}"} for i in range(len(vectors))]
    coll = Collection.from_vectors(vectors, ids=custom_ids, metadatas=metas, backend=backend, rerank_factor=10)
    hits = coll.search(vectors[3], 1)
    assert hits[0].id == custom_ids[3]
    assert hits[0].metadata == {"tag": "item-3"}


def test_id_exceeding_u32_rejected_rabitq_only(vectors: Vectors) -> None:
    """rabitq-only: RabitqIndex stores ids as u32. hnsw has no such
    ceiling (ids are strings on the Rust side) - see
    test_hnsw_large_id_not_rejected."""
    too_big = [2**32 + i for i in range(len(vectors))]
    with pytest.raises(CollectionError):
        Collection.from_vectors(vectors, ids=too_big, backend="rabitq")


def test_hnsw_large_id_not_rejected(vectors: Vectors) -> None:
    big_ids = [2**40 + i for i in range(len(vectors))]
    coll = Collection.from_vectors(vectors, ids=big_ids, backend="hnsw")
    hits = coll.search(vectors[0], 1)
    assert hits[0].id == big_ids[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_roundtrip(vectors: Vectors, backend: str) -> None:
    metas = [{"cat": "a" if i % 2 == 0 else "b"} for i in range(len(vectors))]
    coll = Collection.from_vectors(vectors, metadatas=metas, backend=backend, rerank_factor=10)
    coll.delete(3)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.rbpx"
        coll.save(path)
        assert path.exists()
        assert path.with_suffix(path.suffix + ".meta.json").exists()
        coll2 = Collection.load(path)
    assert len(coll2) == len(coll)
    assert coll2.stats().dim == coll.stats().dim
    assert coll2.stats().backend == backend
    h1 = coll.search(vectors[0], 5)
    h2 = coll2.search(vectors[0], 5)
    assert [h.id for h in h1] == [h.id for h in h2]
    assert coll2.get_metadata(0) == {"cat": "a"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_empty_collection(backend: str) -> None:
    coll = Collection.create(dim=4, backend=backend)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "empty.rbpx"
        coll.save(path)
        coll2 = Collection.load(path)
    assert len(coll2) == 0
    assert coll2.stats().backend == backend


def test_load_missing_sidecar_raises() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "nope.rbpx"
        path.write_bytes(b"not a real index")
        with pytest.raises(CollectionError):
            Collection.load(path)


def test_load_old_sidecar_without_backend_key_defaults_to_rabitq(vectors: Vectors) -> None:
    """Sidecars written before the hnsw backend existed have no "backend"
    key at all - must still load (as rabitq, the only backend that
    existed then), not raise or silently misinterpret."""
    import json

    coll = Collection.from_vectors(vectors, backend="rabitq", rerank_factor=10)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "old.rbpx"
        coll.save(path)
        meta_path = Collection.meta_path(path)
        sidecar = json.loads(meta_path.read_text())
        del sidecar["backend"]
        meta_path.write_text(json.dumps(sidecar))
        coll2 = Collection.load(path)
    assert coll2.stats().backend == "rabitq"
    assert len(coll2) == len(coll)


def test_unknown_backend_rejected() -> None:
    with pytest.raises(CollectionError):
        Collection.create(dim=4, backend="not-a-real-backend")


def test_hnsw_metric_options(rng: np.random.Generator) -> None:
    for metric in ["cosine", "euclidean", "l2", "dot", "manhattan"]:
        coll = Collection.create(dim=4, backend="hnsw", metric=metric)
        coll.insert(rng.standard_normal(4).astype(np.float32))
        assert len(coll) == 1
    with pytest.raises(Exception):
        Collection.create(dim=4, backend="hnsw", metric="not-a-real-metric")


# --- HNSW metadata JSON conversion (hnsw.rs `py_to_json`) -------------------


def _hnsw_one(meta: dict[str, object]) -> Collection:
    vec = np.ones((1, 4), dtype=np.float32)
    return Collection.from_vectors(vec, metadatas=[meta], backend="hnsw")


def test_hnsw_metadata_u64_range_int_is_exact() -> None:
    big = 2**64 - 1
    coll = _hnsw_one({"h": big})
    hit = coll.search(np.ones(4, dtype=np.float32), 1)[0]
    assert hit.metadata == {"h": big}
    assert type(hit.metadata["h"]) is int


def test_hnsw_metadata_int_beyond_u64_raises_instead_of_rounding() -> None:
    with pytest.raises(OverflowError):
        _hnsw_one({"h": 2**64})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_hnsw_metadata_non_finite_float_raises_instead_of_null(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _hnsw_one({"x": bad})


def test_hnsw_metadata_lone_surrogate_str_gives_content_error() -> None:
    with pytest.raises(ValueError, match="surrogate"):
        _hnsw_one({"s": "ab\ud800cd"})


# ── pre-publish hardening: resource bounds, non-finite input, atomic save ──


@pytest.mark.parametrize("backend", BACKENDS)
def test_search_k_upper_bound(vectors: Vectors, backend: str) -> None:
    """k=10**12 used to abort the whole process (an 8 TB allocation in
    Rust) on the hnsw backend; it must now be a clean error."""
    coll = Collection.from_vectors(vectors, backend=backend)
    with pytest.raises(CollectionError, match="k"):
        coll.search(vectors[0], 10**12)
    with pytest.raises(CollectionError, match="k"):
        coll.search(vectors[0], 10_001)
    assert len(coll.search(vectors[0], 10_000)) == 64  # the bound itself is allowed


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_dim_and_rerank_bounds(backend: str) -> None:
    with pytest.raises(CollectionError, match="dim"):
        Collection.create(8193, backend=backend)
    Collection.create(8192, backend=backend)  # boundary allowed
    for bad in (0, -1, 10_001):
        with pytest.raises(CollectionError, match="rerank_factor"):
            Collection.create(4, backend=backend, rerank_factor=bad)
    with pytest.raises(CollectionError, match="dim"):
        Collection.from_vectors(np.zeros((1, 8193), dtype=np.float32), backend=backend)


def test_search_rerank_factor_and_overfetch_bounds(vectors: Vectors) -> None:
    coll = Collection.from_vectors(vectors, backend="rabitq")
    for bad in (0, -5, 10**9):
        with pytest.raises(CollectionError, match="rerank_factor"):
            coll.search(vectors[0], 3, rerank_factor=bad)
    for bad in (0, 10**9):
        with pytest.raises(CollectionError, match="overfetch"):
            coll.search(vectors[0], 3, overfetch=bad)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_vectors_rejected(vectors: Vectors, backend: str, bad: float) -> None:
    coll = Collection.from_vectors(vectors, backend=backend)
    n = len(coll)
    poisoned = vectors[0].copy()
    poisoned[3] = bad
    with pytest.raises(CollectionError, match="finite"):
        coll.insert(poisoned)
    with pytest.raises(CollectionError, match="finite"):
        coll.insert_batch(np.stack([vectors[1], poisoned]))
    with pytest.raises(CollectionError, match="finite"):
        coll.search(poisoned, 3)
    with pytest.raises(CollectionError, match="finite"):
        Collection.from_vectors(np.stack([vectors[1], poisoned]), backend=backend)
    assert len(coll) == n  # nothing was half-inserted
    assert coll.search(vectors[0], 1)[0].id == 0  # index is not poisoned


def test_non_finite_overflow_to_inf_rejected() -> None:
    """1e39 is finite as float64 but becomes inf when cast to float32."""
    coll = Collection.create(2)
    with pytest.raises(CollectionError, match="finite"):
        coll.insert(np.array([1e39, 0.0], dtype=np.float64))  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_is_atomic_on_failed_replace(
    vectors: Vectors, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    import os

    path = tmp_path / "c.rbpx"
    coll = Collection.from_vectors(vectors[:10], backend=backend)
    coll.save(path)
    before = (path.read_bytes(), Collection.meta_path(path).read_bytes())

    coll.insert(vectors[20])

    def boom(src: object, dst: object) -> None:
        raise OSError("simulated crash during replace")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        coll.save(path)
    monkeypatch.undo()

    assert (path.read_bytes(), Collection.meta_path(path).read_bytes()) == before
    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []  # no temp litter
    assert len(Collection.load(path)) == 10


@pytest.mark.parametrize("backend", BACKENDS)
def test_load_rejects_corrupt_sidecar(vectors: Vectors, tmp_path: Path, backend: str) -> None:
    path = tmp_path / "c.rbpx"
    Collection.from_vectors(vectors[:10], backend=backend).save(path)
    meta = Collection.meta_path(path)
    good = meta.read_text()

    meta.write_text("{not json")
    with pytest.raises(CollectionError):
        Collection.load(path)

    meta.write_text("[1, 2, 3]")
    with pytest.raises(CollectionError):
        Collection.load(path)

    meta.write_text("{}")
    with pytest.raises(CollectionError):
        Collection.load(path)

    import json

    bad = json.loads(good)
    bad["dim"] = 10**9
    meta.write_text(json.dumps(bad))
    with pytest.raises(CollectionError, match="dim"):
        Collection.load(path)

    meta.write_text(good)
    assert len(Collection.load(path)) == 10


def test_load_hnsw_rejects_inconsistent_or_non_finite_vectors(vectors: Vectors, tmp_path: Path) -> None:
    path = tmp_path / "c.rbpx"
    Collection.from_vectors(vectors[:10], backend="hnsw").save(path)

    np.save(path.open("wb"), vectors[:9])  # row count != len(ids)
    with pytest.raises(CollectionError):
        Collection.load(path)

    poisoned = vectors[:10].copy()
    poisoned[2, 1] = np.nan
    with path.open("wb") as f:
        np.save(f, poisoned)
    with pytest.raises(CollectionError, match="finite"):
        Collection.load(path)

    path.write_bytes(b"not an npy file")
    with pytest.raises(CollectionError):
        Collection.load(path)
