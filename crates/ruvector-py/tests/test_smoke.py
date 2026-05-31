"""M1 smoke tests for ``ruvector``.

These exercise the user-visible surface of the wheel:

  - ``ruvector.__version__`` is set
  - ``RabitqIndex.build`` accepts an ``(n, dim)`` float32 NumPy array
  - ``RabitqIndex.search`` returns ``k`` ``(id, score)`` tuples
  - first-result self-search returns id 0 at distance ~0
  - dimension mismatch raises ``RuVectorError``
  - save/load roundtrip preserves search results

Run via ``pytest tests/`` after ``maturin develop`` (see README).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import ruvector


def test_version() -> None:
    assert ruvector.__version__
    # Cargo.toml ships 0.1.0; if you bump there, bump here.
    assert ruvector.__version__ == "0.1.0"


def test_build_and_search() -> None:
    rng = np.random.default_rng(42)
    n, dim = 1000, 128
    vectors = rng.standard_normal((n, dim), dtype=np.float32)
    idx = ruvector.RabitqIndex.build(vectors)

    assert len(idx) == n
    assert idx.dim == dim
    assert idx.memory_bytes > 0
    assert idx.rerank_factor == 20  # default

    query = vectors[0]
    results = idx.search(query, k=10)

    assert len(results) == 10
    # First hit must be the query vector itself: id 0, distance ~0 after
    # exact f32 rerank.
    top_id, top_dist = results[0]
    assert top_id == 0, f"expected self-match at id 0, got id {top_id}"
    assert top_dist < 1e-3, f"self-distance {top_dist} should be ~0"

    # Scores must be ascending (squared L2 — lower is closer).
    scores = [s for _, s in results]
    assert scores == sorted(scores), f"scores not ascending: {scores}"


def test_repr_is_diagnostic() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((50, 32), dtype=np.float32)
    idx = ruvector.RabitqIndex.build(vectors)
    r = repr(idx)
    assert "RabitqIndex" in r
    assert "n=50" in r
    assert "dim=32" in r


def test_error_on_dim_mismatch() -> None:
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((100, 64), dtype=np.float32)
    idx = ruvector.RabitqIndex.build(vectors)
    bad_query = rng.standard_normal(32, dtype=np.float32)
    with pytest.raises(ruvector.RuVectorError):
        idx.search(bad_query, k=10)


def test_error_on_wrong_dtype() -> None:
    # float64 must not silently coerce — it should hit the boundary
    # PyO3 numpy crate's strict dtype check.
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((10, 8))  # float64
    with pytest.raises((TypeError, ValueError)):
        ruvector.RabitqIndex.build(vectors)  # type: ignore[arg-type]


def test_save_load_roundtrip() -> None:
    rng = np.random.default_rng(7)
    n, dim = 200, 64
    vectors = rng.standard_normal((n, dim), dtype=np.float32)
    idx = ruvector.RabitqIndex.build(vectors, rerank_factor=5, seed=1234)

    query = rng.standard_normal(dim, dtype=np.float32)
    before = idx.search(query, k=5)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "idx.rbpx")
        idx.save(path)
        loaded = ruvector.RabitqIndex.load(path)

    assert len(loaded) == n
    assert loaded.dim == dim
    assert loaded.rerank_factor == 5

    after = loaded.search(query, k=5)
    # `(dim, seed, items)` deterministic rebuild → bit-identical search.
    assert before == after, f"roundtrip changed results: {before} vs {after}"


def test_search_with_per_call_rerank() -> None:
    rng = np.random.default_rng(99)
    n, dim = 500, 64
    vectors = rng.standard_normal((n, dim), dtype=np.float32)
    idx = ruvector.RabitqIndex.build(vectors, rerank_factor=2)

    query = vectors[10]

    # Override per call — should still self-match at id 10 with distance ~0.
    results = idx.search(query, k=3, rerank_factor=20)
    assert results[0][0] == 10
    assert results[0][1] < 1e-3
