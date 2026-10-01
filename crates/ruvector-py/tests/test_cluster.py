"""Tests for ``ruvector.kmeans`` — the k-means binding over
``ruvector_cluster_rag::cluster::kmeans``.

Covers: separable clusters give correct partitioning, output shapes match
``k``, and every documented error path (k=0, k>n, empty input, non-finite
coordinates) raises ``ValueError`` rather than crashing with a Rust panic.
"""

from __future__ import annotations

import numpy as np
import pytest

import ruvector


def test_two_tight_clusters_are_separated() -> None:
    rng = np.random.default_rng(42)
    # Two well-separated, tight blobs.
    cluster_a = rng.normal(loc=0.0, scale=0.01, size=(50, 8)).astype(np.float32)
    cluster_b = rng.normal(loc=100.0, scale=0.01, size=(50, 8)).astype(np.float32)
    vectors = np.ascontiguousarray(np.vstack([cluster_a, cluster_b]))

    assignments, centroids, cohesion, cluster_sizes = ruvector.kmeans(vectors, k=2)

    assert centroids.shape == (2, 8)
    assert cohesion.shape == (2,)
    assert cluster_sizes.shape == (2,)
    assert len(assignments) == 100

    # Membership check via set, not exact label values — label order
    # isn't guaranteed (k-means++ seeding may pick either blob first).
    group_a_labels = set(assignments[:50].tolist())
    group_b_labels = set(assignments[50:].tolist())
    assert len(group_a_labels) == 1, "cluster A members must share one label"
    assert len(group_b_labels) == 1, "cluster B members must share one label"
    assert group_a_labels != group_b_labels, "the two blobs must get different labels"

    assert int(cluster_sizes.sum()) == 100
    # Cohesion is mean *cosine* similarity to the centroid, not a Euclidean
    # tightness measure — a blob centered near the origin (small magnitude,
    # noise-dominated direction) can have low cosine cohesion even though
    # it is Euclidean-tight. Just check it stays in the documented range.
    assert all(-1.0 <= c <= 1.0 for c in cohesion.tolist())


def test_centroid_count_matches_k() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((60, 4), dtype=np.float32)
    for k in (1, 3, 6):
        assignments, centroids, cohesion, cluster_sizes = ruvector.kmeans(vectors, k=k)
        assert centroids.shape == (k, 4)
        assert cohesion.shape == (k,)
        assert cluster_sizes.shape == (k,)
        assert set(assignments.tolist()) <= set(range(k))


def test_iters_kwarg_is_accepted() -> None:
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((20, 3), dtype=np.float32)
    assignments, centroids, _, _ = ruvector.kmeans(vectors, k=2, iters=1)
    assert len(assignments) == 20
    assert centroids.shape == (2, 3)


def test_error_on_k_zero() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="k must be > 0"):
        ruvector.kmeans(vectors, k=0)


def test_error_on_k_exceeds_n() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((5, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="must not exceed"):
        ruvector.kmeans(vectors, k=6)


def test_error_on_empty_input() -> None:
    vectors = np.empty((0, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="at least 1 row"):
        ruvector.kmeans(vectors, k=1)


def test_error_on_non_finite_coordinates() -> None:
    vectors = np.array([[0.0, 0.0], [float("nan"), 1.0], [2.0, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        ruvector.kmeans(vectors, k=2)


def test_error_on_wrong_dtype() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((10, 4))  # float64, not float32
    with pytest.raises((TypeError, ValueError)):
        ruvector.kmeans(vectors, k=2)  # type: ignore[arg-type]


def test_error_on_non_contiguous_input() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((10, 8), dtype=np.float32)
    non_contig = base[:, ::2]  # a strided view, not C-contiguous
    assert not non_contig.flags["C_CONTIGUOUS"]
    with pytest.raises(TypeError, match="C-contiguous"):
        ruvector.kmeans(non_contig, k=2)
