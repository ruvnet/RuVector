"""Per-call ``ef_search`` on ``HnswIndex.search`` changes the candidate-list size for that call only."""
import numpy as np
import pytest

import ruvector


def _index(n=1200, dim=16, ef_search=10):
    rng = np.random.default_rng(7)
    x = rng.random((n, dim), dtype=np.float32)
    idx = ruvector.HnswIndex.create(dim, metric="euclidean", m=16, ef_construction=200, ef_search=ef_search)
    idx.insert_batch([str(i) for i in range(n)], x)
    return idx, x, rng


def _recall(idx, x, queries, k, **kw):
    hits = 0
    for q in queries:
        truth = set(np.argsort(((x - q) ** 2).sum(axis=1))[:k].tolist())
        got = {int(r[0]) for r in idx.search(q, k, **kw)}
        hits += len(truth & got)
    return hits / (len(queries) * k)


def test_larger_ef_search_raises_recall_and_none_means_configured():
    idx, x, rng = _index()
    queries = rng.random((40, x.shape[1]), dtype=np.float32)
    low = _recall(idx, x, queries, 10, ef_search=10)
    high = _recall(idx, x, queries, 10, ef_search=300)
    assert high >= low
    assert high > 0.97
    assert _recall(idx, x, queries, 10) == _recall(idx, x, queries, 10, ef_search=None)


def test_ef_search_below_k_is_clamped_not_an_error():
    idx, x, _ = _index()
    assert len(idx.search(x[0], 10, ef_search=1)) == 10


def test_results_are_sorted_by_distance_and_the_nearest_is_the_query_itself():
    idx, x, _ = _index()
    res = idx.search(x[5], 10, ef_search=100)
    assert res[0][0] == "5"
    scores = [r[1] for r in res]
    assert scores == sorted(scores)
