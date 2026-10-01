"""ADR-352 benchmark: ruvector (RaBitQ+ via Collection) vs hnswlib.

Real, measured numbers on this host only (32-core, per `nproc`/`free -h`
in-session). No numbers are carried over from ruvector-rabitq/BENCHMARK.md
(different host) or fabricated. Run manually; not part of pytest (too slow
for CI, and the whole point is a side-by-side with a real comparator that
isn't always installed).
"""
import json
import time

import hnswlib
import numpy as np

from ruvector import Collection


def recall_at_k(approx_ids, exact_ids, k):
    hits = 0
    for a, e in zip(approx_ids, exact_ids):
        hits += len(set(a[:k]) & set(e[:k]))
    return hits / (len(approx_ids) * k)


def exact_knn(data, queries, k):
    # brute-force exact L2 for recall ground truth
    out = []
    for q in queries:
        d = np.sum((data - q) ** 2, axis=1)
        idx = np.argpartition(d, k)[:k]
        idx = idx[np.argsort(d[idx])]
        out.append(idx.tolist())
    return out


def bench_one(n, dim, k, n_queries, rerank_factor, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    ground_truth = exact_knn(data, queries, k)

    # --- ruvector (RabitqPlus via Collection) ---
    t0 = time.perf_counter()
    coll = Collection.from_vectors(data, rerank_factor=rerank_factor)
    rv_build_s = time.perf_counter() - t0

    rv_latencies = []
    rv_results = []
    for q in queries:
        t0 = time.perf_counter()
        hits = coll.search(q, k)
        rv_latencies.append(time.perf_counter() - t0)
        rv_results.append([h.id for h in hits])
    rv_latencies.sort()
    rv_recall = recall_at_k(rv_results, ground_truth, k)
    rv_mem = coll.stats().memory_bytes

    # --- hnswlib ---
    t0 = time.perf_counter()
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n, ef_construction=200, M=16)
    index.add_items(data, np.arange(n))
    index.set_ef(max(k * 2, 50))
    hw_build_s = time.perf_counter() - t0

    hw_latencies = []
    hw_results = []
    for q in queries:
        t0 = time.perf_counter()
        labels, _ = index.knn_query(q, k=k)
        hw_latencies.append(time.perf_counter() - t0)
        hw_results.append(labels[0].tolist())
    hw_latencies.sort()
    hw_recall = recall_at_k(hw_results, ground_truth, k)
    hw_mem = index.get_current_count() * dim * 4  # hnswlib doesn't expose a direct byte count; approx data-only floor

    def pctl(sorted_list, p):
        idx = max(0, min(len(sorted_list) - 1, int(len(sorted_list) * p) - 1))
        return sorted_list[idx] * 1000  # ms

    return {
        "n": n, "dim": dim, "k": k, "n_queries": n_queries, "rerank_factor": rerank_factor,
        "ruvector": {
            "build_s": rv_build_s,
            "p50_ms": pctl(rv_latencies, 0.50),
            "p99_ms": pctl(rv_latencies, 0.99),
            "qps": 1.0 / (sum(rv_latencies) / len(rv_latencies)),
            "recall_at_k": rv_recall,
            "memory_bytes": rv_mem,
        },
        "hnswlib": {
            "build_s": hw_build_s,
            "p50_ms": pctl(hw_latencies, 0.50),
            "p99_ms": pctl(hw_latencies, 0.99),
            "qps": 1.0 / (sum(hw_latencies) / len(hw_latencies)),
            "recall_at_k": hw_recall,
            "memory_bytes_floor": hw_mem,
        },
    }


if __name__ == "__main__":
    configs = [
        dict(n=10_000, dim=128, k=10, n_queries=200, rerank_factor=20),
        dict(n=100_000, dim=128, k=10, n_queries=200, rerank_factor=20),
    ]
    results = [bench_one(**c) for c in configs]
    print(json.dumps(results, indent=2))
