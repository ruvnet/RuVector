"""ADR-352 re-benchmark: ruvector's new HnswIndex-backed Collection vs
hnswlib, on REAL MiniLM text embeddings (20 Newsgroups train split) rather
than synthetic random-Gaussian data. Addresses the earlier benchmark's own
flagged caveat ("random-Gaussian is close to worst-case for both
algorithms... re-run on a real embedding dataset"). Real numbers only,
this host, run manually (not part of pytest - too slow for CI and depends
on sentence-transformers/sklearn which aren't core deps).
"""
import json
import time

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

from ruvector import Collection


def recall_at_k(approx_ids, exact_ids, k):
    hits = 0
    for a, e in zip(approx_ids, exact_ids):
        hits += len(set(a[:k]) & set(e[:k]))
    return hits / (len(approx_ids) * k)


def exact_knn_cosine(data, queries, k):
    # brute-force exact cosine distance (1 - cos_sim) for ground truth
    data_n = data / np.linalg.norm(data, axis=1, keepdims=True)
    q_n = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    out = []
    for q in q_n:
        sims = data_n @ q
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
        out.append(idx.tolist())
    return out


def pctl(sorted_list, p):
    idx = max(0, min(len(sorted_list) - 1, int(len(sorted_list) * p) - 1))
    return sorted_list[idx] * 1000


print("Loading 20 Newsgroups...")
news = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
docs = [d.strip() for d in news.data if len(d.strip()) > 20][:10000]
print(f"{len(docs)} documents")

print("Embedding with all-MiniLM-L6-v2 (CPU)...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
t0 = time.perf_counter()
embeddings = model.encode(docs, batch_size=64, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
embed_s = time.perf_counter() - t0
print(f"embedded {len(docs)} docs, dim={embeddings.shape[1]}, in {embed_s:.1f}s")

n_queries = 200
rng = np.random.default_rng(0)
query_idx = rng.choice(len(docs), size=n_queries, replace=False)
queries = embeddings[query_idx]

dim = embeddings.shape[1]
k = 10

print("Computing exact ground truth (cosine)...")
ground_truth = exact_knn_cosine(embeddings, queries, k)

results = {"dataset": "20newsgroups-train (sklearn)", "n_docs": len(docs), "dim": dim, "k": k,
           "n_queries": n_queries, "embed_seconds": embed_s, "embedder": "all-MiniLM-L6-v2 (CPU)"}

configs = [
    {"m": 16, "ef_construction": 200, "ef_search": 50},
    {"m": 16, "ef_construction": 200, "ef_search": 100},
    {"m": 16, "ef_construction": 200, "ef_search": 200},
]

ruvector_rows = []
for cfg in configs:
    t0 = time.perf_counter()
    coll = Collection.from_vectors(
        embeddings, backend="hnsw", metric="cosine",
        m=cfg["m"], ef_construction=cfg["ef_construction"], ef_search=cfg["ef_search"],
    )
    build_s = time.perf_counter() - t0
    latencies = []
    hits_all = []
    for q in queries:
        t0 = time.perf_counter()
        hits = coll.search(q, k)
        latencies.append(time.perf_counter() - t0)
        hits_all.append([h.id for h in hits])
    latencies.sort()
    recall = recall_at_k(hits_all, ground_truth, k)
    ruvector_rows.append({
        **cfg, "build_s": build_s,
        "p50_ms": pctl(latencies, 0.50), "p99_ms": pctl(latencies, 0.99),
        "qps": 1.0 / (sum(latencies) / len(latencies)), "recall_at_10": recall,
    })
    print("ruvector", cfg, "->", ruvector_rows[-1])

hnswlib_rows = []
for cfg in configs:
    t0 = time.perf_counter()
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(docs), ef_construction=cfg["ef_construction"], M=cfg["m"])
    index.add_items(embeddings, np.arange(len(docs)))
    index.set_ef(cfg["ef_search"])
    build_s = time.perf_counter() - t0
    latencies = []
    hits_all = []
    for q in queries:
        t0 = time.perf_counter()
        labels, _ = index.knn_query(q, k=k)
        latencies.append(time.perf_counter() - t0)
        hits_all.append(labels[0].tolist())
    latencies.sort()
    recall = recall_at_k(hits_all, ground_truth, k)
    hnswlib_rows.append({
        **cfg, "build_s": build_s,
        "p50_ms": pctl(latencies, 0.50), "p99_ms": pctl(latencies, 0.99),
        "qps": 1.0 / (sum(latencies) / len(latencies)), "recall_at_10": recall,
    })
    print("hnswlib", cfg, "->", hnswlib_rows[-1])

results["ruvector"] = ruvector_rows
results["hnswlib"] = hnswlib_rows
print(json.dumps(results, indent=2))
with open("/tmp/bench_hnsw_real_results.json", "w") as f:
    json.dump(results, f, indent=2)
