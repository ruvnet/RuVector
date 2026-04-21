"""
Vector Search Benchmark: TurboQuant vs Baseline Quantization Methods

Evaluates quantization approaches for approximate nearest neighbor search
on standard datasets (GloVe d=200, SIFT1M d=128, PKM d=384).

Compares:
  1. No quantization (f32 brute-force baseline)
  2. Scalar int8 quantization (4x compression)
  3. TurboQuant MSE-only (Stage 1: Hadamard rotation + Lloyd-Max scalar)
  4. TurboQuant full (Stage 1 + QJL residual correction)

Metrics:
  - Recall@1, Recall@10, Recall@100
  - Compression ratio (bits per dimension)
  - Search latency (p50, p95, p99)
  - Memory footprint

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)
Paper: arxiv.org/abs/2504.19874
"""

import sys
import os
import time
import json
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch

# turboquant_ref is a symlink to turboquant-pytorch (valid Python package name)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from turboquant_ref import TurboQuantMSE, TurboQuantProd  # noqa: E402
from turboquant_ref.turboquant import generate_rotation_matrix  # noqa: E402
sys.path.pop(0)


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------

def load_fvecs(path: str) -> np.ndarray:
    """Load vectors in the .fvecs format (SIFT1M)."""
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    vectors = []
    while offset < len(data):
        d = struct.unpack("i", data[offset : offset + 4])[0]
        offset += 4
        vec = struct.unpack(f"{d}f", data[offset : offset + d * 4])
        vectors.append(vec)
        offset += d * 4
    return np.array(vectors, dtype=np.float32)


def load_ivecs(path: str) -> np.ndarray:
    """Load integer vectors in the .ivecs format (ground truth)."""
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    vectors = []
    while offset < len(data):
        d = struct.unpack("i", data[offset : offset + 4])[0]
        offset += 4
        vec = struct.unpack(f"{d}i", data[offset : offset + d * 4])
        vectors.append(vec)
        offset += d * 4
    return np.array(vectors, dtype=np.int32)


def load_glove(path: str, max_vectors: int = 0) -> np.ndarray:
    """Load GloVe text format vectors."""
    vectors = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_vectors and i >= max_vectors:
                break
            parts = line.strip().split()
            vec = [float(x) for x in parts[1:]]
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def load_npy(path: str) -> np.ndarray:
    """Load numpy array."""
    return np.load(path).astype(np.float32)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# ---------------------------------------------------------------------------
# Quantization Methods
# ---------------------------------------------------------------------------

def scalar_int8_quantize(vectors: np.ndarray):
    """Scalar quantization to int8 (same approach as ruvector-core ScalarQuantized)."""
    vmin = vectors.min(axis=1, keepdims=True)
    vmax = vectors.max(axis=1, keepdims=True)
    scale = (vmax - vmin) / 255.0
    scale[scale == 0] = 1.0
    quantized = np.round((vectors - vmin) / scale).clip(0, 255).astype(np.uint8)
    return quantized, vmin.squeeze(), scale.squeeze()


def scalar_int8_search(query: np.ndarray, quantized: np.ndarray,
                       vmin: np.ndarray, scale: np.ndarray, k: int):
    """Brute-force search using int8 quantized vectors (reconstruct then dot)."""
    # Reconstruct
    reconstructed = quantized.astype(np.float32) * scale[:, None] + vmin[:, None]
    scores = reconstructed @ query
    top_k = np.argpartition(-scores, k)[:k]
    top_k = top_k[np.argsort(-scores[top_k])]
    return top_k


def int4_quantize(vectors: np.ndarray):
    """Int4 quantization (same approach as ruvector-core Int4Quantized).
    4-bit per value, 16 levels, min-max scaling. 8x compression."""
    vmin = vectors.min(axis=1, keepdims=True)
    vmax = vectors.max(axis=1, keepdims=True)
    scale = (vmax - vmin) / 15.0
    scale[scale == 0] = 1.0
    quantized = np.round((vectors - vmin) / scale).clip(0, 15).astype(np.uint8)
    return quantized, vmin.squeeze(), scale.squeeze()


def binary_quantize(vectors: np.ndarray):
    """Binary quantization (same approach as ruvector-core BinaryQuantized).
    1-bit per value: positive -> 1, non-positive -> -1. 32x compression."""
    return (vectors > 0).astype(np.float32) * 2.0 - 1.0


def product_quantize_train(vectors: np.ndarray, n_subspaces: int = 8,
                           codebook_size: int = 256, n_iter: int = 20):
    """Train product quantization codebooks (simplified k-means per subspace).
    Matches ruvector-core ProductQuantized approach."""
    d = vectors.shape[1]
    sub_d = d // n_subspaces
    # Cap codebook size to number of vectors
    codebook_size = min(codebook_size, vectors.shape[0])
    codebooks = []
    for s in range(n_subspaces):
        sub_vecs = vectors[:, s * sub_d : (s + 1) * sub_d]
        # Simple k-means: random init + Lloyd iterations
        rng = np.random.RandomState(42 + s)
        indices = rng.choice(sub_vecs.shape[0], codebook_size, replace=False)
        centroids = sub_vecs[indices].copy()
        for _ in range(n_iter):
            # Assign
            dists = np.sum((sub_vecs[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assignments = dists.argmin(axis=1)
            # Update
            for c in range(codebook_size):
                mask = assignments == c
                if mask.any():
                    centroids[c] = sub_vecs[mask].mean(axis=0)
        codebooks.append(centroids)
    return codebooks, n_subspaces, sub_d


def product_quantize_encode(vectors: np.ndarray, codebooks, n_subspaces, sub_d):
    """Encode vectors using trained PQ codebooks."""
    codes = np.zeros((vectors.shape[0], n_subspaces), dtype=np.uint8)
    for s in range(n_subspaces):
        sub_vecs = vectors[:, s * sub_d : (s + 1) * sub_d]
        dists = np.sum((sub_vecs[:, None, :] - codebooks[s][None, :, :]) ** 2, axis=2)
        codes[:, s] = dists.argmin(axis=1).astype(np.uint8)
    return codes


def product_quantize_reconstruct(codes, codebooks, n_subspaces, sub_d):
    """Reconstruct vectors from PQ codes."""
    n = codes.shape[0]
    d = n_subspaces * sub_d
    result = np.zeros((n, d), dtype=np.float32)
    for s in range(n_subspaces):
        result[:, s * sub_d : (s + 1) * sub_d] = codebooks[s][codes[:, s]]
    return result


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

N_TRIALS = 3  # Number of independent trials for variance estimation


@dataclass
class BenchmarkResult:
    method: str
    dataset: str
    n_vectors: int
    dimensions: int
    bits_per_dim: float
    compression_ratio: float
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    memory_mb: float
    recall_at_1_std: float = 0.0
    recall_at_10_std: float = 0.0
    recall_at_100_std: float = 0.0
    latency_p50_std: float = 0.0
    n_trials: int = 1
    notes: str = ""


def run_search_trial(search_vectors: np.ndarray, queries: np.ndarray,
                     ground_truth: np.ndarray):
    """Run a single search trial: brute-force matmul, return retrieved indices and latencies."""
    latencies = []
    all_retrieved = np.zeros((queries.shape[0], 100), dtype=np.int32)

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        scores = search_vectors @ q
        top_100 = np.argpartition(-scores, 100)[:100]
        top_100 = top_100[np.argsort(-scores[top_100])]
        latencies.append(time.perf_counter() - t0)
        all_retrieved[i] = top_100

    latencies_ms = np.array(latencies) * 1000
    r1 = compute_recall(all_retrieved, ground_truth, 1)
    r10 = compute_recall(all_retrieved, ground_truth, 10)
    r100 = compute_recall(all_retrieved, ground_truth, 100)
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    return r1, r10, r100, p50, p95, p99


def aggregate_trials(trials):
    """Aggregate trial results into mean/std."""
    r1s = [t[0] for t in trials]
    r10s = [t[1] for t in trials]
    r100s = [t[2] for t in trials]
    p50s = [t[3] for t in trials]
    p95s = [t[4] for t in trials]
    return {
        "recall_at_1": float(np.mean(r1s)),
        "recall_at_10": float(np.mean(r10s)),
        "recall_at_100": float(np.mean(r100s)),
        "latency_p50_ms": float(np.mean(p50s)),
        "latency_p95_ms": float(np.mean(p95s)),
        "latency_p99_ms": float(np.mean([t[5] for t in trials])),
        "recall_at_1_std": float(np.std(r1s)),
        "recall_at_10_std": float(np.std(r10s)),
        "recall_at_100_std": float(np.std(r100s)),
        "latency_p50_std": float(np.std(p50s)),
        "n_trials": len(trials),
    }


def compute_ground_truth_ip(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Compute exact k-NN using inner product (brute force)."""
    gt = np.zeros((queries.shape[0], k), dtype=np.int32)
    for i, q in enumerate(queries):
        scores = base @ q
        top_k = np.argpartition(-scores, k)[:k]
        top_k = top_k[np.argsort(-scores[top_k])]
        gt[i] = top_k
    return gt


def compute_recall(retrieved: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    total = 0
    for i in range(len(retrieved)):
        gt_set = set(ground_truth[i, :k].tolist())
        ret_set = set(retrieved[i, :k].tolist())
        total += len(gt_set & ret_set) / k
    return total / len(retrieved)


def benchmark_baseline(base: np.ndarray, queries: np.ndarray,
                       ground_truth: np.ndarray, dataset_name: str) -> BenchmarkResult:
    """Benchmark f32 brute-force (should give perfect recall)."""
    trials = [run_search_trial(base, queries, ground_truth) for _ in range(N_TRIALS)]
    agg = aggregate_trials(trials)

    return BenchmarkResult(
        method="f32_baseline",
        dataset=dataset_name,
        n_vectors=base.shape[0],
        dimensions=base.shape[1],
        bits_per_dim=32.0,
        compression_ratio=1.0,
        memory_mb=base.nbytes / 1e6,
        **agg,
    )


def benchmark_scalar_int8(base: np.ndarray, queries: np.ndarray,
                          ground_truth: np.ndarray, dataset_name: str) -> BenchmarkResult:
    """Benchmark scalar int8 quantization (ruvector-core ScalarQuantized equivalent)."""
    quantized, vmin, scale = scalar_int8_quantize(base)

    # Pre-reconstruct outside timing loop (fair comparison with TurboQuant MSE)
    reconstructed = quantized.astype(np.float32) * scale[:, None] + vmin[:, None]

    trials = [run_search_trial(reconstructed, queries, ground_truth) for _ in range(N_TRIALS)]
    agg = aggregate_trials(trials)
    mem = quantized.nbytes + vmin.nbytes + scale.nbytes

    return BenchmarkResult(
        method="scalar_int8",
        dataset=dataset_name,
        n_vectors=base.shape[0],
        dimensions=base.shape[1],
        bits_per_dim=8.0,
        compression_ratio=32.0 / 8.0,
        memory_mb=mem / 1e6,
        **agg,
    )


def benchmark_int4(base: np.ndarray, queries: np.ndarray,
                   ground_truth: np.ndarray, dataset_name: str) -> BenchmarkResult:
    """Benchmark Int4 quantization (ruvector-core Int4Quantized equivalent)."""
    quantized, vmin, scale = int4_quantize(base)

    # Pre-reconstruct outside timing loop (fair comparison with TurboQuant MSE)
    reconstructed = quantized.astype(np.float32) * scale[:, None] + vmin[:, None]

    trials = [run_search_trial(reconstructed, queries, ground_truth) for _ in range(N_TRIALS)]
    agg = aggregate_trials(trials)
    mem = quantized.nbytes + vmin.nbytes + scale.nbytes

    return BenchmarkResult(
        method="int4",
        dataset=dataset_name,
        n_vectors=base.shape[0],
        dimensions=base.shape[1],
        bits_per_dim=4.0,
        compression_ratio=8.0,
        memory_mb=mem / 1e6,
        notes="Min-max 4-bit scalar, 16 levels. Same as ruvector-core Int4Quantized.",
        **agg,
    )


def benchmark_binary(base: np.ndarray, queries: np.ndarray,
                     ground_truth: np.ndarray, dataset_name: str) -> BenchmarkResult:
    """Benchmark binary quantization (ruvector-core BinaryQuantized equivalent)."""
    binary_base = binary_quantize(base)

    trials = [run_search_trial(binary_base, queries, ground_truth) for _ in range(N_TRIALS)]
    agg = aggregate_trials(trials)
    # 1 bit per dim, but stored as float32 for matmul. True compressed would be 32x.
    mem = base.shape[0] * base.shape[1] / 8  # True compressed size

    return BenchmarkResult(
        method="binary",
        dataset=dataset_name,
        n_vectors=base.shape[0],
        dimensions=base.shape[1],
        bits_per_dim=1.0,
        compression_ratio=32.0,
        memory_mb=mem / 1e6,
        notes="Sign-bit quantization (>0 -> +1, <=0 -> -1). Same as ruvector-core BinaryQuantized.",
        **agg,
    )


def benchmark_product_quantization(base: np.ndarray, queries: np.ndarray,
                                    ground_truth: np.ndarray, dataset_name: str,
                                    n_subspaces: int = 8) -> BenchmarkResult:
    """Benchmark product quantization (ruvector-core ProductQuantized equivalent)."""
    d = base.shape[1]
    sub_d = d // n_subspaces
    # Trim dimensions to be divisible by n_subspaces
    effective_d = n_subspaces * sub_d
    base_trimmed = base[:, :effective_d]
    queries_trimmed = queries[:, :effective_d]

    # Multi-trial: retrain with different seeds to capture recall variance
    trials = []
    for trial in range(N_TRIALS):
        codebooks, ns, sd = product_quantize_train(base_trimmed, n_subspaces)
        codes = product_quantize_encode(base_trimmed, codebooks, ns, sd)
        reconstructed = product_quantize_reconstruct(codes, codebooks, ns, sd)
        trials.append(run_search_trial(reconstructed, queries_trimmed, ground_truth))

    agg = aggregate_trials(trials)

    # PQ storage: n_subspaces bytes per vector (uint8 codes) + codebook overhead
    codebook_size = min(256, base.shape[0])
    mem_codes = base.shape[0] * n_subspaces  # uint8 codes
    mem_codebooks = n_subspaces * codebook_size * sub_d * 4  # float32 centroids
    mem = mem_codes + mem_codebooks
    bits_per_dim = (n_subspaces * 8) / effective_d  # 8 bits per subspace code

    return BenchmarkResult(
        method=f"product_quant_{n_subspaces}sub",
        dataset=dataset_name,
        n_vectors=base.shape[0],
        dimensions=effective_d,
        bits_per_dim=bits_per_dim,
        compression_ratio=32.0 / bits_per_dim,
        memory_mb=mem / 1e6,
        notes=f"K-means codebooks, {n_subspaces} subspaces, {codebook_size} centroids each. Same as ruvector-core ProductQuantized.",
        **agg,
    )


def benchmark_turboquant_mse(base: np.ndarray, queries: np.ndarray,
                              ground_truth: np.ndarray, dataset_name: str,
                              bits: int = 3) -> BenchmarkResult:
    """Benchmark TurboQuant MSE-only (Stage 1: Hadamard + Lloyd-Max scalar)."""
    d = base.shape[1]
    device = "cpu"
    base_t = torch.from_numpy(base).to(device)

    # Multi-trial: different random rotation matrices to capture recall variance
    trials = []
    for trial in range(N_TRIALS):
        quantizer = TurboQuantMSE(d, bits, seed=42 + trial, device=device)
        with torch.no_grad():
            base_hat, _ = quantizer(base_t)
        base_hat_np = base_hat.numpy()
        trials.append(run_search_trial(base_hat_np, queries, ground_truth))

    agg = aggregate_trials(trials)

    # Memory: bits per dim for indices + negligible codebook
    storage_bits = base.shape[0] * d * bits
    mem = storage_bits / 8

    return BenchmarkResult(
        method=f"turboquant_mse_{bits}bit",
        dataset=dataset_name,
        n_vectors=base.shape[0],
        dimensions=d,
        bits_per_dim=float(bits),
        compression_ratio=32.0 / bits,
        memory_mb=mem / 1e6,
        notes="MSE-only (no QJL correction). Searches on reconstructed vectors.",
        **agg,
    )


def _run_turboquant_full_trial(quantizer, base_t, queries_t, ground_truth, n_base):
    """Run a single TurboQuant full trial (custom inner product estimator)."""
    with torch.no_grad():
        compressed = quantizer.quantize(base_t)

    latencies = []
    all_retrieved = np.zeros((queries_t.shape[0], 100), dtype=np.int32)

    for i in range(queries_t.shape[0]):
        q = queries_t[i].unsqueeze(0).expand(n_base, -1)
        t0 = time.perf_counter()
        scores = quantizer.inner_product(q, compressed).numpy()
        top_100 = np.argpartition(-scores, 100)[:100]
        top_100 = top_100[np.argsort(-scores[top_100])]
        latencies.append(time.perf_counter() - t0)
        all_retrieved[i] = top_100

    latencies_ms = np.array(latencies) * 1000
    r1 = compute_recall(all_retrieved, ground_truth, 1)
    r10 = compute_recall(all_retrieved, ground_truth, 10)
    r100 = compute_recall(all_retrieved, ground_truth, 100)
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    return r1, r10, r100, p50, p95, p99


def benchmark_turboquant_full(base: np.ndarray, queries: np.ndarray,
                               ground_truth: np.ndarray, dataset_name: str,
                               bits: int = 3) -> BenchmarkResult:
    """Benchmark TurboQuant full two-stage (MSE + QJL) with unbiased inner product."""
    d = base.shape[1]
    device = "cpu"
    base_t = torch.from_numpy(base).to(device)
    queries_t = torch.from_numpy(queries).to(device)

    # Multi-trial: different random seeds for rotation + QJL matrices
    trials = []
    for trial in range(N_TRIALS):
        quantizer = TurboQuantProd(d, bits, seed=42 + trial, device=device)
        trials.append(_run_turboquant_full_trial(
            quantizer, base_t, queries_t, ground_truth, base.shape[0]))

    agg = aggregate_trials(trials)

    # Memory: (bits-1) per dim for MSE indices + 1 bit per dim for QJL signs + 16 bits per vector for norm
    storage_bits = base.shape[0] * (d * bits + 16)
    mem = storage_bits / 8

    return BenchmarkResult(
        method=f"turboquant_full_{bits}bit",
        dataset=dataset_name,
        n_vectors=base.shape[0],
        dimensions=d,
        bits_per_dim=float(bits),
        compression_ratio=32.0 / bits,
        memory_mb=mem / 1e6,
        notes="Full two-stage (MSE + QJL). Uses unbiased inner product estimator.",
        **agg,
    )


# ---------------------------------------------------------------------------
# Dataset Configurations
# ---------------------------------------------------------------------------

DATA_DIR = Path("/Volumes/black box/data/ann-benchmarks")

DATASETS = {
    "sift1m": {
        "base": DATA_DIR / "sift" / "sift_base.fvecs",
        "query": DATA_DIR / "sift" / "sift_query.fvecs",
        "gt": None,  # Recompute GT on normalized vectors (provided GT uses L2 on raw)
        "loader": "fvecs",
        "max_base": 100_000,  # Use 100K subset for tractable benchmarking
        "max_query": 1_000,
        "normalize": True,
    },
    "glove200": {
        "base": DATA_DIR / "glove.6B.200d.txt",
        "loader": "glove",
        "max_base": 100_000,
        "max_query": 1_000,
        "normalize": True,
    },
    "pkm384": {
        "base": DATA_DIR / "pkm-embeddings-384d.npy",
        "loader": "npy",
        "max_base": 0,  # Use all (137 vectors)
        "max_query": 20,
        "normalize": False,  # Already unit-normalized
    },
}


def load_dataset(name: str):
    """Load a dataset and return (base, queries, ground_truth)."""
    cfg = DATASETS[name]
    print(f"\nLoading {name}...")

    if cfg["loader"] == "fvecs":
        base = load_fvecs(str(cfg["base"]))
        queries = load_fvecs(str(cfg["query"]))
        gt = load_ivecs(str(cfg["gt"])) if cfg.get("gt") else None
    elif cfg["loader"] == "glove":
        all_vectors = load_glove(str(cfg["base"]),
                                 max_vectors=cfg["max_base"] + cfg["max_query"])
        # Split into base and query
        base = all_vectors[: -cfg["max_query"]]
        queries = all_vectors[-cfg["max_query"] :]
        gt = None  # Compute ourselves
    elif cfg["loader"] == "npy":
        base = load_npy(str(cfg["base"]))
        # Use random subset as queries, remainder as base
        n_query = cfg["max_query"]
        rng = np.random.RandomState(42)
        indices = rng.permutation(base.shape[0])
        queries = base[indices[:n_query]]
        base = base[indices[n_query:]]
        gt = None
    else:
        raise ValueError(f"Unknown loader: {cfg['loader']}")

    # Subset base vectors if needed
    if cfg["max_base"] and base.shape[0] > cfg["max_base"]:
        base = base[: cfg["max_base"]]

    # Subset queries if needed
    if cfg["max_query"] and queries.shape[0] > cfg["max_query"]:
        queries = queries[: cfg["max_query"]]

    # Normalize if needed
    if cfg.get("normalize", False):
        base = normalize_vectors(base)
        queries = normalize_vectors(queries)

    # Compute ground truth if not provided
    if gt is None:
        print(f"  Computing ground truth (brute force, {base.shape[0]} x {queries.shape[0]})...")
        gt = compute_ground_truth_ip(base, queries, 100)
    else:
        gt = gt[: queries.shape[0]]

    print(f"  Base: {base.shape}, Queries: {queries.shape}, GT: {gt.shape}")
    return base, queries, gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def format_results_table(results: list[BenchmarkResult]) -> str:
    """Format results as markdown table with variance when available."""
    lines = [
        "| Dataset | Method | Dims | N | Bits/dim | Compress | R@1 | R@10 | R@100 | p50 ms | Trials | Memory MB |",
        "|---------|--------|------|---|----------|----------|-----|------|-------|--------|--------|-----------|",
    ]
    for r in results:
        # Show std only when non-zero (stochastic methods)
        def fmt_recall(mean, std):
            if std > 0.001:
                return f"{mean:.3f}±{std:.3f}"
            return f"{mean:.3f}"

        def fmt_latency(mean, std):
            if std > 0.01:
                return f"{mean:.2f}±{std:.2f}"
            return f"{mean:.2f}"

        lines.append(
            f"| {r.dataset} | {r.method} | {r.dimensions} | "
            f"{r.n_vectors:,} | {r.bits_per_dim:.1f} | {r.compression_ratio:.1f}x | "
            f"{fmt_recall(r.recall_at_1, r.recall_at_1_std)} | "
            f"{fmt_recall(r.recall_at_10, r.recall_at_10_std)} | "
            f"{fmt_recall(r.recall_at_100, r.recall_at_100_std)} | "
            f"{fmt_latency(r.latency_p50_ms, r.latency_p50_std)} | "
            f"{r.n_trials} | {r.memory_mb:.1f} |"
        )
    return "\n".join(lines)


def main():
    results = []
    datasets_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    for dataset_name in datasets_to_run:
        if dataset_name not in DATASETS:
            print(f"Unknown dataset: {dataset_name}, skipping")
            continue

        base, queries, gt = load_dataset(dataset_name)

        # Skip datasets too small for recall@100
        k_max = min(100, base.shape[0] - 1)
        if k_max < 100:
            print(f"  Dataset too small for R@100 ({base.shape[0]} vectors). Using R@{k_max}.")

        print(f"\n--- Benchmarking {dataset_name} ({base.shape[0]} vectors, d={base.shape[1]}) ---")

        # --- ruvector-core methods (existing implementations) ---
        # 1. Baseline
        print("  [1/8] f32 baseline...")
        results.append(benchmark_baseline(base, queries, gt, dataset_name))

        # 2. Scalar int8 (ruvector-core ScalarQuantized)
        print("  [2/8] Scalar int8 (ScalarQuantized)...")
        results.append(benchmark_scalar_int8(base, queries, gt, dataset_name))

        # 3. Int4 (ruvector-core Int4Quantized)
        print("  [3/8] Int4 (Int4Quantized)...")
        results.append(benchmark_int4(base, queries, gt, dataset_name))

        # 4. Binary (ruvector-core BinaryQuantized)
        print("  [4/8] Binary (BinaryQuantized)...")
        results.append(benchmark_binary(base, queries, gt, dataset_name))

        # 5. Product Quantization (ruvector-core ProductQuantized)
        n_sub = min(8, base.shape[1] // 4)  # Ensure sub_d >= 4
        print(f"  [5/8] Product Quantization ({n_sub} subspaces)...")
        results.append(benchmark_product_quantization(base, queries, gt, dataset_name, n_subspaces=n_sub))

        # --- TurboQuant methods (ruvllm, not integrated with HNSW) ---
        # 6. TurboQuant MSE 3-bit
        print("  [6/8] TurboQuant MSE 3-bit...")
        results.append(benchmark_turboquant_mse(base, queries, gt, dataset_name, bits=3))

        # 7. TurboQuant MSE 4-bit
        print("  [7/8] TurboQuant MSE 4-bit...")
        results.append(benchmark_turboquant_mse(base, queries, gt, dataset_name, bits=4))

        # 8. TurboQuant full 3-bit (MSE + QJL)
        print("  [8/8] TurboQuant full 3-bit (MSE + QJL)...")
        results.append(benchmark_turboquant_full(base, queries, gt, dataset_name, bits=3))

    # Output results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    table = format_results_table(results)
    print(table)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "benchmark_results.md", "w") as f:
        f.write("# TurboQuant Vector Search Benchmark Results\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Platform: {sys.platform}\n")
        f.write(f"Python: {sys.version.split()[0]}\n")
        f.write(f"PyTorch: {torch.__version__}\n\n")
        f.write("## Results\n\n")
        f.write(table)
        f.write("\n\n## Notes\n\n")
        f.write("- All searches are brute-force (no HNSW acceleration) to isolate quantization quality.\n")
        f.write("- Vectors are L2-normalized before quantization (inner product = cosine similarity).\n")
        f.write("- All methods pre-reconstruct to f32 before search (fair latency comparison).\n")
        f.write("- TurboQuant full uses the unbiased inner product estimator from the paper.\n")
        f.write("- Ground truth computed with exact f32 inner product.\n")
        f.write(f"- Each configuration run {N_TRIALS}x with different seeds (stochastic methods) or repeated (deterministic).\n")
        f.write("- ±values show standard deviation across trials.\n")

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
