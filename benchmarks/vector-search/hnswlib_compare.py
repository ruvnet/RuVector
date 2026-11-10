"""
hnswlib and exact-search baseline for ruvector-core HNSW on identical data.

The `sift1m-bench` binary in crates/ruvector-sota-bench measures ruvector-core
on TEXMEX fvecs/ivecs files. This script runs hnswlib (C++, via Python) and a
numpy exact search on the same files, with the same M, ef_construction, ef
sweep and k, and prints the same columns, so the two outputs compare row for
row.

Usage:
  pip install hnswlib numpy

  # 1. Write a seeded synthetic dataset (Gaussian, exact L2 top-100 ground
  #    truth) in the file layout sift1m-bench reads. Skip this for real SIFT1M.
  python benchmarks/vector-search/hnswlib_compare.py gen \
      --out bench_data/random-10k --n 10000 --dim 128 --queries 1000

  # 2. ruvector-core
  cargo run --release -p ruvector-sota-bench --bin sift1m-bench -- \
      --data bench_data/random-10k --m 16 --ef-construction 200

  # 3. hnswlib and numpy exact search on the same files
  python benchmarks/vector-search/hnswlib_compare.py run \
      --data bench_data/random-10k --m 16 --ef-construction 200

Both sides build single-threaded and time one query at a time, and both compute
QPS as queries / sum(per-query latency). The hnswlib per-query numbers include
the Python call overhead, so the run also reports batch_QPS: every query in one
single-threaded knn_query call. hnswlib's hand-written distance kernels are
SSE/AVX only (none for NEON), so say which architecture a result came from.
--build-threads lets hnswlib build in parallel; its build time is then no
longer comparable.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

BASE = "sift_base.fvecs"
QUERY = "sift_query.fvecs"
GROUND_TRUTH = "sift_groundtruth.ivecs"
GT_DEPTH = 100


def write_vecs(path, rows, dtype):
    rows = np.ascontiguousarray(rows, dtype=dtype)
    n, d = rows.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = rows.view(np.int32)
    out.tofile(path)


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    d = int(raw[0])
    rows = raw.reshape(-1, d + 1)
    if not np.all(rows[:, 0] == d):
        raise ValueError(f"{path}: inconsistent dimension")
    return np.ascontiguousarray(rows[:, 1:]).view(dtype)


def exact_topk(base, queries, k, chunk=100):
    """Exact L2 top-k for every query, nearest first."""
    base_sq = np.einsum("ij,ij->i", base, base)
    out = np.empty((len(queries), k), dtype=np.int32)
    for start in range(0, len(queries), chunk):
        q = queries[start : start + chunk]
        # ||b||^2 - 2 q.b ranks the same as ||q - b||^2
        dist = base_sq[None, :] - 2.0 * (q @ base.T)
        part = np.argpartition(dist, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(dist, part, axis=1).argsort(axis=1)
        out[start : start + chunk] = np.take_along_axis(part, order, axis=1)
    return out


def recall(ids, gt_row, k):
    return len(set(ids[:k].tolist()) & set(gt_row[:k].tolist())) / k


def summarize(latencies_ns, recalls):
    lat = np.sort(np.asarray(latencies_ns, dtype=np.float64))
    return {
        "recall": float(np.mean(recalls)),
        "qps": len(lat) / (lat.sum() / 1e9),
        "p50_us": float(lat[int(len(lat) * 0.50)] / 1000.0),
        "p99_us": float(lat[int(len(lat) * 0.99)] / 1000.0),
    }


def print_row(label, r):
    batch = f"  {r['batch_qps']:>10.0f}" if "batch_qps" in r else ""
    print(
        f"{label:<8}  {r['recall']:>10.4f}  {r['qps']:>10.0f}  "
        f"{r['p50_us']:>12.1f}  {r['p99_us']:>10.1f}{batch}"
    )


def hnswlib_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("hnswlib")
    except PackageNotFoundError:
        return "unknown"


def cmd_gen(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    base = rng.standard_normal((args.n, args.dim), dtype=np.float32)
    queries = rng.standard_normal((args.queries, args.dim), dtype=np.float32)
    t0 = time.perf_counter()
    gt = exact_topk(base, queries, min(GT_DEPTH, args.n))
    write_vecs(out / BASE, base, np.float32)
    write_vecs(out / QUERY, queries, np.float32)
    write_vecs(out / GROUND_TRUTH, gt, np.int32)
    print(
        f"wrote {args.n} x {args.dim}d base, {args.queries} queries and "
        f"top-{gt.shape[1]} ground truth to {out} ({time.perf_counter() - t0:.1f}s)"
    )


def cmd_run(args):
    import hnswlib

    data = Path(args.data)
    base = read_vecs(data / BASE, np.float32)
    queries = read_vecs(data / QUERY, np.float32)
    gt = read_vecs(data / GROUND_TRUTH, np.int32)
    if args.query_limit:
        queries, gt = queries[: args.query_limit], gt[: args.query_limit]
    n, dim = base.shape
    k = args.k
    header = f"{'ef':<8}  {'recall@' + str(k):>10}  {'QPS':>10}  {'p50_us':>12}  {'p99_us':>10}"

    print(f"=== hnswlib {hnswlib_version()} / numpy {np.__version__} ===")
    print(f"Data       : {n} x {dim}d, {len(queries)} queries")
    print(f"M / efC    : {args.m} / {args.ef_construction}, k={k}")
    print(f"Machine    : {platform.platform()}, Python {sys.version.split()[0]}")
    results = {
        "n": n,
        "dim": dim,
        "queries": len(queries),
        "m": args.m,
        "ef_construction": args.ef_construction,
        "k": k,
    }

    # Exact search one query at a time: the latency floor, and a check that
    # the ground truth file belongs to this data (recall must be 1.0).
    base_sq = np.einsum("ij,ij->i", base, base)
    lat, rec = [], []
    for qi, q in enumerate(queries):
        t = time.perf_counter_ns()
        dist = base_sq - 2.0 * (base @ q)
        part = np.argpartition(dist, k - 1)[:k]
        ids = part[np.argsort(dist[part])]
        lat.append(time.perf_counter_ns() - t)
        rec.append(recall(ids, gt[qi], k))
    exact = summarize(lat, rec)
    results["numpy_exact"] = exact
    print("\nnumpy exact search")
    print(header)
    print_row("exact", exact)

    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n, ef_construction=args.ef_construction, M=args.m)
    index.set_num_threads(args.build_threads)
    t = time.perf_counter()
    index.add_items(base, np.arange(n))
    build_s = time.perf_counter() - t
    index.set_num_threads(1)
    results["hnswlib_build_s"] = build_s
    results["hnswlib_build_threads"] = args.build_threads
    print(
        f"\nhnswlib build: {build_s:.1f}s ({n / build_s:.0f} vec/s, "
        f"{args.build_threads} thread(s))"
    )
    # Per-query timings include the Python call into hnswlib. batch_QPS runs
    # all queries in one single-threaded call, which leaves that overhead out.
    print(header + f"  {'batch_QPS':>10}")
    results["hnswlib"] = {}
    for ef in args.ef:
        index.set_ef(max(ef, k))
        lat, rec = [], []
        for qi, q in enumerate(queries):
            t = time.perf_counter_ns()
            labels, _ = index.knn_query(q, k=k)
            lat.append(time.perf_counter_ns() - t)
            rec.append(recall(labels[0], gt[qi], k))
        r = summarize(lat, rec)
        t = time.perf_counter_ns()
        index.knn_query(queries, k=k, num_threads=1)
        r["batch_qps"] = len(queries) / ((time.perf_counter_ns() - t) / 1e9)
        results["hnswlib"][str(ef)] = r
        print_row(str(ef), r)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="write a seeded synthetic dataset")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=10_000)
    g.add_argument("--dim", type=int, default=128)
    g.add_argument("--queries", type=int, default=1_000)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_gen)

    r = sub.add_parser("run", help="benchmark hnswlib and numpy exact search")
    r.add_argument("--data", required=True)
    r.add_argument("--m", type=int, default=16)
    r.add_argument("--ef-construction", type=int, default=200)
    r.add_argument(
        "--ef",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[20, 40, 80, 100, 200, 400],
    )
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--query-limit", type=int, default=0)
    r.add_argument("--build-threads", type=int, default=1)
    r.add_argument("--json")
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
