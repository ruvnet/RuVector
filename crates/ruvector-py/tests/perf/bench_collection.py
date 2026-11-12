"""Micro-benchmark for Collection insert / insert_batch / search / save / load.

Not collected by pytest (no ``test_`` prefix). Run on a quiet machine:

    python tests/perf/bench_collection.py --n 50000 --dim 256 --repeat 5

Each figure is the median of ``--repeat`` runs; the spread (min..max) is printed
beside it. Data is seeded, so runs are comparable across commits on one machine.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from ruvector.collection import Collection


def timed(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def summarize(samples: List[float]) -> str:
    return f"{statistics.median(samples) * 1000:9.2f} ms  [{min(samples) * 1000:.2f}..{max(samples) * 1000:.2f}]"


def bench_backend(backend: str, n: int, dim: int, repeat: int, with_meta: bool) -> Dict[str, List[float]]:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((200, dim)).astype(np.float32)
    metas = [{"i": i, "tag": f"t{i % 50}", "txt": "x" * 40} for i in range(n)] if with_meta else None
    n_single = min(2000, n)
    out: Dict[str, List[float]] = {k: [] for k in ("insert_x2000", "insert_batch", "from_vectors", "search_x200", "save", "load")}
    for _ in range(repeat):
        c = Collection.create(dim=dim, backend=backend)
        if backend == "rabitq":  # first row builds; time the add() path after it
            c.insert(vecs[0])
        out["insert_x2000"].append(timed(lambda: [c.insert(vecs[i]) for i in range(1, n_single)]))
        c = Collection.create(dim=dim, backend=backend)
        out["insert_batch"].append(timed(lambda: c.insert_batch(vecs, metadatas=metas)))
        out["from_vectors"].append(timed(lambda: Collection.from_vectors(vecs, backend=backend, metadatas=metas)))
        if backend == "rabitq":
            for i in range(0, n, 10):  # 10% tombstones, so search exercises the filter path
                c.delete(i)
        out["search_x200"].append(timed(lambda: [c.search(q, 10) for q in queries]))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.idx"
            out["save"].append(timed(lambda: c.save(p)))
            out["load"].append(timed(lambda: Collection.load(p)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--backends", default="hnsw,rabitq")
    ap.add_argument("--no-metadata", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    results = {}
    for backend in a.backends.split(","):
        results[backend] = bench_backend(backend, a.n, a.dim, a.repeat, not a.no_metadata)
    if a.json:
        print(json.dumps({b: {k: statistics.median(v) for k, v in r.items()} for b, r in results.items()}))
        return
    print(f"n={a.n} dim={a.dim} repeat={a.repeat} metadata={'no' if a.no_metadata else 'yes'}")
    for backend, r in results.items():
        for name, samples in r.items():
            print(f"{backend:7s} {name:14s} {summarize(samples)}")


if __name__ == "__main__":
    main()
