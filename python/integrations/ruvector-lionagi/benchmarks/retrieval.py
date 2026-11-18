"""Reproducible synthetic cosine ANN benchmark; no embedding/model/network costs."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
import resource
from lionagi import Node
from ruvector_lionagi import RuVectorStore


def run(n: int, dim: int, queries: int, ef: int) -> dict:
    rng = np.random.default_rng(20261011)
    x = rng.standard_normal((n, dim), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = rng.standard_normal((queries, dim), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    nodes = [Node(content=f'document {i}', embedding=v.tolist()) for i, v in enumerate(x)]
    id_rows = {str(node.id): i for i, node in enumerate(nodes)}
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    store = RuVectorStore(dim, capacity=n, ef_search=ef)
    start = time.perf_counter()
    for offset in range(0, n, 1000):
        store.add_nodes(nodes[offset:offset+1000])
    build_s = time.perf_counter() - start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    for query in q[:10]:
        store.search(query, k=10)
    ann_ms, exact_ms, recalls = [], [], []
    for query in q:
        start = time.perf_counter_ns()
        scores = x @ query
        exact = np.argpartition(-scores, 9)[:10]
        exact = exact[np.argsort(-scores[exact])]
        exact_ms.append((time.perf_counter_ns()-start)/1e6)
        start = time.perf_counter_ns()
        hits = store.search(query, k=10)
        ann_ms.append((time.perf_counter_ns()-start)/1e6)
        recalls.append(len(set(exact.tolist()) & {id_rows[h.node_id] for h in hits})/10)
    def summary(a):
        return {'p50_ms':float(np.percentile(a,50)), 'p95_ms':float(np.percentile(a,95))}
    return {'schema':1,'seed':20261011,'n':n,'dimensions':dim,'queries':queries,'k':10,
            'ef_search':ef,'dataset_sha256':hashlib.sha256(x.tobytes()+q.tobytes()).hexdigest(),
            'build_seconds':build_s,'peak_rss_delta_bytes':rss_after-rss_before,
            'process_peak_rss_bytes':rss_after,'recall_at_10':float(np.mean(recalls)),
            'adapter':summary(ann_ms),'numpy_exact':summary(exact_ms),
            'environment':{'python':platform.python_version(),'platform':platform.platform(),
                           'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
                           'packages':{p:metadata.version(p) for p in ('lionagi','ruvector','numpy','ruvector-lionagi')}},
            'limitations':['Synthetic independent Gaussian vectors, not application relevance.',
                           'One process; high-water RSS includes Python and native allocations, not exclusive index memory.',
                           'Includes Python adapter overhead; excludes embedding, network and LLM costs.',
                           'HNSW graph construction is not fully deterministic; repeat for variance.',
                           'Exact baseline uses one BLAS thread and full exact ranking of ten results.']}

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=10000)
    parser.add_argument('--dimensions',type=int,default=64)
    parser.add_argument('--queries',type=int,default=200)
    parser.add_argument('--ef',type=int,default=100)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 10<=args.n<=100000 or not 10<=args.queries<=2000 or not 2<=args.dimensions<=8192:
        parser.error('benchmark parameters outside bounded range')
    result=run(args.n,args.dimensions,args.queries,args.ef)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
