"""Frozen public SNAP workloads with independent igraph minimum-cut oracles.

Drop loops, collapse duplicate/directed pairs, then select the largest connected
component. Core views remove easy peripheral cuts. Email reciprocity weights
count observed directions (1 or 2), not message frequency. Updates are controlled
perturbations of real edges, not a historical traffic trace.
"""
from collections import Counter
import gzip
import hashlib
import json
import pathlib
import sys
import urllib.request
import igraph as ig

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
manifest = []
sources = [
    ('email-Eu-core', 'development', '4b47acdb80197b085fe63c819c357ae488131ee904ed93d1b219a68b0f9e245f'),
    ('ca-GrQc', 'development', 'a254442cdf5d684712578b630c2e0d7543518ab154ef2341cabb607572ce7230'),
    ('facebook_combined', 'holdout', '125e84db872eeba443d270c70315c256b0af43a502fcfe51f50621166ad035d7'),
    ('ca-HepPh', 'holdout', '8d679f64ea507834613f4c09e6f692cc8a2405d5a01bfe1ee3b24fbbea1d807f'),
]

def largest_component(graph):
    vertices = sorted(graph.connected_components(), key=lambda vs: (-len(vs), min(vs)))[0]
    return graph.induced_subgraph(vertices)

for name, role, expected_hash in sources:
    url = f'https://snap.stanford.edu/data/{name}.txt.gz'
    cache = out / f'{name}.txt.gz'
    if not cache.exists():
        cache.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    raw = cache.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise RuntimeError(f'Source checksum changed: {name}')
    directed = set()
    for line in gzip.decompress(raw).decode().splitlines():
        if not line or line.startswith('#'):
            continue
        u, v = map(int, line.split()[:2])
        if u != v:
            directed.add((u, v))
    directions = Counter(tuple(sorted(edge)) for edge in directed)
    ids = sorted({v for edge in directions for v in edge})
    index = {v: i for i, v in enumerate(ids)}
    graph = ig.Graph(n=len(ids), edges=sorted((index[u], index[v]) for u, v in directions))
    graph.vs['original_id'] = ids
    graph = largest_component(graph)
    views = [('lcc', 0, False), ('core2', 2, False)]
    if name in ['email-Eu-core', 'facebook_combined']:
        views.append(('core4', 4, False))
    if name == 'email-Eu-core':
        views.append(('core4-reciprocity', 4, True))
    for view, k, weighted in views:
        g = graph if not k else largest_component(graph.induced_subgraph([i for i, degree in enumerate(graph.coreness()) if degree >= k]))
        def capacity(u, v):
            original = tuple(sorted((g.vs[u]['original_id'], g.vs[v]['original_id'])))
            return directions[original] if weighted else 1
        capacities = [capacity(u, v) for u, v in g.get_edgelist()]
        edges = sorted((min(u, v), max(u, v), w) for (u, v), w in zip(g.get_edgelist(), capacities))
        filename = f'{name}-{view}.edges'
        content = f'{g.vcount()} {len(edges)}\n' + ''.join(f'{u} {v} {w}\n' for u, v, w in edges)
        (out / filename).write_text(content)
        cut = g.mincut(capacity=capacities)
        bridges = g.bridges()
        record = dict(name=f'{name}-{view}', role=role, source=url,
            source_sha256=expected_hash, file=filename,
            normalized_sha256=hashlib.sha256(content.encode()).hexdigest(),
            vertices=g.vcount(), edges=len(edges), minimum_degree=min(g.degree()),
            bridges=len(bridges), oracle_cut=cut.value,
            transformation='simple undirected largest component' + (f' then largest component of {k}-core' if k else ''),
            capacity='observed directions per pair' if weighted else 'unit', trace=[])
        if view == 'lcc' and name in ['email-Eu-core', 'facebook_combined']:
            selected = sorted(set(([bridges[0]] if bridges else []) + [len(edges)//3, 2*len(edges)//3]))
            for eid in selected:
                u, v = g.es[eid].tuple
                changed = g.copy()
                changed.delete_edges([eid])
                record['trace'].append(dict(u=u, v=v, weight=1, after_delete=changed.mincut().value))
        manifest.append(record)
        print(json.dumps(record), flush=True)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
