"""Download public SNAP graphs, record provenance, and freeze deterministic workloads.

Directions, duplicate edges and self loops are removed explicitly. Capacities are
unit weights, not email frequencies. LCC and 2-core views prevent disconnected
zero cuts and leaf-only wins from concealing expensive solver behavior.
"""
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
for name, role in [('email-Eu-core', 'development'), ('ca-GrQc', 'development'), ('facebook_combined', 'holdout'), ('ca-HepPh', 'holdout')]:
    url = f'https://snap.stanford.edu/data/{name}.txt.gz'
    raw = urllib.request.urlopen(url, timeout=60).read()
    edges = set()
    for line in gzip.decompress(raw).decode().splitlines():
        if not line or line.startswith('#'): continue
        u, v = map(int, line.split()[:2])
        if u != v: edges.add(tuple(sorted((u, v))))
    ids = sorted({v for edge in edges for v in edge})
    index = {v: i for i, v in enumerate(ids)}
    graph = ig.Graph(n=len(ids), edges=sorted((index[u], index[v]) for u, v in edges))
    graph.vs['original_id'] = ids
    components = graph.connected_components()
    largest = sorted(components, key=lambda vs: (-len(vs), min(vs)))[0]
    graph = graph.induced_subgraph(largest)
    for view in ['lcc', 'core2']:
        g = graph if view == 'lcc' else graph.induced_subgraph([i for i, k in enumerate(graph.coreness()) if k >= 2])
        if not g.is_connected():
            g = g.induced_subgraph(sorted(g.connected_components(), key=lambda vs: (-len(vs), min(vs)))[0])
        pairs = sorted(tuple(sorted(e)) for e in g.get_edgelist())
        filename = f'{name}-{view}.edges'
        content = f'{g.vcount()} {len(pairs)}\n' + ''.join(f'{u} {v} 1\n' for u, v in pairs)
        (out / filename).write_text(content)
        cut = g.mincut()
        record = dict(name=f'{name}-{view}', role=role, source=url,
            source_sha256=hashlib.sha256(raw).hexdigest(), file=filename,
            normalized_sha256=hashlib.sha256(content.encode()).hexdigest(),
            vertices=g.vcount(), edges=len(pairs), minimum_degree=min(g.degree()),
            bridges=len(g.bridges()), oracle_cut=cut.value,
            transformation='simple undirected unit-capacity largest component' + (' then largest component of 2-core' if view == 'core2' else ''))
        manifest.append(record)
        print(json.dumps(record), flush=True)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
