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
expected_sha256 = {
    'email-Eu-core': '4b47acdb80197b085fe63c819c357ae488131ee904ed93d1b219a68b0f9e245f',
    'ca-GrQc': 'a254442cdf5d684712578b630c2e0d7543518ab154ef2341cabb607572ce7230',
    'facebook_combined': '125e84db872eeba443d270c70315c256b0af43a502fcfe51f50621166ad035d7',
    'ca-HepPh': '8d679f64ea507834613f4c09e6f692cc8a2405d5a01bfe1ee3b24fbbea1d807f',
}
for name, role in [('email-Eu-core', 'development'), ('ca-GrQc', 'development'), ('facebook_combined', 'holdout'), ('ca-HepPh', 'holdout')]:
    url = f'https://snap.stanford.edu/data/{name}.txt.gz'
    cache = out / f'{name}.txt.gz'
    if not cache.exists(): cache.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    raw = cache.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256[name]:
        raise RuntimeError(f'Source checksum changed: {name}')
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
    views = ['lcc', 'core2'] + (['core4'] if name in ['email-Eu-core', 'facebook_combined'] else [])
    for view in views:
        g = graph if view == 'lcc' else graph.induced_subgraph([i for i, k in enumerate(graph.coreness()) if k >= int(view[4:])])
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
            transformation='simple undirected unit-capacity largest component' + (f' then largest component of {view[4:]}-core' if view != 'lcc' else ''))
        # Controlled perturbation of observed edges, not a historical event trace.
        record['trace'] = []
        if view == 'lcc' and name in ['email-Eu-core', 'facebook_combined']:
            selected = sorted(set(([g.bridges()[0]] if g.bridges() else []) + [len(pairs)//3, 2*len(pairs)//3]))
            for eid in selected:
                u, v = g.es[eid].tuple
                changed = g.copy()
                changed.delete_edges([eid])
                record['trace'].append(dict(u=u, v=v, weight=1, after_delete=changed.mincut().value))
        manifest.append(record)
        print(json.dumps(record), flush=True)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
