"""Pinned historical DIMACS roads; normalized binary input and igraph oracles.

These files do not contain present-day OSM access/turn restrictions. Do not use
this benchmark as a road-legality or live-navigation certification.
"""
import argparse
import array
import gzip
import hashlib
import json
import pathlib
import random
import struct
import sys
import time
import urllib.error
import urllib.request
import igraph

SOURCES = {
    'NY': ['7b2446c7ffe6179efbc42af6812448e8cb782f12968ca2bb1d50d979343056d4',
           '02547a628742164e02bd73f59e33a65d23dfac8fc54c09275c8b9169c38eafdf'],
    'BAY': ['630c4a96869b3ecdd631ceb0f63923fb954ee4c7c26d983863612e2c4442462e',
            '67330855e3082dae03609f5b3012fe13ad2d982139d79c3040e77dad55ffd338'],
}

def download(url, limit, attempts=4):
    error = None
    request = urllib.request.Request(url, headers={'User-Agent': 'RuVector-benchmark/1'})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read(limit + 1)
        except (OSError, urllib.error.URLError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f'download failed after {attempts} attempts: {url}') from error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': 20260915, 'oracle': 'igraph ' + igraph.__version__, 'datasets': []}
    for region, hashes in SOURCES.items():
        provenance = []
        for ext, digest in zip(['gr', 'co'], hashes):
            name = f'USA-road-d.{region}.{ext}.gz'
            path = args.output / name
            url = 'https://www.diag.uniroma1.it/challenge9/data/USA-road-d/' + name
            if not path.exists():
                data = download(url, 20_000_000)
                if len(data) > 20_000_000:
                    raise ValueError('Compressed source too large')
                path.write_bytes(data)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Source checksum mismatch: ' + name)
            provenance.append({'url': url, 'sha256': digest})
        edges, weights = [], []
        with gzip.open(args.output / f'USA-road-d.{region}.gr.gz', 'rt') as stream:
            for line in stream:
                row = line.split()
                if not row or row[0] == 'c': continue
                if row[0] == 'p': n, expected_m = map(int, row[2:4])
                if row[0] == 'a':
                    s, t, w = map(int, row[1:4]); edges.append((s-1, t-1)); weights.append(w)
        assert len(edges) == expected_m and n <= 1_000_000
        assert all(0 <= w <= 1_000_000_000 for w in weights)
        coords = [None] * n
        with gzip.open(args.output / f'USA-road-d.{region}.co.gz', 'rt') as stream:
            for line in stream:
                row = line.split()
                if row and row[0] == 'v':
                    node, lon, lat = map(int, row[1:4]); coords[node-1] = (lat/1e6, lon/1e6)
        assert all(c is not None for c in coords)
        # Eight spatially separated landmarks; selection uses no query timings.
        landmarks = [min(range(n), key=lambda i: coords[i])]
        nearest = [float('inf')] * n
        for _ in range(7):
            a = coords[landmarks[-1]]
            for i, b in enumerate(coords):
                nearest[i] = min(nearest[i], (a[0]-b[0])**2 + (a[1]-b[1])**2)
            landmarks.append(max(range(n), key=lambda i: nearest[i]))
        graph = igraph.Graph(n=n, edges=edges, directed=True)
        graph.es['weight'] = weights
        rng = random.Random(20260915)
        queries = []
        for i in range(128):
            s, t = rng.randrange(n), rng.randrange(n)
            if i % 2:
                t = s
                for _ in range(24):
                    neighbors = graph.neighbors(t, mode='out')
                    if not neighbors: break
                    t = rng.choice(neighbors)
            begin = time.perf_counter_ns()
            cost = graph.distances(source=[s], target=[t], weights='weight')[0][0]
            elapsed = time.perf_counter_ns() - begin
            queries.append((s, t, -1 if cost == float('inf') else int(cost), elapsed))
        traces = []
        for s,t,cost,_ in queries[:8]:
            if cost < 0 or s == t: continue
            route = graph.get_shortest_paths(s, to=t, weights='weight', output='epath')[0]
            if not route: continue
            arc = route[len(route)//2]
            graph.es[arc]['weight'] = float('inf')
            closed = graph.distances(source=[s], target=[t], weights='weight')[0][0]
            graph.es[arc]['weight'] = weights[arc]
            traces.append((s,t,arc,-1 if closed == float('inf') else int(closed),cost))
        path = args.output / f'{region}.roads'
        with path.open('wb') as out:
            out.write(struct.pack('<III', n, len(edges), len(queries)))
            flat = array.array('I', (x for (s,t),w in zip(edges,weights) for x in (s,t,w)))
            if sys.byteorder != 'little': flat.byteswap()
            flat.tofile(out)
            flat = array.array('d', (x for c in coords for x in c))
            if sys.byteorder != 'little': flat.byteswap()
            flat.tofile(out)
            out.write(struct.pack('<I', len(landmarks)))
            for v in landmarks: out.write(struct.pack('<I', v))
            for s,t,cost,_ in queries: out.write(struct.pack('<IIq',s,t,cost))
            out.write(struct.pack('<I',len(traces)))
            for s,t,arc,closed,original in traces: out.write(struct.pack('<IIIqq',s,t,arc,closed,original))
        item = {'name': region, 'role': 'development' if region == 'NY' else 'holdout',
                'nodes': n, 'arcs': len(edges), 'sources': provenance,
                'normalized_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'landmarks': landmarks, 'queries': queries, 'closure_traces': traces,
                'query_mix': '64 uniform endpoints, 64 endpoints of 24-step directed random walks'}
        manifest['datasets'].append(item)
        print(json.dumps({k:v for k,v in item.items() if k not in ('queries','sources')}), flush=True)
    (args.output / 'roads-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

if __name__ == '__main__': main()
