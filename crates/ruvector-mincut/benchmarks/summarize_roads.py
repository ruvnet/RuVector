"""Summarize paired kernel samples without hiding local-query regressions."""
import json
import math
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
summary = {}
for name in ['NY', 'BAY']:
    rows = [json.loads(line) for line in (root / f'{name}-results.jsonl').read_text().splitlines()]
    queries = [r for r in rows if r['type'] == 'query']
    assert len(queries) == 128 * 3 * 2
    entry = {'build': rows[0], 'snap': rows[-1]}
    for parity, label in [(0, 'uniform'), (1, 'local')]:
        result = {}
        for alt, algorithm in [(False, 'dijkstra'), (True, 'alt')]:
            samples = sorted(r['ns']/1e6 for r in queries if r['query'] % 2 == parity and r['alt'] == alt)
            result[algorithm] = {'median_ms': statistics.median(samples),
                                 'p95_ms': samples[math.ceil(.95*len(samples))-1],
                                 'samples': len(samples)}
        ratios = []
        for q in range(parity,128,2):
            a = statistics.median(r['ns'] for r in queries if r['query']==q and r['alt'])
            b = statistics.median(r['ns'] for r in queries if r['query']==q and not r['alt'])
            ratios.append(b/a)
        result['paired_geomean_speedup'] = math.exp(statistics.mean(map(math.log,ratios)))
        result['queries_over_10_percent_slower'] = sum(r < 1/1.1 for r in ratios)
        entry[label] = result
    summary[name] = entry
(root/'routing-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary,indent=2))
