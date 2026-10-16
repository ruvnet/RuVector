#!/usr/bin/env python3
"""Compare two trusted agent_memory_bench binaries with alternating execution order."""
import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--before', type=Path, required=True)
parser.add_argument('--after', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--pairs', type=int, default=3)
args = parser.parse_args()
if not 3 <= args.pairs <= 20:
    parser.error('pairs must be in 3..20')
if args.output.exists():
    parser.error('output must be a new path')
paths = {name: path.resolve(strict=True) for name, path in [('before', args.before), ('after', args.after)]}
blocks = []
for pair in range(args.pairs):
    order = ['before', 'after'] if pair % 2 == 0 else ['after', 'before']
    block = {'pair': pair, 'order': order, 'runs': {}}
    for name in order:
        result = subprocess.run([str(paths[name])], check=True, capture_output=True, text=True, timeout=120)
        block['runs'][name] = json.loads(result.stdout)
    blocks.append(block)

summary = []
for index, size in enumerate([128, 1024, 4096]):
    invariant = None
    for block in blocks:
        for run in block['runs'].values():
            row = run['results'][index]
            assert row['nodes'] == size and row['equal_statuses'] and row['exact_replay']
            actual = [row[k] for k in ['receipt_hash', 'runtime_work', 'audit_execution_work', 'audit_status_changes']]
            if invariant is None:
                invariant = actual
            assert actual == invariant, f'Behavior or deterministic work changed at size {size}'
    metrics = {}
    paths_to_compare = ['direct_status_only', 'full_runtime', 'audit_four_executions']
    if all('prepared_runtime' in b['runs']['before']['results'][index] for b in blocks):
        paths_to_compare += ['prepared_runtime', 'preparation']
    for metric in paths_to_compare:
        old = statistics.median(b['runs']['before']['results'][index][metric]['median_ns'] for b in blocks)
        new = statistics.median(b['runs']['after']['results'][index][metric]['median_ns'] for b in blocks)
        metrics[metric] = {'before_median_ns': old, 'after_median_ns': new, 'reduction_percent': 100 * (old - new) / old}
    row = {'nodes': size, 'identical_receipt_and_work': True, 'metrics': metrics}
    if all('prepared_runtime' in b['runs']['after']['results'][index] for b in blocks):
        cold = statistics.median(b['runs']['after']['results'][index]['full_runtime']['median_ns'] for b in blocks)
        warm = statistics.median(b['runs']['after']['results'][index]['prepared_runtime']['median_ns'] for b in blocks)
        preparation = statistics.median(b['runs']['after']['results'][index]['preparation']['median_ns'] for b in blocks)
        row['prepared'] = {'cold_median_ns':cold,'warm_median_ns':warm,'preparation_median_ns':preparation,
                           'warm_reduction_percent':100*(cold-warm)/cold,
                           'break_even_queries_estimate':math.ceil(preparation/(cold-warm)) if cold > warm else None}
    summary.append(row)
report = {'schema_version': 1, 'platform': platform.platform(), 'machine': platform.machine(),
          'binary_sha256': {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()},
          'method': 'Alternating AB/BA process blocks; each binary has 5 warmups and 30 samples per size/path. Summary is median of block medians. No confidence interval or universal speedup claim.',
          'summary': summary, 'blocks': blocks}
with args.output.open('x') as output:
    rendered = json.dumps(report, indent=2)
    rendered = re.sub(r'"samples_ns": \[([\s\d,]+)\]',
                      lambda m: '"samples_ns": ' + json.dumps(json.loads('[' + m.group(1) + ']')), rendered)
    output.write(rendered + '\n')
print(json.dumps(summary, indent=2))
