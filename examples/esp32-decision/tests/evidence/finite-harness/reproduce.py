#!/usr/bin/env python3
"""Reproduce independent ESP32 benchmark comparisons.

This file combines the two original inline evaluator invocations recorded during
this run. It is a reconstructed wrapper, not a byte-identical original script.
The seven malformed cases and the 96-case loop, seed, generation order, parent
source loading, and exact comparison are preserved. No hardware is exercised.
Run from the repository root. The parent commit must exist in local git history.
"""
import hashlib
import json
import random
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, 'examples/esp32-decision/tools')
import rig

PARENT = '6fdb5f12ac7fe16752eeb60405e52567081706db'
FROZEN_EVALUATOR = 'cc5c17ecb2c9513167bb1a040c0e3854c063d5842578511d3c0985ecd70e498a'
evaluator = Path('examples/esp32-decision/tests/test_finite_benchmark.py')
assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == FROZEN_EVALUATOR
source = subprocess.check_output(
    ['git', 'show', PARENT + ':examples/esp32-decision/tools/rig.py'], text=True)
parent = types.ModuleType('frozen_parent_rig')
exec(compile(source, 'frozen_parent_rig.py', 'exec'), parent.__dict__)

# Exact cases from the original parent/candidate malformed-input invocations.
cases = [
    ('infinite_baseline', float('inf'), 1, 11),
    ('nan_baseline', float('nan'), 1, 11),
    ('boolean_baseline', True, .5, 11),
    ('ratio_overflow', 1e308, 1e-308, 11),
    ('ratio_underflow', 1e-308, 1e308, 11),
    ('huge_integer_ratio', 10**1000, 10**999, 11),
    ('even_median_overflow', 1e308, 1, 12),
]
malformed = []
for name, a, b, count in cases:
    records = [{'baseline': {'profile': {'mean_cycles': a}},
                'candidate': {'profile': {'mean_cycles': b}}} for _ in range(count)]
    row = {'name': name}
    for prefix, module in [('parent', parent), ('candidate', rig)]:
        try:
            result = module.retention(records, 'mean_cycles', 'physical', True)
            row[prefix + '_rejected'] = False
            row[prefix + '_false_accept'] = bool(result['retain'])
        except (ValueError, TypeError, OverflowError) as error:
            row[prefix + '_rejected'] = True
            row[prefix + '_false_accept'] = False
            row[prefix + '_exception'] = type(error).__name__
    malformed.append(row)
assert all(row['candidate_rejected'] for row in malformed)
assert sum(row['parent_false_accept'] for row in malformed) == 5

# Exact seed, generation order, arithmetic and equality from the 96-case run.
rng = random.Random(84491)
count = 0
for n in (1, 2, 10, 11, 12, 99):
    for kind in ('integer', 'float', 'large_safe_integer', 'tiny_float'):
        records = []
        for _ in range(n):
            a = rng.randint(1, 10000)
            b = rng.randint(1, 10000)
            if kind == 'float':
                a /= 73
                b /= 79
            if kind == 'large_safe_integer':
                a *= 2**100
                b *= 2**100
            if kind == 'tiny_float':
                a *= 1e-280
                b *= 1e-280
            records.append({'baseline': {'profile': {'mean_cycles': a}},
                            'candidate': {'profile': {'mean_cycles': b}}})
        for execution, correct in [('physical', True), ('physical', False),
                                   ('native', True), ('emulator', True)]:
            actual = rig.retention(records, 'mean_cycles', execution, correct)
            expected = parent.retention(records, 'mean_cycles', execution, correct)
            assert actual == expected, (n, kind, execution, correct)
            count += 1
assert count == 96
report = {
    'parent_sha': PARENT,
    'candidate_sha256': hashlib.sha256(Path(rig.__file__).read_bytes()).hexdigest(),
    'evaluator_sha256': FROZEN_EVALUATOR,
    'finite_legacy_equal': True,
    'finite_parity_cases': count,
    'finite_parity_seed': 84491,
    'finite_parity_kinds': ['integer', 'float', 'large_safe_integer', 'tiny_float'],
    'malformed': malformed,
    'lab_pass': None,
    'note': 'Synthetic software fixtures only. Lab and unittest results are separate receipts.',
    'reproduction_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('build-finite-reproduced.json')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'finite_parity_cases': count, 'malformed_rejected': len(malformed),
                  'parent_false_accepts': 5, 'output': str(output)}))
