"""Audit the all-target normal/build dependency closure of the native SDK.

Derive from the checked-in lockfile without resolving different versions.
Keep package versions and checksums; remove edges to inactive optional dependencies.
Dev-only and unrelated workspace crates are outside shipped-wheel scope.
"""
from pathlib import Path
import re
import json
import subprocess
import sys
import tomllib

root = Path(__file__).resolve().parents[2]
result = subprocess.run(
    ['cargo', 'tree', '--locked', '-p', 'ruvector-py', '--target', 'all',
     '--edges', 'normal,build', '--prefix', 'none', '--format', '{p}'],
    cwd=root, check=True, capture_output=True, text=True,
)
wanted = set()
for line in result.stdout.splitlines():
    match = re.match(r'^([A-Za-z0-9_-]+) v([^ ]+)', line)
    if not match:
        raise ValueError(f'Unrecognized cargo tree record: {line!r}')
    wanted.add(match.groups())
native_version = tomllib.loads((root / 'crates/ruvector-py/Cargo.toml').read_text())['package']['version']
if ('ruvector-py', native_version) not in wanted:
    raise ValueError('Native root missing from dependency closure')
text = (root / 'Cargo.lock').read_text()
blocks = text.split('[[package]]')
selected = []
found = set()
for block in blocks[1:]:
    record = tomllib.loads('[[package]]' + block)['package'][0]
    key = (record['name'], record['version'])
    if key in wanted:
        found.add(key)
        deps = []
        for dep in record.get('dependencies', []):
            parts = dep.split()
            if any(name == parts[0] and (len(parts) == 1 or version == parts[1]) for name, version in wanted):
                deps.append(dep)
        block = re.sub(r'dependencies = \[.*?\n\]', 'dependencies = ' + json.dumps(deps), block, flags=re.S)
        selected.append('[[package]]' + block)
if found != wanted:
    raise ValueError('Cargo tree and lockfile disagree')
output = Path(sys.argv[1])
output.write_text(blocks[0] + ''.join(selected))
print(f'Auditing {len(found)} native runtime/build packages across all targets')
subprocess.run(['cargo', 'audit', '--file', str(output), '--deny', 'unsound', '--no-yanked'], check=True)
