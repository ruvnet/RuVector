"""Publish native wheels only when their version changes on main."""
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib

path = 'crates/ruvector-py/pyproject.toml'
version = tomllib.loads(Path(path).read_text())['project']['version']
event_name = os.environ['GITHUB_EVENT_NAME']
ref = os.environ['GITHUB_REF']
event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
release = False
if event_name == 'push' and ref == 'refs/heads/main':
    before = event.get('before', '')
    if not re.fullmatch('[0-9a-f]{40}', before) or set(before) == {'0'}:
        raise ValueError('Expected an existing previous commit')
    old = subprocess.run(['git', 'show', f'{before}:{path}'], check=True, capture_output=True, text=True)
    release = version != tomllib.loads(old.stdout)['project']['version']
elif event_name == 'push' and ref.startswith('refs/tags/python-v'):
    if ref != f'refs/tags/python-v{version}':
        raise ValueError('Tag does not match native version')
    release = True
elif event_name == 'workflow_dispatch':
    release = ref == 'refs/heads/main' and str(event.get('inputs', {}).get('publish', '')).lower() == 'true'
with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
    output.write(f'release={str(release).lower()}\n')
