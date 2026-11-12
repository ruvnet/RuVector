"""Select releases: a version change on main or an explicit matching tag."""
import json
import os
from pathlib import Path
import subprocess


def requested(event_name, ref, event, version, previous_version):
    if event_name != 'push':
        return False
    if ref.startswith('refs/tags/'):
        return ref == f'refs/tags/python-portfolio-v{version}'
    return ref == 'refs/heads/main' and version != previous_version


def main():
    version = json.loads(Path('python/portfolio/manifest.json').read_text())['version']
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    previous = None
    before = event.get('before', '')
    if os.environ['GITHUB_REF'] == 'refs/heads/main' and before and set(before) != {'0'}:
        # A commit hash from the GitHub event, never arbitrary shell input.
        if len(before) != 40 or any(c not in '0123456789abcdef' for c in before):
            raise ValueError('Invalid before SHA')
        exists = subprocess.run(['git', 'cat-file', '-e', before], capture_output=True)
        if exists.returncode:
            raise RuntimeError('Previous commit unavailable; refusing ambiguous release')
        old = subprocess.run(['git', 'show', f'{before}:python/portfolio/manifest.json'], capture_output=True, text=True)
        if old.returncode == 0:
            previous = json.loads(old.stdout)['version']
        else:
            tree = subprocess.run(['git', 'ls-tree', before, 'python/portfolio/manifest.json'], capture_output=True, text=True, check=True)
            if tree.stdout:
                raise RuntimeError('Previous manifest unreadable')
    release = requested(os.environ['GITHUB_EVENT_NAME'], os.environ['GITHUB_REF'], event, version, previous)
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'release={str(release).lower()}\n')


if __name__ == '__main__':
    main()
