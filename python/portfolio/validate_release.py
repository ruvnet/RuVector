"""Fail closed on version drift or unexpected distribution contents."""
import email
import json
import os
from pathlib import Path
import sys
import tarfile
import tomllib
import zipfile

BASE = Path(__file__).resolve().parent


def validate(directory=None):
    manifest = json.loads((BASE / 'manifest.json').read_text())
    expected = manifest['version']
    ref = os.environ.get('GITHUB_REF', '')
    if ref.startswith('refs/tags/') and ref != f'refs/tags/python-portfolio-v{expected}':
        raise ValueError('Release tag must match portfolio version')
    for name in manifest['packages']:
        project = tomllib.loads((BASE / name / 'pyproject.toml').read_text())['project']
        if project['name'] != name or project['version'] != expected:
            raise ValueError('Package identity mismatch')
        if f"ruvector=={manifest['runtime']}" not in project['dependencies']:
            raise ValueError('Native runtime pin mismatch')
    if directory is None:
        return
    directory = Path(directory)
    expected_files = set()
    for name in manifest['packages']:
        normalized = name.replace('-', '_')
        wheel = f'{normalized}-{expected}-py3-none-any.whl'
        sdist = f'{normalized}-{expected}.tar.gz'
        expected_files.update([wheel, sdist])
        with zipfile.ZipFile(directory / wheel) as archive:
            names = archive.namelist()
            metadata = [n for n in names if n.endswith('.dist-info/METADATA')]
            if len(metadata) != 1:
                raise ValueError('Missing or ambiguous wheel metadata')
            parsed = email.message_from_bytes(archive.read(metadata[0]))
            if parsed['Name'] != name or parsed['Version'] != expected:
                raise ValueError('Wheel metadata mismatch')
            for path in names:
                if any(part.startswith('.') or part in {'__pycache__', 'node_modules', 'target'} for part in Path(path).parts) or Path(path).suffix in {'.pem', '.key', '.so', '.pyd'}:
                    raise ValueError('Forbidden wheel content')
                if not (path.startswith(normalized + '/') or path.startswith(f'{normalized}-{expected}.dist-info/')):
                    raise ValueError('Unexpected wheel path')
                if '..' in Path(path).parts or path.startswith('/'):
                    raise ValueError('Unsafe wheel path')
        with tarfile.open(directory / sdist) as archive:
            prefix = f'{normalized}-{expected}/'
            allowed = {'src', 'README.md', 'LICENSE', 'pyproject.toml', 'PKG-INFO', '.gitignore'}
            for member in archive.getmembers():
                if not member.name.startswith(prefix) or '..' in Path(member.name).parts or member.issym() or member.islnk():
                    raise ValueError('Unsafe sdist path')
                if any((part.startswith('.') and not (part == '.gitignore' and member.name == prefix + '.gitignore')) or part in {'__pycache__', 'node_modules', 'target'} for part in Path(member.name).parts) or Path(member.name).suffix in {'.pem', '.key', '.so', '.pyd'}:
                    raise ValueError('Forbidden sdist content')
                if member.name[len(prefix):].split('/')[0] not in allowed:
                    raise ValueError('Unexpected sdist content')
    if {p.name for p in directory.iterdir()} != expected_files:
        raise ValueError('Missing or unexpected release artifacts')


if __name__ == '__main__':
    validate(sys.argv[1] if len(sys.argv) > 1 else None)
    print('Release identity and artifact validation passed')
