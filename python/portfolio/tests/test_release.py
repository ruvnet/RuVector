import importlib.util
import json
from pathlib import Path
import tomllib
import pytest

spec = importlib.util.spec_from_file_location('release_check', Path(__file__).parents[1] / 'validate_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_matching_tag(monkeypatch):
    monkeypatch.setenv('GITHUB_REF', 'refs/tags/python-portfolio-v0.2.1')
    release.validate()


@pytest.mark.parametrize('ref', ['refs/tags/python-portfolio-v9.9.9', 'refs/tags/python-v0.1.0', 'refs/tags/python-portfolio-v0.2.0'])
def test_wrong_tag_blocks_release(monkeypatch, ref):
    monkeypatch.setenv('GITHUB_REF', ref)
    with pytest.raises(ValueError, match='tag'):
        release.validate()


def test_native_runtime_identity_matches_companions():
    root = Path(__file__).resolve().parents[3]
    runtime = json.loads((release.BASE / 'manifest.json').read_text())['runtime']
    native = root / 'crates' / 'ruvector-py'
    assert tomllib.loads((native / 'pyproject.toml').read_text())['project']['version'] == runtime
    assert tomllib.loads((native / 'Cargo.toml').read_text())['package']['version'] == runtime
    packages = tomllib.loads((root / 'Cargo.lock').read_text())['package']
    assert [p['version'] for p in packages if p['name'] == 'ruvector-py'] == [runtime]
