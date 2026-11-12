import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('release_check', Path(__file__).parents[1] / 'validate_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_matching_tag(monkeypatch):
    monkeypatch.setenv('GITHUB_REF', 'refs/tags/python-portfolio-v0.2.0')
    release.validate()


@pytest.mark.parametrize('ref', ['refs/tags/python-portfolio-v9.9.9', 'refs/tags/python-v0.1.0'])
def test_wrong_tag_blocks_release(monkeypatch, ref):
    monkeypatch.setenv('GITHUB_REF', ref)
    with pytest.raises(ValueError, match='tag'):
        release.validate()
