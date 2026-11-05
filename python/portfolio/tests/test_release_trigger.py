import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import pytest

spec = importlib.util.spec_from_file_location('trigger', Path(__file__).parents[1] / 'release_requested.py')
trigger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trigger)

@pytest.mark.parametrize('event,ref,previous,expected', [
    ('push','refs/heads/main',None,True),
    ('push','refs/heads/main','0.0.9',True),
    ('push','refs/heads/main','0.1.0',False),
    ('push','refs/heads/feature','0.0.9',False),
    ('pull_request','refs/pull/1185/merge',None,False),
    ('workflow_dispatch','refs/heads/main',None,False),
    ('push','refs/tags/python-portfolio-v0.1.0','0.1.0',True),
    ('push','refs/tags/python-portfolio-v9.9.9',None,False),
])
def test_release_scope(event,ref,previous,expected):
    assert trigger.requested(event,ref,{},'0.1.0',previous) is expected


def test_pr_workflow_uses_candidate_native_and_skips_publication():
    root = Path(__file__).resolve().parents[3]
    workflow = (root / '.github/workflows/python-portfolio.yml').read_text()
    candidate = workflow.split('name: Verify candidate runtime compatibility', 1)[1].split('      - uses:', 1)[0]
    install = next(line.strip() for line in candidate.splitlines() if '--only-binary=' in line)
    arguments = shlex.split(install)
    assert '/tmp/native/*.whl' in arguments and 'dist/*.whl' in arguments
    assert '/tmp/candidate-test/bin/python -m pytest python/portfolio/tests -q' in candidate
    assert 'python -m pip_audit --path /tmp/candidate-test/lib/python3.12/site-packages' in candidate
    assert "if: needs.test-build.outputs.release == 'true'" in workflow
    assert not trigger.requested('pull_request', 'refs/pull/1242/merge', {}, '0.2.1', '0.2.0')


def test_postpublication_install_pins_public_native_and_companions(tmp_path):
    root = Path(__file__).resolve().parents[3]
    workflow = (root / '.github/workflows/python-portfolio.yml').read_text()
    published = workflow.split('  verify-published:', 1)[1]
    assert 'needs: publish' in published
    generator = next(line.strip() for line in published.splitlines() if "python -c 'import json,pathlib;" in line)
    manifest = json.loads((root / 'python/portfolio/manifest.json').read_text())
    path = tmp_path / 'python/portfolio/manifest.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    subprocess.run([sys.executable, '-c', shlex.split(generator)[2]], cwd=tmp_path, check=True)
    assert (tmp_path / 'release-requirements.txt').read_text().splitlines() == [
        'ruvector==' + manifest['runtime'],
        *[name + '==' + manifest['version'] for name in manifest['packages']],
    ]
    install = next(line.strip() for line in published.splitlines() if '-m pip install' in line)
    arguments = shlex.split(install)
    assert arguments[arguments.index('--index-url') + 1] == 'https://pypi.org/simple'
    assert '--no-cache-dir' in arguments
    assert arguments[arguments.index('-r') + 1] == 'release-requirements.txt'
    assert not any(token in arguments for token in ['--find-links', '--extra-index-url', '/tmp/native/*.whl', 'dist/*.whl'])
