import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WRITERS = {'build-native.yml': 'commit-binaries', 'build-attention.yml': 'commit-binaries',
           'build-graph-transformer.yml': 'commit-binaries', 'build-rvf-node.yml': 'commit-binaries',
           'build-gnn.yml': 'commit-binaries', 'edge-net-models.yml': 'update-registry'}


class MainMutation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.repo, self.remote = base / 'source', base / 'remote.git'
        self.repo.mkdir()
        self.git('init', '--bare', str(self.remote))
        self.git('init', '-b', 'main')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('config', 'user.name', 'Mutation guard fixture')
        self.git('remote', 'add', 'origin', str(self.remote))
        (self.repo / 'source').write_text('original source')
        self.git('add', 'source')
        self.git('commit', '-m', 'source')
        self.source = self.git('rev-parse', 'HEAD')
        self.git('push', 'origin', 'HEAD:refs/heads/main')

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.repo, stderr=subprocess.PIPE, text=True).strip()

    def guard(self, sha=None, ref='refs/heads/main'):
        return subprocess.run(['bash', str(ROOT / '.github/scripts/require-build-head.sh')], cwd=self.repo,
                              env={**os.environ, 'GITHUB_SHA': sha or self.source, 'GITHUB_REF': ref},
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def test_exact_source_accepts_main_and_explicit_manual_branch(self):
        self.assertEqual(self.guard().returncode, 0)
        self.git('push', 'origin', 'HEAD:refs/heads/topic')
        self.assertEqual(self.guard(ref='refs/heads/topic').returncode, 0)

    def test_stale_remote_rejects_artifacts_without_rebasing_source(self):
        (self.repo / 'source').write_text('new source')
        self.git('commit', '-am', 'advance main')
        self.git('push', 'origin', 'HEAD:refs/heads/main')
        self.git('checkout', '--detach', self.source)
        result = self.guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('stale build', result.stdout)
        self.assertEqual(self.git('rev-parse', 'HEAD'), self.source)

    def test_wrong_source_dirty_checkout_and_tag_targets_reject(self):
        for sha, ref, message in [('a' * 40, 'refs/heads/main', 'Checkout differs'),
                                  (self.source, 'refs/tags/v1', 'branch ref')]:
            with self.subTest(ref=ref, sha=sha):
                result = self.guard(sha, ref)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout)
        (self.repo / 'unowned-input').write_text('dirty')
        result = self.guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('clean source', result.stdout)

    def test_nonforce_push_cannot_replace_main_that_advanced_after_guard(self):
        self.assertEqual(self.guard().returncode, 0)
        other = Path(self.temp.name) / 'other'
        subprocess.run(['git', 'clone', '--branch', 'main', str(self.remote), str(other)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (other / 'source').write_text('concurrent source')
        subprocess.run(['git', '-c', 'user.name=Concurrent fixture', '-c', 'user.email=fixture@example.invalid',
                        'commit', '-am', 'advance main after guard'], cwd=other, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/main'], cwd=other, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        newer = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=other, text=True).strip()
        (self.repo / 'artifact').write_text('built from old source')
        self.git('add', 'artifact')
        self.git('commit', '-m', 'old build artifacts')
        result = subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/main'], cwd=self.repo,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.git('ls-remote', 'origin', 'refs/heads/main').split()[0], newer)

    def test_all_writers_and_publisher_share_only_job_mutation_lock(self):
        for name, job in {**WRITERS, 'e-means-protected-release.yml': 'publish'}.items():
            with self.subTest(workflow=name):
                text = (ROOT / '.github/workflows' / name).read_text()
                block = text.split('  ' + job + ':\n', 1)[1]
                block = re.split(r'\n  [\w-]+:\n', block, maxsplit=1)[0]
                self.assertRegex(block, r'    concurrency:\n      group: ruvector-main-mutation\n      queue: max\n      cancel-in-progress: false\n')
                if name in WRITERS:
                    self.assertIn('ref: ${{ github.sha }}', block)
                    guard = block.index('run: bash .github/scripts/require-build-head.sh')
                    self.assertLess(guard, block.index('- name: Download all artifacts'))
                    self.assertIn('git push origin HEAD:refs/heads/main', block)
                    self.assertIn('git push origin "HEAD:$GITHUB_REF"', block)
                    self.assertNotRegex(block, r'git (?:pull|rebase)|git push[^\n]*--force')
