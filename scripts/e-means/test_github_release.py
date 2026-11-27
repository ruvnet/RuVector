"""Mutation tests use synthetic evidence and mocked GitHub; never publish."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import github_release as github
import release

SOURCE = 'a' * 40
NODE = '# pass 2\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n'


class GitHubProtocol(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.folder = self.base / 'assets'
        self.folder.mkdir()
        self.payload = {'root.crate': b'canonical crate', 'receipt.json': b'actual receipt fixture'}
        for name, data in self.payload.items():
            (self.folder / name).write_bytes(data)
        self.manifest = {name: hashlib.sha256(data).hexdigest() for name, data in self.payload.items()}
        self.tag = None
        self.row = None
        self.remote = {}
        self.mutations = []
        self.downloads = []

    def gh_fixture(self, *args):
        self.mutations.append(args)
        if args[:3] == ('api', '--method', 'POST'):
            self.assertIn('sha=' + SOURCE, args)
            self.tag = SOURCE
            return '{}'
        if args[:2] == ('release', 'create'):
            self.assertEqual(self.tag, SOURCE)
            self.assertIn('--verify-tag', args)
            self.assertIn('--draft', args)
            self.row = {'id': 42, 'tag_name': github.TAG, 'name': github.TITLE, 'draft': True, 'target_commitish': SOURCE}
        elif args[:2] == ('release', 'upload'):
            path = Path(args[3])
            self.assertNotIn('--clobber', args)
            self.remote[path.name] = path.read_bytes()
        elif args[:2] == ('release', 'edit'):
            self.assertEqual(set(self.remote), set(self.payload))
            self.row['draft'] = False
        else:
            self.fail('Unexpected mutating GitHub command: ' + repr(args))
        return ''

    def remote_inventory(self, endpoint):
        return [{'id': index, 'name': name, 'size': len(data), 'state': 'uploaded',
                 'digest': 'sha256:' + hashlib.sha256(data).hexdigest()}
                for index, (name, data) in enumerate(sorted(self.remote.items()), 1)]

    def download(self, args, stdout, check):
        index = int(args[2].rsplit('/', 1)[1]) - 1
        name, data = sorted(self.remote.items())[index]
        self.downloads.append((name, self.row['draft']))
        stdout.write(data)
        return SimpleNamespace(returncode=0)

    def run_protocol(self):
        with patch.object(github, 'guard'), patch.object(github, 'assets', return_value=(self.folder, self.manifest)), \
                patch.object(github, 'tag_source', side_effect=lambda: self.tag), \
                patch.object(github, 'existing_release', side_effect=lambda: copy.deepcopy(self.row)), \
                patch.object(github, 'inventory', side_effect=self.remote_inventory), \
                patch.object(github, 'gh', side_effect=self.gh_fixture), \
                patch.object(github.subprocess, 'run', side_effect=self.download):
            github.publish(SOURCE, self.base)

    def test_missing_tag_created_before_verify_tag_draft_and_all_assets_downloaded_twice(self):
        self.run_protocol()
        self.assertEqual(self.mutations[0][:3], ('api', '--method', 'POST'))
        self.assertEqual({name for name, draft in self.downloads if draft}, set(self.payload))
        self.assertEqual({name for name, draft in self.downloads if not draft}, set(self.payload))
        self.assertEqual(json.loads((self.base / 'github-release.json').read_text())['tagSourceSha'], SOURCE)

    def test_partial_tag_and_partial_draft_resume_without_overwrite(self):
        self.tag = SOURCE
        self.run_protocol()
        self.assertFalse(any(args[:3] == ('api', '--method', 'POST') for args in self.mutations))
        self.row['draft'] = True
        self.remote.pop('receipt.json')
        self.mutations.clear()
        self.run_protocol()
        uploads = [args for args in self.mutations if args[:2] == ('release', 'upload')]
        self.assertEqual([Path(args[3]).name for args in uploads], ['receipt.json'])

    def test_conflicting_tag_blocks_every_mutation(self):
        self.tag = 'b' * 40
        with self.assertRaisesRegex(ValueError, 'immutable release tag'):
            self.run_protocol()
        self.assertEqual(self.mutations, [])

    def test_existing_draft_without_tag_and_wrong_source_fail_closed(self):
        self.row = {'id': 42, 'tag_name': github.TAG, 'name': github.TITLE, 'draft': True, 'target_commitish': SOURCE}
        with self.assertRaisesRegex(ValueError, 'immutable source'):
            self.run_protocol()
        self.tag = SOURCE
        self.row['target_commitish'] = 'b' * 40
        with self.assertRaisesRegex(ValueError, 'immutable source'):
            self.run_protocol()
        self.assertEqual(self.mutations, [])

    def test_conflicting_existing_asset_bytes_never_overwritten(self):
        self.tag = SOURCE
        self.row = {'id': 42, 'tag_name': github.TAG, 'name': github.TITLE, 'draft': True, 'target_commitish': SOURCE}
        self.remote['root.crate'] = b'changed content'
        with self.assertRaisesRegex(ValueError, 'incomplete|bytes'):
            self.run_protocol()
        self.assertEqual(self.mutations, [])

    def test_published_release_is_verified_without_mutation(self):
        self.run_protocol()
        self.mutations.clear()
        self.downloads.clear()
        self.run_protocol()
        self.assertEqual(self.mutations, [])
        self.assertEqual({name for name, _ in self.downloads}, set(self.payload))

    def test_extra_or_missing_published_asset_rejected(self):
        self.run_protocol()
        self.remote['extra.bin'] = b'not approved'
        self.mutations.clear()
        with self.assertRaisesRegex(ValueError, 'asset inventory'):
            self.run_protocol()
        self.assertEqual(self.mutations, [])
        self.remote.pop('extra.bin')
        self.remote.pop('receipt.json')
        with self.assertRaisesRegex(ValueError, 'asset inventory'):
            self.run_protocol()

    def test_nonprotected_execution_cannot_read_assets_or_mutate_github(self):
        with patch.object(release, 'identity'), patch.dict(os.environ, {}, clear=True), \
                patch.object(github, 'assets') as assets, patch.object(github, 'gh') as gh:
            with self.assertRaisesRegex(ValueError, 'protected main'):
                github.publish(SOURCE, self.base)
            assets.assert_not_called()
            gh.assert_not_called()

    def test_environment_guard_required_even_when_main_guard_passes(self):
        with patch.object(release, 'protected'), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'npm-production'):
                github.guard(SOURCE)

    def test_authenticated_listing_includes_drafts_and_rejects_duplicate_tags(self):
        draft = {'id': 42, 'tag_name': github.TAG, 'draft': True}
        with patch.object(github, 'inventory', return_value=[draft]):
            self.assertEqual(github.existing_release(), draft)
        with patch.object(github, 'inventory', return_value=[draft, draft]):
            with self.assertRaisesRegex(ValueError, 'Multiple releases'):
                github.existing_release()

    def test_only_real_404_means_absent_and_other_errors_fail_closed(self):
        with patch.object(github.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='gh: Not Found (HTTP 404)')):
            self.assertIsNone(github.api_optional('endpoint'))
        with patch.object(github.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='gh: unauthorized (HTTP 403)')):
            with self.assertRaisesRegex(ValueError, 'identity or absence'):
                github.api_optional('endpoint')


class PublicEvidence(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.fresh = self.base / 'fresh'
        self.installed = self.fresh / 'npm/node_modules/@ruvector/e-means'
        (self.base / 'candidate/harness/test').mkdir(parents=True)
        (self.base / 'candidate/harness/test/one.test.mjs').write_text('test fixture')
        (self.fresh / 'cargo/src').mkdir(parents=True)
        (self.fresh / 'cargo/src/main.rs').write_bytes((release.ROOT / 'scripts/e-means/public-consumer.rs').read_bytes())
        (self.base / 'public').mkdir()
        self.canonical = {f'{release.CRATE}-{release.VERSION}.crate': b'public crate fixture',
                          f'ruvector-e-means-{release.VERSION}.tgz': b'public npm fixture'}
        self.candidate = {'artifacts': {name: hashlib.sha256(data).hexdigest() for name, data in self.canonical.items()}}
        for name, data in self.canonical.items():
            (self.base / 'public' / name).write_bytes(data)
        self.registry = 'registry+https://github.com/rust-lang/crates.io-index'
        self.metadata = {'packages': [{'name': release.CRATE, 'version': release.VERSION, 'source': self.registry}]}
        self.lock = f'[[package]]\nname="{release.CRATE}"\nversion="{release.VERSION}"\nsource="{self.registry}"\nchecksum="{self.candidate["artifacts"][f"{release.CRATE}-{release.VERSION}.crate"]}"\n'
        (self.fresh / 'cargo/Cargo.lock').write_text(self.lock)
        manifest = str(self.fresh / 'cargo/Cargo.toml')
        args = [(['cargo', 'run', '--manifest-path', manifest], 'E_MEANS_PUBLIC_CHECKS=12\n'),
                (['cargo', 'metadata', '--quiet', '--locked', '--format-version', '1', '--manifest-path', manifest], json.dumps(self.metadata)),
                (['npm', 'install', '--ignore-scripts', '--workspaces=false', '--no-audit', '--no-fund'], 'installed'),
                (['node', '--test', '--test-reporter=tap', str(self.installed / 'test/one.test.mjs')], NODE),
                (['node', str(self.fresh / 'npm/node_modules/typescript/bin/tsc'), '--noEmit', '-p', str(self.installed / 'test/tsconfig.json')], ''),
                (['node', str(self.installed / 'test/browser-consumer.mjs')], release.BROWSER_MARKER + '\n')]
        self.commands = []
        for index, (argv, text) in enumerate(args):
            path = self.fresh / f'command-{index}.log'
            path.write_text(text)
            self.commands.append({'args': argv, 'exitCode': 0, 'log': path.name, 'logSha256': release.sha(path)})
        self.commands[3]['testsPassed'] = 2
        self.commands[5]['browserPassed'] = True
        self.report = {'schema': 'ruvector.e-means-public-consumers/1', 'sourceSha': SOURCE,
                       'artifacts': self.candidate['artifacts'], 'rustChecks': 12, 'commands': self.commands}
        (self.base / 'publication.json').write_text(json.dumps({'schema': 'ruvector.e-means-publication/1',
            'sourceSha': SOURCE, 'artifacts': self.candidate['artifacts']}))
        self.write()

    def write(self):
        (self.base / 'public-consumers.json').write_text(json.dumps(self.report))

    def validate(self):
        with patch.object(release.registry, 'downloaded', return_value=self.canonical[f'{release.CRATE}-{release.VERSION}.crate']), \
                patch.object(release, 'npm_public', return_value=self.canonical[f'ruvector-e-means-{release.VERSION}.tgz']):
            return github.validate_public(self.base, SOURCE, self.candidate)

    def test_real_command_and_log_evidence_accepted(self):
        self.assertEqual(self.validate()['rustChecks'], 12)

    def test_forged_checks_source_missing_commands_and_wrong_args_rejected(self):
        original = copy.deepcopy(self.report)
        for mutation in ('checks', 'source', 'missing', 'args', 'failed', 'log-path'):
            with self.subTest(mutation=mutation):
                self.report = copy.deepcopy(original)
                if mutation == 'checks': self.report['rustChecks'] = 9000
                elif mutation == 'source': self.report['sourceSha'] = 'b' * 40
                elif mutation == 'missing': self.report['commands'].pop()
                elif mutation == 'args': self.report['commands'][0]['args'] = ['cargo', '--version']
                elif mutation == 'failed': self.report['commands'][0]['exitCode'] = 1
                else: self.report['commands'][0]['log'] = '../outside.log'
                self.write()
                with self.assertRaises(ValueError): self.validate()

    def test_skipped_or_zero_node_tests_cannot_be_hidden_by_updated_hash(self):
        for text in (NODE.replace('# pass 2', '# pass 0'), NODE.replace('# skipped 0', '# skipped 1')):
            path = self.fresh / 'command-3.log'
            path.write_text(text)
            self.commands[3]['logSha256'] = release.sha(path)
            self.write()
            with self.assertRaisesRegex(ValueError, 'Node qualification'): self.validate()

    def test_browser_claim_without_actual_marker_rejected(self):
        path = self.fresh / 'command-5.log'
        path.write_text('browser started')
        self.commands[5]['logSha256'] = release.sha(path)
        self.write()
        with self.assertRaisesRegex(ValueError, 'Actual browser'): self.validate()

    def test_path_dependency_and_wrong_public_lock_checksum_rejected(self):
        path = self.fresh / 'command-1.log'
        self.metadata['packages'][0]['source'] = None
        path.write_text(json.dumps(self.metadata))
        self.commands[1]['logSha256'] = release.sha(path)
        self.write()
        with self.assertRaisesRegex(ValueError, 'canonical registry'): self.validate()
        self.metadata['packages'][0]['source'] = self.registry
        path.write_text(json.dumps(self.metadata))
        self.commands[1]['logSha256'] = release.sha(path)
        self.write()
        (self.fresh / 'cargo/Cargo.lock').write_text(self.lock.replace('checksum="', 'checksum="b'))
        with self.assertRaisesRegex(ValueError, 'canonical registry'): self.validate()

    def test_changed_canonical_bytes_rejected(self):
        name = next(iter(self.canonical))
        (self.base / 'public' / name).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'public archive changed'): self.validate()


class AcceptedArtifact(unittest.TestCase):
    def test_actual_zip_and_github_source_jobs_artifact_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / 'candidate').mkdir()
            (base / 'candidate/FILES.json').write_text('{}')
            with zipfile.ZipFile(base / 'candidate.zip', 'w') as archive:
                archive.write(base / 'candidate/FILES.json', 'FILES.json')
            jobs = [{'id': index, 'name': name, 'status': 'completed', 'conclusion': 'success'}
                    for index, name in enumerate(('qualify', 'node20'), 1)]
            artifact = {'id': 7, 'name': 'e-means-candidate-' + SOURCE, 'expired': False,
                        'digest': 'sha256:' + release.sha(base / 'candidate.zip'),
                        'workflow_run': {'id': 8, 'head_sha': SOURCE}}
            ci = {'sourceSha': SOURCE, 'runId': 8, 'jobs': jobs, 'artifact': artifact}
            (base / 'ci.json').write_text(json.dumps(ci))
            run = {'head_sha': SOURCE, 'event': 'push', 'head_branch': 'release/e-means',
                   'path': release.PREFLIGHT, 'status': 'completed', 'conclusion': 'success'}
            with patch.object(release, 'gh', side_effect=lambda endpoint: copy.deepcopy(artifact if '/artifacts/' in endpoint else run)), \
                    patch.object(release, 'pages', return_value=jobs):
                github.validate_ci(base, SOURCE)
                for mutation in ('source', 'event', 'workflow', 'artifact-run', 'node20-skipped', 'candidate-bytes'):
                    with self.subTest(mutation=mutation):
                        original_run, original_artifact, original_jobs = copy.deepcopy(run), copy.deepcopy(artifact), copy.deepcopy(jobs)
                        if mutation == 'source': run['head_sha'] = 'b' * 40
                        elif mutation == 'event': run['event'] = 'workflow_dispatch'
                        elif mutation == 'workflow': run['path'] = 'other.yml'
                        elif mutation == 'artifact-run': artifact['workflow_run']['id'] = 9
                        elif mutation == 'node20-skipped': jobs[1]['conclusion'] = 'skipped'
                        else: (base / 'candidate/FILES.json').write_text('{"changed":true}')
                        with self.assertRaises(ValueError): github.validate_ci(base, SOURCE)
                        run.clear(); run.update(original_run)
                        artifact.clear(); artifact.update(original_artifact)
                        jobs[:] = original_jobs
                        (base / 'candidate/FILES.json').write_text('{}')

    def test_workflow_protects_github_step_and_retains_all_proof_assets(self):
        workflow = (release.ROOT / '.github/workflows/e-means-protected-release.yml').read_text()
        self.assertIn('    environment: npm-production', workflow)
        self.assertIn('    permissions:\n      contents: write', workflow)
        start = workflow.index('      - name: Publish exact-source GitHub release')
        end = workflow.index('      - uses: actions/upload-artifact@', start)
        step = workflow[start:end]
        self.assertIn('E_MEANS_RELEASE_ENVIRONMENT: npm-production', step)
        self.assertIn('python3 scripts/e-means/github_release.py --expected-sha "$EXPECTED_SHA"', step)
        self.assertNotIn('CARGO_REGISTRY_TOKEN', step)
        self.assertNotIn('NODE_AUTH_TOKEN', step)
        self.assertLess(workflow.index('release.py verify-public'), start)
        self.assertIn('e-means-publication/github-release.json', workflow)
        self.assertIn('e-means-publication/github-release-assets/*', workflow)


if __name__ == '__main__':
    unittest.main()
