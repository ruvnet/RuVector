import base64
import contextlib
import copy
import hashlib
import json
import shutil
import tarfile
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import release


def role_index(name):
    return list(release.QUALIFICATION).index(name)


class ReleaseGuards(unittest.TestCase):
    def test_local_publication_cannot_reach_registry_or_github(self):
        with patch.object(release, 'identity'), patch.object(release, 'gh') as github:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, 'protected main'):
                    release.protected('a' * 40)
            github.assert_not_called()

    def test_wrong_workflow_ref_or_source_cannot_publish(self):
        valid = {'GITHUB_ACTIONS': 'true', 'GITHUB_WORKFLOW': release.PUBLISHER,
                 'GITHUB_REF': 'refs/heads/main', 'GITHUB_SHA': 'a' * 40}
        for field, value in [('GITHUB_WORKFLOW', 'other'), ('GITHUB_REF', 'refs/heads/release/x'), ('GITHUB_SHA', 'b' * 40)]:
            with self.subTest(field=field), patch.object(release, 'identity'), patch.object(release, 'gh') as github:
                with patch.dict(os.environ, {**valid, field: value}, clear=True):
                    with self.assertRaisesRegex(ValueError, 'protected main'):
                        release.protected('a' * 40)
                github.assert_not_called()

    def test_moved_main_and_release_blockers_fail_closed(self):
        valid = {'GITHUB_ACTIONS': 'true', 'GITHUB_WORKFLOW': release.PUBLISHER,
                 'GITHUB_REF': 'refs/heads/main', 'GITHUB_SHA': 'a' * 40}
        with patch.object(release, 'identity'), patch.dict(os.environ, valid, clear=True):
            with patch.object(release, 'gh', return_value={'object': {'sha': 'b' * 40}}):
                with self.assertRaisesRegex(ValueError, 'Main moved'):
                    release.protected('a' * 40)
            with patch.object(release, 'gh', side_effect=[{'object': {'sha': 'a' * 40}}, [{'number': 1}]]):
                with self.assertRaisesRegex(ValueError, 'release blockers'):
                    release.protected('a' * 40)

    def test_success_exit_with_zero_or_skipped_node_tests_is_rejected(self):
        for passed, skipped in [(0, 0), (1, 1)]:
            with self.subTest(passed=passed, skipped=skipped), tempfile.TemporaryDirectory() as temp:
                text = f'# pass {passed}\n# fail 0\n# cancelled 0\n# skipped {skipped}\n# todo 0'
                with self.assertRaisesRegex(ValueError, 'nonzero tests'):
                    release.execute([sys.executable, '-c', f'print({text!r})'], Path(temp), Path(temp), [], 'node')

    def test_success_exit_with_ignored_rust_tests_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'ignored tests'):
                release.execute([sys.executable, '-c', 'print("test result: ok. 3 passed; 0 failed; 1 ignored;")'], Path(temp), Path(temp), [], 'rust')

    def test_browser_process_success_without_actual_oracle_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'Actual browser'):
                release.execute([sys.executable, '-c', 'print("browser started")'], Path(temp), Path(temp), [], 'browser')

    def test_failed_subprocess_cannot_become_success(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            commands = []
            with self.assertRaises(RuntimeError):
                release.execute([sys.executable, '-c', 'raise SystemExit(23)'], Path(temp), Path(temp), commands)
            self.assertEqual(commands[0]['exitCode'], 23)

    def test_immutable_sha_validation_rejects_branches(self):
        with self.assertRaisesRegex(ValueError, 'immutable SHA'):
            release.identity('main')


RUST_RESULT = 'test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n'
NODE_RESULT = '# pass 7\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n'


class CandidateProof(unittest.TestCase):
    """Synthetic archives/logs are validator mutation fixtures, never receipts."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        npm = release.ROOT / 'npm/packages/e-means'
        shutil.copytree(npm / 'test', self.base / 'harness/test')
        for name in ('package.json', 'package-lock.json'):
            shutil.copyfile(npm / name, self.base / 'harness' / name)
        self.commands = []
        for role, (args, cwd, kind, suites) in release.QUALIFICATION.items():
            log = f'command-{len(self.commands)}.log'
            text = (RUST_RESULT * suites if kind == 'rust' else NODE_RESULT if kind == 'node'
                    else release.BROWSER_MARKER + '\n' if kind == 'browser'
                    else '\n'.join(release.CONSUMER_MARKERS) if kind == 'consumer' else 'command completed\n')
            (self.base / log).write_text(text)
            command = {'role': role, 'args': list(args), 'cwd': cwd, 'exitCode': 0,
                       'log': log, 'logSha256': release.sha(self.base / log)}
            if kind in ('rust', 'node'):
                command['testsPassed'] = 3 * suites if kind == 'rust' else 7
            if kind == 'browser':
                command['browserPassed'] = True
            self.commands.append(command)
        crate = f'{release.CRATE}-{release.VERSION}.crate'
        npm_archive = f'ruvector-e-means-{release.VERSION}.tgz'
        wasm = b'validator-fixture-not-an-executable-wasm'
        self.archive(crate, {f'{release.CRATE}-{release.VERSION}/Cargo.toml':
                            f'[package]\nname="{release.CRATE}"\nversion="{release.VERSION}"\n'.encode()})
        self.archive(npm_archive, {'package/package.json': json.dumps({'name': release.NPM_NAME, 'version': release.VERSION}).encode(),
                                  'package/wasm/ruvector_e_means_bg.wasm': wasm})
        self.receipt = {'schema': 'ruvector.e-means-candidate/1', 'sourceSha': 'a' * 40,
                        'commands': self.commands, 'wasmBindgen': 'wasm-bindgen 0.2.125',
                        'wasmSha256': hashlib.sha256(wasm).hexdigest(),
                        'artifacts': {name: release.sha(self.base / name) for name in (crate, npm_archive)}}
        self.write_receipt()

    def archive(self, filename, members):
        with tarfile.open(self.base / filename, 'w:gz') as tar:
            for name, data in members.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))

    def manifest(self):
        files = {path.relative_to(self.base).as_posix(): release.sha(path)
                 for path in self.base.rglob('*') if path.is_file() and path.name != 'FILES.json'}
        (self.base / 'FILES.json').write_text(json.dumps(files))
        return files

    def write_receipt(self):
        (self.base / 'candidate.json').write_text(json.dumps(self.receipt))
        self.manifest()

    def validate(self):
        return release.validate_candidate(self.base, 'a' * 40)

    def test_valid_complete_candidate_survives_transfer_to_new_directory(self):
        self.assertEqual(self.validate()['sourceSha'], 'a' * 40)
        with tempfile.TemporaryDirectory() as moved:
            destination = Path(moved) / 'transferred'
            shutil.copytree(self.base, destination)
            self.assertEqual(release.validate_candidate(destination, 'a' * 40)['commands'], self.commands)

    def test_real_seal_command_wiring_matches_required_roles_before_transfer(self):
        # Exercise the producer orchestration and its argument/cwd checks without
        # building code or writing to a registry. Only subprocess effects are fake.
        generated = self.base / 'generated'
        original_logs = {role: (self.base / command['log']).read_text()
                         for role, command in zip(release.QUALIFICATION, self.commands)}
        wasm = b'validator-fixture-not-an-executable-wasm'

        def subprocess_fixture(args, cwd, stdout, stderr, env):
            index = int(Path(stdout.name).stem.split('-')[1])
            role = list(release.QUALIFICATION)[index]
            stdout.write(original_logs[role])
            if role == 'rust-package':
                destination = generated / 'package-build/package'
                destination.mkdir(parents=True)
                shutil.copyfile(self.base / f'{release.CRATE}-{release.VERSION}.crate', destination / f'{release.CRATE}-{release.VERSION}.crate')
            if role == 'wasm-bindgen':
                destination = generated / 'npm-source/wasm/ruvector_e_means_bg.wasm'
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(wasm)
            if role == 'npm-package':
                shutil.copyfile(self.base / f'ruvector-e-means-{release.VERSION}.tgz', generated / f'ruvector-e-means-{release.VERSION}.tgz')
            return SimpleNamespace(returncode=0)

        with patch.object(release, 'identity'), patch.object(release.subprocess, 'run', side_effect=subprocess_fixture), \
                patch.object(release.subprocess, 'check_output', return_value='wasm-bindgen 0.2.125'), \
                contextlib.redirect_stdout(io.StringIO()):
            release.seal('a' * 40, generated)
        validated = release.validate_candidate(generated, 'a' * 40)
        self.assertEqual([row['role'] for row in validated['commands']], list(release.QUALIFICATION))
        with tempfile.TemporaryDirectory() as moved:
            destination = Path(moved) / 'candidate'
            shutil.copytree(generated, destination)
            self.assertEqual(release.validate_candidate(destination, 'a' * 40), validated)

    def test_missing_duplicate_reordered_or_unknown_roles_fail(self):
        valid = copy.deepcopy(self.commands)
        for mutation in ('missing', 'duplicate', 'reordered', 'unknown'):
            with self.subTest(mutation=mutation):
                self.commands[:] = copy.deepcopy(valid)
                if mutation == 'missing': self.commands.pop(role_index('rust-packed'))
                elif mutation == 'duplicate': self.commands[role_index('rust-packed')] = copy.deepcopy(self.commands[role_index('rust-msrv')])
                elif mutation == 'reordered': self.commands[role_index('rust-msrv')], self.commands[role_index('rust-packed')] = self.commands[role_index('rust-packed')], self.commands[role_index('rust-msrv')]
                else: self.commands[role_index('rust-packed')]['role'] = 'not-required'
                self.write_receipt()
                with self.assertRaisesRegex(ValueError, 'required qualification roles'):
                    self.validate()

    def test_wrong_args_cwd_failed_exit_unowned_or_reused_logs_fail(self):
        valid = copy.deepcopy(self.commands)
        for key, value in [('args', ['cargo', '--version']), ('cwd', '/tmp/other-source'),
                           ('exitCode', 1), ('exitCode', False), ('log', '../command-0.log'), ('log', 'command-0.log'),
                           ('logSha256', 'b' * 64)]:
            with self.subTest(key=key, value=value):
                self.commands[:] = copy.deepcopy(valid)
                self.commands[role_index('rust-packed')][key] = value
                self.write_receipt()
                with self.assertRaisesRegex(ValueError, 'command/log differs'):
                    self.validate()

    def test_forged_evidence_claims_fail_even_when_outer_manifest_is_resealed(self):
        valid = copy.deepcopy(self.commands)
        for index, key, value in [(role_index('rust-packed'), 'testsPassed', 9000), (role_index('npm-browser'), 'browserPassed', False), (role_index('npm-browser'), 'browserPassed', 1), (role_index('clippy'), 'testsPassed', 1), (0, 'browserPassed', True)]:
            with self.subTest(index=index, key=key):
                self.commands[:] = copy.deepcopy(valid)
                self.commands[index][key] = value
                self.write_receipt()
                with self.assertRaisesRegex(ValueError, 'claims differ'):
                    self.validate()

    def replace_log(self, index, text):
        log = self.base / self.commands[index]['log']
        log.write_text(text)
        self.commands[index]['logSha256'] = release.sha(log)
        self.write_receipt()

    def test_zero_missing_failed_ignored_measured_and_filtered_rust_suites_fail(self):
        for text in ('no tests executed\n', RUST_RESULT, RUST_RESULT.replace('3 passed', '0 passed') * 4,
                     RUST_RESULT.replace('0 failed', '1 failed') * 4,
                     RUST_RESULT.replace('0 ignored', '1 ignored') * 4,
                     RUST_RESULT.replace('0 measured', '1 measured') * 4,
                     RUST_RESULT.replace('0 filtered out', '1 filtered out') * 4):
            with self.subTest(text=text):
                self.replace_log(role_index('rust-packed'), text)
                with self.assertRaisesRegex(ValueError, 'Rust qualification'):
                    self.validate()

    def test_zero_failed_cancelled_skipped_todo_and_duplicate_node_summaries_fail(self):
        variants = [NODE_RESULT.replace('# pass 7', '# pass 0'), NODE_RESULT * 2]
        variants.extend(NODE_RESULT.replace(f'# {key} 0', f'# {key} 1') for key in ('fail', 'cancelled', 'skipped', 'todo'))
        for text in variants:
            with self.subTest(text=text):
                self.replace_log(role_index('npm-tests'), text)
                with self.assertRaisesRegex(ValueError, 'Node qualification'):
                    self.validate()

    def test_successful_browser_and_installed_consumer_without_oracles_fail(self):
        for index, text, message in [(role_index('npm-browser'), 'browser started', 'Actual browser'),
                                     (role_index('npm-consumer'), release.CONSUMER_MARKERS[0], 'Installed archive consumer')]:
            with self.subTest(index=index):
                self.replace_log(index, text)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()
                # Restore this role before testing the next one.
                original = release.BROWSER_MARKER if index == role_index('npm-browser') else '\n'.join(release.CONSUMER_MARKERS)
                self.replace_log(index, original)

    def test_files_manifest_cannot_omit_logs_or_accept_unlisted_payload(self):
        files = self.manifest()
        files.pop('command-0.log')
        (self.base / 'FILES.json').write_text(json.dumps(files))
        with self.assertRaisesRegex(ValueError, 'complete payload'):
            self.validate()
        self.manifest()
        (self.base / 'unlisted.txt').write_text('not covered')
        with self.assertRaisesRegex(ValueError, 'complete payload'):
            self.validate()

    def test_harness_changes_cannot_be_hidden_by_resealing_manifest(self):
        (self.base / 'harness/test/browser-consumer.mjs').write_text('console.log("passed")')
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'exact-source checkout'):
            self.validate()

    def test_manifest_traversal_and_symlinks_fail(self):
        files = self.manifest()
        files['../outside.txt'] = 'a' * 64
        (self.base / 'FILES.json').write_text(json.dumps(files))
        with self.assertRaisesRegex(ValueError, 'complete payload'):
            self.validate()
        self.manifest()
        (self.base / 'alias').symlink_to(self.base / 'command-0.log')
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            self.validate()

    def test_source_and_pinned_binding_tool_are_required(self):
        for key, value, message in [('sourceSha', 'b' * 40, 'identity'), ('wasmBindgen', 'wasm-bindgen 0.2.124', 'pinned version')]:
            with self.subTest(key=key):
                original = self.receipt[key]
                self.receipt[key] = value
                self.write_receipt()
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()
                self.receipt[key] = original


class RegistryIntegrity(unittest.TestCase):
    def test_npm_public_bytes_must_match_registry_sha512_integrity(self):
        payload = b'canonical public archive fixture'
        integrity = 'sha512-' + base64.b64encode(hashlib.sha512(payload).digest()).decode()
        metadata = {'name': release.NPM_NAME, 'version': release.VERSION,
                    'dist': {'tarball': 'https://registry.npmjs.org/archive.tgz', 'integrity': integrity}}
        with patch.object(release.urllib.request, 'urlopen', side_effect=[io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(payload)]):
            self.assertEqual(release.npm_public(), payload)
        with patch.object(release.urllib.request, 'urlopen', side_effect=[io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(payload + b'changed')]):
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                release.npm_public()


if __name__ == '__main__':
    unittest.main()
