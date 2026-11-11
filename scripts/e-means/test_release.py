import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import release


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


if __name__ == '__main__':
    unittest.main()
