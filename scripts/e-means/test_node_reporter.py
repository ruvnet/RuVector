"""Exercise the actual Node reporter contract; synthetic probes are not receipts."""
import ast
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import release


class NodeReporter(unittest.TestCase):
    def test_actual_default_reporter_and_explicit_tap(self):
        node = shutil.which('node')
        self.assertIsNotNone(node, 'The release guard requires the actual Node runtime')
        major = int(subprocess.check_output([node, '--version'], text=True).strip().lstrip('v').split('.')[0])
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / 'reporter.test.mjs'
            fixture.write_text("import test from 'node:test'; test('actual reporter regression probe', () => {});\n")
            baseline = subprocess.run([node, '--test', str(fixture)], capture_output=True, text=True, check=True)
            if major >= 23:
                # Node 23 changed non-TTY default output from TAP to spec.
                with self.assertRaisesRegex(ValueError, 'Node qualification'):
                    release.evidence(baseline.stdout, 'node')
            else:
                self.assertEqual(release.evidence(baseline.stdout, 'node'), {'testsPassed': 1})
            tap = subprocess.run([node, '--test', '--test-reporter=tap', str(fixture)], capture_output=True, text=True, check=True)
            self.assertEqual(release.evidence(tap.stdout, 'node'), {'testsPassed': 1})
            fixture.write_text("import test from 'node:test'; test('must not qualify', { skip: true }, () => {});\n")
            skipped = subprocess.run([node, '--test', '--test-reporter=tap', str(fixture)], capture_output=True, text=True, check=True)
            with self.assertRaisesRegex(ValueError, 'Node qualification'):
                release.evidence(skipped.stdout, 'node')

    def test_parser_failure_prints_the_actual_owned_command_log(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stderr(io.StringIO()) as diagnostic:
            commands = []
            with self.assertRaisesRegex(ValueError, 'Node qualification'):
                release.execute([sys.executable, '-c', 'print("actual reporter output without TAP")'],
                                Path(temporary), Path(temporary), commands, 'node')
            self.assertIn('actual reporter output without TAP', diagnostic.getvalue())
            self.assertEqual(commands[0]['exitCode'], 0)
            self.assertNotIn('testsPassed', commands[0])

    def test_package_and_all_direct_release_tests_pin_tap(self):
        manifest = json.loads((release.ROOT / 'npm/packages/e-means/package.json').read_text())
        self.assertEqual(manifest['scripts']['test'], 'node --test --test-reporter=tap test/*.test.mjs')
        for filename, count in (('release.py', 2), ('github_release.py', 1)):
            with self.subTest(filename=filename):
                tree = ast.parse((Path(__file__).parent / filename).read_text())
                arguments = [node for node in ast.walk(tree) if isinstance(node, ast.List)
                             and len(node.elts) >= 2 and all(isinstance(value, ast.Constant) for value in node.elts[:2])
                             and [value.value for value in node.elts[:2]] == ['node', '--test']]
                self.assertEqual(len(arguments), count)
                for argv in arguments:
                    self.assertIsInstance(argv.elts[2], ast.Constant)
                    self.assertEqual(argv.elts[2].value, '--test-reporter=tap')

    def test_failure_upload_is_conditional_and_contains_only_owned_logs(self):
        workflow = (release.ROOT / '.github/workflows/e-means-preflight.yml').read_text()
        start = workflow.index('      - name: Preserve owned command logs when qualification fails')
        step = workflow[start:workflow.index('  node20:', start)]
        self.assertIn('        if: failure()', step)
        self.assertIn('name: e-means-failure-logs-${{ github.sha }}', step)
        paths = [line.strip() for line in step.splitlines() if line.strip().startswith('path:')]
        self.assertEqual(paths, ['path: ${{ runner.temp }}/e-means-seal/command-*.log'])
        self.assertIn('if-no-files-found: ignore', step)
        start = workflow.index('      - name: Preserve Node minimum command logs when qualification fails')
        step = workflow[start:]
        self.assertIn('        if: failure()', step)
        self.assertIn('name: e-means-failure-logs-${{ github.sha }}', step)
        paths = [line.strip() for line in step.splitlines() if line.strip().startswith('path:')]
        self.assertEqual(paths, ['path: ${{ runner.temp }}/node20-consumer/command-*.log'])


if __name__ == '__main__':
    unittest.main()
