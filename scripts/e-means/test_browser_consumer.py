"""Exercise the actual pinned browser and a broken module graph; never publish."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import release


class ActualBrowser(unittest.TestCase):
    def fixture(self, directory):
        source = release.ROOT / 'npm/packages/e-means'
        root = Path(directory)
        (root / 'test').mkdir()
        for name in ('e-means.mjs', 'runtime.mjs'):
            shutil.copyfile(source / name, root / name)
        shutil.copytree(source / 'wasm', root / 'wasm')
        script = root / 'test/browser-consumer.mjs'
        shutil.copyfile(source / 'test/browser-consumer.mjs', script)
        return script

    def invoke(self, script, env=None):
        return subprocess.run(['node', str(script)], capture_output=True, text=True,
                              env={**os.environ, **(env or {})}, timeout=40)

    def test_real_browser_passes_missing_runtime_fails_and_restoration_passes(self):
        self.assertTrue(os.environ.get('E_MEANS_PLAYWRIGHT_ROOT'), 'Bootstrap external pinned Playwright and set E_MEANS_PLAYWRIGHT_ROOT')
        with tempfile.TemporaryDirectory() as directory:
            script = self.fixture(directory)
            original = script.read_text()
            good = self.invoke(script)
            self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
            self.assertEqual(good.stdout.splitlines().count(release.BROWSER_MARKER), 1)
            route = "  ['/runtime.mjs', ['../runtime.mjs', 'application/javascript']],\n"
            self.assertEqual(original.count(route), 1, 'The mutation must remove exactly the actual runtime route')
            script.write_text(original.replace(route, ''))
            broken = self.invoke(script)
            self.assertNotEqual(broken.returncode, 0, 'A missing runtime module must fail actual Chromium')
            self.assertNotIn(release.BROWSER_MARKER, broken.stdout)
            self.assertRegex(broken.stderr, '404|Failed to load|request failed')
            script.write_text(original)
            restored = self.invoke(script)
            self.assertEqual(restored.returncode, 0, restored.stdout + restored.stderr)
            self.assertEqual(restored.stdout.splitlines().count(release.BROWSER_MARKER), 1)

    def test_wrong_driver_version_rejected_before_browser_success(self):
        with tempfile.TemporaryDirectory() as directory:
            script = self.fixture(directory)
            tooling = Path(directory) / 'wrong-tools'
            package = tooling / 'node_modules/playwright'
            package.mkdir(parents=True)
            (package / 'package.json').write_text(json.dumps({'version': '1.62.0'}))
            result = self.invoke(script, {'E_MEANS_PLAYWRIGHT_ROOT': str(tooling)})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires Playwright 1.62.1', result.stderr)
            self.assertNotIn(release.BROWSER_MARKER, result.stdout)

    def test_both_workflows_supply_exact_external_driver_and_real_chromium(self):
        for filename in ('e-means-preflight.yml', 'e-means-protected-release.yml'):
            with self.subTest(filename=filename):
                workflow = (release.ROOT / '.github/workflows' / filename).read_text()
                self.assertIn('--ignore-scripts --no-save playwright@1.62.1', workflow)
                self.assertIn('"$RUNNER_TEMP/browser-tools/node_modules/.bin/playwright" install chromium', workflow)
                self.assertIn('echo "E_MEANS_PLAYWRIGHT_ROOT=$RUNNER_TEMP/browser-tools" >> "$GITHUB_ENV"', workflow)
                self.assertIn('console.log("E_MEANS_CHROMIUM="+p.chromium.executablePath())', workflow)


if __name__ == '__main__':
    unittest.main()
