import json
from pathlib import Path
import sys
import tomllib
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import github_release
import release

# Registry consumers stay on the immutable public predecessor until the new
# shared archives have completed publication and public-byte verification.
FACADE_VERSION = '0.1.0'


class SharedGeneration(unittest.TestCase):
    def test_reviewed_generation_and_staged_consumer_pins_are_consistent(self):
        root = release.ROOT
        self.assertEqual(release.VERSION, '0.1.1')
        self.assertEqual(github_release.TAG, 'e-means-v' + release.VERSION)
        self.assertIn(release.VERSION, github_release.TITLE)
        crate = tomllib.loads((root / 'crates/ruvector-e-means/Cargo.toml').read_text())
        self.assertEqual(crate['package']['version'], release.VERSION)
        for name in ('ruvector-core', 'ruvector-pq-search', 'ruvector-rairs'):
            manifest = tomllib.loads((root / 'crates' / name / 'Cargo.toml').read_text())
            self.assertEqual(manifest['dependencies'][release.CRATE]['version'], '=' + release.VERSION)
        locked = [row for row in tomllib.loads((root / 'Cargo.lock').read_text())['package'] if row['name'] == release.CRATE]
        self.assertEqual(len(locked), 1)
        self.assertEqual(locked[0]['version'], release.VERSION)
        self.assertNotIn('source', locked[0])
        shared = root / 'npm/packages/e-means'
        self.assertEqual(json.loads((shared / 'package.json').read_text())['version'], release.VERSION)
        isolated = json.loads((shared / 'package-lock.json').read_text())
        self.assertEqual(isolated['version'], release.VERSION)
        self.assertEqual(isolated['packages']['']['version'], release.VERSION)
        workspace = json.loads((root / 'npm/package-lock.json').read_text())
        self.assertEqual(workspace['packages']['packages/e-means']['version'], release.VERSION)
        for name in ('core', 'all', 'ruvector'):
            manifest = json.loads((root / 'npm/packages' / name / 'package.json').read_text())
            self.assertEqual(workspace['packages']['packages/' + name]['version'], manifest['version'])
            self.assertEqual(manifest['dependencies'][release.NPM_NAME], FACADE_VERSION)
            self.assertEqual(workspace['packages']['packages/' + name]['dependencies'][release.NPM_NAME], FACADE_VERSION)
            public = workspace['packages']['packages/' + name + '/node_modules/' + release.NPM_NAME]
            self.assertEqual(public['version'], FACADE_VERSION)
            self.assertEqual(public['resolved'], 'https://registry.npmjs.org/@ruvector/e-means/-/e-means-' + FACADE_VERSION + '.tgz')
            self.assertTrue(public['integrity'].startswith('sha512-'))
            self.assertNotIn('link', public)
        wasm = json.loads((root / 'crates/ruvector-wasm/package.json').read_text())
        self.assertEqual(wasm['dependencies'][release.NPM_NAME], FACADE_VERSION)
        for path in ('test/offline-tooling-lock.mjs', 'test/packed-consumer.mjs', 'test/ecosystem-consumers.mjs'):
            self.assertNotIn("'0.1.0'", (shared / path).read_text(), path)
        self.assertNotIn('ruvector-e-means="=0.1.0"', (root / 'scripts/e-means/release.py').read_text())
