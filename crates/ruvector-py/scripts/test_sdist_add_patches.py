"""Source-archive helper regression fixtures; no network or real Cargo resolution."""
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

HELPER = Path(__file__).with_name("sdist-add-patches.sh")
LOCK = '''version = 4
[[package]]
name = "kept"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
[[package]]
name = "pruned"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
'''

class SdistPatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ruvector-sdist-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        patch = self.workspace / "patches/example"
        patch.mkdir(parents=True)
        (patch / "Cargo.toml").write_text('[package]\nname="example"\nversion="1.0.0"\n')
        (patch / "LICENSE").write_text("fixture license")
        sdk = self.workspace / "crates/ruvector-py"
        sdk.mkdir(parents=True)
        (sdk / "LICENSE").write_text("SDK fixture license")
        (self.workspace / "Cargo.toml").write_text('[patch.crates-io]\nexample={path="patches/example"}\n')
        (self.workspace / "Cargo.lock").write_text(LOCK)
        package = self.root / "source/ruvector-fixture"
        package.mkdir(parents=True)
        (package / "Cargo.toml").write_text((self.workspace / "Cargo.toml").read_text())
        (package / "Cargo.lock").write_text(LOCK)
        self.archive = self.root / "source.tar.gz"
        with tarfile.open(self.archive, "w:gz") as output:
            output.add(package, arcname=package.name)
        tools = self.root / "tools"
        tools.mkdir()
        cargo = tools / "cargo"
        cargo.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
manifest = pathlib.Path(sys.argv[sys.argv.index("--manifest-path") + 1])
lock = manifest.with_name("Cargo.lock")
text = lock.read_text().split('[[package]]')[0] + '[[package]]' + lock.read_text().split('[[package]]')[1]
if os.environ.get("TEST_REGISTRY_DRIFT"):
    text = text.replace('version = "1.0.0"', 'version = "1.0.1"')
lock.write_text(text)
print("{}")
''')
        cargo.chmod(0o755)
        self.environment = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"])

    def run_helper(self):
        return subprocess.run(["bash", str(HELPER), str(self.archive), str(self.workspace)],
                              env=self.environment, capture_output=True, text=True)

    def assert_failure_preserves_archive(self, message):
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        result = self.run_helper()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, result.stderr)
        self.assertEqual(hashlib.sha256(self.archive.read_bytes()).hexdigest(), digest)

    def test_copies_patch_and_licenses_and_prunes_lock_without_upgrades(self):
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        with tarfile.open(self.archive) as archive:
            for member in ["patches/example/Cargo.toml", "patches/example/LICENSE", "LICENSE"]:
                self.assertIsNotNone(archive.extractfile("ruvector-fixture/" + member))
            lock = archive.extractfile("ruvector-fixture/Cargo.lock").read().decode()
            self.assertIn('name = "kept"', lock)
            self.assertNotIn('name = "pruned"', lock)

    def test_missing_patch_manifest(self):
        (self.workspace / "patches/example/Cargo.toml").unlink()
        self.assert_failure_preserves_archive("Patch manifest or retained license missing")

    def test_missing_patch_license(self):
        (self.workspace / "patches/example/LICENSE").unlink()
        self.assert_failure_preserves_archive("Patch manifest or retained license missing")

    def test_missing_sdk_license(self):
        (self.workspace / "crates/ruvector-py/LICENSE").unlink()
        self.assert_failure_preserves_archive("LICENSE")

    def test_registry_upgrade_is_rejected(self):
        self.environment["TEST_REGISTRY_DRIFT"] = "1"
        self.assert_failure_preserves_archive("changed locked registry packages")

if __name__ == "__main__":
    unittest.main()
