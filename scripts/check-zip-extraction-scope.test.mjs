import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

const script = await readFile(new URL("./check-zip-extraction-scope.mjs", import.meta.url), "utf8");
const lock = await readFile(new URL("../Cargo.lock", import.meta.url), "utf8");

async function runGuard(candidateLock) {
  const root = await mkdtemp(join(tmpdir(), "ruvector-zip-guard-test-"));
  try {
    await mkdir(join(root, "scripts"));
    await writeFile(join(root, "scripts", "guard.mjs"), script);
    await writeFile(join(root, "Cargo.lock"), candidateLock);
    return spawnSync(process.execPath, [join(root, "scripts", "guard.mjs")], {
      encoding: "utf8", timeout: 60000,
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

for (const dependency of [
  "zip 1.1.4",
  "zip 1.1.4 (registry+https://github.com/rust-lang/crates.io-index)",
]) {
  test("rejects an additional ZIP caller: " + dependency, async () => {
    const changed = lock + `\n[[package]]\nname = "new-extractor"\nversion = "0.1.0"\ndependencies = [\n "${dependency}",\n]\n`;
    const result = await runGuard(changed);
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /Unexpected reverse dependency for zip 1\.1\.4/);
  });
}

test("recognizes a source-qualified existing caller", async () => {
  const changed = lock.replace('"zip 1.1.4",',
    '"zip 1.1.4 (registry+https://github.com/rust-lang/crates.io-index)",');
  assert.notEqual(changed, lock);
  const result = await runGuard(changed);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /PASS: 126 parent Rust source files scanned/);
});

test("rejects ZIP checksum drift", async () => {
  const result = await runGuard(lock.replace(
    "9cc23c04387f4da0374be4533ad1208cbb091d5c11d070dfef13676ad6497164", "0".repeat(64)));
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /ZIP checksum changed: 1\.1\.4/);
});

test("rejects an additional ZIP version", async () => {
  const result = await runGuard(lock + '\n[[package]]\nname = "zip"\nversion = "9.0.5"\n');
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /ZIP lock graph changed/);
});
