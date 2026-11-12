#!/usr/bin/env node
/**
 * ruflo ADR-150 compliance guard (MetaHarness Integration Surfaces).
 *
 * Rule 2 (verbatim): "@metaharness/* packages MUST appear in optionalDependencies
 * or peerDependencies (optional), never in dependencies".
 * Rule 4: the package must keep working on an --ignore-optional install path.
 *
 * Part (a) parses package.json and asserts the manifest shape.
 * Part (b) simulates a MetaHarness-less install: the compiled bridge is copied to a
 * scratch directory with no reachable node_modules, so every lazy import() fails the
 * way it would after `npm install --ignore-optional`, and the bridge must degrade
 * gracefully with a clear error naming the missing optional package.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const META_PACKAGES = [
  'metaharness',
  '@metaharness/darwin',
  '@metaharness/flywheel',
  '@metaharness/harness',
  '@metaharness/redblue',
  '@metaharness/router',
  '@metaharness/weight-eft',
  '@metaharness/workspace-lens',
  '@metaharness/workspace-probe',
];

function checkManifest() {
  const manifestPath = path.join(__dirname, '..', 'package.json');
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const dependencies = manifest.dependencies || {};
  const optionalDependencies = manifest.optionalDependencies || {};
  const peerDependencies = manifest.peerDependencies || {};
  const peerDependenciesMeta = manifest.peerDependenciesMeta || {};

  for (const name of META_PACKAGES) {
    assert.ok(
      !(name in dependencies),
      `${name} must never be a hard dependency (ruflo ADR-150 rule 2)`,
    );
    const inOptional = name in optionalDependencies;
    const inPeer = name in peerDependencies;
    assert.ok(
      inOptional || inPeer,
      `${name} must be declared in optionalDependencies or peerDependencies (ruflo ADR-150 rule 2)`,
    );
    if (inPeer) {
      assert.strictEqual(
        peerDependenciesMeta[name] && peerDependenciesMeta[name].optional,
        true,
        `${name} peer dependency must be marked optional in peerDependenciesMeta (ruflo ADR-150 rule 2)`,
      );
    }
  }
  console.log(`  PASS  all ${META_PACKAGES.length} MetaHarness packages are optional-only in package.json`);
}

async function checkAbsenceDegradation() {
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-adr150-'));
  try {
    // The bridge module is self-contained CJS; loading a copy from a directory with
    // no node_modules on its resolution path makes every lazy import() fail exactly
    // as it does on an --ignore-optional install.
    const bridgeSource = path.join(__dirname, '..', 'dist', 'metaharness', 'index.js');
    const bridgeCopy = path.join(scratch, 'metaharness-bridge.js');
    fs.copyFileSync(bridgeSource, bridgeCopy);
    const bridge = require(bridgeCopy);

    const capabilities = await bridge.getMetaHarnessCapabilities();
    assert.strictEqual(capabilities.length, META_PACKAGES.length);
    assert.deepStrictEqual(
      capabilities.filter((capability) => capability.available),
      [],
      'no MetaHarness capability may report available when the packages are absent',
    );
    console.log('  PASS  getMetaHarnessCapabilities degrades to available:false without MetaHarness');

    await assert.rejects(
      () =>
        bridge.routeWithMetaHarness({
          rows: [{ embedding: [1, 0], scores: { cheap: 0.95 } }],
          prices: { cheap: 1 },
          queryEmbedding: [1, 0],
        }),
      (error) =>
        error.code === 'ERR_METAHARNESS_OPTIONAL_MISSING' &&
        error.message.includes('@metaharness/router'),
      'a MetaHarness-backed call must fail with a clear error naming the missing optional package',
    );
    console.log('  PASS  missing-package calls throw ERR_METAHARNESS_OPTIONAL_MISSING naming the package');
  } finally {
    fs.rmSync(scratch, { recursive: true, force: true });
  }
}

async function main() {
  console.log('\nruflo ADR-150 optional-dependency compliance guard');
  console.log('-'.repeat(50));
  checkManifest();
  await checkAbsenceDegradation();
  console.log('ADR-150 optional-dependency compliance checks passed');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
