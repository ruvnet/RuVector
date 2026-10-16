#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const meta = require('../dist/metaharness');

function isolatedStatus(root, name, metadata) {
  const directory = path.join(root, name);
  fs.mkdirSync(directory);
  fs.copyFileSync(path.join(__dirname, '../dist/metaharness/index.js'), path.join(directory, 'bridge.js'));
  if (metadata) {
    const packageDir = path.join(directory, 'node_modules/@metaharness/darwin');
    fs.mkdirSync(packageDir, { recursive: true });
    fs.writeFileSync(path.join(packageDir, 'package.json'), JSON.stringify({
      name: '@metaharness/darwin', type: 'module',
      exports: { '.': { import: './index.js' } }, ...metadata,
    }));
    fs.writeFileSync(path.join(packageDir, 'index.js'), 'export function evolve() {}\n');
  }
  fs.writeFileSync(path.join(directory, 'status.js'), `
    require('./bridge').getMetaHarnessCapabilities().then(
      statuses => process.stdout.write(JSON.stringify(statuses)),
      error => { console.error(error); process.exitCode = 1; }
    );
  `);
  const env = { ...process.env };
  delete env.NODE_PATH;
  const result = spawnSync(process.execPath, [path.join(directory, 'status.js')], { encoding: 'utf8', cwd: directory, env });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

async function main() {
  const statuses = await meta.getMetaHarnessCapabilities();
  let installed = 0;
  for (const status of statuses) {
    assert.equal(status.baselineVersion, meta.METAHARNESS_VERSIONS[status.name]);
    const location = (require.resolve.paths(status.package) || [])
      .map(directory => path.join(directory, status.package, 'package.json'))
      .find(file => fs.existsSync(file));
    if (location) {
      const metadata = JSON.parse(fs.readFileSync(location, 'utf8'));
      assert.equal(status.version, metadata.version, `${status.package} must report its actual installed version`);
      installed++;
    } else {
      assert.equal(status.version, 'unknown');
    }
  }
  assert.ok(installed > 0, 'Install optional MetaHarness packages to run the real-package version check');

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-meta-versions-'));
  try {
    const missing = isolatedStatus(root, 'missing');
    assert.equal(missing.length, 9);
    for (const status of missing) {
      assert.equal(status.version, 'unknown');
      assert.equal(status.available, false);
      assert.equal(status.baselineVersion, meta.METAHARNESS_VERSIONS[status.name]);
    }
    // Metadata remains readable when an ESM package deliberately blocks the
    // package.json subpath and has no require export, as real MetaHarness does.
    const newer = isolatedStatus(root, 'newer', { version: '0.10.3' }).find(status => status.name === 'darwin');
    assert.equal(newer.version, '0.10.3');
    assert.equal(newer.baselineVersion, '0.9.2');
    assert.equal(newer.available, true);

    const unknown = isolatedStatus(root, 'unknown-version', {}).find(status => status.name === 'darwin');
    assert.equal(unknown.version, 'unknown');
    assert.equal(unknown.available, true, 'Unknown metadata must not change runtime export checks');

    const mismatched = isolatedStatus(root, 'wrong-identity', { name: 'some-other-package', version: '9.9.9' })
      .find(status => status.name === 'darwin');
    assert.equal(mismatched.version, 'unknown');
    assert.equal(mismatched.available, true);
    console.log(`PASS MetaHarness versions: ${installed} real installed packages, missing metadata and import-only exports`);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
