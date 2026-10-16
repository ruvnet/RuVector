'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const dist = path.resolve(__dirname, '../dist');
function isolated(t, packageFiles) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-optional-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  fs.cpSync(path.join(dist, 'integrations'), path.join(dir, 'integrations'), { recursive: true });
  for (const [file, contents] of Object.entries(packageFiles || {})) {
    const target = path.join(dir, 'node_modules', file);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, contents);
  }
  return require(path.join(dir, 'integrations/optional-package.js'));
}

test('importing the core SDK does not resolve or load feature packages', () => {
  const child = spawnSync(process.execPath, ['-e', `
    const Module = require('node:module');
    const feature = name => ['@ruvector/kge', '@ruvector/rvforge', '@ruvnet/ruview'].some(p => name === p || name.startsWith(p + '/'));
    for (const method of ['_load', '_resolveFilename']) {
      const original = Module[method];
      Module[method] = function(name, ...args) {
        if (feature(name)) throw new Error('eager feature dependency: ' + name);
        return original.call(this, name, ...args);
      };
    }
    const sdk = require(${JSON.stringify(dist)});
    for (const name of ['createRuViewIntegration', 'createGraphReasoner', 'validateRvfArtifact']) {
      if (typeof sdk[name] !== 'function') throw new Error('missing core export ' + name);
    }
  `], { encoding: 'utf8', timeout: 15000 });
  assert.equal(child.status, 0, child.stderr || child.stdout);
});

test('missing package gives explicit install command, including release-only dev scope', async t => {
  const { loadOptionalPackage } = isolated(t);
  await assert.rejects(loadOptionalPackage('@ruvnet/ruview', '@ruvnet/ruview/sdk'), e =>
    e.code === 'ERR_INTEGRATION_MISSING' && e.message.includes('npm install @ruvnet/ruview'));
  await assert.rejects(loadOptionalPackage('@ruvector/rvforge', undefined, true), e =>
    e.code === 'ERR_INTEGRATION_MISSING' && e.message.includes('npm install --save-dev @ruvector/rvforge'));
});

test('installed but broken package remains a load error with its original cause', async t => {
  const { loadOptionalPackage } = isolated(t, {
    '@ruvnet/ruview/package.json': JSON.stringify({ name: '@ruvnet/ruview', exports: { './sdk': './sdk.js' } }),
    '@ruvnet/ruview/sdk.js': "require('ruvector-deliberately-missing-transitive');",
  });
  await assert.rejects(loadOptionalPackage('@ruvnet/ruview', '@ruvnet/ruview/sdk'), e =>
    e.code === 'ERR_INTEGRATION_LOAD' && e.cause?.code === 'MODULE_NOT_FOUND');
});

test('unsupported package entry is an API compatibility error', async t => {
  const { loadOptionalPackage } = isolated(t, {
    '@ruvnet/ruview/package.json': JSON.stringify({ name: '@ruvnet/ruview', exports: { '.': './index.js' } }),
    '@ruvnet/ruview/index.js': 'module.exports = {};',
  });
  await assert.rejects(loadOptionalPackage('@ruvnet/ruview', '@ruvnet/ruview/sdk'), e =>
    e.code === 'ERR_INTEGRATION_API' && e.cause?.code === 'ERR_PACKAGE_PATH_NOT_EXPORTED');
});

test('new feature packages are optional peers, not default runtime installs', () => {
  const manifest = require('../package.json');
  for (const name of ['@ruvector/kge', '@ruvector/rvforge', '@ruvnet/ruview']) {
    assert.equal(manifest.dependencies?.[name], undefined);
    assert.equal(manifest.optionalDependencies?.[name], undefined);
    assert.equal(manifest.peerDependenciesMeta[name].optional, true);
    assert.ok(manifest.peerDependencies[name]);
  }
  assert.equal(manifest.devDependencies['@ruvector/rvforge'], '0.2.0');
});


test('installed package with missing entry is not misreported as an absent peer', async t => {
  const { loadOptionalPackage } = isolated(t, {
    '@ruvnet/ruview/package.json': JSON.stringify({ name: '@ruvnet/ruview', exports: { './sdk': './missing.js' } }),
  });
  await assert.rejects(loadOptionalPackage('@ruvnet/ruview', '@ruvnet/ruview/sdk'), e =>
    e.code === 'ERR_INTEGRATION_API' && e.cause?.code === 'MODULE_NOT_FOUND');
});
