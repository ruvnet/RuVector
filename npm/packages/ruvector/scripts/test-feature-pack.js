#!/usr/bin/env node
'use strict';

// Test the installable tarball in a fresh consumer, outside the workspace.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-feature-pack-'));
const consumer = path.join(scratch, 'consumer');
const env = { ...process.env };
delete env.NODE_PATH;
delete env.KGE_BACKEND;

function run(command, args, cwd = consumer, extraEnv = {}) {
  const result = spawnSync(command, args, {
    cwd, env: { ...env, ...extraEnv }, encoding: 'utf8', timeout: 300000,
    maxBuffer: 16 * 1024 * 1024,
  });
  assert.equal(result.status, 0, `${command} ${args.join(' ')}\n${result.stdout}\n${result.stderr}\n${result.error || ''}`);
  return result.stdout;
}

try {
  fs.mkdirSync(consumer);
  fs.writeFileSync(path.join(consumer, 'package.json'), JSON.stringify({ name: 'feature-consumer', version: '1.0.0', private: true }));
  run('npm', ['pack', '--workspaces=false', '--pack-destination', scratch], root);
  const tarball = fs.readdirSync(scratch).find(file => file.endsWith('.tgz'));
  assert.ok(tarball, 'npm pack must produce an artifact');
  run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', path.join(scratch, tarball)]);
  run(process.execPath, ['-e', `
    const assert = require('node:assert/strict');
    const { VectorDB, isNative, createRuViewIntegration, createGraphReasoner, validateRvfArtifact } = require('ruvector');
    (async () => {
      for (const name of ['@ruvector/kge', '@ruvector/rvforge', '@ruvnet/ruview']) {
        assert.throws(() => require.resolve(name), { code: 'MODULE_NOT_FOUND' });
      }
      assert.equal(isNative(), true, 'core must use its real native backend');
      const db = new VectorDB({ dimensions: 3, storagePath: './core.db' });
      await db.insert({ id: 'core-only', vector: [1, 0, 0] });
      assert.equal((await db.search({ vector: [1, 0, 0], k: 1 }))[0].id, 'core-only');
      await assert.rejects(createRuViewIntegration(), { code: 'ERR_INTEGRATION_MISSING' });
      await assert.rejects(createGraphReasoner(), { code: 'ERR_INTEGRATION_MISSING' });
      await assert.rejects(validateRvfArtifact('unused.rvf', { expectedSha256: 'a'.repeat(64) }), { code: 'ERR_INTEGRATION_MISSING' });
    })().catch(error => { console.error(error); process.exitCode = 1; });
  `]);
  console.log('PASS packed core: native insert/search; new feature peers absent on ordinary install');

  run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', '@ruvector/kge@0.2.0', '@ruvector/rvforge@0.2.0', '@ruvnet/ruview@0.9.1']);
  const installed = path.join(consumer, 'node_modules/ruvector');
  const tests = ['kge-integration.js', 'rvforge-integration.js', 'ruview-integration.js', 'metaharness-versions.js'];
  fs.mkdirSync(path.join(installed, 'test'), { recursive: true });
  for (const file of tests) fs.copyFileSync(path.join(root, 'test', file), path.join(installed, 'test', file));
  process.stdout.write(run(process.execPath, ['--test', ...tests.map(file => `test/${file}`)], installed));
  process.stdout.write(run(process.execPath, ['--test', 'test/kge-integration.js'], installed, { KGE_BACKEND: 'wasm' }));
  // These source consumers must work with the packed main SDK too.
  const downstream = [
    '../agentic-synth/native-tests/ruvector.js',
    '../ruvector-extensions/native-tests/persistence.js',
  ].map(file => path.resolve(root, file));
  process.stdout.write(run(process.execPath, ['--test', ...downstream], consumer, { RUVECTOR_NATIVE_TEST_PREFIX: consumer }));
  console.log('PASS packed integrations and downstream native consumers');
} finally {
  fs.rmSync(scratch, { recursive: true, force: true });
}
