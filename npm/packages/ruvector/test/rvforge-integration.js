#!/usr/bin/env node
'use strict';

// Functional checks use the installed, published RVForge writer and validator.
// Only unavailable/broken dependency contracts use synthetic package exports.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const { validateRvfArtifact, stageRvfArtifact } = require('../dist');

function run(script, args, options = {}) {
  const result = spawnSync(process.execPath, [script, ...args], { encoding: 'utf8', ...options });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}\n${result.error || ''}`);
  return result;
}

function dependencyContract(root, name, body, expected) {
  const isolated = path.join(root, name);
  fs.mkdirSync(isolated);
  for (const filename of ['rvforge.js', 'optional-package.js']) {
    fs.copyFileSync(path.join(__dirname, '../dist/integrations', filename), path.join(isolated, filename));
  }
  if (body !== null) {
    const packageDir = path.join(isolated, 'node_modules/@ruvector/rvforge');
    fs.mkdirSync(packageDir, { recursive: true });
    fs.writeFileSync(path.join(packageDir, 'package.json'), JSON.stringify({ name: '@ruvector/rvforge', main: 'index.js' }));
    fs.writeFileSync(path.join(packageDir, 'index.js'), body);
  }
  const check = path.join(isolated, 'check.js');
  fs.writeFileSync(check, `
    const assert = require('node:assert/strict');
    const api = require('./rvforge');
    api.validateRvfArtifact('unused.rvf', { expectedSha256: 'a'.repeat(64) }).then(
      () => { throw new Error('Expected integration failure'); },
      error => {
        assert.equal(error.code, ${JSON.stringify(expected)});
        assert.equal(error.packageName, '@ruvector/rvforge');
        if (error.code === 'ERR_INTEGRATION_MISSING') assert.ok(error.message.includes('npm install --save-dev @ruvector/rvforge'));
      }
    ).catch(error => { console.error(error); process.exitCode = 1; });
  `);
  const env = { ...process.env };
  delete env.NODE_PATH;
  run(check, [], { cwd: isolated, env });
}

async function main() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-rvforge-test-'));
  try {
    const forgeEntry = require.resolve('@ruvector/rvforge');
    const cli = path.join(path.dirname(forgeEntry), 'cli.js');
    const fixtureDir = path.join(root, 'fixture');
    fs.mkdirSync(fixtureDir);
    run(cli, ['init', '--keygen'], { cwd: fixtureDir });
    const payloadDir = path.join(fixtureDir, 'payload');
    fs.mkdirSync(payloadDir);
    const payload = Buffer.from('Canonical RVF integration payload; never executed.');
    fs.writeFileSync(path.join(payloadDir, 'data.txt'), payload);
    run(cli, ['create', 'artifact.rvf', '--from', payloadDir], { cwd: fixtureDir });
    const rvfPath = path.join(fixtureDir, 'artifact.rvf');
    const bytes = fs.readFileSync(rvfPath);
    const expectedSha256 = crypto.createHash('sha256').update(bytes).digest('hex');
    const validation = await validateRvfArtifact(rvfPath, { expectedSha256: expectedSha256.toUpperCase() });
    assert.equal(validation.sha256, expectedSha256);
    assert.equal(validation.size, bytes.length);
    assert.equal(validation.formatVersion, 1);
    assert.equal(validation.signaturePresent, true); // Presence, not authentication.
    assert.equal(validation.expectedSha256Matched, true);
    assert.ok(validation.segmentCount >= 2);

    await assert.rejects(validateRvfArtifact(rvfPath), { code: 'ERR_INTEGRATION_INVALID' });
    await assert.rejects(validateRvfArtifact(rvfPath, { expectedSha256: 'not a digest' }), { code: 'ERR_INTEGRATION_INVALID' });
    await assert.rejects(validateRvfArtifact(rvfPath, { expectedSha256, allowUnsigned: 'false' }), { code: 'ERR_INTEGRATION_INVALID' });
    await assert.rejects(validateRvfArtifact(path.join(root, 'absent.rvf'), { expectedSha256 }), { code: 'FORGE_E_IO' });

    const changed = Buffer.from(bytes);
    const payloadOffset = changed.indexOf(payload);
    assert.ok(payloadOffset >= 0, 'Real RVForge writer must embed the supplied payload');
    changed[payloadOffset] ^= 1;
    const changedPath = path.join(root, 'changed.rvf');
    fs.writeFileSync(changedPath, changed);
    await assert.rejects(validateRvfArtifact(changedPath, { expectedSha256 }), { code: 'ERR_INTEGRATION_INVALID' });

    const malformed = Buffer.from(bytes);
    malformed[malformed.length - 1] ^= 1; // Root CRC corruption.
    const malformedPath = path.join(root, 'malformed.rvf');
    fs.writeFileSync(malformedPath, malformed);
    await assert.rejects(validateRvfArtifact(malformedPath, { expectedSha256 }), { code: 'FORGE_E_INVALID_RVF' });

    const outDir = path.join(root, 'bundles');
    const options = {
      rvfPath, expectedSha256, outDir,
      app: { name: 'RVF integration test', version: '1.0.0', publisher: 'Test', identifier: 'io.ruvector.test' },
      targets: ['linux-x64', 'windows-x64'],
      runtime: { profile: 'wasm', rvmVersion: '0.1.0', rvmCommit: 'test-revision' },
    };
    const staged = await stageRvfArtifact(options);
    assert.equal(staged.status, 'staged');
    assert.equal(staged.packaged, false, 'RVForge 0.2.0 does not produce installers');
    assert.equal(staged.sha256, expectedSha256);
    assert.equal(staged.bundles.length, 2);
    for (const bundle of staged.bundles) {
      assert.deepEqual(fs.readFileSync(path.join(outDir, bundle.rvfPath)), bytes);
      const reader = JSON.parse(fs.readFileSync(path.join(outDir, bundle.readerSlotPath), 'utf8'));
      assert.equal(reader.status, 'placeholder');
    }
    for (const file of ['build-manifest.json', 'provenance.json', 'inventory.json', 'checksums.txt']) {
      assert.ok(fs.statSync(path.join(outDir, file)).isFile());
    }
    await assert.rejects(stageRvfArtifact(options), { code: 'FORGE_E_IO' });
    const corruptOut = path.join(root, 'must-not-exist');
    await assert.rejects(stageRvfArtifact({ ...options, rvfPath: changedPath, outDir: corruptOut }), { code: 'ERR_INTEGRATION_INVALID' });
    assert.equal(fs.existsSync(corruptOut), false);

    dependencyContract(root, 'missing', null, 'ERR_INTEGRATION_MISSING');
    dependencyContract(root, 'wrong-api', 'module.exports = {};', 'ERR_INTEGRATION_API');
    dependencyContract(root, 'broken-load', "throw new Error('broken installation');", 'ERR_INTEGRATION_LOAD');
    dependencyContract(root, 'wrong-result', 'exports.validateRvf = async () => ({ok:true, deep:true});', 'ERR_INTEGRATION_API');
    console.log('PASS RVForge: real signed RVF validation, corruption rejection, two-target staging, optional dependency failures');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
