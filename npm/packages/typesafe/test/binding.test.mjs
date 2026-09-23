// Binding smoke tests — run against both backends (node --test).
// The core `Engine::decide` may still be the "not implemented yet" stub when
// these run; the decide assertions accept EITHER a real DecisionResponse OR the
// documented error JSON, and tighten automatically once answers appear.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync, existsSync, mkdtempSync, mkdirSync, copyFileSync, writeFileSync, rmSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';

const require = createRequire(import.meta.url);
const testDir = dirname(fileURLToPath(import.meta.url));
const pkgDir = join(testDir, '..');
const repoRoot = join(pkgDir, '..', '..', '..');
const indexPath = join(pkgDir, 'index.js');

// Cargo workspace version from the root Cargo.toml [workspace.package].
function workspaceVersion() {
  const toml = readFileSync(join(repoRoot, 'Cargo.toml'), 'utf8');
  const section = toml.slice(toml.indexOf('[workspace.package]'));
  const m = section.match(/version\s*=\s*"([^"]+)"/);
  assert.ok(m, 'workspace version found in Cargo.toml');
  return m[1];
}
const WORKSPACE_VERSION = workspaceVersion();

// Load index.js fresh with a chosen backend, clearing the require cache so the
// backend selection (which index.js decides at load time) re-runs.
function loadBackend(forced) {
  for (const key of Object.keys(require.cache)) delete require.cache[key];
  if (forced) process.env.TYPESAFE_BACKEND = forced;
  else delete process.env.TYPESAFE_BACKEND;
  return require(indexPath);
}

const nativeBuilt = ['linux-x64-gnu', 'linux-arm64-gnu', 'darwin-x64', 'darwin-arm64', 'win32-x64-msvc']
  .some((t) => existsSync(join(pkgDir, 'native', `typesafe.${t}.node`)) ||
    existsSync(join(pkgDir, 'native', t, `typesafe.${t}.node`)));
const wasmBuilt = existsSync(join(pkgDir, 'wasm', 'ruvector_typesafe_wasm.js'));

const twoOptionRequest = JSON.stringify({
  state: 'my card was charged twice, please help',
  questions: {
    dept: {
      type: 'choice',
      instructions: 'route to a department',
      criteria: { billing: 'charges and refunds', fraud: 'unauthorised card use' },
    },
  },
});

function runContract(mod, expectedBackend) {
  assert.equal(typeof mod.version, 'function', 'exports version()');
  assert.equal(typeof mod.Engine, 'function', 'exports Engine class');
  assert.equal(mod.backend, expectedBackend, `backend is ${expectedBackend}`);
  assert.equal(mod.version(), WORKSPACE_VERSION, 'version() matches Cargo workspace version');

  const engine = new mod.Engine('{"embedder":"hash","dims":128}');

  // stats
  const stats = JSON.parse(engine.statsJson());
  assert.equal(stats.dims, 128, 'stats reports the configured dims');
  assert.equal(typeof stats.embedderId, 'string', 'stats has embedderId');

  // decide: either a real response or the documented error JSON.
  const res = JSON.parse(engine.decideJson(twoOptionRequest));
  if ('answers' in res) {
    assert.ok(res.usage, 'response carries usage');
    assert.ok(res.answers.dept, 'response has the dept answer');
    assert.ok('choice' in res.answers.dept, 'choice answer exposes .choice');
  } else {
    assert.ok(res.error, 'error JSON present');
    assert.ok(['limit', 'invalid', 'embedder'].includes(res.error.kind), 'known error kind');
    assert.equal(typeof res.error.message, 'string', 'error has a message');
  }

  // train: report-or-error shape (the sole &mut method on the wasm backend).
  const trained = JSON.parse(
    engine.trainJson('{"question":"dept","examples":[{"text":"refund my duplicate charge","label":"billing"}]}'),
  );
  if ('error' in trained) {
    assert.ok(['limit', 'invalid', 'embedder'].includes(trained.error.kind), 'train error kind');
  } else {
    assert.equal(trained.question, 'dept', 'train report echoes the question id');
    assert.equal(typeof trained.accepted, 'number', 'train report has an accepted count');
  }

  // Bad options JSON is a programmer error -> throws.
  assert.throws(() => new mod.Engine('{ not valid json'), 'bad options JSON throws');

  // >16 KiB state -> limit error (implemented in the core today).
  const big = JSON.stringify({
    state: 'x'.repeat(16 * 1024 + 1),
    questions: { q: { type: 'choice', criteria: { a: 'alpha', b: 'beta' } } },
  });
  const limit = JSON.parse(engine.decideJson(big));
  assert.ok(limit.error, 'oversized state returns an error');
  assert.equal(limit.error.kind, 'limit', 'oversized state -> kind "limit"');
}

test('native backend contract', { skip: !nativeBuilt && 'native binary not built' }, () => {
  const mod = loadBackend(undefined);
  runContract(mod, 'native');
});

test('native async decide resolves to JSON', {
  skip: !nativeBuilt && 'native binary not built',
}, async () => {
  const mod = loadBackend(undefined);
  const engine = new mod.Engine('{"embedder":"hash","dims":64}');
  assert.equal(typeof engine.decide, 'function', 'native exposes async decide');
  const out = JSON.parse(await engine.decide(twoOptionRequest));
  assert.ok('answers' in out || out.error, 'async decide returns response-or-error JSON');
});

test('wasm backend contract', { skip: !wasmBuilt && 'wasm module not built' }, () => {
  const mod = loadBackend('wasm');
  runContract(mod, 'wasm');
});

test('default binding preserves a present native binary load error and tolerates absent artifacts', {
  skip: !({ linux: ['x64', 'arm64'], darwin: ['x64', 'arm64'], win32: ['x64'] }[process.platform]
    ?.includes(process.arch)) && 'unsupported platform',
}, () => {
  const triple = {
    linux: { x64: 'linux-x64-gnu', arm64: 'linux-arm64-gnu' },
    darwin: { x64: 'darwin-x64', arm64: 'darwin-arm64' },
    win32: { x64: 'win32-x64-msvc' },
  }[process.platform][process.arch];
  const temp = mkdtempSync(join(tmpdir(), 'typesafe-binding-'));
  const nativeFile = join(temp, 'native', `typesafe.${triple}.node`);
  const pairedFile = join(temp, 'native', triple, `typesafe.${triple}.node`);
  const script = `
    const { resolveDefaultBinding } = require('./dist/binding.js');
    try {
      process.stdout.write(JSON.stringify({ binding: resolveDefaultBinding() }));
    } catch (error) {
      process.stdout.write(JSON.stringify({ code: error.code, message: error.message }));
    }
  `;
  const run = () => {
    const child = spawnSync(process.execPath, ['-e', script], {
      cwd: temp,
      encoding: 'utf8',
      env: { ...process.env, TYPESAFE_BACKEND: '' },
    });
    assert.equal(child.status, 0, child.stderr);
    return JSON.parse(child.stdout);
  };
  try {
    mkdirSync(join(temp, 'native'));
    mkdirSync(join(temp, 'dist'));
    copyFileSync(indexPath, join(temp, 'index.js'));
    copyFileSync(join(pkgDir, 'dist', 'binding.js'), join(temp, 'dist', 'binding.js'));

    writeFileSync(nativeFile, 'invalid native addon');
    const broken = run();
    assert.equal(broken.code, 'ERR_DLOPEN_FAILED');
    assert.match(broken.message, /typesafe\..*\.node/);

    if (process.platform === 'linux') {
      mkdirSync(dirname(pairedFile));
      writeFileSync(pairedFile, 'invalid paired native addon');
      const paired = run();
      assert.equal(paired.code, 'ERR_DLOPEN_FAILED');
      assert.match(paired.message, new RegExp(`native/${triple}/typesafe\\.${triple}\\.node`));
      rmSync(pairedFile);
    }

    rmSync(nativeFile);
    assert.deepEqual(run(), { binding: null }, 'both missing artifacts are a normal absence');
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
});
