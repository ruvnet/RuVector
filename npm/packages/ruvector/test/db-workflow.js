#!/usr/bin/env node

/**
 * End-to-end CLI database workflow guard — regression suite for #508.
 *
 * The stats/search/insert commands shipped for multiple releases calling
 * VectorDB methods that do not exist (db.load/db.save/db.stats), because
 * the CLI test suite only checked help text and exit codes of --help paths.
 * This suite runs the real workflow a user runs:
 *
 *   demo --basic  ->  stats  ->  search  ->  insert  ->  stats again
 *
 * in a temp directory, against the actual native/wasm implementation, and
 * asserts on real output values. Export must fail HONESTLY (no enumeration
 * API exists), never write a fake backup.
 *
 * Skips (exit 0 with a notice) when no vector implementation is available,
 * mirroring the behavior of the CLI itself in restricted environments.
 */

const { execSync } = require('child_process');
const assert = require('assert');
const path = require('path');
const fs = require('fs');
const os = require('os');

const CLI = path.join(__dirname, '..', 'bin', 'cli.js');

let passed = 0;
let failed = 0;

function run(args, opts = {}) {
  try {
    const stdout = execSync(`node "${CLI}" ${args}`, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
      timeout: 120000,
      ...opts,
    });
    return { code: 0, out: stdout };
  } catch (e) {
    return { code: e.status ?? 1, out: `${e.stdout || ''}${e.stderr || ''}` };
  }
}

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL  ${name}\n        ${e.message.split('\n')[0]}`);
  }
}

// ---------------------------------------------------------------------------

const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-db-workflow-'));
const cwd = { cwd: workDir };

console.log('ruvector CLI database workflow tests (#508 guard)');
console.log('='.repeat(60));

// Probe: is a vector implementation available at all?
const probe = run('demo --basic', cwd);
if (probe.code !== 0 && /implementation|not available|Cannot find module/i.test(probe.out)) {
  console.log('  SKIP  no vector implementation available in this environment');
  process.exit(0);
}

test('demo --basic creates ./demo.db and exits 0', () => {
  assert.strictEqual(probe.code, 0, `demo exited ${probe.code}: ${probe.out.slice(-200)}`);
  assert.ok(fs.existsSync(path.join(workDir, 'demo.db')), 'demo.db not created');
  assert.ok(fs.existsSync(path.join(workDir, 'demo.db.meta.json')), 'dimension sidecar not created');
});

test('stats opens the demo db with the right dimension and count', () => {
  const r = run('stats ./demo.db', cwd);
  assert.strictEqual(r.code, 0, `stats exited ${r.code}: ${r.out.slice(-200)}`);
  assert.match(r.out, /Vector Count:\s*\S*4/, `expected count 4 in:\n${r.out}`);
  assert.match(r.out, /Dimension:\s*\S*4\b/, `expected dimension 4 in:\n${r.out}`);
});

test('search returns the correct nearest neighbor', () => {
  const r = run('search ./demo.db --vector "[0.8, 0.6, 0, 0]" -k 3', cwd);
  assert.strictEqual(r.code, 0, `search exited ${r.code}: ${r.out.slice(-200)}`);
  // vec4 = [0.7, 0.7, 0, 0] is the closest by cosine to [0.8, 0.6, 0, 0]
  const firstResult = r.out.split('\n').find(l => /1\.\s+ID:/.test(l)) || '';
  assert.match(firstResult, /vec4/, `expected vec4 first in:\n${r.out}`);
});

test('insert adds vectors to an existing db and persists', () => {
  const vecFile = path.join(workDir, 'vecs.json');
  fs.writeFileSync(vecFile, JSON.stringify([
    { vector: [0.1, 0.2, 0.3, 0.4], metadata: { label: 't1' } },
  ]));
  const r = run('insert ./demo.db vecs.json', cwd);
  assert.strictEqual(r.code, 0, `insert exited ${r.code}: ${r.out.slice(-200)}`);

  const after = run('stats ./demo.db', cwd);
  assert.match(after.out, /Vector Count:\s*\S*5/, `expected count 5 after insert in:\n${after.out}`);
});

test('insert creates a NEW db and writes the dimension sidecar', () => {
  const vecFile = path.join(workDir, 'vecs8.json');
  fs.writeFileSync(vecFile, JSON.stringify([
    { vector: [1, 0, 0, 0, 0, 0, 0, 0], metadata: { label: 'dim8' } },
  ]));
  const r = run('insert ./fresh.db vecs8.json', cwd);
  assert.strictEqual(r.code, 0, `insert exited ${r.code}: ${r.out.slice(-200)}`);
  const meta = JSON.parse(fs.readFileSync(path.join(workDir, 'fresh.db.meta.json'), 'utf8'));
  assert.strictEqual(meta.dimension, 8, 'sidecar dimension should come from the data');
  const stats = run('stats ./fresh.db', cwd);
  assert.match(stats.out, /Dimension:\s*\S*8\b/, `expected dimension 8 in:\n${stats.out}`);
});

test('export fails honestly instead of writing a fake backup', () => {
  const r = run('export ./demo.db -o out.json', cwd);
  assert.notStrictEqual(r.code, 0, 'export must exit non-zero (no enumeration API exists)');
  assert.ok(!fs.existsSync(path.join(workDir, 'out.json')), 'export must not write a vector-less file');
  assert.match(r.out, /not supported|portable artifact/i, `expected honest message in:\n${r.out}`);
});

test('stats/search on a missing db fail cleanly (no phantom-API error)', () => {
  const r = run('stats ./nope.db', cwd);
  assert.notStrictEqual(r.code, 0);
  assert.ok(!/is not a function/.test(r.out), `phantom API error resurfaced:\n${r.out}`);
  assert.match(r.out, /not found/i, `expected clean not-found message in:\n${r.out}`);
});

// ---------------------------------------------------------------------------

fs.rmSync(workDir, { recursive: true, force: true });

console.log('='.repeat(60));
console.log(`Passed:  ${passed}`);
console.log(`Failed:  ${failed}`);
process.exit(failed > 0 ? 1 : 0);
