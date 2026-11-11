#!/usr/bin/env node
/**
 * `ruvector weights` and the weights_* MCP tools against the real
 * @ruvector/weight-space. Prints SKIP and exits 0 when the optional peer is not
 * installed (CI installs it from the sibling package; see ruvector-npm-ci.yml).
 *
 * Uses explicit --embedding values only, so nothing is downloaded.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn, spawnSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const CLI = path.join(ROOT, 'bin', 'cli.js');

let wsPath;
try {
  wsPath = require.resolve('@ruvector/weight-space', { paths: [ROOT] });
} catch {
  console.log('  SKIP  @ruvector/weight-space not installed (npm i @ruvector/weight-space@0.1.0)');
  process.exit(0);
}

const work = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-ws-cli-')));
const outside = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-ws-outside-')));
const ENV = { ...process.env, HOME: work, NO_COLOR: '1' };
const REGISTRY = path.join(work, '.ruvector', 'weights.jsonl');

function model(scale) {
  return {
    schema: 'ruv-fann.dense-layers/1',
    layers: [
      { inputs: 3, outputs: 2, weights: [0.1, -0.2, 0.3, 0.4, -0.5, 0.6].map((w) => w * scale), bias: [0.01, -0.02], activation: 'Tanh', steepness: 1 },
      { inputs: 2, outputs: 1, weights: [0.7, -0.8], bias: [0.05], activation: 'Linear', steepness: 1 },
    ],
  };
}

function cli(args) {
  const r = spawnSync(process.execPath, [CLI, ...args], { cwd: work, env: ENV, encoding: 'utf8', timeout: 60000 });
  let json = null;
  try { json = JSON.parse(r.stdout); } catch { /* human output */ }
  return { status: r.status, stdout: r.stdout, stderr: r.stderr, json };
}

function ok(r, what) {
  assert.strictEqual(r.status, 0, `${what} failed (exit ${r.status}): ${r.stdout}\n${r.stderr}`);
  assert.ok(r.json && r.json.success === true, `${what}: expected success JSON, got ${r.stdout}`);
  return r.json;
}

let passed = 0;
let failed = 0;
async function test(name, fn) {
  try { await fn(); passed++; console.log(`  PASS  ${name}`); }
  catch (e) { failed++; console.log(`  FAIL  ${name}\n        ${e.stack || e.message}`); }
}

function mcpSession() {
  const child = spawn(process.execPath, [CLI, 'mcp', 'start'], { cwd: work, env: ENV, stdio: ['pipe', 'pipe', 'pipe'] });
  let nextId = 1;
  let buffer = '';
  const pending = new Map();
  child.stdout.on('data', (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.trim()) continue;
      let msg;
      try { msg = JSON.parse(line); } catch { continue; }
      const waiter = pending.get(msg.id);
      if (waiter) { pending.delete(msg.id); waiter(msg); }
    }
  });
  const request = (method, params) => new Promise((resolve, reject) => {
    const id = nextId++;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`timed out waiting for ${method}`)); }, 20000);
    pending.set(id, (msg) => { clearTimeout(timer); resolve(msg); });
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
  });
  const call = async (name, args) => {
    const res = await request('tools/call', { name, arguments: args });
    return { isError: res.result.isError === true, payload: JSON.parse(res.result.content[0].text) };
  };
  return { child, request, call };
}

async function main() {
  console.log(`\nruvector weights (real @ruvector/weight-space at ${path.relative(ROOT, wsPath)})`);
  fs.writeFileSync(path.join(work, 'model-a.json'), JSON.stringify(model(1)));
  fs.writeFileSync(path.join(work, 'model-b.json'), JSON.stringify(model(1.5)));
  fs.writeFileSync(path.join(outside, 'model.json'), JSON.stringify(model(1)));

  let idA;
  let idB;
  let childId;

  await test('save: two snapshots with different embeddings', () => {
    const a = ok(cli(['weights', 'save', '--model', 'model-a.json', '--embedding', '1,0,0', '--task', 'task-a', '--tags', 'x,y', '--note', 'first', '--json']), 'save a');
    assert.match(a.snapshot.id, /^ws-[0-9a-f]{16}$/);
    assert.strictEqual(a.snapshot.version, 1);
    assert.strictEqual(a.snapshot.parent, null);
    assert.match(a.snapshot.hash, /^[0-9a-f]{64}$/);
    assert.deepStrictEqual(a.snapshot.topology.sizes, [3, 2, 1]);
    assert.deepStrictEqual(a.snapshot.tags, ['x', 'y']);
    assert.strictEqual(a.snapshot.task, 'task-a');
    assert.strictEqual(a.registry, REGISTRY);
    assert.ok(fs.existsSync(REGISTRY), 'default registry .ruvector/weights.jsonl was written');
    const b = ok(cli(['weights', 'save', '--model', 'model-b.json', '--embedding', '[0,1,0]', '--task', 'task-b', '--json']), 'save b');
    assert.notStrictEqual(b.snapshot.id, a.snapshot.id);
    assert.notStrictEqual(b.snapshot.hash, a.snapshot.hash);
    idA = a.snapshot.id;
    idB = b.snapshot.id;
  });

  await test('save: human output shows id, version, hash, topology', () => {
    const r = cli(['weights', 'save', '--model', 'model-a.json', '--embedding', '1,0,0', '--task', 'task-a', '--tags', 'x,y', '--note', 'first']);
    assert.strictEqual(r.status, 0, r.stderr);
    assert.ok(r.stdout.includes(idA), 'identical content returns the same snapshot id');
    assert.match(r.stdout, /version:\s+1/);
    assert.match(r.stdout, /hash:\s+[0-9a-f]{64}/);
    assert.match(r.stdout, /topology:\s+3->2->1/);
  });

  await test('nearest: query closest to A ranks A first', () => {
    const n = ok(cli(['weights', 'nearest', '--embedding', '0.9,0.1,0', '-k', '2', '--json']), 'nearest');
    assert.ok(n.results.length >= 1);
    assert.strictEqual(n.results[0].id, idA);
    assert.ok(n.results[0].similarity > 0.9, `similarity ${n.results[0].similarity}`);
    const ids = n.results.map((r) => r.id);
    if (n.results.length > 1) assert.ok(ids.includes(idB), `results: ${ids.join(',')}`);
    const nb = ok(cli(['weights', 'nearest', '--embedding', '0,1,0', '-k', '1', '--json']), 'nearest b');
    assert.strictEqual(nb.results[0].id, idB);
    assert.ok(Math.abs(nb.results[0].similarity - 1) < 1e-5, `similarity ${nb.results[0].similarity}`);
  });

  await test('diff: same topology, every changed weight is counted', () => {
    const d = ok(cli(['weights', 'diff', idA, idB, '--json']), 'diff');
    assert.strictEqual(d.diff.compatible, true);
    assert.strictEqual(d.diff.layers.length, 2);
    assert.strictEqual(d.diff.layers[0].changed, 6, 'the 6 scaled weights of layer 0 changed');
    assert.strictEqual(d.diff.layers[1].changed, 0);
    assert.ok(d.diff.l2 > 0);
  });

  await test('lineage: root, then a child committed through the library', async () => {
    const root = ok(cli(['weights', 'lineage', idA, '--json']), 'lineage root');
    assert.deepStrictEqual(root.lineage.map((s) => s.id), [idA]);
    const { WeightRegistry } = require(wsPath);
    const reg = await WeightRegistry.open({ path: REGISTRY });
    try {
      childId = (await reg.perturb(idA, { scale: 0.01, seed: 7 }, { note: 'perturbed' })).id;
    } finally {
      await reg.close();
    }
    const l = ok(cli(['weights', 'lineage', childId, '--json']), 'lineage child');
    assert.deepStrictEqual(l.lineage.map((s) => s.id), [idA, childId]);
    assert.deepStrictEqual(l.lineage.map((s) => s.version), [1, 2]);
    assert.strictEqual(l.lineage[1].op, 'perturb');
    const human = cli(['weights', 'lineage', childId]);
    assert.strictEqual(human.status, 0, human.stderr);
    assert.ok(human.stdout.includes(idA) && human.stdout.includes(childId));
  });

  await test('get: returns the model and embedding', () => {
    const g = ok(cli(['weights', 'get', idB, '--json']), 'get');
    assert.deepStrictEqual(g.snapshot.embedding, [0, 1, 0]);
    assert.deepStrictEqual(g.snapshot.model.layers[0].weights, model(1.5).layers[0].weights);
  });

  await test('errors: bad input exits non-zero with a message', () => {
    const dim = cli(['weights', 'save', '--model', 'model-a.json', '--embedding', '1,0,0,0', '--json']);
    assert.notStrictEqual(dim.status, 0);
    assert.strictEqual(dim.json.success, false);
    const none = cli(['weights', 'nearest', '--json']);
    assert.notStrictEqual(none.status, 0);
    assert.match(none.json.error, /embedding or a text/);
    const both = cli(['weights', 'nearest', '--embedding', '1,0,0', '--text', 'x', '--json']);
    assert.notStrictEqual(both.status, 0);
    const unknown = cli(['weights', 'lineage', 'ws-ffffffffffffffff', '--json']);
    assert.notStrictEqual(unknown.status, 0);
    assert.strictEqual(unknown.json.code, 'ENOTFOUND');
    fs.writeFileSync(path.join(work, 'bad-model.json'), JSON.stringify({ schema: 'ruv-fann.dense-layers/1', layers: [] }));
    const bad = cli(['weights', 'save', '--model', 'bad-model.json', '--embedding', '1,0,0', '--json']);
    assert.notStrictEqual(bad.status, 0);
  });

  await test('confinement: model and registry paths outside the working directory are refused', () => {
    const m = cli(['weights', 'save', '--model', path.join(outside, 'model.json'), '--embedding', '1,0,0', '--json']);
    assert.notStrictEqual(m.status, 0);
    assert.match(m.json.error, /outside the working directory/);
    const r = cli(['weights', 'nearest', '--embedding', '1,0,0', '--registry', path.join(outside, 'w.jsonl'), '--json']);
    assert.notStrictEqual(r.status, 0);
    assert.match(r.json.error, /outside the working directory/);
    fs.symlinkSync(outside, path.join(work, 'link-out'));
    const s = cli(['weights', 'save', '--model', 'model-a.json', '--embedding', '1,0,0', '--registry', 'link-out/w.jsonl', '--json']);
    assert.notStrictEqual(s.status, 0, 'a symlink out of the workspace is followed and refused');
    assert.deepStrictEqual(fs.readdirSync(outside).sort(), ['model.json'], 'nothing written outside the workspace');
  });

  await test('MCP: weights_nearest / weights_save / weights_diff / weights_lineage with path confinement', async () => {
    const { child, request, call } = mcpSession();
    try {
      const init = await request('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'ws-cli-test', version: '1.0.0' } });
      assert.ok(init.result);
      child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`);
      const near = await call('weights_nearest', { embedding: [0, 1, 0], k: 1 });
      assert.strictEqual(near.isError, false, JSON.stringify(near.payload));
      assert.strictEqual(near.payload.results[0].id, idB);
      const saved = await call('weights_save', { model: model(2), embedding: [0, 0, 1], task: 'task-c', tags: ['mcp'] });
      assert.strictEqual(saved.isError, false, JSON.stringify(saved.payload));
      assert.strictEqual(saved.payload.snapshot.version, 1);
      const d = await call('weights_diff', { id_a: idA, id_b: saved.payload.snapshot.id });
      assert.strictEqual(d.payload.diff.compatible, true);
      const lin = await call('weights_lineage', { id: childId });
      assert.deepStrictEqual(lin.payload.lineage.map((s) => s.id), [idA, childId]);
      const esc = await call('weights_lineage', { id: idA, registry: path.join(outside, 'w.jsonl') });
      assert.strictEqual(esc.isError, true);
      assert.match(esc.payload.error, /outside the working directory/);
      const escModel = await call('weights_save', { model_path: '../' + path.basename(outside) + '/model.json', embedding: [1, 0, 0] });
      assert.strictEqual(escModel.isError, true);
      assert.match(escModel.payload.error, /outside the working directory/);
      const badVec = await call('weights_nearest', { embedding: [1, 'x', 0] });
      assert.strictEqual(badVec.isError, true);
    } finally {
      child.stdin.end();
      child.kill('SIGTERM');
    }
  });

  await test("require('ruvector') (CJS entry) loads alongside the package", () => {
    const r = require(ROOT);
    assert.strictEqual(typeof r.VectorDB, 'function');
  });

  console.log(`\nResults: ${passed} passed, ${failed} failed`);
  process.exitCode = failed ? 1 : 0;
}

main()
  .catch((e) => { console.error(e); process.exitCode = 1; })
  .finally(() => {
    fs.rmSync(work, { recursive: true, force: true });
    fs.rmSync(outside, { recursive: true, force: true });
  });
