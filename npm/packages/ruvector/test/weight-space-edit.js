#!/usr/bin/env node
/**
 * `ruvector weights set-layer | perturb | learn` and the weights_set_layer / weights_perturb /
 * weights_learn MCP tools against the real @ruvector/weight-space.
 *
 * Each operation writes a CHILD snapshot (the registry is append-only), so these tests check
 * lineage, determinism, that a rejected fine-tune writes nothing, input bounds, and that file
 * inputs are confined to the working directory.
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

let ws;
try {
  ws = require(require.resolve('@ruvector/weight-space', { paths: [ROOT] }));
} catch {
  console.error('  FAIL  @ruvector/weight-space is a dependency of ruvector but does not resolve (run npm install)');
  process.exit(1);
}

const work = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-ws-edit-')));
const outside = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-ws-edit-outside-')));
const ENV = { ...process.env, HOME: work, NO_COLOR: '1' };
const REGISTRY = path.join(work, '.ruvector', 'weights.jsonl');

const registryLines = () => fs.readFileSync(REGISTRY, 'utf8').split('\n').filter(Boolean).length;

function cli(args) {
  const r = spawnSync(process.execPath, [CLI, ...args], { cwd: work, env: ENV, encoding: 'utf8', timeout: 90000 });
  let json = null;
  try { json = JSON.parse(r.stdout); } catch { /* human output */ }
  return { status: r.status, stdout: r.stdout, stderr: r.stderr, json };
}

function ok(r, what) {
  assert.strictEqual(r.status, 0, `${what} failed (exit ${r.status}): ${r.stdout}\n${r.stderr}`);
  assert.ok(r.json && r.json.success === true, `${what}: expected success JSON, got ${r.stdout}`);
  return r.json;
}

function fails(r, pattern, what) {
  assert.notStrictEqual(r.status, 0, `${what}: expected a non-zero exit`);
  assert.ok(r.json && r.json.success === false, `${what}: expected an error JSON, got ${r.stdout}`);
  if (pattern) assert.match(r.json.error, pattern, `${what}: ${r.json.error}`);
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
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`timed out waiting for ${method}`)); }, 60000);
    pending.set(id, (msg) => { clearTimeout(timer); resolve(msg); });
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
  });
  const call = async (name, args) => {
    const res = await request('tools/call', { name, arguments: args });
    return { isError: res.result.isError === true, payload: JSON.parse(res.result.content[0].text) };
  };
  return { child, request, call };
}

/** A [2, 4, 1] network: 2 inputs, 4 hidden units, 1 output. */
const mlp = () => ws.createMlp([2, 4, 1], { seed: 3 });

/** Trajectories the model can learn: y = a*b. */
const trajectories = (n = 64) => Array.from({ length: n }, (_, i) => {
  const a = Math.sin(i * 1.7);
  const b = Math.cos(i * 2.3);
  return { input: [a, b], output: [a * b], score: 1 };
});

async function main() {
  console.log(`\nruvector weights edit + fine-tune (real @ruvector/weight-space)`);
  fs.writeFileSync(path.join(work, 'model.json'), JSON.stringify(mlp()));
  fs.writeFileSync(path.join(work, 'traj.json'), JSON.stringify(trajectories()));
  fs.writeFileSync(path.join(outside, 'patch.json'), JSON.stringify({ bias: [0, 0, 0, 0] }));
  fs.writeFileSync(path.join(outside, 'traj.json'), JSON.stringify(trajectories()));

  const root = ok(cli(['weights', 'save', '--model', 'model.json', '--embedding', '1,0,0', '--task', 'edit-root', '--json']), 'save root').snapshot;

  await test('set-layer: writes a child with op=setLayer, version+1 and the parent link; only that layer changes', () => {
    fs.writeFileSync(path.join(work, 'patch.json'), JSON.stringify({ bias: [0.5, 0.5, 0.5, 0.5] }));
    const r = ok(cli(['weights', 'set-layer', root.id, '0', '--patch', 'patch.json', '--note', 'raise bias', '--tags', 'edit', '--json']), 'set-layer');
    assert.strictEqual(r.parent, root.id);
    assert.strictEqual(r.snapshot.op, 'setLayer');
    assert.strictEqual(r.snapshot.version, root.version + 1);
    assert.strictEqual(r.snapshot.parent, root.id);
    assert.strictEqual(r.snapshot.note, 'raise bias');
    assert.deepStrictEqual(r.snapshot.tags, ['edit']);
    const d = ok(cli(['weights', 'diff', root.id, r.snapshot.id, '--json']), 'diff').diff;
    assert.strictEqual(d.compatible, true);
    assert.ok(d.layers.find((l) => l.index === 0).changed > 0, 'layer 0 changed');
    assert.strictEqual(d.layers.find((l) => l.index === 1).changed, 0, 'layer 1 untouched');
    const lin = ok(cli(['weights', 'lineage', r.snapshot.id, '--json']), 'lineage').lineage;
    assert.deepStrictEqual(lin.map((s) => s.id), [root.id, r.snapshot.id]);
  });

  await test('set-layer: a shape change or an unsupported field is refused and writes nothing', () => {
    const before = registryLines();
    fs.writeFileSync(path.join(work, 'bad-shape.json'), JSON.stringify({ weights: [1, 2, 3] }));
    fails(cli(['weights', 'set-layer', root.id, '0', '--patch', 'bad-shape.json', '--json']), undefined, 'wrong weight count');
    fs.writeFileSync(path.join(work, 'bad-field.json'), JSON.stringify({ outputs: 9 }));
    fails(cli(['weights', 'set-layer', root.id, '0', '--patch', 'bad-field.json', '--json']), /unsupported field 'outputs'/, 'unsupported field');
    fs.writeFileSync(path.join(work, 'empty.json'), '{}');
    fails(cli(['weights', 'set-layer', root.id, '0', '--patch', 'empty.json', '--json']), /at least one of/, 'empty patch');
    fails(cli(['weights', 'set-layer', root.id, '99', '--patch', 'patch.json', '--json']), /layer must be at most/, 'layer out of range');
    fails(cli(['weights', 'set-layer', 'ws-doesnotexist', '0', '--patch', 'patch.json', '--json']), undefined, 'unknown snapshot');
    assert.strictEqual(registryLines(), before, 'no snapshot was written by a refused edit');
  });

  await test('perturb: seeded and deterministic; --no-bias leaves biases untouched', () => {
    const a = ok(cli(['weights', 'perturb', root.id, '--scale', '0.05', '--seed', '7', '--json']), 'perturb a').snapshot;
    const b = ok(cli(['weights', 'perturb', root.id, '--scale', '0.05', '--seed', '7', '--json']), 'perturb b').snapshot;
    const c = ok(cli(['weights', 'perturb', root.id, '--scale', '0.05', '--seed', '8', '--json']), 'perturb c').snapshot;
    assert.strictEqual(a.op, 'perturb');
    assert.strictEqual(a.parent, root.id);
    assert.strictEqual(a.hash, b.hash, 'same seed gives the same model');
    assert.notStrictEqual(a.hash, c.hash, 'a different seed gives a different model');
    const nb = ok(cli(['weights', 'perturb', root.id, '--scale', '0.05', '--seed', '7', '--no-bias', '--json']), 'perturb no-bias').snapshot;
    const parent = ok(cli(['weights', 'get', root.id, '--json']), 'get parent').snapshot.model;
    const child = ok(cli(['weights', 'get', nb.id, '--json']), 'get child').snapshot.model;
    parent.layers.forEach((l, i) => assert.deepStrictEqual(child.layers[i].bias, l.bias, `layer ${i} bias unchanged`));
    assert.notDeepStrictEqual(child.layers[0].weights, parent.layers[0].weights, 'weights did change');
  });

  await test('perturb: bounds are enforced before anything is written', () => {
    const before = registryLines();
    fails(cli(['weights', 'perturb', root.id, '--scale', '1000', '--json']), /scale must be at most/, 'scale too big');
    fails(cli(['weights', 'perturb', root.id, '--scale', 'NaN', '--json']), /finite number/, 'NaN scale');
    fails(cli(['weights', 'perturb', root.id, '--seed', '1.5', '--json']), /integer/, 'fractional seed');
    fails(cli(['weights', 'perturb', root.id, '--layers', '0,abc', '--json']), /layers\[1\]/, 'bad layer list');
    fails(cli(['weights', 'perturb', 'not an id', '--json']), /snapshot id/, 'bad id');
    assert.strictEqual(registryLines(), before);
  });

  await test('learn: an accepted fine-tune writes a child with op=finetune and a better held-out error', () => {
    const r = ok(cli(['weights', 'learn', root.id, '--trajectories', 'traj.json', '--epochs', '300', '--learning-rate', '0.2', '--json']), 'learn');
    assert.strictEqual(r.accepted, true, r.reason);
    assert.strictEqual(r.parent, root.id);
    assert.strictEqual(r.child.op, 'finetune');
    assert.strictEqual(r.child.parent, root.id);
    assert.ok(r.holdoutMse.after < r.holdoutMse.before, `holdout MSE improved: ${r.holdoutMse.before} -> ${r.holdoutMse.after}`);
    assert.ok(r.rows.train > 0 && r.rows.holdout > 0);
  });

  await test('learn: a fine-tune that does not clear the held-out check is reported, not an error, and writes nothing', () => {
    const before = registryLines();
    // Requiring a 100% error reduction cannot be met, so the result must be a rejection.
    const r = ok(cli(['weights', 'learn', root.id, '--trajectories', 'traj.json', '--epochs', '5', '--min-improvement', '1', '--json']), 'learn rejected');
    assert.strictEqual(r.accepted, false);
    assert.strictEqual(r.child, null);
    assert.ok(typeof r.reason === 'string' && r.reason.length > 0, 'a reason is given');
    assert.strictEqual(registryLines(), before, 'a rejected fine-tune appends nothing');
    const human = cli(['weights', 'learn', root.id, '--trajectories', 'traj.json', '--epochs', '5', '--min-improvement', '1']);
    assert.strictEqual(human.status, 0);
    assert.match(human.stdout, /Not accepted:/);
  });

  await test('learn: bad trajectories and out-of-range options are refused before training', () => {
    const before = registryLines();
    fs.writeFileSync(path.join(work, 'bad-rows.json'), JSON.stringify([{ input: [1, 2], output: [3] }]));
    fails(cli(['weights', 'learn', root.id, '--trajectories', 'bad-rows.json', '--json']), /score/, 'missing score');
    fs.writeFileSync(path.join(work, 'bad-num.json'), JSON.stringify([{ input: [1, 'x'], output: [3], score: 1 }]));
    fails(cli(['weights', 'learn', root.id, '--trajectories', 'bad-num.json', '--json']), /finite numbers/, 'non-numeric input');
    fs.writeFileSync(path.join(work, 'not-array.json'), JSON.stringify({ input: [1] }));
    fails(cli(['weights', 'learn', root.id, '--trajectories', 'not-array.json', '--json']), /array of 1 to/, 'not an array');
    fails(cli(['weights', 'learn', root.id, '--trajectories', 'traj.json', '--epochs', '99999999', '--json']), /epochs must be at most/, 'epochs cap');
    fails(cli(['weights', 'learn', root.id, '--trajectories', 'traj.json', '--optimizer', 'rmsprop', '--json']), /optimizer must be/, 'optimizer');
    // Wrong input width for this model: the library rejects it.
    fs.writeFileSync(path.join(work, 'wrong-width.json'), JSON.stringify([{ input: [1, 2, 3], output: [1], score: 1 }]));
    fails(cli(['weights', 'learn', root.id, '--trajectories', 'wrong-width.json', '--json']), undefined, 'input width does not match the model');
    assert.strictEqual(registryLines(), before);
  });

  await test('confinement: patch and trajectories files outside the working directory are refused', () => {
    const before = registryLines();
    fails(cli(['weights', 'set-layer', root.id, '0', '--patch', path.join(outside, 'patch.json'), '--json']), /outside the working directory/, 'patch outside');
    fails(cli(['weights', 'learn', root.id, '--trajectories', path.join(outside, 'traj.json'), '--json']), /outside the working directory/, 'trajectories outside');
    fails(cli(['weights', 'perturb', root.id, '--registry', path.join(outside, 'w.jsonl'), '--json']), /outside the working directory/, 'registry outside');
    assert.strictEqual(registryLines(), before);
    assert.deepStrictEqual(fs.readdirSync(outside).sort(), ['patch.json', 'traj.json'], 'nothing written outside the workspace');
  });

  await test('MCP: the three tools are listed, work, and enforce the same bounds and path confinement', async () => {
    const { child, request, call } = mcpSession();
    try {
      const init = await request('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'ws-edit-test', version: '1.0.0' } });
      assert.ok(init.result);
      child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`);
      const list = await request('tools/list', {});
      const names = list.result.tools.map((t) => t.name);
      for (const n of ['weights_set_layer', 'weights_perturb', 'weights_learn']) assert.ok(names.includes(n), `${n} listed`);

      const set = await call('weights_set_layer', { id: root.id, layer: 1, patch: { bias: [0.25] }, note: 'via mcp' });
      assert.strictEqual(set.isError, false, JSON.stringify(set.payload));
      assert.strictEqual(set.payload.snapshot.op, 'setLayer');
      assert.strictEqual(set.payload.snapshot.parent, root.id);

      const per = await call('weights_perturb', { id: root.id, scale: 0.02, seed: 3, layers: [0], bias: false });
      assert.strictEqual(per.isError, false, JSON.stringify(per.payload));
      assert.strictEqual(per.payload.snapshot.op, 'perturb');

      const learn = await call('weights_learn', { id: root.id, trajectories: trajectories(), epochs: 300, learning_rate: 0.2 });
      assert.strictEqual(learn.isError, false, JSON.stringify(learn.payload));
      assert.strictEqual(learn.payload.accepted, true, learn.payload.reason);
      assert.strictEqual(learn.payload.child.op, 'finetune');

      const before = registryLines();
      const rejected = await call('weights_learn', { id: root.id, trajectories: trajectories(), epochs: 5, min_improvement: 1 });
      assert.strictEqual(rejected.isError, false);
      assert.strictEqual(rejected.payload.accepted, false);
      assert.strictEqual(registryLines(), before);

      const both = await call('weights_set_layer', { id: root.id, layer: 0, patch: { bias: [0, 0, 0, 0] }, patch_path: 'patch.json' });
      assert.strictEqual(both.isError, true);
      assert.match(both.payload.error, /exactly one of/);
      const neither = await call('weights_learn', { id: root.id });
      assert.strictEqual(neither.isError, true);
      assert.match(neither.payload.error, /exactly one of/);
      const big = await call('weights_perturb', { id: root.id, scale: 99999 });
      assert.strictEqual(big.isError, true);
      assert.match(big.payload.error, /scale must be at most/);
      const epochs = await call('weights_learn', { id: root.id, trajectories: trajectories(4), epochs: 1e9 });
      assert.strictEqual(epochs.isError, true);

      const escPatch = await call('weights_set_layer', { id: root.id, layer: 0, patch_path: path.join(outside, 'patch.json') });
      assert.strictEqual(escPatch.isError, true);
      assert.match(escPatch.payload.error, /outside the working directory/);
      const escTraj = await call('weights_learn', { id: root.id, trajectories_path: '../' + path.basename(outside) + '/traj.json' });
      assert.strictEqual(escTraj.isError, true);
      assert.match(escTraj.payload.error, /outside the working directory/);
      const escReg = await call('weights_perturb', { id: root.id, registry: path.join(outside, 'w.jsonl') });
      assert.strictEqual(escReg.isError, true);
      assert.match(escReg.payload.error, /outside the working directory/);
    } finally {
      child.stdin.end();
      child.kill('SIGTERM');
    }
  });

  await test('the registry still verifies after all the writes (hash chain replays)', () => {
    const lin = ok(cli(['weights', 'lineage', root.id, '--json']), 'lineage after edits').lineage;
    assert.strictEqual(lin[0].id, root.id);
    const nearest = ok(cli(['weights', 'nearest', '--embedding', '1,0,0', '-k', '3', '--json']), 'nearest after edits');
    assert.strictEqual(nearest.results.length, 3);
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
