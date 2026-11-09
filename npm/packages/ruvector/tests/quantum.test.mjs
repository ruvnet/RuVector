/**
 * Quantum (ruqu) integration: `ruvector quantum`, the quantum_* MCP tools and
 * the lazy `ruvector.quantum` library accessor.
 *
 *   - Real-import tests run against the installed @ruvector/ruqu-wasm (it is an
 *     optional dependency; they SKIP with a logged reason when it is absent).
 *   - The absent-engine path is tested for real: bin/ + dist/ are copied into a
 *     temp package whose node_modules links every dependency EXCEPT ruqu-wasm.
 *   - MCP validation limits are enforced before the engine loads, so they are
 *     checked regardless of whether the engine is installed.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PKG = path.join(__dirname, '..');
const CLI = path.join(PKG, 'bin', 'cli.js');
const require = createRequire(import.meta.url);
const q = require(path.join(PKG, 'dist', 'core', 'quantum-wrapper.js'));

const HAVE_ENGINE = q.isQuantumAvailable();
const skipNoEngine = HAVE_ENGINE ? false : '@ruvector/ruqu-wasm not installed (optional dependency)';
if (!HAVE_ENGINE) console.log(`# SKIP real-engine tests: ${skipNoEngine}`);

function run(args, opts = {}) {
  return spawnSync(process.execPath, [opts.cli || CLI, ...args], {
    cwd: opts.cwd || PKG,
    env: { ...process.env, NO_COLOR: '1', FORCE_COLOR: '0' },
    encoding: 'utf8',
    timeout: 60_000,
  });
}

// ── real engine ─────────────────────────────────────────────────────────────

test('Bell state: |00> and |11> each 0.5, shots split ~50/50', { skip: skipNoEngine }, async () => {
  const r = await q.simulate(q.ghzCircuit(2), { shots: 4000, seed: 1 });
  assert.equal(r.numQubits, 2);
  assert.deepEqual(r.top.map((s) => s.state).sort(), ['00', '11']);
  for (const s of r.top) assert.ok(Math.abs(s.probability - 0.5) < 1e-9, `${s.state}=${s.probability}`);
  assert.deepEqual(Object.keys(r.counts).sort(), ['00', '11']);
  assert.equal(r.counts['00'] + r.counts['11'], 4000);
  // 4000 shots: sd = sqrt(4000*.25) ~ 32; 5 sd band
  assert.ok(Math.abs(r.counts['00'] - 2000) < 160, JSON.stringify(r.counts));
  // seeded sampling is reproducible
  const again = await q.simulate(q.ghzCircuit(2), { shots: 4000, seed: 1 });
  assert.deepEqual(again.counts, r.counts);
});

test('n-qubit GHZ state: only all-zeros and all-ones, 0.5 each', { skip: skipNoEngine }, async () => {
  for (const n of [3, 8, 12]) {
    const r = await q.simulate(q.ghzCircuit(n), { top: 4 });
    assert.equal(r.top.length, 2, `GHZ(${n}) support`);
    assert.deepEqual(r.top.map((s) => s.state).sort(), ['0'.repeat(n), '1'.repeat(n)]);
    for (const s of r.top) assert.ok(Math.abs(s.probability - 0.5) < 1e-9);
    assert.equal(r.gateCount, n);
  }
});

test('bit order: x on qubit 0 of 3 gives |001>', { skip: skipNoEngine }, async () => {
  const r = await q.simulate({ qubits: 3, gates: [{ gate: 'x', qubits: [0] }] });
  assert.equal(r.top[0].state, '001');
  assert.ok(Math.abs(r.top[0].probability - 1) < 1e-9);
});

test('text circuit form and library accessor ruvector.quantum', { skip: skipNoEngine }, async () => {
  const rv = require(path.join(PKG, 'dist', 'index.js'));
  const spec = rv.quantum.parseCircuitText('h 0; cx 0 1');
  const r = await rv.quantum.simulate(spec);
  assert.deepEqual(r.top.map((s) => s.state).sort(), ['00', '11']);
  const g = await rv.quantum.grover(4, [11], 7);
  assert.equal(g.targetFound, true);
  assert.equal(g.measured, '1011');
});

test('CLI: quantum simulate bell --json, ruqu alias, bench, info', { skip: skipNoEngine }, () => {
  const sim = run(['quantum', 'simulate', 'bell', '--json', '--shots', '1000', '--seed', '3']);
  assert.equal(sim.status, 0, sim.stderr);
  const r = JSON.parse(sim.stdout);
  assert.deepEqual(r.top.map((s) => s.state).sort(), ['00', '11']);
  assert.equal(r.counts['00'] + r.counts['11'], 1000);

  const alias = run(['ruqu', 'simulate', 'ghz:4', '--json', '--shots', '0']);
  assert.equal(alias.status, 0, alias.stderr);
  assert.deepEqual(JSON.parse(alias.stdout).top.map((s) => s.state).sort(), ['0000', '1111']);

  const bench = run(['quantum', 'bench', '--qubits', '4,8', '--repeats', '2', '--json']);
  assert.equal(bench.status, 0, bench.stderr);
  assert.deepEqual(JSON.parse(bench.stdout).rows.map((row) => row.qubits), [4, 8]);

  const info = run(['quantum', 'info', '--json']);
  assert.equal(info.status, 0, info.stderr);
  const i = JSON.parse(info.stdout);
  assert.equal(i.available, true);
  assert.equal(i.engineMaxQubits, 25);
});

test('CLI rejects measure ops and over-cap circuits with a non-zero exit', { skip: skipNoEngine }, () => {
  const m = run(['quantum', 'simulate', 'measure 0']);
  assert.notEqual(m.status, 0);
  assert.match(m.stderr, /use shots/);
  const big = run(['quantum', 'simulate', 'ghz:26']);
  assert.notEqual(big.status, 0);
  assert.match(big.stderr, /exceeds the cap of 25/);
});

// ── validation (engine-independent) ─────────────────────────────────────────

test('validateCircuit enforces qubit, gate, angle and index limits', () => {
  assert.throws(() => q.validateCircuit({ qubits: 21, gates: [] }), /exceeds the cap of 20.*16\.0 MiB|exceeds the cap of 20/);
  assert.throws(() => q.validateCircuit({ qubits: 0, gates: [] }), /positive integer/);
  assert.throws(() => q.validateCircuit({ qubits: 2.5, gates: [] }), /positive integer/);
  assert.throws(() => q.validateCircuit({ qubits: 2, gates: [{ gate: 'rx', qubits: [0], angle: NaN }] }), /finite/);
  assert.throws(() => q.validateCircuit({ qubits: 2, gates: [{ gate: 'rx', qubits: [0], angle: Infinity }] }), /finite/);
  assert.throws(() => q.validateCircuit({ qubits: 2, gates: [{ gate: 'h', qubits: [2] }] }), /out of range/);
  assert.throws(() => q.validateCircuit({ qubits: 2, gates: [{ gate: 'cnot', qubits: [1, 1] }] }), /distinct/);
  assert.throws(() => q.validateCircuit({ qubits: 2, gates: [{ gate: 'toffoli', qubits: [0] }] }), /unknown gate/);
  assert.throws(() => q.validateCircuit({ qubits: 2, gates: [{ gate: 'measure', qubits: [0] }] }), /use shots/);
  assert.throws(() => q.validateCircuit({ qubits: 2, gates: [{ gate: '__proto__', qubits: [0] }] }), /unknown gate/);
  const many = Array.from({ length: 10001 }, () => ({ gate: 'h', qubits: [0] }));
  assert.throws(() => q.validateCircuit({ qubits: 1, gates: many }), /exceeds the cap of 10000/);
  // the engine ceiling cannot be raised by a caller
  assert.throws(() => q.validateCircuit({ qubits: 26, gates: [] }, { maxQubits: 64 }), /cap of 25/);
  const ok = q.validateCircuit({ qubits: 2, gates: [{ op: 'cx', control: 0, target: 1 }, { gate: 'rzz', qubits: [0, 1], angle: 0.5 }] });
  assert.deepEqual(ok.gates[0], { gate: 'cx', qubits: [0, 1] });
});

// ── absent engine (real absence, not a mock) ────────────────────────────────

function packageWithoutEngine() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-noquantum-'));
  fs.cpSync(path.join(PKG, 'bin'), path.join(dir, 'bin'), { recursive: true });
  fs.cpSync(path.join(PKG, 'dist'), path.join(dir, 'dist'), { recursive: true });
  fs.copyFileSync(path.join(PKG, 'package.json'), path.join(dir, 'package.json'));
  const src = path.join(PKG, 'node_modules');
  const dst = path.join(dir, 'node_modules');
  fs.mkdirSync(dst);
  for (const name of fs.readdirSync(src)) {
    if (name === '@ruvector') {
      fs.mkdirSync(path.join(dst, name));
      for (const sub of fs.readdirSync(path.join(src, name))) {
        if (sub === 'ruqu-wasm') continue;
        fs.symlinkSync(path.join(src, name, sub), path.join(dst, name, sub), 'junction');
      }
    } else {
      fs.symlinkSync(path.join(src, name), path.join(dst, name), 'junction');
    }
  }
  return dir;
}

test('without @ruvector/ruqu-wasm: CLI prints the install hint and exits non-zero', () => {
  const dir = packageWithoutEngine();
  try {
    const cli = path.join(dir, 'bin', 'cli.js');
    const r = run(['quantum', 'simulate', 'bell'], { cli, cwd: dir });
    assert.notEqual(r.status, 0, 'must exit non-zero');
    assert.match(r.stderr, /npm i @ruvector\/ruqu-wasm/);
    assert.doesNotMatch(r.stderr, /at .*\.js:\d+/, 'no stack trace');
    const alias = run(['ruqu', 'bench'], { cli, cwd: dir });
    assert.notEqual(alias.status, 0);
    assert.match(alias.stderr, /npm i @ruvector\/ruqu-wasm/);
    const info = run(['quantum', 'info', '--json'], { cli, cwd: dir });
    assert.notEqual(info.status, 0);
    assert.equal(JSON.parse(info.stdout).available, false);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ── MCP over stdio ──────────────────────────────────────────────────────────

function mcpSession(cwd, cli = CLI) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-mcp-quantum-'));
  const child = spawn(process.execPath, [cli, 'mcp', 'start'], {
    cwd: cwd || home,
    env: { ...process.env, HOME: home, NO_COLOR: '1' },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
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
      const w = pending.get(msg.id);
      if (w) { pending.delete(msg.id); w(msg); }
    }
  });
  const request = (method, params) => {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`timeout: ${method}`)); }, 20_000);
      pending.set(id, (m) => { clearTimeout(timer); resolve(m); });
      child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
    });
  };
  const call = async (name, args) => {
    const res = await request('tools/call', { name, arguments: args });
    return { isError: !!res.result.isError, body: JSON.parse(res.result.content[0].text) };
  };
  const close = () => {
    child.kill('SIGKILL');
    fs.rmSync(home, { recursive: true, force: true });
  };
  return { request, call, close, child };
}

test('MCP: quantum tools listed; validation limits enforced; Bell over stdio', async () => {
  const s = mcpSession();
  try {
    const init = await s.request('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'quantum-test', version: '1' } });
    assert.ok(init.result);
    s.child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`);
    const list = await s.request('tools/list', {});
    const names = list.result.tools.map((t) => t.name);
    assert.ok(names.includes('quantum_simulate') && names.includes('quantum_info'));

    const reject = async (args, re) => {
      const r = await s.call('quantum_simulate', args);
      assert.equal(r.isError, true, JSON.stringify(args));
      assert.equal(r.body.success, false);
      assert.match(r.body.error, re);
    };
    await reject({ circuit: { qubits: 21, gates: [] } }, /cap of 20/);
    await reject({ preset: 'ghz', qubits: 30 }, /cap of 20/);
    await reject({ circuit: { qubits: 1, gates: Array.from({ length: 10001 }, () => ({ gate: 'h', qubits: [0] })) } }, /cap of 10000/);
    await reject({ circuit: { qubits: 1, gates: [{ gate: 'rx', qubits: [0], angle: 'NaN' }] } }, /finite/);
    await reject({ circuit: { qubits: 2, gates: [] }, shots: 100001 }, /shots/);
    await reject({ circuit: { qubits: 2, gates: [] }, shots: -1 }, /shots/);
    await reject({ circuit: { qubits: 2, gates: [] }, shots: 1.5 }, /shots/);
    await reject({ circuit: { qubits: 2, gates: [] }, top: 257 }, /top/);
    await reject({}, /required/);
    await reject({ preset: 'qft' }, /unknown preset/);

    const info = await s.call('quantum_info', {});
    assert.equal(info.body.success, true);
    assert.equal(info.body.available, HAVE_ENGINE);
    assert.equal(info.body.limits.maxQubits, 20);

    if (HAVE_ENGINE) {
      const r = await s.call('quantum_simulate', { preset: 'bell', shots: 2000, seed: 9 });
      assert.equal(r.isError, false, JSON.stringify(r.body));
      assert.deepEqual(r.body.top.map((x) => x.state).sort(), ['00', '11']);
      assert.ok(Math.abs(r.body.counts['00'] - 1000) < 120, JSON.stringify(r.body.counts));
      assert.equal(r.body.probabilities, undefined, 'MCP never returns the raw 2^n vector');
      const ghz = await s.call('quantum_simulate', { circuit: { qubits: 20, gates: q.ghzCircuit(20).gates }, top: 4 });
      assert.equal(ghz.isError, false, JSON.stringify(ghz.body));
      assert.deepEqual(ghz.body.top.map((x) => x.state).sort(), ['0'.repeat(20), '1'.repeat(20)]);
    } else {
      const r = await s.call('quantum_simulate', { preset: 'bell' });
      assert.equal(r.isError, true);
      assert.match(r.body.error, /npm i @ruvector\/ruqu-wasm/);
    }
  } finally {
    s.close();
  }
});
