#!/usr/bin/env node
/**
 * End-to-end check of `hooks graph-mincut` and `hooks graph-cluster`
 * (louvain + spectral), run through the real CLI on fixture source files, and
 * of the MCP tools that shell out to them (hooks_graph_mincut /
 * hooks_graph_cluster).
 *
 * Fixture: two import triangles {a,b,c} and {d,e,f} joined by one bridge
 * import c -> d. Every algorithm must split it into exactly those two groups;
 * the min cut is the single bridge (weight 1).
 *
 * @ruvector/mincut-wasm is optional. The mincut case runs twice: once as
 * installed (native engine when present) and once with the module hidden via a
 * --require shim, which must fall back to the TS solver with identical output.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn, execFileSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const CLI = path.join(ROOT, 'bin', 'cli.js');
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-graph-cli-'));

const FILES = {
  'a.ts': "import { b } from './b';\nimport { c } from './c';\n",
  'b.ts': "import { c } from './c';\n",
  'c.ts': "import { d } from './d';\nexport const c = 1;\n",
  'd.ts': "import { e } from './e';\nimport { f } from './f.js';\n",
  'e.ts': "const f = require('./f');\n",
  'f.js': 'export const f = 1;\n',
};
for (const [name, body] of Object.entries(FILES)) fs.writeFileSync(path.join(work, name), body);
const fileArgs = Object.keys(FILES);

// Preload shim that makes @ruvector/mincut-wasm unresolvable, simulating an
// install without the optional dependency.
const HIDE_SHIM = path.join(work, 'hide-mincut-wasm.js');
fs.writeFileSync(
  HIDE_SHIM,
  "const M = require('module');\n" +
    'const orig = M._resolveFilename;\n' +
    "M._resolveFilename = function (req, ...rest) {\n" +
    "  if (req === '@ruvector/mincut-wasm' || req.startsWith('@ruvector/mincut-wasm/')) {\n" +
    "    const e = new Error(\"Cannot find module '\" + req + \"'\"); e.code = 'MODULE_NOT_FOUND'; throw e;\n" +
    '  }\n' +
    '  return orig.call(this, req, ...rest);\n' +
    '};\n',
);

const EXPECTED_GROUPS = [['a', 'b', 'c'], ['d', 'e', 'f']];
const norm = (groups) => groups.map((g) => [...g].sort().join(',')).sort();

function cli(args, { hideWasm = false } = {}) {
  const env = { ...process.env, HOME: work, NO_COLOR: '1' };
  if (hideWasm) env.NODE_OPTIONS = `${env.NODE_OPTIONS || ''} --require ${HIDE_SHIM}`.trim();
  const out = execFileSync(process.execPath, [CLI, ...args], { cwd: work, env, encoding: 'utf8', timeout: 60_000 });
  const line = out.trim().split(/\r?\n/).filter((l) => l.startsWith('{')).pop();
  assert.ok(line, `no JSON output from ${args.join(' ')}:\n${out}`);
  return JSON.parse(line);
}

function assertTwoTriangles(r, label) {
  assert.strictEqual(r.success, true, `${label}: ${JSON.stringify(r)}`);
  assert.strictEqual(r.nodes, 6, `${label}: 6 nodes`);
  assert.strictEqual(r.edges, 7, `${label}: 7 import edges`);
  assert.deepStrictEqual(norm(r.groups), norm(EXPECTED_GROUPS), `${label}: groups ${JSON.stringify(r.groups)}`);
}

let wasmInstalled = false;
try { require.resolve('@ruvector/mincut-wasm'); wasmInstalled = true; } catch { /* optional */ }

function testMincut() {
  const r = cli(['hooks', 'graph-mincut', ...fileArgs]);
  assertTwoTriangles(r, 'graph-mincut');
  assert.strictEqual(r.cutWeight, 1, 'min cut is the single bridge');
  assert.strictEqual(r.engine, wasmInstalled ? 'mincut-wasm' : 'typescript');
  console.log(`  PASS  graph-mincut (${r.engine}): cut ${r.cutWeight}, groups ${JSON.stringify(r.groups)}`);

  const ts = cli(['hooks', 'graph-mincut', ...fileArgs], { hideWasm: true });
  assertTwoTriangles(ts, 'graph-mincut (no wasm)');
  assert.strictEqual(ts.engine, 'typescript', 'falls back to the TS solver');
  assert.strictEqual(ts.cutWeight, r.cutWeight, 'same cut with and without wasm');
  assert.strictEqual(ts.modularity, r.modularity, 'same modularity with and without wasm');
  console.log('  PASS  graph-mincut without @ruvector/mincut-wasm falls back to TS with identical result');
}

function testLouvain() {
  const r = cli(['hooks', 'graph-cluster', ...fileArgs, '--method', 'louvain']);
  assertTwoTriangles(r, 'louvain');
  assert.strictEqual(r.method, 'louvain');
  // Q = 2 * (3/7 - (7/14)^2) for two triangles + one bridge.
  assert.ok(Math.abs(r.modularity - 2 * (3 / 7 - 0.25)) < 1e-9, `louvain modularity ${r.modularity}`);
  assert.strictEqual(r.assignments.a, r.assignments.b);
  assert.notStrictEqual(r.assignments.a, r.assignments.d);
  console.log(`  PASS  graph-cluster louvain: ${JSON.stringify(r.groups)}, Q=${r.modularity.toFixed(4)}`);
}

function testSpectral() {
  const r = cli(['hooks', 'graph-cluster', ...fileArgs, '--method', 'spectral', '--clusters', '2']);
  assertTwoTriangles(r, 'spectral');
  assert.strictEqual(r.method, 'spectral');
  assert.strictEqual(r.eigenvalues.length, 2);
  // Smallest normalized-Laplacian eigenvalue of a connected graph is 0.
  assert.ok(Math.abs(r.eigenvalues[0]) < 1e-6, `lambda_0 ~ 0, got ${r.eigenvalues[0]}`);
  assert.ok(r.eigenvalues[1] > 0 && r.eigenvalues[1] < 1, `Fiedler value in (0,1), got ${r.eigenvalues[1]}`);
  const again = cli(['hooks', 'graph-cluster', ...fileArgs, '--method', 'spectral', '--clusters', '2']);
  assert.deepStrictEqual(again.assignments, r.assignments, 'spectral is deterministic');
  console.log(`  PASS  graph-cluster spectral: ${JSON.stringify(r.groups)}, eigenvalues ${r.eigenvalues.map((v) => v.toFixed(4))}`);

  const bad = cli(['hooks', 'graph-cluster', ...fileArgs, '--method', 'bogus']);
  assert.strictEqual(bad.success, false, 'unknown method is rejected');
  console.log('  PASS  graph-cluster rejects an unknown method');
}

async function testMcp() {
  const child = spawn(process.execPath, [CLI, 'mcp', 'start'], {
    cwd: work,
    env: { ...process.env, HOME: work, NO_COLOR: '1' },
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
      if (!line) continue;
      const message = JSON.parse(line);
      const waiter = pending.get(message.id);
      if (waiter) { pending.delete(message.id); waiter(message); }
    }
  });
  const request = (method, params) => {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out waiting for MCP ${method}`)); }, 60_000);
      pending.set(id, (m) => { clearTimeout(timer); resolve(m); });
      child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
    });
  };
  try {
    const init = await request('initialize', {
      protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'graph-cli-e2e', version: '1.0.0' },
    });
    assert.ok(init.result);
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`);

    const mc = await request('tools/call', { name: 'hooks_graph_mincut', arguments: { files: fileArgs } });
    const mcOut = JSON.parse(mc.result.content[0].text.trim().split(/\r?\n/).pop());
    assertTwoTriangles(mcOut, 'MCP hooks_graph_mincut');
    assert.strictEqual(mcOut.cutWeight, 1);
    console.log('  PASS  MCP hooks_graph_mincut');

    for (const method of ['louvain', 'spectral']) {
      const res = await request('tools/call', { name: 'hooks_graph_cluster', arguments: { files: fileArgs, method, clusters: 2 } });
      const out = JSON.parse(res.result.content[0].text.trim().split(/\r?\n/).pop());
      assertTwoTriangles(out, `MCP hooks_graph_cluster ${method}`);
      console.log(`  PASS  MCP hooks_graph_cluster ${method}`);
    }
  } finally {
    child.stdin.end();
    child.kill('SIGTERM');
  }
}

async function main() {
  console.log(`  @ruvector/mincut-wasm installed: ${wasmInstalled}`);
  testMincut();
  testLouvain();
  testSpectral();
  await testMcp();
}

main()
  .catch((error) => {
    console.error(`  FAIL  ${error.stack || error.message}`);
    process.exitCode = 1;
  })
  .finally(() => {
    fs.rmSync(work, { recursive: true, force: true });
  });
