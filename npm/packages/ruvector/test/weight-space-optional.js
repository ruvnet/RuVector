#!/usr/bin/env node
/**
 * @ruvector/weight-space is an optional peer of ruvector.
 *
 * (a) manifest: exact optional peer (0.1.0), never a dependency of any kind, so a
 *     default install does not fetch it.
 * (b) absent: with the package hidden by a module-resolution shim (works whether or
 *     not it is installed), `ruvector weights ...` prints the install hint and exits
 *     non-zero, the weights_* MCP tools return a structured error with the hint, and
 *     require('ruvector') still loads.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn, spawnSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const CLI = path.join(ROOT, 'bin', 'cli.js');
const HINT = 'npm i @ruvector/weight-space@0.1.0';
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-ws-optional-'));

const shim = path.join(work, 'hide-weight-space.js');
fs.writeFileSync(shim,
  "const M = require('module'); const orig = M._resolveFilename;\n" +
  "M._resolveFilename = function (req, ...rest) {\n" +
  "  if (req === '@ruvector/weight-space' || req.startsWith('@ruvector/weight-space/')) {\n" +
  "    const e = new Error(\"Cannot find module '\" + req + \"'\"); e.code = 'MODULE_NOT_FOUND'; throw e;\n" +
  '  }\n  return orig.call(this, req, ...rest);\n};\n');
const HIDDEN_ENV = {
  ...process.env,
  HOME: work,
  NO_COLOR: '1',
  NODE_OPTIONS: `${process.env.NODE_OPTIONS || ''} --require ${shim}`.trim(),
};

let passed = 0;
let failed = 0;
async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL  ${name}\n        ${e.stack || e.message}`);
  }
}

function cli(args) {
  return spawnSync(process.execPath, [CLI, ...args], { cwd: work, env: HIDDEN_ENV, encoding: 'utf8', timeout: 60000 });
}

function mcpSession() {
  const child = spawn(process.execPath, [CLI, 'mcp', 'start'], { cwd: work, env: HIDDEN_ENV, stdio: ['pipe', 'pipe', 'pipe'] });
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
  return { child, request };
}

async function main() {
  console.log('\n@ruvector/weight-space optional peer');

  await test('manifest: exact optional peer 0.1.0, never a dependency', () => {
    const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
    const name = '@ruvector/weight-space';
    assert.strictEqual(pkg.peerDependencies[name], '0.1.0');
    assert.strictEqual(pkg.peerDependenciesMeta[name].optional, true);
    for (const section of ['dependencies', 'devDependencies', 'optionalDependencies']) {
      assert.ok(!(name in (pkg[section] || {})), `${name} must not be in ${section}`);
    }
  });

  await test('CLI: `weights nearest` without the package prints the install hint and exits non-zero', () => {
    const r = cli(['weights', 'nearest', '--embedding', '1,0,0']);
    assert.notStrictEqual(r.status, 0, `expected non-zero exit, got ${r.status}`);
    assert.ok(r.stderr.includes(HINT), `stderr must contain "${HINT}":\n${r.stderr}`);
    assert.ok(!fs.existsSync(path.join(work, '.ruvector', 'weights.jsonl')), 'no registry is created');
  });

  await test('CLI: --json error carries code and install hint', () => {
    const r = cli(['weights', 'lineage', 'ws-0000000000000000', '--json']);
    assert.notStrictEqual(r.status, 0);
    const out = JSON.parse(r.stdout);
    assert.strictEqual(out.success, false);
    assert.strictEqual(out.code, 'ERR_WEIGHT_SPACE_MISSING');
    assert.strictEqual(out.install, HINT);
  });

  await test('CLI: --help still lists the weights group (lazy load)', () => {
    const r = cli(['weights', '--help']);
    assert.strictEqual(r.status, 0, r.stderr);
    for (const sub of ['save', 'nearest', 'diff', 'lineage']) assert.match(r.stdout, new RegExp(`\\b${sub}\\b`));
  });

  await test('MCP: weights_* tools are listed and return a structured install-hint error', async () => {
    const { child, request } = mcpSession();
    try {
      const init = await request('initialize', {
        protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'ws-optional-test', version: '1.0.0' },
      });
      assert.ok(init.result, 'initialize succeeded');
      child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`);
      const list = await request('tools/list', {});
      const names = list.result.tools.map((t) => t.name);
      for (const n of ['weights_save', 'weights_nearest', 'weights_diff', 'weights_lineage']) {
        assert.ok(names.includes(n), `${n} listed`);
      }
      const res = await request('tools/call', { name: 'weights_nearest', arguments: { embedding: [1, 0, 0] } });
      assert.strictEqual(res.result.isError, true);
      const payload = JSON.parse(res.result.content[0].text);
      assert.strictEqual(payload.success, false);
      assert.strictEqual(payload.code, 'ERR_WEIGHT_SPACE_MISSING');
      assert.strictEqual(payload.install, HINT);
      assert.ok(payload.error.includes(HINT));
    } finally {
      child.stdin.end();
      child.kill('SIGTERM');
    }
  });

  await test("require('ruvector') (CJS entry) still loads without the package", () => {
    const r = spawnSync(process.execPath, ['-e',
      `const r = require(${JSON.stringify(ROOT)}); if (typeof r.VectorDB !== 'function') process.exit(2);` +
      "let hidden = false; try { require.resolve('@ruvector/weight-space'); } catch { hidden = true; } if (!hidden) process.exit(3);"],
    { cwd: ROOT, env: HIDDEN_ENV, encoding: 'utf8', timeout: 60000 });
    assert.strictEqual(r.status, 0, `exit ${r.status}: ${r.stderr}`);
  });

  console.log(`\nResults: ${passed} passed, ${failed} failed`);
  process.exitCode = failed ? 1 : 0;
}

main()
  .catch((e) => { console.error(e); process.exitCode = 1; })
  .finally(() => fs.rmSync(work, { recursive: true, force: true }));
