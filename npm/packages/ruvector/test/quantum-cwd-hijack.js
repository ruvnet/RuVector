#!/usr/bin/env node
/**
 * Regression test for GHSA-8pv4-hqq5-m2wv / GHSA-7258-fhh3-8pq8.
 *
 * The quantum wrapper used to resolve @ruvector/ruqu-wasm with
 * `require.resolve(PKG, { paths: [process.cwd(), __dirname] })`, so a
 * node_modules/@ruvector/ruqu-wasm committed to a cloned repository shadowed
 * the engine installed with ruvector, and the first quantum_info call (a
 * readonly-profile MCP tool) imported and ran it.
 *
 * A fake engine whose top-level code writes a marker file is planted in the
 * working directory. The MCP server (quantum_info, quantum_simulate), the
 * library and the CLI must all ignore it.
 */

const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const assert = require('assert');

const PKG_ROOT = path.join(__dirname, '..');
const SERVER = path.join(PKG_ROOT, 'bin', 'mcp-server.js');
const CLI = path.join(PKG_ROOT, 'bin', 'cli.js');
const WRAPPER = path.join(PKG_ROOT, 'dist', 'core', 'quantum-wrapper.js');
const TIMEOUT_MS = 60000;

function plantFakeEngine(proj, marker) {
  const dir = path.join(proj, 'node_modules', '@ruvector', 'ruqu-wasm');
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({
    name: '@ruvector/ruqu-wasm', version: '9.9.9-hijack', type: 'module', main: 'index.mjs',
  }));
  fs.writeFileSync(path.join(dir, 'index.mjs'), [
    "import { writeFileSync } from 'node:fs';",
    `writeFileSync(${JSON.stringify(marker)}, 'pwned');`,
    'export function initSync() {}',
    'export function max_qubits() { return 25; }',
    'export function estimate_memory(n) { return 2 ** n * 16; }',
    '',
  ].join('\n'));
  fs.writeFileSync(path.join(dir, 'ruqu_wasm_bg.wasm'), '');
}

function callTools(cwd, calls) {
  return new Promise((resolve, reject) => {
    const p = spawn(process.execPath, [SERVER], {
      cwd,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, HOME: cwd, RUVECTOR_MCP_PROFILE: 'readonly' },
    });
    const results = new Map();
    let buf = '';
    const send = (o) => p.stdin.write(JSON.stringify(o) + '\n');
    const timer = setTimeout(() => { p.kill(); reject(new Error('MCP server timed out')); }, TIMEOUT_MS);
    p.stdout.on('data', (d) => {
      buf += d;
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 1);
        let m;
        try { m = JSON.parse(line); } catch { continue; }
        if (m.id === 0) {
          send({ jsonrpc: '2.0', method: 'notifications/initialized' });
          calls.forEach(([name, args], idx) =>
            send({ jsonrpc: '2.0', id: idx + 1, method: 'tools/call', params: { name, arguments: args } }));
        } else if (typeof m.id === 'number') {
          results.set(m.id, m.result || m.error);
          if (results.size === calls.length) {
            clearTimeout(timer);
            p.kill();
            resolve(calls.map((_, idx) => JSON.stringify(results.get(idx + 1))));
          }
        }
      }
    });
    send({ jsonrpc: '2.0', id: 0, method: 'initialize', params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'quantum-cwd-hijack', version: '1' } } });
  });
}

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-ruqu-hijack-'));
  const proj = path.join(root, 'proj');
  fs.mkdirSync(proj);
  const marker = path.join(root, 'PWNED');
  plantFakeEngine(proj, marker);

  let passed = 0;
  const check = (name, fn) => { fn(); passed++; console.log(`  PASS  ${name}`); };
  const notRun = (where) => assert.ok(!fs.existsSync(marker), `${where} executed the planted @ruvector/ruqu-wasm from the working directory`);

  try {
    const [info, sim] = await callTools(proj, [
      ['quantum_info', {}],
      ['quantum_simulate', { circuit: { qubits: 2, gates: [{ gate: 'h', qubits: [0] }] } }],
    ]);
    check('MCP quantum_info / quantum_simulate do not import the cwd engine', () => notRun('MCP server'));
    check('MCP quantum_info does not report the planted engine version', () =>
      assert.ok(!info.includes('9.9.9-hijack'), info));
    assert.ok(sim);

    // Library: resolve + load from a process whose cwd holds the planted engine.
    const lib = spawnSync(process.execPath, ['-e', `
      const q = require(${JSON.stringify(WRAPPER)});
      q.info().then((i) => { console.log(JSON.stringify(i)); }, (e) => { console.log('ERR ' + e.message); });
    `], { cwd: proj, encoding: 'utf8', timeout: TIMEOUT_MS });
    check('library info() does not import the cwd engine', () => notRun('library'));
    check('library info() does not report the planted engine version', () =>
      assert.ok(!lib.stdout.includes('9.9.9-hijack'), lib.stdout + lib.stderr));

    spawnSync(process.execPath, [CLI, 'quantum', 'info', '--json'], { cwd: proj, encoding: 'utf8', timeout: TIMEOUT_MS, env: { ...process.env, HOME: proj } });
    check('CLI `quantum info` does not import the cwd engine', () => notRun('CLI'));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
  console.log(`\nquantum-cwd-hijack: ${passed} passed`);
})().catch((e) => {
  console.error(`  FAIL  ${e.message}`);
  process.exit(1);
});
