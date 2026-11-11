#!/usr/bin/env node
/**
 * hooks_capabilities reports the opt-in extras (read-only listing; installing
 * stays a CLI action — `ruvector add <extra>` — never an MCP side effect).
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const ROOT = path.join(__dirname, '..');
const CLI = path.join(ROOT, 'bin', 'cli.js');
const home = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-mcp-extras-'));
const child = spawn(process.execPath, [CLI, 'mcp', 'start'], {
  cwd: home,
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
    if (!line) continue;
    const message = JSON.parse(line);
    const waiter = pending.get(message.id);
    if (waiter) {
      pending.delete(message.id);
      waiter(message);
    }
  }
});

function request(method, params) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`Timed out waiting for MCP ${method}`));
    }, 15_000);
    pending.set(id, (message) => {
      clearTimeout(timer);
      resolve(message);
    });
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
  });
}

async function main() {
  const init = await request('initialize', {
    protocolVersion: '2024-11-05',
    capabilities: {},
    clientInfo: { name: 'extras-test', version: '1.0.0' },
  });
  assert.ok(init.result);
  child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`);

  const res = await request('tools/call', { name: 'hooks_capabilities', arguments: {} });
  const payload = JSON.parse(res.result.content[0].text);
  assert.strictEqual(payload.success, true);
  assert.ok(Array.isArray(payload.extras), 'hooks_capabilities must list extras');
  const rabitq = payload.extras.find((e) => e.key === 'rabitq');
  assert.ok(rabitq, 'rabitq extra listed');
  assert.strictEqual(rabitq.package, '@ruvector/rabitq-wasm');
  assert.strictEqual(typeof rabitq.installed, 'boolean');
  assert.strictEqual(rabitq.install, 'npx ruvector add rabitq');
  const sona = payload.extras.find((e) => e.key === 'sona');
  assert.strictEqual(sona.bundled, true);
  assert.strictEqual(sona.install, null);
  console.log('  PASS  hooks_capabilities lists opt-in extras with install hints');
}

main()
  .catch((error) => {
    console.error(`  FAIL  ${error.message}`);
    process.exitCode = 1;
  })
  .finally(() => {
    child.stdin.end();
    child.kill('SIGTERM');
  });
