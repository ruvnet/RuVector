#!/usr/bin/env node

const assert = require('assert');
const path = require('path');
const { spawn } = require('child_process');

const ROOT = path.join(__dirname, '..');
const CLI = path.join(ROOT, 'bin', 'cli.js');
const child = spawn(process.execPath, [CLI, 'mcp', 'start'], {
  cwd: ROOT,
  env: { ...process.env, RUVECTOR_MCP_PROFILE: 'readonly', NO_COLOR: '1' },
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
      waiter.resolve(message);
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
    pending.set(id, {
      resolve: (message) => {
        clearTimeout(timer);
        resolve(message);
      },
    });
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
  });
}

async function main() {
  const initialized = await request('initialize', {
    protocolVersion: '2024-11-05',
    capabilities: {},
    clientInfo: { name: 'metaharness-integration-test', version: '1.0.0' },
  });
  assert.ok(initialized.result);
  child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`);

  const listed = await request('tools/list', {});
  const names = listed.result.tools.map((tool) => tool.name);
  for (const name of [
    'metaharness_status',
    'metaharness_route',
    'metaharness_replay_verify',
    'metaharness_flywheel_gate',
    'metaharness_workspace_probe',
    'metaharness_reward_hack_scan',
  ]) {
    assert.ok(names.includes(name), `${name} must be available in the readonly profile`);
  }

  const status = await request('tools/call', {
    name: 'metaharness_status',
    arguments: {},
  });
  const statusPayload = JSON.parse(status.result.content[0].text);
  assert.strictEqual(statusPayload.success, true);

  const route = await request('tools/call', {
    name: 'metaharness_route',
    arguments: {
      rows: [{ embedding: [1, 0], scores: { cheap: 0.91, frontier: 0.99 } }],
      prices: { cheap: 1, frontier: 20 },
      query_embedding: [1, 0],
      quality_bar: 0.9,
    },
  });
  const routePayload = JSON.parse(route.result.content[0].text);
  assert.strictEqual(routePayload.route.id, 'cheap');

  console.log('MetaHarness MCP integration checks passed');
}

main()
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  })
  .finally(() => {
    child.stdin.end();
    child.kill('SIGTERM');
  });
