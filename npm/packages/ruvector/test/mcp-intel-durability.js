#!/usr/bin/env node

/**
 * Regression coverage for #995: the MCP server's own Intelligence copy must
 * (a) quarantine a corrupt intelligence.json instead of silently treating it as
 * empty and later overwriting it, and (b) save through temp-file + rename so a
 * concurrent hook process never reads a torn file.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const MCP_SERVER = path.join(__dirname, '..', 'bin', 'mcp-server.js');

function rpc(child, id, method, params, timeoutMs = 20_000) {
  return new Promise((resolve, reject) => {
    let buf = '';
    const timer = setTimeout(() => reject(new Error(`timeout waiting for ${method}`)), timeoutMs);
    const onData = (chunk) => {
      buf += chunk.toString();
      for (const line of buf.split(/\r?\n/).filter(Boolean)) {
        let msg;
        try { msg = JSON.parse(line); } catch { continue; }
        if (msg.id === id) {
          clearTimeout(timer);
          child.stdout.off('data', onData);
          resolve(msg);
          return;
        }
      }
    };
    child.stdout.on('data', onData);
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
  });
}

async function main() {
  const source = fs.readFileSync(MCP_SERVER, 'utf8');
  assert.doesNotMatch(
    source,
    /fs\.writeFileSync\(this\.intelPath/,
    'mcp-server.js must not write intelligence.json with a plain writeFileSync',
  );

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ruv995-'));
  const store = path.join(tmp, '.ruvector', 'intelligence.json');
  fs.mkdirSync(path.dirname(store), { recursive: true });
  const torn = '{"memories": [{"content": "keep me"}, {"content": "trunc';
  fs.writeFileSync(store, torn);

  const child = spawn(process.execPath, [MCP_SERVER], {
    cwd: tmp,
    env: { ...process.env, NO_COLOR: '1', HOME: tmp },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  let stderr = '';
  child.stderr.on('data', (c) => { stderr += c.toString(); });

  try {
    const init = await rpc(child, 1, 'initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'ruvector-995-regression', version: '1.0.0' },
    });
    assert.ok(init.result, `initialize failed: ${JSON.stringify(init)}`);

    const call = await rpc(child, 2, 'tools/call', {
      name: 'hooks_remember',
      arguments: { content: 'after quarantine', type: 'test' },
    });
    assert.ok(call.result, `hooks_remember failed: ${JSON.stringify(call)}`);

    const entries = fs.readdirSync(path.dirname(store));
    const quarantined = entries.filter((f) => f.startsWith('intelligence.json.corrupt-'));
    assert.strictEqual(quarantined.length, 1, `expected one quarantine file, got ${entries.join(', ')}`);
    assert.strictEqual(
      fs.readFileSync(path.join(path.dirname(store), quarantined[0]), 'utf8'),
      torn,
      'quarantined file must hold the original bytes',
    );
    assert.match(stderr, /is corrupt/, 'corrupt load must be reported on stderr');

    const saved = JSON.parse(fs.readFileSync(store, 'utf8'));
    assert.ok(Array.isArray(saved.memories), 'new store must be valid JSON with memories[]');
    assert.deepStrictEqual(
      entries.filter((f) => f.includes('.tmp.')),
      [],
      'atomic save must not leave temp files behind',
    );
  } finally {
    child.stdin.end();
    child.kill('SIGTERM');
    fs.rmSync(tmp, { recursive: true, force: true });
  }

  console.log('MCP intelligence store durability checks passed (#995)');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
