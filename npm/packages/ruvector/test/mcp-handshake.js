#!/usr/bin/env node
/**
 * MCP handshake regression test.
 *
 * The static checks in `ruvector mcp test` (file exists, syntax parses, SDK
 * resolves, TOOLS array greps to N entries) all passed against a server that
 * never started: `bin/cli.js` loaded it with `require(mcpServerPath)`, so
 * `require.main === module` was false and `main()` never ran. Nothing
 * constructed a transport and `initialize` went unanswered.
 *
 * This test spawns the CLI exactly as a client does — `ruvector mcp start` —
 * and asserts the protocol actually works. It deliberately does NOT invoke
 * bin/mcp-server.js directly: run as main, that file always worked, so a
 * direct probe would have passed throughout the outage.
 */

const { spawn } = require('child_process');
const path = require('path');
const assert = require('assert');

const CLI = path.join(__dirname, '..', 'bin', 'cli.js');
const TIMEOUT_MS = Number(process.env.RUVECTOR_MCP_TEST_TIMEOUT || 45000);
// Grace window after the last expected reply, to catch stdout written *after*
// the handshake (model-loading chatter arrives late and would otherwise be missed).
const DRAIN_MS = Number(process.env.RUVECTOR_MCP_TEST_DRAIN || 3000);

const INIT = {
  jsonrpc: '2.0',
  id: 1,
  method: 'initialize',
  params: {
    protocolVersion: '2025-06-18',
    capabilities: {},
    clientInfo: { name: 'ruvector-handshake-test', version: '1.0.0' },
  },
};
const LIST = { jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} };

/**
 * Drive a full handshake against `ruvector mcp start`.
 * Resolves { init, tools, stdoutNoise } — never rejects on protocol failure,
 * so the assertions below produce readable diagnostics instead of a stack.
 */
function handshake() {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [CLI, 'mcp', 'start'], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let buf = '';
    const noise = [];
    const seen = new Map();
    let settled = false;
    let draining = false;

    const finish = (extra = {}) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      // The server exits on stdin close once a transport is connected; kill is
      // a backstop so a regression cannot leak a worker-heavy process here.
      try { child.stdin.end(); } catch { /* already closed */ }
      try { child.kill('SIGTERM'); } catch { /* already gone */ }
      resolve({
        init: seen.get(1) || null,
        tools: seen.get(2) || null,
        stdoutNoise: noise,
        ...extra,
      });
    };

    const timer = setTimeout(() => finish({ timedOut: true }), TIMEOUT_MS);

    child.stdout.on('data', (chunk) => {
      buf += chunk.toString();
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const raw of lines) {
        const line = raw.trim();
        if (!line) continue;
        if (!line.startsWith('{')) {
          // stdio MCP requires stdout to carry ONLY newline-delimited
          // JSON-RPC. Anything else corrupts the stream for a real client.
          noise.push(line);
          continue;
        }
        try {
          const msg = JSON.parse(line);
          if (msg.id != null) seen.set(msg.id, msg);
        } catch {
          noise.push(line);
        }
      }
      // Do NOT finish the moment both responses land. The stdout noise this
      // test exists to catch (e.g. the ONNX loader's cache messages) is
      // emitted *after* the initialize reply, so returning immediately races
      // past it and the noise assertion silently passes. Drain briefly first.
      if (seen.has(1) && seen.has(2) && !draining) {
        draining = true;
        setTimeout(() => finish(), DRAIN_MS);
      }
    });

    child.on('error', (err) => finish({ spawnError: err.message }));
    child.on('exit', (code) => finish({ exitedEarly: true, exitCode: code }));

    child.stdin.write(JSON.stringify(INIT) + '\n');
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) + '\n');
    child.stdin.write(JSON.stringify(LIST) + '\n');
  });
}

async function main() {
  console.log('\nMCP Handshake Test');
  console.log('-'.repeat(40));

  const r = await handshake();
  let failed = 0;
  const check = (name, fn) => {
    try {
      fn();
      console.log(`  PASS  ${name}`);
    } catch (e) {
      failed++;
      console.log(`  FAIL  ${name}\n        ${e.message}`);
    }
  };

  check('`ruvector mcp start` answers initialize', () => {
    assert.ok(!r.spawnError, `spawn failed: ${r.spawnError}`);
    assert.ok(
      !r.timedOut || r.init,
      `no initialize response within ${TIMEOUT_MS}ms — the server started but ` +
      'never connected a transport (see bin/cli.js `mcp start`)'
    );
    assert.ok(r.init, 'no initialize response received');
    assert.ok(r.init.result, `initialize returned an error: ${JSON.stringify(r.init.error)}`);
  });

  check('initialize reports a protocol version and serverInfo', () => {
    assert.ok(r.init && r.init.result, 'no initialize result to inspect');
    assert.ok(r.init.result.protocolVersion, 'missing protocolVersion');
    assert.strictEqual(r.init.result.serverInfo.name, 'ruvector');
  });

  check('tools/list returns a non-empty tool set', () => {
    assert.ok(r.tools, 'no tools/list response received');
    assert.ok(r.tools.result, `tools/list returned an error: ${JSON.stringify(r.tools.error)}`);
    const tools = r.tools.result.tools || [];
    assert.ok(tools.length > 0, 'tools/list returned zero tools');
    assert.ok(tools.every((t) => t.name), 'a tool is missing its name');
  });

  check('stdout carries only JSON-RPC (no diagnostic noise)', () => {
    assert.strictEqual(
      r.stdoutNoise.length,
      0,
      `non-JSON line(s) on stdout would corrupt the stream for a real client: ` +
      JSON.stringify(r.stdoutNoise.slice(0, 3))
    );
  });

  const total = 4;
  console.log(`\nResults: ${total - failed} passed, ${failed} failed`);
  if (failed) {
    console.log('MCP HANDSHAKE TEST FAILED');
    process.exit(1);
  }
  console.log('ALL TESTS PASSED');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
