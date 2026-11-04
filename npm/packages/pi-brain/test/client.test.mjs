// Regression tests for the π Brain client and MCP stdio server.
// Run with `npm test` (builds first). Uses a loopback HTTP mock — no network.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { PiBrainClient } from '../dist/index.js';

const CLI = fileURLToPath(new URL('../dist/cli.js', import.meta.url));
const TEST_KEY = 'test-key-not-real';
let server;
let base;
const seen = [];

before(async () => {
  server = http.createServer((req, res) => {
    seen.push({ method: req.method, url: req.url, auth: req.headers.authorization });
    if (req.method === 'DELETE') {
      res.writeHead(204);
      return res.end();
    }
    if (req.url.startsWith('/v1/hang')) return; // never respond
    if (req.url.startsWith('/v1/html')) {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      return res.end('<html>not json</html>');
    }
    if (req.url.startsWith('/v1/huge-error')) {
      res.writeHead(500);
      return res.end('x'.repeat(100_000));
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true, url: req.url }));
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => server.close());

const client = () => new PiBrainClient({ url: base, apiKey: TEST_KEY });

test('delete resolves on 204 No Content instead of throwing', async () => {
  const out = await client().delete('00000000-0000-0000-0000-000000000001');
  assert.equal(out, null);
});

test('memory ids are encoded so they cannot escape their path segment', async () => {
  seen.length = 0;
  await client().get('../../v1/transfer');
  await client().vote('a/../../b?x=1#f', 'up');
  await client().delete('../status');
  assert.deepEqual(
    seen.map((s) => `${s.method} ${s.url}`),
    [
      'GET /v1/memories/..%2F..%2Fv1%2Ftransfer',
      'POST /v1/memories/a%2F..%2F..%2Fb%3Fx%3D1%23f/vote',
      'DELETE /v1/memories/..%2Fstatus',
    ],
  );
});

test('refuses to send the bearer token over plain http to a non-loopback host', () => {
  assert.throws(
    () => new PiBrainClient({ url: 'http://pi.example.com', apiKey: TEST_KEY }),
    /https/,
  );
  assert.throws(() => new PiBrainClient({ url: 'not a url', apiKey: TEST_KEY }), /Invalid/);
  assert.doesNotThrow(() => new PiBrainClient({ url: 'http://localhost:9', apiKey: TEST_KEY }));
  assert.doesNotThrow(() => new PiBrainClient({ url: 'https://pi.ruv.io', apiKey: TEST_KEY }));
});

test('trailing slash on base URL does not produce a double slash', async () => {
  seen.length = 0;
  await new PiBrainClient({ url: base + '/', apiKey: TEST_KEY }).status();
  assert.equal(seen[0].url, '/v1/status');
});

test('requests time out instead of hanging forever', async () => {
  const c = new PiBrainClient({ url: base, apiKey: TEST_KEY, timeoutMs: 200 });
  await assert.rejects(c['request']('GET', '/v1/hang'), /timed out/);
});

test('non-JSON 2xx body yields a clear error', async () => {
  await assert.rejects(client()['request']('GET', '/v1/html'), /non-JSON/);
});

test('error bodies are truncated', async () => {
  await assert.rejects(client()['request']('GET', '/v1/huge-error'), (e) => {
    assert.match(e.message, /π error \(500\)/);
    assert.ok(e.message.length < 1000, `message length ${e.message.length}`);
    return true;
  });
});

function runMcp(lines) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [CLI, 'mcp'], {
      env: { ...process.env, BRAIN_URL: base, PI: TEST_KEY },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let out = '';
    child.stdout.on('data', (d) => (out += d));
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error('mcp server did not exit after stdin closed'));
    }, 5000);
    child.on('exit', () => {
      clearTimeout(timer);
      resolve(out.split('\n').filter(Boolean).map((l) => JSON.parse(l)));
    });
    for (const l of lines) child.stdin.write(l + '\n');
    child.stdin.end();
  });
}

test('mcp stdio: no reply to notifications, ping works, exits on stdin close', async () => {
  const replies = await runMcp([
    JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: {} }),
    JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }),
    JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'ping' }),
    JSON.stringify({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'brain_delete', arguments: { id: 'abc' } } }),
  ]);
  assert.deepEqual(replies.map((r) => r.id).sort(), [1, 2, 3]);
  const ping = replies.find((r) => r.id === 2);
  assert.deepEqual(ping.result, {});
  const del = replies.find((r) => r.id === 3);
  assert.notEqual(del.result.isError, true, JSON.stringify(del));
});
