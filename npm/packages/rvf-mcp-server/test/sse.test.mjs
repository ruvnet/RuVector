// SSE transport tests: spawn the CLI and talk raw HTTP to it.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import http from 'node:http';
import { fileURLToPath } from 'node:url';

const CLI = fileURLToPath(new URL('../dist/cli.js', import.meta.url));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function startServer(extraArgs = []) {
  const port = 39000 + Math.floor(Math.random() * 2000);
  const proc = spawn(process.execPath, [CLI, '--transport', 'sse', '--port', String(port), ...extraArgs], {
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  let stderr = '';
  proc.stderr.on('data', (d) => { stderr += d; });
  const state = { exited: null };
  proc.on('exit', (code) => { state.exited = code; });
  for (let i = 0; i < 50 && !/listening/.test(stderr); i++) await wait(50);
  return { port, proc, state, stderr: () => stderr };
}

/** Open an SSE stream; resolves with { req, endpoint, events } once the endpoint event arrives. */
function openSse(port) {
  return new Promise((resolve, reject) => {
    const events = [];
    const req = http.get({ host: '127.0.0.1', port, path: '/sse' }, (res) => {
      if (res.statusCode !== 200) {
        res.resume();
        resolve({ req, status: res.statusCode, events });
        return;
      }
      let buf = '';
      let endpoint;
      res.setEncoding('utf8');
      res.on('data', (chunk) => {
        buf += chunk;
        let idx;
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const block = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const ev = /event: (.*)/.exec(block)?.[1];
          const data = /data: (.*)/.exec(block)?.[1];
          events.push({ ev, data });
          if (ev === 'endpoint' && !endpoint) {
            endpoint = data;
            resolve({ req, status: 200, endpoint, events });
          }
        }
      });
    });
    req.on('error', reject);
  });
}

function post(port, path, body) {
  return new Promise((resolve) => {
    const req = http.request({
      host: '127.0.0.1', port, path, method: 'POST', headers: { 'content-type': 'application/json' },
    }, (res) => { res.resume(); res.on('end', () => resolve(res.statusCode)); });
    req.on('error', (e) => resolve(e.code));
    req.end(JSON.stringify(body));
  });
}

test('POST after the SSE stream closes does not crash the server', { timeout: 10_000 }, async (t) => {
  const s = await startServer();
  t.after(() => s.proc.kill());
  const a = await openSse(s.port);
  a.req.destroy();
  await wait(150);
  const status = await post(s.port, a.endpoint, { jsonrpc: '2.0', id: 1, method: 'ping' });
  await wait(300);
  assert.equal(s.state.exited, null, `server exited; stderr:\n${s.stderr()}`);
  assert.ok(typeof status === 'number' && status >= 400 && status < 500, `got ${status}`);

  // A fresh client can still connect and get answers on its stream.
  const b = await openSse(s.port);
  assert.equal(b.status, 200);
  assert.equal(await post(s.port, b.endpoint, { jsonrpc: '2.0', id: 7, method: 'ping' }), 202);
  await wait(150);
  assert.ok(b.events.some((e) => e.ev === 'message' && /"id":7/.test(e.data)));
  b.req.destroy();
});

test('a second concurrent SSE client cannot hijack the first session', { timeout: 10_000 }, async (t) => {
  const s = await startServer();
  t.after(() => s.proc.kill());
  const a = await openSse(s.port);
  const b = await openSse(s.port);
  assert.equal(b.status, 409);
  b.req.destroy();

  // First client is unaffected.
  assert.equal(await post(s.port, a.endpoint, { jsonrpc: '2.0', id: 3, method: 'ping' }), 202);
  // Messages without the right sessionId are refused.
  assert.equal(await post(s.port, '/messages?sessionId=bogus', { jsonrpc: '2.0', id: 4, method: 'ping' }), 404);
  assert.equal(await post(s.port, '/messages', { jsonrpc: '2.0', id: 5, method: 'ping' }), 404);
  await wait(150);
  assert.ok(a.events.some((e) => e.ev === 'message' && /"id":3/.test(e.data)));
  assert.ok(!a.events.some((e) => e.ev === 'message' && /"id":4/.test(e.data)));
  a.req.destroy();
  assert.equal(s.state.exited, null);
});

test('SSE binds to loopback by default and rejects foreign Host headers', { timeout: 10_000 }, async (t) => {
  const s = await startServer();
  t.after(() => s.proc.kill());
  assert.match(s.stderr(), /127\.0\.0\.1/);
  const a = await openSse(s.port);
  const status = await new Promise((resolve) => {
    const req = http.request({
      host: '127.0.0.1', port: s.port, path: a.endpoint, method: 'POST',
      headers: { 'content-type': 'application/json', host: 'evil.example:80' },
    }, (res) => { res.resume(); res.on('end', () => resolve(res.statusCode)); });
    req.end(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'ping' }));
  });
  assert.equal(status, 403);
  a.req.destroy();

  const getStatus = await new Promise((resolve) => {
    const req = http.get({
      host: '127.0.0.1', port: s.port, path: '/sse', headers: { host: 'evil.example:80' },
    }, (res) => { res.resume(); resolve(res.statusCode); req.destroy(); });
  });
  assert.equal(getStatus, 403);
});
