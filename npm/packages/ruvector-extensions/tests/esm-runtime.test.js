/**
 * Regression tests for the ESM build: the package is "type": "module", so
 * CommonJS globals (`require`, `__dirname`) do not exist at runtime.
 */

import { test } from 'node:test';
import assert from 'node:assert';
import { existsSync } from 'node:fs';
import { AnthropicEmbeddings } from '../dist/embeddings.js';
import { UIServer } from '../dist/ui-server.js';
import { DatabasePersistence } from '../dist/persistence.js';

test('AnthropicEmbeddings loads its (installed) SDK under ESM', () => {
  assert.doesNotThrow(() => new AnthropicEmbeddings({ apiKey: 'test-key' }));
});

const fakeDb = {
  list: async () => [],
  get: async () => null,
  query: async () => [],
  getStats: async () => ({ count: 0 }),
};

test('UIServer constructs under ESM and serves the bundled UI', async () => {
  const server = new UIServer(fakeDb, 0);
  await server.start();
  try {
    const addr = server.address();
    assert.strictEqual(addr.address, '127.0.0.1', 'binds loopback by default');
    const res = await fetch(`http://127.0.0.1:${addr.port}/`);
    assert.strictEqual(res.status, 200);
    assert.match(await res.text(), /<html/i);
    const health = await fetch(`http://127.0.0.1:${addr.port}/health`);
    assert.strictEqual(health.headers.get('access-control-allow-origin'), null);
  } finally {
    await server.stop();
  }
});

test('UIServer rejects cross-origin WebSocket connections', async () => {
  const { WebSocket } = await import('ws');
  const server = new UIServer(fakeDb, 0);
  await server.start();
  const url = `ws://127.0.0.1:${server.address().port}`;
  const open = (origin) => new Promise((resolve) => {
    const ws = new WebSocket(url, { origin });
    ws.on('open', () => { ws.close(); resolve('open'); });
    ws.on('error', () => resolve('rejected'));
  });
  try {
    assert.strictEqual(await open('https://evil.example'), 'rejected');
    assert.strictEqual(await open(`http://127.0.0.1:${server.address().port}`), 'open');
  } finally {
    await server.stop();
  }
});

test('UIServer.start rejects when the port is already in use', async () => {
  const first = new UIServer(fakeDb, 0);
  await first.start();
  try {
    const second = new UIServer(fakeDb, first.address().port);
    await assert.rejects(second.start(), /EADDRINUSE/);
  } finally {
    await first.stop();
  }
});

test('build copies the UI assets into dist/ui', () => {
  assert.ok(existsSync(new URL('../dist/ui/index.html', import.meta.url)));
});

test('unwritable baseDir surfaces on save instead of an unhandled rejection', async () => {
  const p = new DatabasePersistence({ stats: () => ({}) }, { baseDir: '/etc/hostname/ruvx-not-a-dir' });
  await assert.rejects(p.save());
});

test('snapshot ids cannot escape the snapshots directory', async () => {
  const p = new DatabasePersistence({ stats: () => ({}) }, { baseDir: '/etc/hostname/ruvx-not-a-dir' });
  for (const id of ['../../etc/passwd', 'a/b', '..\\x', '']) {
    await assert.rejects(p.deleteSnapshot(id), /Invalid snapshot id/);
    await assert.rejects(p.restoreSnapshot(id), /Invalid snapshot id/);
  }
});
