/**
 * Edge-Net Relay Brain API Bridge — Tests
 *
 * Tests the relay server's WebSocket handling, identity derivation,
 * rate limiting, rUv accounting, and brain API proxy routing.
 *
 * Uses Node.js built-in test runner (node:test).
 */

import { describe, it, before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import http from 'node:http';
import { WebSocket } from 'ws';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Generate a valid 32-byte Ed25519 public key (hex) for testing. */
function makePublicKey(seed = 0) {
  const buf = Buffer.alloc(32);
  buf[0] = seed & 0xff;
  buf[1] = (seed >> 8) & 0xff;
  buf.fill(0xab, 2);
  return buf.toString('hex');
}

/** Derive expected SHAKE-256 pseudonym for a key (mirrors relay logic). */
function expectedPseudonym(publicKeyHex) {
  const h = createHash('shake256', { outputLength: 16 });
  h.update(Buffer.from(publicKeyHex, 'hex'));
  return h.digest('hex');
}

/**
 * Connect a WebSocket to the relay and optionally authenticate.
 * @param {number} port
 * @param {string|null} publicKey - If provided, sends auth message and waits for auth_result.
 * @returns {Promise<{ws: WebSocket, messages: Object[], pseudonym: string|null}>}
 */
function connectClient(port, publicKey = null) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
    const messages = [];
    let pseudonym = null;

    ws.on('error', reject);

    ws.on('message', (data) => {
      const msg = JSON.parse(data.toString());
      messages.push(msg);

      // After receiving welcome, optionally authenticate.
      if (msg.type === 'welcome' && publicKey) {
        ws.send(JSON.stringify({ id: 'auth-1', type: 'auth', payload: { public_key: publicKey } }));
      }

      if (msg.type === 'auth_result' && msg.ok) {
        pseudonym = msg.data.pseudonym;
        resolve({ ws, messages, pseudonym });
      }

      if (msg.type === 'auth_result' && !msg.ok) {
        resolve({ ws, messages, pseudonym: null });
      }
    });

    // If no auth needed, resolve after welcome.
    if (!publicKey) {
      ws.on('open', () => {
        // Wait for welcome message.
        const check = setInterval(() => {
          if (messages.length > 0) {
            clearInterval(check);
            resolve({ ws, messages, pseudonym: null });
          }
        }, 10);
      });
    }
  });
}

/**
 * Send a message and wait for a response of the expected type.
 * @param {WebSocket} ws
 * @param {Object} msg
 * @param {string} expectedType
 * @param {number} timeout
 * @returns {Promise<Object>}
 */
function sendAndWait(ws, msg, expectedType, timeout = 5000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Timeout waiting for ${expectedType}`)), timeout);

    const handler = (data) => {
      const parsed = JSON.parse(data.toString());
      if (parsed.type === expectedType) {
        ws.off('message', handler);
        clearTimeout(timer);
        resolve(parsed);
      }
    };

    ws.on('message', handler);
    ws.send(JSON.stringify(msg));
  });
}

/** Fetch JSON from the relay's HTTP endpoint. */
async function httpGet(port, path) {
  return new Promise((resolve, reject) => {
    http.get(`http://127.0.0.1:${port}${path}`, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(body) });
        } catch {
          resolve({ status: res.statusCode, data: body });
        }
      });
    }).on('error', reject);
  });
}

// ---------------------------------------------------------------------------
// Test Suite
// ---------------------------------------------------------------------------

describe('Edge-Net Relay Brain API Bridge', () => {
  /** @type {import('child_process').ChildProcess} */
  let relayProcess;
  let relayPort;

  before(async () => {
    // Start the relay on a random port for test isolation.
    relayPort = 10000 + Math.floor(Math.random() * 50000);

    const { spawn } = await import('node:child_process');
    relayProcess = spawn('node', ['index.js'], {
      cwd: '/workspaces/ruvector/examples/edge-net/relay',
      env: { ...process.env, PORT: String(relayPort), BRAIN_API_BASE: 'http://127.0.0.1:19999' },
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    // Wait for the relay to start listening.
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Relay failed to start within 5s')), 5000);
      relayProcess.stdout.on('data', (data) => {
        if (data.toString().includes('listening on port')) {
          clearTimeout(timer);
          resolve();
        }
      });
      relayProcess.stderr.on('data', (data) => {
        // Log stderr but don't fail — Node warnings are fine.
      });
      relayProcess.on('exit', (code) => {
        if (code !== null) {
          clearTimeout(timer);
          reject(new Error(`Relay exited with code ${code}`));
        }
      });
    });
  });

  after(() => {
    if (relayProcess) {
      relayProcess.kill('SIGTERM');
    }
  });

  // ---- HTTP Health Endpoint ----

  describe('HTTP health endpoint', () => {
    it('returns health status at /', async () => {
      const { status, data } = await httpGet(relayPort, '/');
      assert.equal(status, 200);
      assert.equal(data.status, 'ok');
      assert.equal(data.service, 'edge-net-relay');
      assert.equal(data.version, '0.2.0');
      assert.equal(typeof data.connected_nodes, 'number');
      assert.equal(typeof data.uptime_seconds, 'number');
    });

    it('returns health status at /health', async () => {
      const { status, data } = await httpGet(relayPort, '/health');
      assert.equal(status, 200);
      assert.equal(data.status, 'ok');
    });

    it('returns stats at /stats', async () => {
      const { status, data } = await httpGet(relayPort, '/stats');
      assert.equal(status, 200);
      assert.equal(typeof data.nodes, 'number');
      assert.ok(data.rate_limits);
      assert.equal(data.rate_limits.reads_per_hour, 1000);
      assert.equal(data.rate_limits.writes_per_hour, 100);
    });

    it('returns 404 for unknown paths', async () => {
      const { status, data } = await httpGet(relayPort, '/nonexistent');
      assert.equal(status, 404);
      assert.equal(data.error, 'Not found');
    });
  });

  // ---- WebSocket Connection & Auth ----

  describe('WebSocket auth handshake', () => {
    it('sends welcome message on connect', async () => {
      const { ws, messages } = await connectClient(relayPort);
      assert.equal(messages[0].type, 'welcome');
      assert.ok(messages[0].data.supported_types);
      assert.equal(messages[0].data.auth_required, true);
      ws.close();
    });

    it('authenticates with valid Pi-Key public key', async () => {
      const pubKey = makePublicKey(1);
      const { ws, pseudonym } = await connectClient(relayPort, pubKey);
      assert.ok(pseudonym);
      assert.equal(pseudonym, expectedPseudonym(pubKey));
      assert.equal(pseudonym.length, 32); // 16 bytes = 32 hex chars
      ws.close();
    });

    it('rejects invalid public key (wrong length)', async () => {
      const { ws, messages } = await connectClient(relayPort);
      const resp = await sendAndWait(
        ws,
        { id: 'bad-auth', type: 'auth', payload: { public_key: 'deadbeef' } },
        'auth_result',
      );
      assert.equal(resp.ok, false);
      assert.ok(resp.error.includes('Invalid public key'));
      ws.close();
    });

    it('rejects non-hex public key', async () => {
      const { ws, messages } = await connectClient(relayPort);
      const resp = await sendAndWait(
        ws,
        { id: 'bad-auth2', type: 'auth', payload: { public_key: 'zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz' } },
        'auth_result',
      );
      assert.equal(resp.ok, false);
      ws.close();
    });

    it('rejects operations before auth', async () => {
      const { ws } = await connectClient(relayPort);
      const resp = await sendAndWait(
        ws,
        { id: 'pre-auth', type: 'brain_status', payload: {} },
        'brain_status_result',
      );
      assert.equal(resp.ok, false);
      assert.ok(resp.error.includes('Not authenticated'));
      ws.close();
    });
  });

  // ---- Identity Derivation ----

  describe('SHAKE-256 pseudonym derivation', () => {
    it('produces deterministic pseudonyms', async () => {
      const pubKey = makePublicKey(42);
      const { ws: ws1, pseudonym: p1 } = await connectClient(relayPort, pubKey);
      ws1.close();

      // Allow cleanup
      await new Promise((r) => setTimeout(r, 100));

      const { ws: ws2, pseudonym: p2 } = await connectClient(relayPort, pubKey);
      assert.equal(p1, p2);
      ws2.close();
    });

    it('produces different pseudonyms for different keys', async () => {
      const { ws: ws1, pseudonym: p1 } = await connectClient(relayPort, makePublicKey(1));
      const { ws: ws2, pseudonym: p2 } = await connectClient(relayPort, makePublicKey(2));
      assert.notEqual(p1, p2);
      ws1.close();
      ws2.close();
    });
  });

  // ---- rUv Accounting (local operations) ----

  describe('rUv accounting', () => {
    it('returns zero balance for new nodes', async () => {
      const { ws } = await connectClient(relayPort, makePublicKey(100));
      const resp = await sendAndWait(ws, { id: 'bal-1', type: 'ruv_balance', payload: {} }, 'ruv_balance_result');
      assert.equal(resp.ok, true);
      assert.equal(resp.data.balance, 0);
      assert.equal(resp.data.operations, 0);
      ws.close();
    });

    it('credits rUv via ruv_earn', async () => {
      const { ws } = await connectClient(relayPort, makePublicKey(101));

      const earnResp = await sendAndWait(
        ws,
        { id: 'earn-1', type: 'ruv_earn', payload: { amount: 3.5, reason: 'embedding_gen' } },
        'ruv_earn_result',
      );
      assert.equal(earnResp.ok, true);
      assert.equal(earnResp.data.credited, 3.5);
      assert.equal(earnResp.data.balance, 3.5);

      const balResp = await sendAndWait(ws, { id: 'bal-2', type: 'ruv_balance', payload: {} }, 'ruv_balance_result');
      assert.equal(balResp.data.balance, 3.5);
      assert.equal(balResp.data.operations, 1);
      ws.close();
    });

    it('rejects ruv_earn with non-positive amount', async () => {
      const { ws } = await connectClient(relayPort, makePublicKey(102));
      const resp = await sendAndWait(
        ws,
        { id: 'earn-bad', type: 'ruv_earn', payload: { amount: -1, reason: 'hack' } },
        'ruv_earn_result',
      );
      assert.equal(resp.ok, false);
      assert.ok(resp.error.includes('positive'));
      ws.close();
    });

    it('accumulates multiple earn operations', async () => {
      const { ws } = await connectClient(relayPort, makePublicKey(103));

      await sendAndWait(ws, { id: 'e1', type: 'ruv_earn', payload: { amount: 1.0, reason: 'a' } }, 'ruv_earn_result');
      await sendAndWait(ws, { id: 'e2', type: 'ruv_earn', payload: { amount: 2.0, reason: 'b' } }, 'ruv_earn_result');
      await sendAndWait(ws, { id: 'e3', type: 'ruv_earn', payload: { amount: 0.5, reason: 'c' } }, 'ruv_earn_result');

      const balResp = await sendAndWait(ws, { id: 'bal-3', type: 'ruv_balance', payload: {} }, 'ruv_balance_result');
      assert.equal(balResp.data.balance, 3.5);
      assert.equal(balResp.data.operations, 3);
      ws.close();
    });
  });

  // ---- Brain API Proxy (with mock brain unavailable) ----
  // Since the mock brain API at 127.0.0.1:19999 is not running, these
  // should return errors from the fetch failure, testing error handling.

  describe('brain API proxy (brain offline)', () => {
    it('handles brain_status when brain is unreachable', async () => {
      const { ws } = await connectClient(relayPort, makePublicKey(200));
      const resp = await sendAndWait(
        ws,
        { id: 'status-1', type: 'brain_status', payload: {} },
        'brain_status_result',
      );
      // Brain is offline, so we expect an error response (not a crash).
      assert.equal(resp.ok, false);
      assert.ok(resp.error);
      ws.close();
    });

    it('handles brain_search when brain is unreachable', async () => {
      const { ws } = await connectClient(relayPort, makePublicKey(201));
      const resp = await sendAndWait(
        ws,
        { id: 'search-1', type: 'brain_search', payload: { query: 'test query', limit: 5 } },
        'brain_search_result',
      );
      assert.equal(resp.ok, false);
      assert.ok(resp.error);
      ws.close();
    });

    it('handles brain_share when brain is unreachable', async () => {
      const { ws } = await connectClient(relayPort, makePublicKey(202));
      const resp = await sendAndWait(
        ws,
        { id: 'share-1', type: 'brain_share', payload: { title: 'Test', content: 'Hello', category: 'debug' } },
        'brain_share_result',
      );
      assert.equal(resp.ok, false);
      assert.ok(resp.error);
      ws.close();
    });

    it('handles brain_vote when brain is unreachable', async () => {
      const { ws } = await connectClient(relayPort, makePublicKey(203));
      const resp = await sendAndWait(
        ws,
        { id: 'vote-1', type: 'brain_vote', payload: { id: 'mem-123', direction: 'up' } },
        'brain_vote_result',
      );
      assert.equal(resp.ok, false);
      assert.ok(resp.error);
      ws.close();
    });

    it('handles brain_list when brain is unreachable', async () => {
      const { ws } = await connectClient(relayPort, makePublicKey(204));
      const resp = await sendAndWait(
        ws,
        { id: 'list-1', type: 'brain_list', payload: { limit: 10 } },
        'brain_list_result',
      );
      assert.equal(resp.ok, false);
      assert.ok(resp.error);
      ws.close();
    });

    it('handles brain_lora_latest when brain is unreachable', async () => {
      const { ws } = await connectClient(relayPort, makePublicKey(205));
      const resp = await sendAndWait(
        ws,
        { id: 'lora-1', type: 'brain_lora_latest', payload: {} },
        'brain_lora_latest_result',
      );
      assert.equal(resp.ok, false);
      assert.ok(resp.error);
      ws.close();
    });

    it('rejects unknown message types', async () => {
      const { ws } = await connectClient(relayPort, makePublicKey(206));
      const resp = await sendAndWait(
        ws,
        { id: 'bad-1', type: 'nonexistent_op', payload: {} },
        'nonexistent_op_result',
      );
      assert.equal(resp.ok, false);
      assert.ok(resp.error.includes('Unknown'));
      ws.close();
    });
  });

  // ---- Invalid JSON handling ----

  describe('error handling', () => {
    it('handles invalid JSON gracefully', async () => {
      const { ws, messages } = await connectClient(relayPort);
      // Wait for welcome
      await new Promise((r) => setTimeout(r, 50));

      return new Promise((resolve) => {
        ws.on('message', (data) => {
          const msg = JSON.parse(data.toString());
          if (msg.type === 'error') {
            assert.equal(msg.ok, false);
            assert.ok(msg.error.includes('Invalid JSON'));
            ws.close();
            resolve();
          }
        });
        ws.send('not valid json {{{');
      });
    });
  });

  // ---- Concurrent connections ----

  describe('concurrent connections', () => {
    it('handles multiple simultaneous clients', async () => {
      const clients = await Promise.all([
        connectClient(relayPort, makePublicKey(300)),
        connectClient(relayPort, makePublicKey(301)),
        connectClient(relayPort, makePublicKey(302)),
      ]);

      // Each client should have a unique pseudonym.
      const pseudonyms = new Set(clients.map((c) => c.pseudonym));
      assert.equal(pseudonyms.size, 3);

      // Each client can query rUv balance independently.
      const results = await Promise.all(
        clients.map((c) =>
          sendAndWait(c.ws, { id: 'bal', type: 'ruv_balance', payload: {} }, 'ruv_balance_result'),
        ),
      );

      for (const r of results) {
        assert.equal(r.ok, true);
        assert.equal(r.data.balance, 0);
      }

      for (const c of clients) c.ws.close();
    });
  });
});
