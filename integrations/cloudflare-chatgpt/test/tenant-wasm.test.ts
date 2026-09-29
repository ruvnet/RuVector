import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { SignJWT, generateKeyPair } from 'jose';
import { AccessError, authenticate, authConfigured, requireTenant, type AuthEnv } from '../src/auth.ts';
import { loadVectors } from '../src/storage.ts';
import { searchExact } from '../src/vector.ts';
import { calculateMinCut } from '../src/mincut.ts';
import { invokeService, serviceReady, type ServiceEnv } from '../src/services.ts';
import { requestTooLarge } from '../src/limits.ts';

const wasm = (name: string) => new WebAssembly.Module(readFileSync(fileURLToPath(import.meta.resolve(name))));

test('RVF WASM ranks real vectors and rejects malformed dimensions', async () => {
  const records = [
    { id: 'near', vector: [1, 0, 0] },
    { id: 'far', vector: [0, 1, 0] },
  ];
  const hits = await searchExact(wasm('@ruvector/rvf-wasm/wasm'), records, [0.95, 0.05, 0], 2);
  assert.deepEqual(hits.map(h => h.id), ['near', 'far']);
  assert.ok(hits[0].distance < hits[1].distance);
  await assert.rejects(searchExact(wasm('@ruvector/rvf-wasm/wasm'), records, [1, 0], 2), /invalid_vector/);
});

test('RuVector mincut WASM executes weighted graph analysis', () => {
  const result = calculateMinCut(wasm('@ruvector/mincut-wasm/wasm'), [
    [0, 1, 1], [1, 2, 1], [0, 2, 5],
  ]);
  assert.equal(result.value, 2);
  assert.equal(result.edgeCount, 3);
});

test('verified subject can use only enrolled tenant and viewer cannot write', async () => {
  const { publicKey, privateKey } = await generateKeyPair('RS256');
  const issuer = 'https://id.example.test';
  const token = await new SignJWT({ scope: 'ruvector.read' }).setProtectedHeader({ alg: 'RS256' })
    .setIssuer(issuer).setAudience('https://ruvector.example/mcp').setSubject('alice')
    .setIssuedAt().setExpirationTime('5m').sign(privateKey);
  const queries: unknown[][] = [];
  const database = {
    prepare() {
      return {
        bind(...args: unknown[]) { queries.push(args); return this; },
        async all() { return { results: [{ tenant_id: 'acme', role: 'viewer' }] }; },
      };
    },
  } as unknown as D1Database;
  const env: AuthEnv = { DB: database, OAUTH_ISSUER: issuer,
    OAUTH_AUDIENCE: 'https://ruvector.example/mcp', OAUTH_JWKS_URL: `${issuer}/keys` };
  assert.equal(authConfigured(env), true);
  assert.equal(authConfigured({ ...env, OAUTH_AUDIENCE: 'ruvector-chatgpt' }), false);
  const principal = await authenticate(new Request('https://worker.example/mcp', {
    headers: { authorization: `Bearer ${token}` },
  }), env, async () => publicKey);
  assert.deepEqual(queries, [[issuer, 'alice']]);
  assert.equal(requireTenant(principal, 'acme').role, 'viewer');
  assert.throws(() => requireTenant(principal, 'other'), (e: unknown) => e instanceof AccessError && e.status === 403);
  assert.throws(() => requireTenant(principal, 'acme', true), (e: unknown) => e instanceof AccessError && e.status === 403);
  const editor = { ...principal, memberships: [{ tenant_id: 'acme', role: 'editor' as const }] };
  assert.throws(() => requireTenant(editor, 'acme', true), (e: unknown) => e instanceof AccessError && e.status === 403);
  await assert.rejects(authenticate(new Request('https://worker.example/mcp', {
    headers: { authorization: `Bearer ${token}tampered` },
  }), env, async () => publicKey), (e: unknown) => e instanceof AccessError && e.status === 401);
});

test('D1 collection reads bind tenant and never fetch another tenant rows', async () => {
  const calls: unknown[][] = [];
  const db = {
    prepare(sql: string) {
      assert.match(sql, /WHERE tenant_id = \? AND collection = \?/);
      return {
        bind(...args: unknown[]) { calls.push(args); return this; },
        async all() { return { results: [{ id: 'one', dimension: 2, vector_json: '[1,0]', metadata_json: '{}' }] }; },
      };
    },
  } as unknown as D1Database;
  const env = { DB: db } as AuthEnv;
  const rows = await loadVectors(env, 'acme', 'notes');
  assert.deepEqual(calls, [['acme', 'notes']]);
  assert.deepEqual(rows.map(row => row.id), ['one']);
});

test('service binding receives signed identity context at a fixed internal route', async () => {
  let inspected = false;
  const secret = 'this-is-a-test-key-with-more-than-32-bytes';
  const env: ServiceEnv = {
    SERVICE_SIGNING_KEY: secret,
    METAHARNESS: { async fetch(input: RequestInfo | URL, init?: RequestInit) {
      assert.equal(String(input), 'https://ruvector.internal/ruvector-internal/v1');
      const payload = String(init?.body);
      const context = JSON.parse(payload);
      assert.equal(context.tenantId, 'acme');
      assert.equal(context.actor, 'alice');
      assert.equal(context.input.promote, false);
      const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(secret),
        { name: 'HMAC', hash: 'SHA-256' }, false, ['verify']);
      const hex = new Headers(init?.headers).get('x-ruvector-signature')!;
      const signature = Uint8Array.from(hex.match(/../g)!, byte => parseInt(byte, 16));
      assert.equal(await crypto.subtle.verify('HMAC', key, signature, new TextEncoder().encode(payload)), true);
      inspected = true;
      return new Response(JSON.stringify({ accepted: true }));
    } } as Fetcher,
  };
  assert.equal(serviceReady(env, 'METAHARNESS'), true);
  const result = await invokeService(env, 'METAHARNESS', {
    issuer: 'https://id.example.test', subject: 'alice', scopes: new Set(['ruvector.read', 'ruvector.write']),
    memberships: [{ tenant_id: 'acme', role: 'editor' }],
  }, 'acme', 'evaluate_candidate', { promote: false });
  assert.equal(result.accepted, true);
  assert.equal(inspected, true);
});

test('chunked requests are bounded without trusting Content Length', async () => {
  const request = new Request('https://worker.example/mcp', { method: 'POST',
    body: new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array(8)); controller.enqueue(new Uint8Array(8)); controller.close(); } }),
    duplex: 'half',
  } as RequestInit);
  assert.equal(await requestTooLarge(request, 12), true);
});
