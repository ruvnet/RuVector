/**
 * ADR-210 wave 2 — D0 bypass closure (bin/mcp-server.js) + security pass.
 *
 * D0 closure: the MCP server's Intelligence class writes to the same
 * .ruvector/intelligence.json as the CLI and previously bypassed the
 * embedding-provenance invariant. It now enforces the same contract through
 * the same shared dist module: mismatched vector writes refused naming both
 * sides, legacy stores read-only, hooks_import gated.
 *
 * Security pass (untrusted inputs must not crash):
 *   - env rollout flags: unexpected values fall back safely (auto/refuse)
 *   - provenance JSON from disk: malformed sidecar/intelligence.json records
 *     are sanitized (treated as absent), never crash
 *   - reembed on adversarial stores: missing fields refused honestly
 *   - prefix handling: text *containing* prefix-like strings is inert
 *
 * All tests here are offline-safe (RUVECTOR_EMBEDDER=hash where embedding
 * happens). The MCP tests skip when the MCP SDK cannot be resolved.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PKG = path.join(__dirname, '..');
const CLI = path.join(PKG, 'bin', 'cli.js');
const MCP_SERVER = path.join(PKG, 'bin', 'mcp-server.js');

function cleanEnv(overrides = {}) {
  const env = { ...process.env, FORCE_COLOR: '0', NO_COLOR: '1' };
  delete env.RUVECTOR_EMBEDDER;
  delete env.RUVECTOR_ONNX;
  delete env.RUVECTOR_REEMBED;
  return { ...env, ...overrides };
}

function cli(cwd, args, envOverrides = {}) {
  const r = spawnSync(process.execPath, [CLI, ...args], {
    cwd,
    encoding: 'utf8',
    timeout: 120000,
    env: cleanEnv(envOverrides),
  });
  return { code: r.status, stdout: r.stdout || '', stderr: r.stderr || '' };
}

function makeProject(intelligence) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-adr210w2-'));
  fs.mkdirSync(path.join(dir, '.ruvector'), { recursive: true });
  if (intelligence !== undefined) {
    fs.writeFileSync(
      path.join(dir, '.ruvector', 'intelligence.json'),
      typeof intelligence === 'string' ? intelligence : JSON.stringify(intelligence, null, 2)
    );
  }
  return dir;
}

const LEGACY_256_STORE = {
  memories: [
    { id: 'm1', memory_type: 'note', content: 'legacy hash-era memory one', embedding: Array(256).fill(0.0625), timestamp: 1 },
    { id: 'm2', memory_type: 'note', content: 'legacy hash-era memory two', embedding: Array(256).fill(0.05), timestamp: 2 },
  ],
  stats: { total_memories: 2 },
};

// ---------------------------------------------------------------------------
// Minimal MCP stdio client (newline-delimited JSON-RPC)
// ---------------------------------------------------------------------------

class McpClient {
  constructor(cwd, envOverrides = {}) {
    this.proc = spawn(process.execPath, [MCP_SERVER], {
      cwd,
      env: cleanEnv({ RUVECTOR_EMBEDDER: 'hash', ...envOverrides }),
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.stderr = '';
    this.proc.stderr.on('data', (d) => { this.stderr += d; });
    this.buffer = '';
    this.pending = new Map();
    this.exited = new Promise((resolve) => this.proc.on('exit', resolve));
    this.startupError = null;
    this.proc.on('error', (e) => { this.startupError = e; });
    this.proc.stdout.on('data', (d) => {
      this.buffer += d;
      let nl;
      while ((nl = this.buffer.indexOf('\n')) >= 0) {
        const line = this.buffer.slice(0, nl).trim();
        this.buffer = this.buffer.slice(nl + 1);
        if (!line) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.id !== undefined && this.pending.has(msg.id)) {
            const { resolve } = this.pending.get(msg.id);
            this.pending.delete(msg.id);
            resolve(msg);
          }
        } catch { /* ignore non-JSON stdout noise */ }
      }
    });
    this.seq = 0;
  }

  request(method, params, timeoutMs = 30000) {
    const id = ++this.seq;
    const msg = { jsonrpc: '2.0', id, method, params };
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP request ${method} timed out. stderr:\n${this.stderr.slice(-2000)}`));
      }, timeoutMs);
      timer.unref?.();
      this.pending.set(id, { resolve: (m) => { clearTimeout(timer); resolve(m); } });
      this.proc.stdin.write(JSON.stringify(msg) + '\n');
    });
  }

  notify(method, params) {
    this.proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n');
  }

  async init() {
    const r = await this.request('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'adr210-wave2-test', version: '0.0.0' },
    });
    this.notify('notifications/initialized', {});
    return r;
  }

  async callTool(name, args) {
    const r = await this.request('tools/call', { name, arguments: args });
    if (r.error) throw new Error(`tools/call error: ${JSON.stringify(r.error)}`);
    const text = r.result?.content?.[0]?.text ?? '{}';
    return { isError: !!r.result?.isError, body: JSON.parse(text) };
  }

  async close() {
    try { this.proc.stdin.end(); } catch {}
    const killer = setTimeout(() => { try { this.proc.kill('SIGKILL'); } catch {} }, 5000);
    killer.unref?.();
    await this.exited;
    clearTimeout(killer);
  }
}

/** Start an MCP client; returns null when the server cannot start (SDK missing). */
async function startMcp(cwd, envOverrides = {}) {
  const client = new McpClient(cwd, envOverrides);
  try {
    await client.init();
    return client;
  } catch (e) {
    console.error(`startMcp failed: ${e.message}; spawn error: ${client.startupError?.message}; stderr: ${client.stderr.slice(-500)}`);
    await client.close().catch(() => {});
    return null;
  }
}

// ---------------------------------------------------------------------------
// D0 closure — bin/mcp-server.js enforces the same provenance contract
// ---------------------------------------------------------------------------

test('mcp D0: hooks_remember on a fresh store succeeds and stamps provenance', async (t) => {
  const dir = makeProject({});
  const mcp = await startMcp(dir);
  // Close the server BEFORE removing its cwd (Windows: EPERM otherwise).
  t.after(async () => {
    if (mcp) await mcp.close();
    fs.rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  });
  if (!mcp) { t.skip('MCP server could not start (SDK unavailable)'); return; }

  const r = await mcp.callTool('hooks_remember', { content: 'wave-2 fresh memory', type: 'note' });
  assert.equal(r.isError, false, JSON.stringify(r.body));
  assert.equal(r.body.success, true);
  assert.equal(r.body.stored, true);

  const saved = JSON.parse(fs.readFileSync(path.join(dir, '.ruvector', 'intelligence.json'), 'utf8'));
  assert.ok(saved.embeddingProvenance, 'first MCP vector write stamps provenance');
  assert.equal(saved.embeddingProvenance.embedderKind, 'hash');
  assert.ok(saved.memories.length >= 1);
});

test('mcp D0: a legacy store (vectors, no provenance) refuses hooks_remember writes', async (t) => {
  const dir = makeProject(LEGACY_256_STORE);
  const mcp = await startMcp(dir);
  // Close the server BEFORE removing its cwd (Windows: EPERM otherwise).
  t.after(async () => {
    if (mcp) await mcp.close();
    fs.rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  });
  if (!mcp) { t.skip('MCP server could not start (SDK unavailable)'); return; }

  const r = await mcp.callTool('hooks_remember', { content: 'must be refused', type: 'note' });
  assert.equal(r.isError, true, 'legacy vector write through MCP must be refused');
  assert.equal(r.body.success, false);
  assert.match(r.body.error, /read-only/);
  assert.match(r.body.error, /hooks reembed/);

  // Nothing was persisted.
  const saved = JSON.parse(fs.readFileSync(path.join(dir, '.ruvector', 'intelligence.json'), 'utf8'));
  assert.equal(saved.memories.length, 2);
  assert.ok(!saved.embeddingProvenance, 'refusal must not stamp provenance');
});

test('mcp D0: mismatched provenance refused naming both sides; reads still warn once', async (t) => {
  const dir = makeProject({
    embeddingProvenance: { embedderKind: 'onnx-minilm', modelId: 'all-MiniLM-L6-v2', dimension: 384, normalize: true, prefixPolicy: 'none' },
    memories: [
      { content: 'semantic memory', type: 'note', embedding: Array(384).fill(0.05), created: 't' },
    ],
    stats: { total_memories: 1 },
  });
  const mcp = await startMcp(dir); // RUVECTOR_EMBEDDER=hash → active is hash
  // Close the server BEFORE removing its cwd (Windows: EPERM otherwise).
  t.after(async () => {
    if (mcp) await mcp.close();
    fs.rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  });
  if (!mcp) { t.skip('MCP server could not start (SDK unavailable)'); return; }

  const r = await mcp.callTool('hooks_remember', { content: 'hash write into minilm store', type: 'note' });
  assert.equal(r.isError, true, 'mismatched write must be refused');
  assert.match(r.body.error, /onnx-minilm/, 'names the store side');
  assert.match(r.body.error, /hash/, 'names the active side');

  // Reads stay allowed but warn (once per process) about degraded recall.
  const rec1 = await mcp.callTool('hooks_recall', { query: 'semantic memory' });
  assert.equal(rec1.isError, false);
  await mcp.callTool('hooks_recall', { query: 'semantic memory again' });
  const warnings = (mcp.stderr.match(/recall quality degraded/g) || []).length;
  assert.equal(warnings, 1, `expected exactly 1 degraded-recall warning, got ${warnings}:\n${mcp.stderr}`);
});

test('mcp D0: hooks_import of vector memories is gated by store provenance', async (t) => {
  const dir = makeProject(LEGACY_256_STORE);
  const mcp = await startMcp(dir);
  // Close the server BEFORE removing its cwd (Windows: EPERM otherwise).
  t.after(async () => {
    if (mcp) await mcp.close();
    fs.rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  });
  if (!mcp) { t.skip('MCP server could not start (SDK unavailable)'); return; }

  // Importing vector-bearing memories into a legacy store is refused.
  const r = await mcp.callTool('hooks_import', {
    data: { memories: [{ content: 'imported', type: 'note', embedding: Array(256).fill(0.1) }] },
    merge: true,
  });
  assert.equal(r.isError, true, 'vector import into a legacy store must be refused');
  assert.match(r.body.error, /read-only|reembed/);

  // Content-only memories (no vectors) still import fine.
  const ok = await mcp.callTool('hooks_import', {
    data: { memories: [{ content: 'content-only memory', type: 'note' }] },
    merge: true,
  });
  assert.equal(ok.isError, false, JSON.stringify(ok.body));
});

// ---------------------------------------------------------------------------
// Security pass — env flags fall back safely
// ---------------------------------------------------------------------------

test('security: unexpected rollout-flag values fall back safely (no throw)', async () => {
  const prov = await import(pathToFileURL(path.join(PKG, 'dist', 'core', 'embedding-provenance.js')).href);
  const cases = [
    [{ RUVECTOR_EMBEDDER: 'bogus; rm -rf /' }, 'auto'],
    [{ RUVECTOR_EMBEDDER: 'MINILM ' }, 'minilm'], // trim+lowercase is fine
    [{ RUVECTOR_EMBEDDER: '' }, 'auto'],
    [{ RUVECTOR_ONNX: '2' }, 'auto'],
    [{ RUVECTOR_ONNX: 'true' }, 'auto'],
    [{ RUVECTOR_ONNX: '0' }, 'hash'],
    [{}, 'auto'],
  ];
  for (const [env, expected] of cases) {
    assert.equal(prov.resolveEmbedderSelection(env), expected, JSON.stringify(env));
  }
  assert.equal(prov.resolveReembedPolicy({ RUVECTOR_REEMBED: 'YES PLEASE' }), 'refuse');
  assert.equal(prov.resolveReembedPolicy({ RUVECTOR_REEMBED: ' WARN ' }), 'warn');
  assert.equal(prov.resolveReembedPolicy({}), 'refuse');
});

test('security: sanitizeProvenance rejects malformed disk records, accepts valid ones', async () => {
  const prov = await import(pathToFileURL(path.join(PKG, 'dist', 'core', 'embedding-provenance.js')).href);
  const bad = [
    null, undefined, 'hash', 42, [], ['hash'],
    {}, { embedderKind: 'hash' }, // missing dimension
    { embedderKind: 'hash', dimension: 0 },
    { embedderKind: 'hash', dimension: -5 },
    { embedderKind: 'hash', dimension: 1.5 },
    { embedderKind: 'hash', dimension: 1e9 }, // bounded
    { embedderKind: 'hash', dimension: '256' },
    { embedderKind: '', dimension: 256 },
    { embedderKind: 'x'.repeat(65), dimension: 256 },
    { embedderKind: 'hash', dimension: 256, modelId: { evil: true } },
    { embedderKind: 'hash', dimension: 256, modelId: 'x'.repeat(257) },
    { embedderKind: 'hash', dimension: 256, prefixPolicy: 'evil' },
    { embedderKind: 'hash', dimension: 256, prefixPolicy: 7 },
  ];
  for (const v of bad) {
    assert.equal(prov.sanitizeProvenance(v), null, `should reject: ${JSON.stringify(v)}`);
  }
  const ok = prov.sanitizeProvenance({ embedderKind: 'onnx-minilm', modelId: 'all-MiniLM-L6-v2', dimension: 384, normalize: 1, prefixPolicy: 'none', extra: 'ignored' });
  assert.deepEqual(ok, { embedderKind: 'onnx-minilm', modelId: 'all-MiniLM-L6-v2', dimension: 384, normalize: true, prefixPolicy: 'none' });
  // Missing optional fields default safely.
  const minimal = prov.sanitizeProvenance({ embedderKind: 'hash', dimension: 256 });
  assert.deepEqual(minimal, { embedderKind: 'hash', modelId: null, dimension: 256, normalize: false, prefixPolicy: 'none' });
});

// ---------------------------------------------------------------------------
// Security pass — malformed on-disk JSON must not crash the CLI
// ---------------------------------------------------------------------------

test('security: malformed sidecar provenance is ignored (insert neither crashes nor bogus-refuses)', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-adr210w2-db-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }));
  const dbPath = path.join(dir, 'mal.db');

  const create = cli(dir, ['create', dbPath, '-d', '4']);
  if (create.code !== 0) {
    t.skip(`VectorDB implementation unavailable: ${create.stderr.slice(-200)}`);
    return;
  }

  for (const provenance of ['garbage-string', 12345, { dimension: 1e99 }, { embedderKind: ['a'] }, []]) {
    fs.writeFileSync(`${dbPath}.meta.json`, JSON.stringify({ dimension: 4, provenance }));
    const seed = path.join(dir, 'seed.json');
    fs.writeFileSync(seed, JSON.stringify([{ id: `v-${typeof provenance}-${Math.random()}`, vector: [1, 0, 0, 0] }]));
    const r = cli(dir, ['insert', dbPath, seed]);
    assert.equal(r.code, 0, `insert must survive malformed provenance ${JSON.stringify(provenance)}: ${r.stdout}${r.stderr}`);
  }

  // A malformed dimension in the sidecar is bounded, not trusted.
  fs.writeFileSync(`${dbPath}.meta.json`, JSON.stringify({ dimension: 'enormous' }));
  const stats = cli(dir, ['stats', dbPath]);
  assert.equal(stats.code, 0, stats.stdout + stats.stderr);
});

test('security: corrupted intelligence.json (wrong shapes) does not crash hooks commands', (t) => {
  const dir = makeProject({
    memories: { not: 'an array' },
    patterns: ['not', 'an', 'object'],
    stats: 'not an object',
    embeddingProvenance: 'garbage',
    trajectories: 'nope',
    errors: 7,
  });
  t.after(() => fs.rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }));

  const stats = cli(dir, ['hooks', 'stats']);
  assert.equal(stats.code, 0, stats.stdout + stats.stderr);

  // With memories normalized away, the store is fresh: write succeeds and stamps.
  const rem = cli(dir, ['hooks', 'remember', '-t', 'note', 'recovered store write']);
  assert.equal(rem.code, 0, rem.stdout + rem.stderr);
  assert.equal(JSON.parse(rem.stdout).success, true);
  const saved = JSON.parse(fs.readFileSync(path.join(dir, '.ruvector', 'intelligence.json'), 'utf8'));
  assert.ok(saved.embeddingProvenance && typeof saved.embeddingProvenance === 'object');

  const rec = cli(dir, ['hooks', 'recall', 'anything']);
  assert.equal(rec.code, 0, rec.stdout + rec.stderr);
});

test('security: reembed on adversarial stores refuses honestly, --drop-missing recovers', (t) => {
  const dir = makeProject({
    memories: [
      { id: 'a', content: 'real text', embedding: Array(256).fill(0.1), timestamp: 1 },
      { id: 'b', embedding: Array(256).fill(0.2), timestamp: 2 }, // no source text
      null, // hostile entry
      { id: 'c', content: 42, embedding: Array(256).fill(0.3), timestamp: 3 }, // non-string content
    ],
    stats: { total_memories: 4 },
  });
  t.after(() => fs.rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }));

  const refuse = cli(dir, ['hooks', 'reembed'], { RUVECTOR_EMBEDDER: 'hash' });
  assert.notEqual(refuse.code, 0, 'reembed must refuse when source text is missing');
  const ref = JSON.parse(refuse.stdout);
  assert.equal(ref.success, false);
  assert.match(ref.error, /no retained source text/);
  assert.match(ref.hint, /--drop-missing/);

  const ok = cli(dir, ['hooks', 'reembed', '--drop-missing'], { RUVECTOR_EMBEDDER: 'hash' });
  assert.equal(ok.code, 0, ok.stdout + ok.stderr);
  const res = JSON.parse(ok.stdout);
  assert.equal(res.success, true);
  assert.equal(res.reembedded, 1);
  assert.equal(res.dropped, 3);
  assert.equal(res.provenance.embedderKind, 'hash');
});

// ---------------------------------------------------------------------------
// Security pass — prefix-like strings inside text are inert (D4)
// ---------------------------------------------------------------------------

test('security: text containing prefix-like strings is never parsed or stripped', async () => {
  const prov = await import(pathToFileURL(path.join(PKG, 'dist', 'core', 'embedding-provenance.js')).href);

  // MiniLM: no prefixes, text passes through verbatim — including text that
  // *looks* like an E5 prefix. Nothing detects or strips prefixes from content.
  assert.equal(prov.prefixText('all-MiniLM-L6-v2', 'query', 'query: drop table users'), 'query: drop table users');
  assert.equal(prov.prefixText('all-MiniLM-L6-v2', 'passage', 'passage: fake'), 'passage: fake');

  // E5: the registered prefix is ALWAYS applied, regardless of content — a
  // text pre-claiming to be a query cannot skip or duplicate-collapse the
  // policy (prefix application is concatenation, never detection).
  assert.equal(prov.prefixText('e5-small-v2', 'query', 'query: x'), 'query: query: x');
  assert.equal(prov.prefixText('e5-small-v2', 'passage', 'query: x'), 'passage: query: x');

  // Unknown/hostile model ids get the no-prefix policy (registry is trusted
  // code; lookups never execute or interpolate the id).
  assert.equal(prov.prefixText('__proto__', 'query', 'safe'), 'safe');
  assert.equal(prov.prefixText('constructor', 'query', 'safe'), 'safe');
  assert.deepEqual(prov.getModelPrefixSpec('__proto__'), { prefixPolicy: 'none', queryPrefix: '', passagePrefix: '' });
});
