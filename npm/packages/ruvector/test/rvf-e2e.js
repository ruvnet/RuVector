#!/usr/bin/env node
/**
 * End-to-end check of the RVF integration against the real @ruvector/rvf
 * that is installed (an optional dependency, so the test skips cleanly when it
 * is absent):
 *
 *   1. library: rvf-wrapper create -> ingest -> query -> close -> reopen ->
 *      query (data survives the reopen) -> ingest more -> close.
 *   2. MCP: rvf_create -> rvf_ingest -> rvf_query -> rvf_status over stdio.
 *
 * No version is hardcoded; the installed @ruvector/rvf version is logged.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const ROOT = path.join(__dirname, '..');
const CLI = path.join(ROOT, 'bin', 'cli.js');
const wrapper = require(path.join(ROOT, 'dist', 'core', 'rvf-wrapper.js'));

if (!wrapper.isRvfAvailable()) {
  console.log('  SKIP  @ruvector/rvf not installed');
  process.exit(0);
}

let rvfVersion = 'unknown';
try { rvfVersion = require('@ruvector/rvf/package.json').version; } catch { /* exports may hide it */ }
if (rvfVersion === 'unknown') {
  try {
    const pkgPath = path.join(path.dirname(require.resolve('@ruvector/rvf')), '..', 'package.json');
    rvfVersion = JSON.parse(fs.readFileSync(pkgPath, 'utf8')).version;
  } catch { /* leave unknown */ }
}
console.log(`  @ruvector/rvf ${rvfVersion}`);

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-rvf-e2e-'));

const ENTRIES = [
  { id: 'a', vector: [1, 0, 0, 0] },
  { id: 'b', vector: [0, 1, 0, 0] },
  { id: 'c', vector: [0, 0, 1, 0] },
];

async function libraryRoundTrip() {
  const file = path.join(work, 'lib.rvf');
  let store = await wrapper.createRvfStore(file, { dimensions: 4, metric: 'cosine' });
  const ing = await wrapper.rvfIngest(store, ENTRIES);
  assert.strictEqual(ing.accepted, 3, `accepted 3, got ${JSON.stringify(ing)}`);
  assert.strictEqual(ing.rejected, 0);
  let hits = await wrapper.rvfQuery(store, [0, 0.95, 0.05, 0], 2);
  assert.strictEqual(hits.length, 2);
  assert.strictEqual(hits[0].id, 'b', `nearest is b, got ${JSON.stringify(hits)}`);
  assert.strictEqual(typeof hits[0].distance, 'number');
  await wrapper.rvfClose(store);

  store = await wrapper.openRvfStore(file);
  const status = await wrapper.rvfStatus(store);
  assert.strictEqual(status.totalVectors, 3, `reopened store has 3 vectors, got ${JSON.stringify(status)}`);
  hits = await wrapper.rvfQuery(store, [0, 0, 1, 0], 1);
  assert.strictEqual(hits[0].id, 'c', `string ids survive reopen, got ${JSON.stringify(hits)}`);
  await wrapper.rvfIngest(store, [{ id: 'd', vector: [0, 0, 0, 1] }]);
  hits = await wrapper.rvfQuery(store, [0, 0, 0, 1], 1);
  assert.strictEqual(hits[0].id, 'd');
  const del = await wrapper.rvfDelete(store, ['a']);
  assert.strictEqual(del.deleted, 1);
  await wrapper.rvfClose(store);

  store = await wrapper.openRvfStore(file);
  assert.strictEqual((await wrapper.rvfStatus(store)).totalVectors, 3);
  const compacted = await wrapper.rvfCompact(store);
  assert.strictEqual(typeof compacted.epoch, 'number', JSON.stringify(compacted));
  const segs = await store.segments();
  assert.ok(Array.isArray(segs) && segs.length > 0, `segments listed, got ${JSON.stringify(segs)}`);
  const derived = await wrapper.rvfDerive(store, path.join(work, 'derived.rvf'));
  await wrapper.rvfClose(derived);
  await wrapper.rvfClose(store);
  console.log('  PASS  rvf-wrapper create/ingest/query/close/reopen/delete/compact/segments/derive');

  // Copy-on-write branch: freeze the parent, branch it, the child sees the
  // parent's vectors plus its own edits.
  store = await wrapper.openRvfStore(file);
  const epoch = await wrapper.rvfFreeze(store);
  assert.strictEqual(typeof epoch, 'number');
  const branch = await wrapper.rvfBranch(store, path.join(work, 'branch.rvf'));
  hits = await wrapper.rvfQuery(branch, [0, 1, 0, 0], 1);
  assert.strictEqual(hits[0].id, 'b', `branch inherits parent vectors, got ${JSON.stringify(hits)}`);
  await wrapper.rvfIngest(branch, [{ id: 'e', vector: [0.5, 0.5, 0, 0] }]);
  hits = await wrapper.rvfQuery(branch, [0.5, 0.5, 0, 0], 1);
  assert.strictEqual(hits[0].id, 'e');
  await wrapper.rvfClose(branch);
  await wrapper.rvfClose(store);
  console.log('  PASS  rvf-wrapper freeze/branch (child inherits parent, accepts own edits)');
}

async function mcpRoundTrip() {
  const home = fs.mkdtempSync(path.join(work, 'mcp-'));
  const child = spawn(process.execPath, [CLI, 'mcp', 'start'], {
    cwd: home,
    env: { ...process.env, HOME: home, NO_COLOR: '1' },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  let stderr = '';
  child.stderr.on('data', (c) => { stderr += c.toString(); });

  let nextId = 1;
  let buffer = '';
  const pending = new Map();
  child.stdout.on('data', (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (!line) continue;
      let message;
      try { message = JSON.parse(line); } catch { continue; }
      const waiter = pending.get(message.id);
      if (waiter) { pending.delete(message.id); waiter(message); }
    }
  });
  const request = (method, params) => {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out waiting for MCP ${method}\n${stderr}`)); }, 20_000);
      pending.set(id, (m) => { clearTimeout(timer); resolve(m); });
      child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
    });
  };
  const call = async (name, args) => {
    const res = await request('tools/call', { name, arguments: args });
    const payload = JSON.parse(res.result.content[0].text);
    assert.strictEqual(payload.success, true, `${name} failed: ${JSON.stringify(payload)}`);
    return payload;
  };

  try {
    const init = await request('initialize', {
      protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'rvf-e2e', version: '1.0.0' },
    });
    assert.ok(init.result);
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`);

    const file = path.join(home, 'mcp.rvf');
    await call('rvf_create', { path: file, dimensions: 4, metric: 'cosine' });
    const ing = await call('rvf_ingest', { path: file, entries: ENTRIES });
    assert.strictEqual(ing.accepted, 3, JSON.stringify(ing));
    const q = await call('rvf_query', { path: file, vector: [1, 0.05, 0, 0], k: 2 });
    assert.strictEqual(q.results.length, 2);
    assert.strictEqual(q.results[0].id, 'a', JSON.stringify(q.results));
    const st = await call('rvf_status', { path: file });
    assert.strictEqual(st.totalVectors, 3, JSON.stringify(st));
    console.log('  PASS  MCP rvf_create/rvf_ingest/rvf_query/rvf_status');
  } finally {
    child.kill();
  }
}

// A failing `ruvector rvf` command must not leave the store locked: the writer
// lock file outlives the process, so a missed close() wedged the store until
// someone deleted <store>.lock by hand.
function cliFailureThenSuccess() {
  const { spawnSync } = require('child_process');
  const dir = fs.mkdtempSync(path.join(work, 'cli-'));
  const run = (...args) => {
    const r = spawnSync(process.execPath, [CLI, 'rvf', ...args], {
      cwd: dir, env: { ...process.env, HOME: dir, NO_COLOR: '1' }, encoding: 'utf8', timeout: 60_000,
    });
    return { code: r.status, out: `${r.stdout}${r.stderr}` };
  };
  const lock = path.join(dir, 's.rvf.lock');
  const noLock = (after) => assert.ok(!fs.existsSync(lock), `stale ${lock} left after ${after}`);

  fs.writeFileSync(path.join(dir, 'in.json'), JSON.stringify(ENTRIES));
  fs.writeFileSync(path.join(dir, 'bad.json'), '{not json');
  assert.strictEqual(run('create', 's.rvf', '-d', '4').code, 0);
  assert.strictEqual(run('ingest', 's.rvf', '-i', 'in.json').code, 0);
  noLock('create/ingest');

  // Fails after the store is opened (unparseable input file).
  const badIngest = run('ingest', 's.rvf', '-i', 'bad.json');
  assert.notStrictEqual(badIngest.code, 0, badIngest.out);
  noLock('a failed ingest');

  // Bad vectors are rejected up front with a clear message.
  for (const bad of ['abc,1,0,0', '1,,0,0', '[0,1,0']) {
    const r = run('query', 's.rvf', '-v', bad, '-k', '1');
    assert.notStrictEqual(r.code, 0, `query -v ${bad} should fail`);
    assert.match(r.out, /invalid vector/i, `query -v ${bad}: ${r.out}`);
    noLock(`query -v ${bad}`);
  }

  // Fails inside the backend (dimension mismatch).
  const wrongDim = run('query', 's.rvf', '-v', '0,1', '-k', '1');
  assert.notStrictEqual(wrongDim.code, 0, wrongDim.out);
  noLock('a wrong-dimension query');

  for (const good of ['0,1,0,0', '[0, 1, 0, 0]']) {
    const r = run('query', 's.rvf', '-v', good, '-k', '1');
    assert.strictEqual(r.code, 0, `query -v ${good}: ${r.out}`);
    assert.match(r.out, /id=b\b/, r.out);
    noLock(`query -v ${good}`);
  }
  for (const cmd of [['status', 's.rvf'], ['segments', 's.rvf'], ['compact', 's.rvf'], ['export', 's.rvf'], ['derive', 's.rvf', 'child.rvf']]) {
    const r = run(...cmd);
    assert.strictEqual(r.code, 0, `${cmd.join(' ')}: ${r.out}`);
    noLock(cmd[0]);
  }
  assert.ok(!fs.existsSync(path.join(dir, 'child.rvf.lock')), 'derived child left locked');
  console.log('  PASS  CLI: failed rvf commands release the lock; bad vectors rejected clearly');
}

(async () => {
  await libraryRoundTrip();
  await mcpRoundTrip();
  cliFailureThenSuccess();
  fs.rmSync(work, { recursive: true, force: true });
})().catch((e) => {
  console.error('  FAIL ', e && e.stack || e);
  process.exit(1);
});
