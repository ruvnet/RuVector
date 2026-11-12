#!/usr/bin/env node
/**
 * `ruvector diskann build|search` and the diskann_build / diskann_search MCP
 * tools against the real @ruvector/diskann (a devDependency here; users get it
 * with `npx ruvector add diskann`). Skips cleanly when it is absent, and checks
 * the not-installed message with the module hidden.
 *
 * Correctness oracle: on 300 random 16-dim vectors, DiskANN's top-10 must
 * agree with exact L2 brute force (recall@10 >= 0.95; it is usually 1.0).
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn, spawnSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const CLI = path.join(ROOT, 'bin', 'cli.js');
const { isDiskAnnAvailable } = require(path.join(ROOT, 'dist', 'core', 'diskann-wrapper.js'));

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-diskann-'));
const N = 300;
const DIM = 16;
let seed = 11;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const records = Array.from({ length: N }, (_, i) => ({ id: `v${i}`, vector: Array.from({ length: DIM }, rnd) }));
const l2 = (a, b) => a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0);
const bruteTop = (q, k) => records.map((r) => [r.id, l2(q, r.vector)]).sort((a, b) => a[1] - b[1]).slice(0, k).map((x) => x[0]);

function cli(args, env = {}) {
  const r = spawnSync(process.execPath, [CLI, ...args], { cwd: work, env: { ...process.env, HOME: work, NO_COLOR: '1', ...env }, encoding: 'utf8', timeout: 60_000 });
  const line = (r.stdout || '').trim().split(/\r?\n/).filter((l) => l.startsWith('{')).pop();
  return { status: r.status, json: line ? JSON.parse(line) : null, stdout: r.stdout, stderr: r.stderr };
}

function testCli() {
  const input = path.join(work, 'vectors.json');
  fs.writeFileSync(input, JSON.stringify(records));
  const built = cli(['diskann', 'build', 'idx', '-i', input, '--json']);
  assert.strictEqual(built.status, 0, built.stderr);
  assert.strictEqual(built.json.success, true);
  assert.strictEqual(built.json.count, N);
  assert.strictEqual(built.json.dim, DIM);

  let hit = 0;
  const queries = records.slice(0, 20).map((r) => r.vector.map((x) => x + 0.01 * (rnd() - 0.5)));
  for (const q of queries) {
    const res = cli(['diskann', 'search', 'idx', '-v', q.join(','), '-k', '10', '--json']);
    assert.strictEqual(res.status, 0, res.stderr);
    const ids = res.json.results.map((h) => h.id);
    assert.strictEqual(ids.length, 10);
    const truth = new Set(bruteTop(q, 10));
    hit += ids.filter((id) => truth.has(id)).length;
  }
  const recall = hit / (queries.length * 10);
  assert.ok(recall >= 0.95, `DiskANN recall@10 vs brute force ${recall}`);
  console.log(`  PASS  diskann build/search CLI: ${N}x${DIM}, recall@10 vs exact L2 = ${recall.toFixed(3)}`);

  const bad = cli(['diskann', 'search', 'idx', '-v', '1,2,3', '--json']);
  assert.notStrictEqual(bad.status, 0);
  assert.match(bad.json.error, /[Dd]imension/);
  const dup = path.join(work, 'dup.json');
  fs.writeFileSync(dup, JSON.stringify([{ id: 'a', vector: [1, 2] }, { id: 'a', vector: [3, 4] }]));
  assert.match(cli(['diskann', 'build', 'dup', '-i', dup, '--json']).json.error, /duplicate id/);
  console.log('  PASS  diskann rejects a wrong-dimension query and duplicate ids');
}

async function testMcp() {
  const child = spawn(process.execPath, [CLI, 'mcp', 'start'], {
    cwd: work, env: { ...process.env, HOME: work, NO_COLOR: '1' }, stdio: ['pipe', 'pipe', 'pipe'],
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
      const m = JSON.parse(line);
      const w = pending.get(m.id);
      if (w) { pending.delete(m.id); w(m); }
    }
  });
  const request = (method, params) => {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out waiting for MCP ${method}`)); }, 30_000);
      pending.set(id, (m) => { clearTimeout(t); resolve(m); });
      child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
    });
  };
  const call = async (name, args) => JSON.parse((await request('tools/call', { name, arguments: args })).result.content[0].text);
  try {
    await request('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'diskann-test', version: '1.0.0' } });
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`);
    const tools = (await request('tools/list', {})).result.tools.map((t) => t.name);
    assert.ok(tools.includes('diskann_build') && tools.includes('diskann_search'));

    const b = await call('diskann_build', { path: 'mcp-idx', vectors: records.slice(0, 50) });
    assert.strictEqual(b.success, true, JSON.stringify(b));
    assert.strictEqual(b.count, 50);
    const s = await call('diskann_search', { path: 'mcp-idx', vector: records[7].vector, k: 3 });
    assert.strictEqual(s.success, true);
    assert.strictEqual(s.results[0].id, 'v7', 'a stored vector is its own nearest neighbour');
    const escape = await call('diskann_build', { path: '../../outside-idx', vectors: records.slice(0, 2) });
    assert.strictEqual(escape.success, false, 'paths outside the working directory are rejected');

    // Index files planted as symlinks inside a confined directory must not be
    // followed: the save would write (and the load read) outside the workspace.
    const outsideDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-diskann-outside-'));
    try {
      const linked = path.join(work, 'linked-idx');
      fs.mkdirSync(linked);
      const victim = path.join(outsideDir, 'victim.txt');
      fs.writeFileSync(victim, 'ORIGINAL\n');
      fs.symlinkSync(victim, path.join(linked, 'ids.json'));
      fs.symlinkSync(path.join(outsideDir, 'created.bin'), path.join(linked, 'graph.bin'));
      const linkedBuild = await call('diskann_build', { path: 'linked-idx', vectors: records.slice(0, 2) });
      assert.strictEqual(linkedBuild.success, false, 'symlinked index files are refused');
      assert.match(linkedBuild.error, /symbolic link|not a regular file/);
      assert.strictEqual(fs.readFileSync(victim, 'utf8'), 'ORIGINAL\n', 'file outside the workspace was not overwritten');
      assert.ok(!fs.existsSync(path.join(outsideDir, 'created.bin')), 'no file was created outside the workspace');
      const linkedSearch = await call('diskann_search', { path: 'linked-idx', vector: records[0].vector, k: 1 });
      assert.strictEqual(linkedSearch.success, false, 'loading through symlinked index files is refused');
    } finally {
      fs.rmSync(outsideDir, { recursive: true, force: true });
    }
    console.log('  PASS  MCP diskann_build/diskann_search (and path confinement)');
  } finally {
    child.stdin.end();
    child.kill('SIGTERM');
  }
}

function testMissing() {
  const shim = path.join(work, 'hide-diskann.js');
  fs.writeFileSync(shim,
    "const M = require('module'); const orig = M._resolveFilename;\n" +
    "M._resolveFilename = function (req, ...rest) {\n" +
    "  if (req.startsWith('@ruvector/diskann')) { const e = new Error(\"Cannot find module '\" + req + \"'\"); e.code = 'MODULE_NOT_FOUND'; throw e; }\n" +
    '  return orig.call(this, req, ...rest);\n};\n');
  const r = cli(['diskann', 'search', 'idx', '-v', '1,2', '--json'], {
    NODE_OPTIONS: `${process.env.NODE_OPTIONS || ''} --require ${shim}`.trim(),
  });
  assert.notStrictEqual(r.status, 0);
  assert.match(r.json.error, /npx ruvector add diskann/);
  console.log('  PASS  without @ruvector/diskann the CLI points at `npx ruvector add diskann`');
}

(async () => {
  testMissing();
  if (!isDiskAnnAvailable()) {
    console.log('  SKIP  @ruvector/diskann not installed (build/search paths)');
    return;
  }
  testCli();
  await testMcp();
})()
  .catch((e) => {
    console.error(`  FAIL  ${e.stack || e.message}`);
    process.exitCode = 1;
  })
  .finally(() => fs.rmSync(work, { recursive: true, force: true }));
