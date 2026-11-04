// Regression tests for the 2026-11-04 correctness/security review.
// Run against the compiled output: `npm run build && npm test`.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import pg from 'pg';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(here, '..', 'dist');
const cli = path.join(dist, 'cli.js');
// Port 1 refuses immediately, so every DB command fails fast.
const DEAD_DB = 'postgresql://127.0.0.1:1/none';

// --- Fake pg.Pool: installed before client.js is imported ------------------
const pools = [];
class FakePool {
  constructor(options) {
    this.options = options;
    this.queries = [];
    this.listeners = {};
    pools.push(this);
  }
  on(event, fn) { this.listeners[event] = fn; return this; }
  async connect() {
    return { query: async (sql) => { this.queries.push(sql); return { rows: [] }; }, release() {} };
  }
  async query(sql, params) { this.queries.push(sql); return { rows: [{ id: 1 }], params }; }
  async end() {}
}
pg.Pool = FakePool;
const { RuVectorClient } = await import(path.join(dist, 'client.js'));
const args = await import(path.join(dist, 'commands', 'install-args.js'));

// --- CLI exit codes ---------------------------------------------------------
for (const argv of [
  ['vector', 'create', 't', '--dim', '3'],
  ['graph', 'create', 'g'],
]) {
  test(`cli exits non-zero when "${argv.join(' ')}" fails`, () => {
    const r = spawnSync(process.execPath, [cli, '-c', DEAD_DB, ...argv], { encoding: 'utf-8' });
    assert.notEqual(r.status, 0, `expected failure exit code, got ${r.status}\n${r.stderr}`);
  });
}

// --- client.ts --------------------------------------------------------------
test('statement_timeout is a pool option, so every pooled connection gets it', async () => {
  const c = new RuVectorClient(DEAD_DB);
  await c.connect();
  const pool = pools.at(-1);
  assert.equal(pool.options.statement_timeout, 30000);
  assert.equal(typeof pool.listeners.error, 'function', 'idle-client errors must not crash the process');
  await c.disconnect();
});

test('createVectorTable rejects non-integer dimensions before building SQL', async () => {
  const c = new RuVectorClient(DEAD_DB);
  await c.connect();
  const pool = pools.at(-1);
  for (const bad of ['3) ; DROP TABLE users; --', NaN, 1.5, '384']) {
    await assert.rejects(c.createVectorTable('t', bad), /Dimensions/);
  }
  assert.equal(pool.queries.filter(q => /CREATE TABLE/.test(q)).length, 0);
  await c.disconnect();
});

test('createVectorTable accepts a 63-char table name (index name is derived safely)', async () => {
  const c = new RuVectorClient(DEAD_DB);
  await c.connect();
  await c.createVectorTable('a'.repeat(63), 3);
  await c.disconnect();
});

// --- install argument handling ---------------------------------------------
test('install validators reject shell metacharacters', () => {
  assert.throws(() => args.assertPgVersion('16; echo pwned'));
  assert.throws(() => args.assertPgVersion('13'));
  assert.equal(args.assertPgVersion('16'), '16');
  assert.throws(() => args.assertContainerName('x; rm -rf ~'));
  assert.throws(() => args.assertContainerName('-rm'));
  assert.equal(args.assertContainerName('ruvector-postgres'), 'ruvector-postgres');
  assert.throws(() => args.assertImageTag('0.2.5 && id'));
  assert.equal(args.assertImageTag('0.2.5'), '0.2.5');
  assert.throws(() => args.assertPgIdentifier('ruvector; DROP'));
  assert.throws(() => args.assertPort(0));
  assert.throws(() => args.assertPort(70000));
  assert.throws(() => args.assertPort(NaN));
  assert.equal(args.assertPort(5432), 5432);
});

test('native configure creates the extension inside the target database', () => {
  const steps = args.configureSteps('ruvector', "p'w$(id)\"", 'mydb');
  const ext = steps.find(s => (s.input || s.args.join(' ')).includes('CREATE EXTENSION'));
  assert.ok(ext, 'extension step missing');
  const i = ext.args.indexOf('-d');
  assert.ok(i >= 0 && ext.args[i + 1] === 'mydb', 'CREATE EXTENSION must run with -d <database>');
  const user = steps.find(s => (s.input || '').includes('CREATE USER'));
  assert.ok(user.input.includes("PASSWORD 'p''w$(id)\"'"), user.input);
  for (const s of steps) assert.ok(!s.args.join(' ').includes('p\'w'), 'password must not be in argv');
});

test('docker run keeps the password off the command line', () => {
  const { args: argv, env } = args.dockerRunArgs({
    name: 'rv', port: 5432, user: 'ruvector', password: 'sec ret;$(id)', database: 'ruvector',
    image: 'ruvnet/ruvector-postgres:0.2.5',
  });
  assert.ok(!argv.join(' ').includes('sec ret'));
  assert.ok(argv.includes('POSTGRES_PASSWORD'));
  assert.equal(env.POSTGRES_PASSWORD, 'sec ret;$(id)');
});

test('psql passes the SQL as a single argv element', () => {
  const sql = `SELECT "id", '$HOME' FROM t`;
  const argv = args.dockerPsqlArgs('rv', sql);
  assert.equal(argv.at(-1), sql);
  assert.equal(argv[argv.length - 2], '-c');
});
