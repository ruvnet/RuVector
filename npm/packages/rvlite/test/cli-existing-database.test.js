import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const cli = fileURLToPath(new URL('../bin/cli.js', import.meta.url));
function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvlite-existing-db-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}
function run(args) {
  return spawnSync(process.execPath, [cli, ...args], {
    encoding: 'utf8', timeout: 10000, env: { ...process.env, NO_COLOR: '1' }
  });
}
for (const action of ['insert', 'delete', 'stats', 'export']) {
  test(`corrupt database refuses ${action} without losing bytes`, t => {
    const dir = fixture(t), db = path.join(dir, 'db.json');
    const original = Buffer.from('{"vectors":{"valuable":{"vector":[1,0]}}');
    fs.writeFileSync(db, original);
    const args = action === 'insert' ? ['insert', '[0,1]'] :
      action === 'delete' ? ['delete', 'valuable'] :
      action === 'export' ? ['export', path.join(dir, 'export.json')] : ['stats'];
    const result = run([...args, '--path', db]);
    assert.equal(result.status, 1, result.stdout + result.stderr);
    assert.deepEqual(fs.readFileSync(db), original);
    if (action === 'export') assert.equal(fs.existsSync(path.join(dir, 'export.json')), false);
  });
}
test('directory database refuses to report empty statistics', t => {
  const dir = fixture(t);
  assert.equal(run(['stats', '--path', dir]).status, 1);
});
test('permission-denied database refuses to report empty statistics', t => {
  if (process.getuid?.() === 0) { t.skip('permission test requires a non-root process'); return; }
  const dir = fixture(t), db = path.join(dir, 'db.json');
  fs.writeFileSync(db, '{}'); fs.chmodSync(db, 0);
  try { assert.equal(run(['stats', '--path', db]).status, 1); }
  finally { fs.chmodSync(db, 0o600); }
});
test('missing database stays valid for first insert', t => {
  const db = path.join(fixture(t), 'missing', 'db.json');
  const result = run(['insert', '[1,0]', '--id', 'one', '--path', db]);
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(fs.readFileSync(db)).vectors.one.vector, [1,0]);
});
test('valid saved database retains its records after another insert', t => {
  const db = path.join(fixture(t), 'db.json');
  assert.equal(run(['insert', '[1,0]', '--id', 'one', '--path', db]).status, 0);
  assert.equal(run(['insert', '[0,1]', '--id', 'two', '--path', db]).status, 0);
  assert.deepEqual(Object.keys(JSON.parse(fs.readFileSync(db)).vectors).sort(), ['one','two']);
  assert.equal(run(['stats', '--path', db]).status, 0);
});
