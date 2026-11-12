// CLI behaviour tests (no build needed: bin/cli.js is plain JS).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const cli = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'bin', 'cli.js');
const run = (...args) => spawnSync(process.execPath, [cli, ...args], { encoding: 'utf8' });

test('a corrupt database file is reported, not silently replaced', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvlite-cli-'));
  try {
    const db = path.join(dir, 'db.json');
    const corrupt = '{"vectors": {"keep": {"vector": [1, 2, 3]}}'; // truncated write
    fs.writeFileSync(db, corrupt);

    const res = run('insert', '[1, 2, 3]', '-p', db);
    assert.notEqual(res.status, 0, 'insert must fail on an unreadable database');
    assert.equal(fs.readFileSync(db, 'utf8'), corrupt, 'existing file must be left untouched');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('insert into a fresh path still works', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvlite-cli-'));
  try {
    const db = path.join(dir, 'sub', 'db.json');
    const res = run('insert', '[1, 2, 3]', '-i', 'a', '-p', db);
    assert.equal(res.status, 0, res.stderr);
    assert.deepEqual(JSON.parse(fs.readFileSync(db, 'utf8')).vectors.a.vector, [1, 2, 3]);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
