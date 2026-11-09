// node --test tests/derive.test.mjs  (after building the addon)
// Issue #1289: derive() records the derivation type the caller asked for,
// validated at the JS boundary, instead of always recording 'filter'.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadAddon } from './load-addon.mjs';

const { RvfDatabase } = loadAddon();

function withDir(fn) {
  const dir = mkdtempSync(join(tmpdir(), 'rvf-node-derive-'));
  return Promise.resolve(fn(dir)).finally(() => rmSync(dir, { recursive: true, force: true }));
}

const TYPES = ['clone', 'filter', 'quantize', 'reindex', 'transform', 'snapshot', 'user_defined'];

test('derive records each requested type, and it survives reopen', () =>
  withDir((dir) => {
    const parent = RvfDatabase.create(join(dir, 'p.rvf'), { dimension: 2, metric: 'l2' });
    assert.equal(parent.derivationType(), null, 'a root store records no derivation');
    for (const type of TYPES) {
      const path = join(dir, `${type}.rvf`);
      const child = parent.derive(path, undefined, type);
      assert.equal(child.derivationType(), type);
      assert.equal(child.parentId(), parent.fileId());
      child.close();
      const reopened = RvfDatabase.openReadonly(path);
      assert.equal(reopened.derivationType(), type, `${type} after reopen`);
      reopened.close();
    }
    parent.close();
  }));

test("derive without a type keeps recording 'filter'", () =>
  withDir((dir) => {
    const parent = RvfDatabase.create(join(dir, 'p.rvf'), { dimension: 2 });
    const child = parent.derive(join(dir, 'c.rvf'));
    assert.equal(child.derivationType(), 'filter');
    child.close();
    parent.close();
  }));

test('derive rejects unknown types and merge at the JS boundary', () =>
  withDir((dir) => {
    const parent = RvfDatabase.create(join(dir, 'p.rvf'), { dimension: 2 });
    for (const bad of ['merge', 'Clone', 'FILTER', 'copy', '']) {
      const path = join(dir, `bad-${bad || 'empty'}.rvf`);
      assert.throws(() => parent.derive(path, undefined, bad), /Invalid derivationType/, bad);
      assert.equal(existsSync(path), false, `no file created for '${bad}'`);
    }
    assert.throws(() => parent.derive(join(dir, 'n.rvf'), undefined, 7), /string/i);
    parent.close();
  }));

test("branch records 'clone'", () =>
  withDir((dir) => {
    const parent = RvfDatabase.create(join(dir, 'p.rvf'), { dimension: 2 });
    parent.ingestBatch(new Float32Array([1, 0, 0, 1]), [1, 2]);
    const branch = parent.branch(join(dir, 'b.rvf'));
    assert.equal(branch.derivationType(), 'clone');
    branch.close();
    parent.close();
  }));
