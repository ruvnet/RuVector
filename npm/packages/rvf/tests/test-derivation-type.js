'use strict';
/**
 * Issue #1289: `derive(childPath, options, derivationType)` records the type
 * the caller asked for, and `derivationType()` reads it back.
 *
 * The end-to-end half needs @ruvector/rvf-node >= 0.4.7 and is skipped when
 * the loadable addon is older. The guard half uses a mock native handle.
 *
 *   node tests/test-derivation-type.js
 */
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { RvfDatabase } = require(process.env.RVF_SDK || '../dist/index');
const { NodeBackend } = require('../dist/backend');
const { RvfErrorCode } = require('../dist/errors');

let passed = 0, failed = 0, skipped = 0;
async function test(name, fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-derive-type-'));
  try {
    const result = await fn(dir);
    if (result === 'skip') {
      console.log(`  SKIP  ${name}`);
      skipped++;
    } else {
      console.log(`  PASS  ${name}`);
      passed++;
    }
  } catch (err) {
    console.log(`  FAIL  ${name}: ${err && err.stack || err}`);
    failed++;
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

async function nativeHasDerivationType(dir) {
  try {
    const db = await RvfDatabase.create(path.join(dir, 'probe.rvf'), { dimensions: 2 });
    const ok = typeof db.backend.handle.derivationType === 'function';
    await db.close();
    return ok;
  } catch {
    return false;
  }
}

/** A NodeBackend over a mock handle whose derive() records its arguments. */
function mockBackend(withDerivationType) {
  const calls = [];
  const handle = {
    derive: (...args) => {
      calls.push(args);
      throw new Error('stop after recording the call');
    },
  };
  if (withDerivationType) handle.derivationType = () => 'filter';
  const backend = new NodeBackend();
  backend.handle = handle;
  backend.native = {};
  backend.storePath = '/nonexistent/parent.rvf';
  return { backend, calls };
}

(async () => {
  await test('derive records the requested type; default stays filter', async (dir) => {
    if (!(await nativeHasDerivationType(dir))) return 'skip';
    const parent = await RvfDatabase.create(path.join(dir, 'p.rvf'), { dimensions: 2 });
    assert.strictEqual(await parent.derivationType(), null);
    for (const type of ['clone', 'snapshot', 'user_defined']) {
      const child = await parent.derive(path.join(dir, `${type}.rvf`), undefined, type);
      assert.strictEqual(await child.derivationType(), type);
      await child.close();
    }
    const plain = await parent.derive(path.join(dir, 'plain.rvf'));
    assert.strictEqual(await plain.derivationType(), 'filter');
    await plain.close();
    await assert.rejects(parent.derive(path.join(dir, 'm.rvf'), undefined, 'merge'), /merge/);
    await parent.close();
  });

  await test('derive passes the type through to the native handle', async () => {
    const { backend, calls } = mockBackend(true);
    await assert.rejects(backend.derive('/nonexistent/c.rvf', undefined, 'snapshot'));
    assert.deepStrictEqual(calls[0], ['/nonexistent/c.rvf', null, 'snapshot']);
    await assert.rejects(backend.derive('/nonexistent/d.rvf'));
    assert.deepStrictEqual(calls[1], ['/nonexistent/d.rvf', undefined]);
  });

  await test('an older addon cannot silently drop the requested type', async () => {
    const { backend, calls } = mockBackend(false);
    await assert.rejects(
      backend.derive('/nonexistent/c.rvf', undefined, 'clone'),
      (err) => err.code === RvfErrorCode.InvalidArgument && /0\.4\.7/.test(err.message),
    );
    assert.strictEqual(calls.length, 0, 'native derive never called');
    assert.strictEqual(await backend.derivationType(), null);
  });

  console.log(`\nResults: ${passed} passed, ${failed} failed, ${skipped} skipped`);
  process.exit(failed ? 1 : 0);
})();
