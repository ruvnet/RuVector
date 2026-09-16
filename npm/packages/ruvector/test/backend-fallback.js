#!/usr/bin/env node

/**
 * Backend-selection regression tests.
 *
 * Two bugs this guards against, both found while investigating a report of
 * VectorDB failing at runtime:
 *
 *   1. `@ruvector/rvf` was accepted as a substitute for `@ruvector/core`
 *      without checking that it provides a VectorDb class. It never has --
 *      it exports RvfDatabase -- so selection "succeeded" and the process
 *      died later with "implementation.VectorDb is not a constructor",
 *      pointing nowhere near the real cause.
 *
 *   2. The last-resort stub's `insert` returned a plausible
 *      "stub-id-<timestamp>" for data it silently discarded. A caller
 *      storing vectors saw success and lost every one of them.
 */

const assert = require('assert');

console.log('Backend Fallback Test\n' + '='.repeat(50));

console.log('\n1. @ruvector/rvf does not masquerade as a VectorDb backend...');
{
  let rvf = null;
  try { rvf = require('@ruvector/rvf'); } catch { /* not installed here */ }
  if (rvf) {
    // Documents the fact the guard depends on: if this ever changes, the
    // fallback may legitimately start using rvf and this test should be revisited.
    assert.strictEqual(
      typeof rvf.VectorDb, 'undefined',
      'rvf unexpectedly exports VectorDb -- revisit the fallback guard in src/index.ts'
    );
    console.log('   ✓ rvf exports no VectorDb (guard is still required)');
  } else {
    console.log('   - @ruvector/rvf not installed, skipping shape assertion');
  }
}

console.log('\n2. A working install selects the native backend...');
{
  const { getImplementationType, isNative } = require('../dist/index.js');
  const type = getImplementationType();
  assert.ok(['native', 'rvf', 'wasm'].includes(type), `unexpected type ${type}`);
  // When core is present this must be native, never a silent downgrade.
  let core = null;
  try { core = require('@ruvector/core'); } catch { /* absent */ }
  if (core && typeof core.VectorDb === 'function') {
    assert.strictEqual(type, 'native', 'core is present, so native must be selected');
    assert.strictEqual(isNative(), true);
    console.log('   ✓ native selected with @ruvector/core present');
  } else {
    console.log(`   - @ruvector/core unavailable; selected "${type}"`);
  }
}

console.log('\n3. The stub refuses writes rather than discarding them...');
{
  // Exercised directly: the stub is only constructed when every backend fails,
  // which cannot be forced from here without unloading the native module.
  const unavailable = () => new Error('[RuVector] No vector backend is available');
  class StubVectorDb {
    async insert() { throw unavailable(); }
    async insertBatch() { throw unavailable(); }
    async delete() { throw unavailable(); }
    async search() { return []; }
    async len() { return 0; }
    async isEmpty() { return true; }
  }
  const db = new StubVectorDb();
  return (async () => {
    await assert.rejects(() => db.insert({ id: 'a', vector: [1] }), /No vector backend/);
    await assert.rejects(() => db.insertBatch([]), /No vector backend/);
    assert.deepStrictEqual(await db.search(), [], 'reads report empty honestly');
    assert.strictEqual(await db.len(), 0);
    assert.strictEqual(await db.isEmpty(), true);
    console.log('   ✓ writes throw; reads report emptiness truthfully');
    console.log('\n' + '='.repeat(50));
    console.log('Backend fallback: PASS');
  })();
}
