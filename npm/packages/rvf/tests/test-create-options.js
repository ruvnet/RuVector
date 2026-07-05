'use strict';
/**
 * Tests for RVF store-creation options normalization (issue #641 fix).
 *
 * `RvfDatabase.create` documents `dimensions` (plural) but the native layer
 * names the field `dimension` (singular). The SDK now accepts both, and a
 * missing/invalid value fails fast with an error naming the public option
 * instead of the native field.
 *
 * Uses a mock native handle so tests work without the N-API addon.
 */

const assert = require('assert');

const { NodeBackend } = require('../dist/backend');
const { RvfErrorCode } = require('../dist/errors');

class MockNativeHandle {
  constructor(opts) {
    this.opts = opts;
  }
  status() { return { total_vectors: 0, total_segments: 1, file_size: 0, current_epoch: 0, profile_id: 0, compaction_state: 'idle', dead_space_ratio: 0, read_only: false }; }
  close() {}
}

function createTestBackend() {
  const backend = new NodeBackend();
  let captured = null;
  backend['native'] = {
    create: (_path, nativeOpts) => {
      captured = nativeOpts;
      return new MockNativeHandle(nativeOpts);
    },
  };
  // loadNative is a no-op once `native` is set
  backend['loadNative'] = async () => {};
  return { backend, getCaptured: () => captured };
}

let passed = 0, failed = 0;

async function test(name, fn) {
  try {
    await fn();
    console.log(`  PASS  ${name}`);
    passed++;
  } catch (err) {
    console.log(`  FAIL  ${name}: ${err.message}`);
    failed++;
  }
}

(async () => {
  console.log('RVF create-options tests (issue #641)\n');

  await test('`dimensions` (plural, documented) maps to native `dimension`', async () => {
    const { backend, getCaptured } = createTestBackend();
    await backend.create('/tmp/a.rvf', { dimensions: 384, metric: 'cosine' });
    assert.strictEqual(getCaptured().dimension, 384);
  });

  await test('`dimension` (singular alias) is accepted', async () => {
    const { backend, getCaptured } = createTestBackend();
    await backend.create('/tmp/b.rvf', { dimension: 384, metric: 'cosine' });
    assert.strictEqual(getCaptured().dimension, 384);
  });

  await test('`dimensions` wins when both are set', async () => {
    const { backend, getCaptured } = createTestBackend();
    await backend.create('/tmp/c.rvf', { dimensions: 128, dimension: 999 });
    assert.strictEqual(getCaptured().dimension, 128);
  });

  await test('missing dimensionality throws an error naming `dimensions`', async () => {
    const { backend } = createTestBackend();
    await assert.rejects(
      () => backend.create('/tmp/d.rvf', { metric: 'cosine' }),
      (err) => {
        assert.strictEqual(err.code, RvfErrorCode.InvalidOptions);
        assert.ok(err.message.includes('`dimensions`'), `error should name the public option, got: ${err.message}`);
        assert.ok(err.message.includes('plural'), `error should point at the plural spelling, got: ${err.message}`);
        return true;
      },
    );
  });

  await test('non-positive / non-integer dimensionality is rejected', async () => {
    const { backend } = createTestBackend();
    for (const bad of [0, -4, 1.5, '384', NaN]) {
      await assert.rejects(
        () => backend.create('/tmp/e.rvf', { dimensions: bad }),
        (err) => err.code === RvfErrorCode.InvalidOptions,
        `expected rejection for dimensions=${JSON.stringify(bad)}`,
      );
    }
  });

  console.log(`\n${'='.repeat(60)}`);
  console.log(`Results: ${passed} passed, ${failed} failed, ${passed + failed} total`);
  console.log('='.repeat(60));
  process.exit(failed > 0 ? 1 : 0);
})();
