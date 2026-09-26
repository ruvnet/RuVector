'use strict';
/**
 * Tests for WasmBackend status/compact honesty (issue #753).
 *
 * `WasmBackend.status()` allocated the full 20-byte status buffer, read only
 * offsets 0 and 4, and then hardcoded `deadSpaceRatio: 0` — discarding the two
 * values the WASM layer had just written for it. `crates/rvf/rvf-wasm/src/store.rs`
 * writes:
 *
 *   0 live · 4 dimension · 8 metric · 12 total entries · 16 deleted entries
 *
 * so `deleted / total` is the dead-space ratio. Reporting a constant 0 left every
 * caller that gates on the documented policy (dead_space_ratio > 0.20, see
 * `crates/rvf/rvf-runtime/src/compaction.rs`) permanently inert — and silently,
 * because the field was present and plausible.
 *
 * `compact()` returned `{segmentsCompacted: 0, bytesReclaimed: 0}` without calling
 * WASM. There is no `rvf_store_compact` export, so "not implemented here" and
 * "ran, nothing to reclaim" were indistinguishable.
 *
 * Uses a fake WASM module that reproduces the documented buffer layout, so these
 * tests run without building the WASM artifact.
 */

const assert = require('assert');
const { WasmBackend } = require('../dist/backend.js');

const memory = new WebAssembly.Memory({ initial: 2 });

/** Writes the status buffer exactly as `rvf-wasm/src/store.rs` does. */
function fakeWasm({ live, dim, metric, total, deleted }) {
  return {
    memory,
    rvf_alloc: () => 1024,
    rvf_free: () => {},
    rvf_store_status(_handle, outPtr) {
      const v = new DataView(memory.buffer);
      v.setUint32(outPtr + 0, live, true);
      v.setUint32(outPtr + 4, dim, true);
      v.setUint32(outPtr + 8, metric, true);
      v.setUint32(outPtr + 12, total, true);
      v.setUint32(outPtr + 16, deleted, true);
      return 20;
    },
  };
}

function backendWith(counts) {
  const be = new WasmBackend();
  be.wasm = fakeWasm(counts);
  be.handle = 1;
  return be;
}

async function main() {
  // THE POINT: tombstones the WASM layer reported must reach deadSpaceRatio.
  {
    const be = backendWith({ live: 3, dim: 128, metric: 0, total: 10, deleted: 7 });
    const s = await be.status();
    assert.strictEqual(s.deadSpaceRatio, 0.7,
      'deadSpaceRatio must be deleted/total, not a constant');
    assert.strictEqual(s.totalVectors, 3, 'live count still read from offset 0');
  }

  // A store with no deletions is genuinely 0 — the fix must not invent dead space.
  {
    const be = backendWith({ live: 5, dim: 8, metric: 0, total: 5, deleted: 0 });
    const s = await be.status();
    assert.strictEqual(s.deadSpaceRatio, 0);
  }

  // An empty store must not divide by zero.
  {
    const be = backendWith({ live: 0, dim: 8, metric: 0, total: 0, deleted: 0 });
    const s = await be.status();
    assert.strictEqual(s.deadSpaceRatio, 0, 'empty store is 0, not NaN');
    assert.ok(Number.isFinite(s.deadSpaceRatio));
  }

  // The documented policy gate must actually be able to fire.
  {
    const be = backendWith({ live: 3, dim: 8, metric: 0, total: 10, deleted: 7 });
    const s = await be.status();
    assert.ok(s.deadSpaceRatio > 0.2,
      'a caller gating on the spec-10 threshold must see 70% dead space');
  }

  // compact() must not report success it did not achieve.
  {
    const be = backendWith({ live: 3, dim: 8, metric: 0, total: 10, deleted: 7 });
    await assert.rejects(
      () => be.compact(),
      (err) => /not supported in WASM backend/.test(String(err && err.message)),
      'compact() must signal unsupported, like every other unsupported op in this class',
    );
  }

  console.log('test-wasm-status-deadspace: all assertions passed');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
