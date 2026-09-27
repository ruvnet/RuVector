// Validation view of a public suite for `run.mjs --no-test` (ADR-008 §2,
// v0-plan Step 1.2). Mirrors the OpenJev exporter (crates/ruvector-typesafe-train
// src/prep/public.rs) so selection runs score exactly the rows the trainer used
// for early stopping — never the test split:
//   Banking77 / HWU64: the loader's train rows whose id falls in the 10 % bucket
//     `u32(sha256(id)[0..8]) % 10 == 0` are validation; the rest train the engine.
//   CLINC150: the official `val` (+ `oos_val`, flagged OOS) is validation; the
//     engine trains on the full official `train`.
// Validation rows whose normalised text collides with ANY held-out text are
// dropped, as the exporter drops them: its held-out set is the union of every
// suite's test rows plus the tickets test/transfer/calibration splits
// (heldoutUnion). Test rows are only hashed, never scored.

import { createHash } from 'node:crypto';
import { sha256Norm } from './norm.mjs';
import { loadTickets } from './fixture.mjs';

/** The exporter's held-out hash set: all public test splits + tickets held-out. */
export async function heldoutUnion(loadDataset, opts = {}) {
  const set = new Set();
  for (const s of ['banking77', 'clinc150', 'hwu64']) {
    const ds = await loadDataset(s, opts);
    if (ds.skipped) throw new Error(`held-out union needs ${s}: ${ds.skipped}`);
    for (const it of ds.testItems) set.add(sha256Norm(it.text));
  }
  const t = loadTickets();
  for (const sp of ['test', 'transfer', 'calibration']) for (const it of t.bySplit[sp]) set.add(sha256Norm(it.text));
  return set;
}

/** Exporter-identical 10 % validation bucket. */
export function isValBucket(id) {
  const h = createHash('sha256').update(String(id), 'utf8').digest('hex');
  return parseInt(h.slice(0, 8), 16) % 10 === 0;
}

/**
 * @param suite 'banking77' | 'clinc150' | 'hwu64'
 * @param ds    the loader's result ({ trainItems, testItems, valItems? })
 * @param heldout sha256Norm set to drop from validation (default: this suite's test)
 * @returns { trainItems, evalItems, counts }
 */
export function validationView(suite, ds, heldout = new Set((ds.testItems ?? []).map((it) => sha256Norm(it.text)))) {
  let trainItems;
  let evalItems;
  if (suite === 'clinc150') {
    if (!Array.isArray(ds.valItems)) throw new Error('clinc150 loader exposes no official val split');
    trainItems = ds.trainItems;
    evalItems = ds.valItems;
  } else {
    trainItems = ds.trainItems.filter((it) => !isValBucket(it.id));
    evalItems = ds.trainItems.filter((it) => isValBucket(it.id));
  }
  const before = evalItems.length;
  evalItems = evalItems.filter((it) => !heldout.has(sha256Norm(it.text)));
  return {
    trainItems,
    evalItems,
    counts: { train: trainItems.length, validation: evalItems.length, dropped_heldout_collisions: before - evalItems.length },
  };
}
