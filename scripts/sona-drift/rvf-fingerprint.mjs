#!/usr/bin/env node
/**
 * RVF-backed SONA behavioral-fingerprint reference.
 *
 * Stores each implementation's 6-component behavioral fingerprint (see
 * harness.mjs) as a vector in a real RVF store file — dogfooding the
 * RuVector Format via @ruvector/rvf's NodeBackend (@ruvector/rvf-node
 * N-API addon; the WASM backend is in-memory only and cannot produce a
 * file artifact).
 *
 * Store layout:
 *   scripts/sona-drift/reference.rvf            RVF store, dimensions=6, L2
 *   scripts/sona-drift/reference.rvf.idmap.json sidecar written by the rvf
 *       SDK mapping string ids -> native i64 labels (required to resolve
 *       ids on reopen — commit it together with reference.rvf)
 *   one vector per implementation, id = sha256("sona-fingerprint:<impl>")
 *       truncated to 16 hex chars (stable hash of the impl name)
 *
 * Modes:
 *   node scripts/sona-drift/rvf-fingerprint.mjs            validate (default)
 *   node scripts/sona-drift/rvf-fingerprint.mjs --update   regenerate reference
 *   (dev only) --perturb   corrupt m2 in memory before validating, to prove
 *                          the guard fires; never use in CI
 *
 * Validation queries each current fingerprint against the stored reference
 * vector for the same impl and FAILS (exit 1) if the L2 distance exceeds
 *
 *   tolerance = max(1e-9, 1e-3 * ||current fingerprint||)
 *
 * Justification: every fingerprint component is a deterministic metric
 * (harness runs each scenario twice and errors on any bit-level jitter;
 * anything timing-based is excluded by construction), so the only legitimate
 * sources of distance are (a) float32 quantization inside the RVF store,
 * ~1e-7 relative, and (b) cross-platform FP reassociation in the Rust
 * engine's SIMD paths, empirically <1e-6 relative. 1e-3 relative leaves
 * >100x headroom over both while remaining ~100x tighter than the smallest
 * real regression we guard against (a no-op stub zeroes m2/m4/m6, moving the
 * fingerprint by >0.5 relative). The 1e-9 absolute floor covers the
 * degenerate all-near-zero fingerprint case.
 */

import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { existsSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { collectFingerprints, REPO, FINGERPRINT_DIM } from './harness.mjs';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const REFERENCE = path.join(__dirname, 'reference.rvf');
const SIDECAR = `${REFERENCE}.idmap.json`;

// Require the rvf SDK dist directly by path (avoids any registry fetch; the
// package's index.js also re-exports @ruvector/rvf-solver which is not built
// in a clean checkout, so we go straight to database.js).
const { RvfDatabase } = require(path.join(REPO, 'npm', 'packages', 'rvf', 'dist', 'database.js'));

function implId(name) {
  return createHash('sha256').update(`sona-fingerprint:${name}`).digest('hex').slice(0, 16);
}

function norm(v) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

function tolerance(fingerprint) {
  return Math.max(1e-9, 1e-3 * norm(fingerprint));
}

function log(...args) {
  console.error('[rvf-fingerprint]', ...args);
}

async function update(results) {
  const entries = results
    .filter((r) => r.fingerprint)
    .map((r) => ({ id: implId(r.impl), vector: new Float32Array(r.fingerprint), impl: r.impl }));
  if (entries.length === 0) {
    log('ERROR: no fingerprints produced — nothing to store');
    return 1;
  }
  // RvfDatabase.create fails on an existing store file — remove old artifacts.
  for (const f of [REFERENCE, SIDECAR]) {
    if (existsSync(f)) rmSync(f);
  }
  const db = await RvfDatabase.create(REFERENCE, { dimensions: FINGERPRINT_DIM, metric: 'l2' });
  const res = await db.ingestBatch(entries.map(({ id, vector }) => ({ id, vector })));
  const status = await db.status();
  await db.close();
  for (const e of entries) log(`stored ${e.impl} as id=${e.id}`);
  log(`reference updated: ${REFERENCE} (${status.totalVectors} vectors, accepted=${res.accepted})`);
  return 0;
}

async function validate(results, { perturb = false } = {}) {
  if (!existsSync(REFERENCE)) {
    log(`ERROR: reference store not found at ${REFERENCE} — run with --update first`);
    return 1;
  }
  const db = await RvfDatabase.openReadonly(REFERENCE);
  let failures = 0;
  try {
    for (const r of results) {
      if (!r.fingerprint) {
        log(`SKIP ${r.impl}: ${r.reason ?? r.status}`);
        continue;
      }
      const fp = [...r.fingerprint];
      if (perturb) {
        fp[1] = 0; // simulate the #519/#553 regression: single-positive delta collapses to zero
      }
      const matches = await db.query(new Float32Array(fp), 16);
      const id = implId(r.impl);
      const match = matches.find((m) => m.id === id);
      if (!match) {
        log(`FAIL ${r.impl}: no reference vector for id=${id} — new impl? run --update intentionally`);
        failures++;
        continue;
      }
      // The native index returns SQUARED L2 distance.
      const dist = Math.sqrt(Math.max(0, match.distance));
      const tol = tolerance(fp);
      if (dist > tol) {
        log(`FAIL ${r.impl}: fingerprint drifted from reference — L2 distance ${dist} > tolerance ${tol}`);
        log(`     current fingerprint: ${JSON.stringify(fp)}`);
        failures++;
      } else {
        log(`OK   ${r.impl}: L2 distance ${dist} <= tolerance ${tol}`);
      }
    }
  } finally {
    // close() on a readonly handle fails fsync in the native layer (there is
    // nothing to persist) — ignore.
    await db.close().catch(() => {});
  }
  if (failures > 0) {
    log(`VALIDATION FAILED: ${failures} implementation(s) drifted. If the behavior change is`);
    log('intentional, regenerate the reference with --update and commit reference.rvf + sidecar.');
    return 1;
  }
  log('VALIDATION PASSED');
  return 0;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const args = process.argv.slice(2);
  const report = await collectFingerprints();
  if (!report.pass) {
    log('ERROR: harness contracts failed — fix behavior before touching the reference');
    process.exit(1);
  }
  const code = args.includes('--update')
    ? await update(report.results)
    : await validate(report.results, { perturb: args.includes('--perturb') });
  process.exit(code);
}
