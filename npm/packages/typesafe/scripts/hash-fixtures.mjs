#!/usr/bin/env node
// Compute a content hash (sha256 of raw bytes) for every frozen bench fixture
// and write bench/fixtures/HASHES.json. The harness refuses to run when a live
// file's hash does not match this manifest (ADR-006 §protocol: "committed with a
// content hash, and never regenerated between arms").
//
// HASHES.json lists only the frozen inputs; it never lists itself. Re-run this
// only when a fixture is deliberately re-frozen (a dated, reviewed event).

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
export const PKG_ROOT = join(HERE, '..');
export const BENCH_DIR = join(PKG_ROOT, 'bench');
export const FIXTURE_DIR = join(BENCH_DIR, 'fixtures');

// The frozen inputs, relative to bench/. Read-only, committed, hash-pinned.
export const FROZEN_FILES = [
  'jev-baseline-2026-09-21.json',
  'ruvector-router-2026-09-21.json',
  'fixtures/tickets-corpus.json',
  'fixtures/tickets-decisions.json',
  'fixtures/tickets-truth.json',
];

export function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

/** Hash every frozen file. Returns { relPath: sha256 }. Throws if one is absent. */
export function computeHashes(benchDir = BENCH_DIR) {
  const out = {};
  for (const rel of FROZEN_FILES) {
    const abs = join(benchDir, rel);
    if (!existsSync(abs)) throw new Error(`frozen fixture missing: ${rel} (${abs})`);
    out[rel] = sha256(readFileSync(abs));
  }
  return out;
}

export const HASHES_PATH = join(FIXTURE_DIR, 'HASHES.json');

/** Compare live hashes to the committed manifest without writing. Exit 1 on drift. */
export function checkHashes() {
  if (!existsSync(HASHES_PATH)) {
    console.error(`HASHES.json missing — run scripts/hash-fixtures.mjs to create it`);
    return 1;
  }
  const manifest = JSON.parse(readFileSync(HASHES_PATH, 'utf8'));
  const expected = manifest.files ?? manifest;
  const actual = computeHashes();
  let drift = 0;
  for (const rel of FROZEN_FILES) {
    if (expected[rel] !== actual[rel]) {
      console.error(`MISMATCH ${rel}\n  manifest ${expected[rel]}\n  actual   ${actual[rel]}`);
      drift++;
    }
  }
  if (drift) {
    console.error(`\n${drift} frozen fixture(s) drifted from HASHES.json — refusing.`);
    return 1;
  }
  console.log(`frozen fixtures verified against HASHES.json (${FROZEN_FILES.length} files)`);
  return 0;
}

function main(argv) {
  if (argv.includes('--check')) process.exit(checkHashes());
  const hashes = computeHashes();
  const doc = {
    _comment:
      'sha256 of the raw bytes of each frozen bench fixture. The harness ' +
      'refuses to run on mismatch. Regenerate only on a deliberate re-freeze.',
    generated_by: 'scripts/hash-fixtures.mjs',
    algorithm: 'sha256',
    files: hashes,
  };
  writeFileSync(HASHES_PATH, JSON.stringify(doc, null, 2) + '\n');
  for (const [k, v] of Object.entries(hashes)) console.log(`${v}  ${k}`);
  console.log(`\nwrote ${HASHES_PATH}`);
}

if (import.meta.url === `file://${process.argv[1]}`) main(process.argv.slice(2));
