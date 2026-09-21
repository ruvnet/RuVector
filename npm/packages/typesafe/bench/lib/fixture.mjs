// Frozen-fixture loader with hash verification and deterministic split
// assignment (ADR-006 §protocol 1,3; ADR-004 promotion gate 1,3).
//
// The harness refuses to run on any hash mismatch — a fixture that drifted is
// not the fixture the Jev baseline was measured against. Splits are assigned
// from item ids deterministically and are instance-ID disjoint
// (assert_train_eval_disjoint, ADR-004).

import { createHash } from 'node:crypto';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { computeHashes, FROZEN_FILES } from '../../scripts/hash-fixtures.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
export const BENCH_DIR = join(HERE, '..');
export const FIXTURE_DIR = join(BENCH_DIR, 'fixtures');

/** The five splits, in canonical order. */
export const SPLITS = ['train', 'calibration', 'validation', 'transfer', 'test'];

function sha256hex(s) {
  return createHash('sha256').update(String(s)).digest('hex');
}

/**
 * Verify every frozen file against HASHES.json. Injectable dirs/manifest so
 * tests can point at a temp tree. Throws with the offending file on mismatch.
 */
export function verifyFixtureHashes({ benchDir = BENCH_DIR, fixtureDir = FIXTURE_DIR } = {}) {
  const manifestPath = join(fixtureDir, 'HASHES.json');
  if (!existsSync(manifestPath)) {
    throw new Error(`HASHES.json missing at ${manifestPath}; run scripts/hash-fixtures.mjs`);
  }
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const expected = manifest.files ?? manifest;
  const actual = computeHashes(benchDir);
  const mismatches = [];
  for (const rel of FROZEN_FILES) {
    if (expected[rel] !== actual[rel]) {
      mismatches.push({ file: rel, expected: expected[rel], actual: actual[rel] });
    }
  }
  if (mismatches.length) {
    const lines = mismatches
      .map((m) => `  ${m.file}\n    expected ${m.expected}\n    actual   ${m.actual}`)
      .join('\n');
    throw new Error(`frozen fixture hash mismatch — refusing to run:\n${lines}`);
  }
  return { verified: FROZEN_FILES.length, hashes: actual };
}

/**
 * Deterministic split assignment for one item id.
 * Native fixture splits anchor validation (`val`) and test; the training pool
 * is carved by sha256(id) into train / calibration / transfer so the loop has
 * a calibration set and a held-out transfer set (ADR-004 gate 3), all disjoint.
 */
export function assignSplit(id, nativeSplit) {
  if (nativeSplit === 'test') return 'test';
  if (nativeSplit === 'val') return 'validation';
  // nativeSplit === 'train' → carve by a stable hash bucket in [0,99]
  const bucket = parseInt(sha256hex(id).slice(0, 8), 16) % 100;
  if (bucket < 15) return 'calibration';
  if (bucket < 30) return 'transfer';
  return 'train';
}

/**
 * Load and verify the ticket decision fixture. Returns the items grouped by
 * assigned split, the question definitions, the department label set, and a
 * stable hash of the split assignment (for the receipt).
 */
export function loadTickets({ benchDir = BENCH_DIR, fixtureDir = FIXTURE_DIR, verify = true } = {}) {
  if (verify) verifyFixtureHashes({ benchDir, fixtureDir });
  const raw = JSON.parse(readFileSync(join(fixtureDir, 'tickets-decisions.json'), 'utf8'));

  const bySplit = Object.fromEntries(SPLITS.map((s) => [s, []]));
  const assignment = {}; // id -> split
  for (const nativeSplit of ['train', 'val', 'test']) {
    for (const it of raw.split[nativeSplit]) {
      const split = assignSplit(it.id, nativeSplit);
      const item = {
        id: it.id,
        text: it.text,
        label: it.label,
        ambiguous: !!it.ambiguous,
        secondary: it.secondary ?? null,
        split,
      };
      bySplit[split].push(item);
      assignment[it.id] = split;
    }
  }

  return {
    departments: raw.departments,
    questions: raw.gen0_questions,
    bySplit,
    assignment,
    splitsHash: hashAssignment(assignment),
    counts: Object.fromEntries(SPLITS.map((s) => [s, bySplit[s].length])),
  };
}

/** Stable content hash of the id→split map (order-independent). */
export function hashAssignment(assignment) {
  const sorted = Object.keys(assignment)
    .sort()
    .map((id) => `${id}:${assignment[id]}`)
    .join('|');
  return sha256hex(sorted);
}

/**
 * Assert the splits are pairwise disjoint by instance id and cover every item
 * exactly once (ADR-004 assert_train_eval_disjoint). Throws on any overlap.
 */
export function assertDisjoint(bySplit) {
  const seen = new Map(); // id -> split
  for (const split of SPLITS) {
    for (const it of bySplit[split] ?? []) {
      if (seen.has(it.id)) {
        throw new Error(
          `split overlap: id ${it.id} is in both ${seen.get(it.id)} and ${split}`,
        );
      }
      seen.set(it.id, split);
    }
  }
  return { disjoint: true, total: seen.size };
}

/**
 * Build the wire request `questions` object (ADR crate types) from the
 * fixture's gen0 question definitions. `choice` criteria pass through as text;
 * `score` maps its `criteria` levels to the wire `legend`; `noul` keeps its
 * instructions. This is the identical question batch every arm scores.
 */
export function buildQuestions(questionDefs) {
  const out = {};
  for (const [name, def] of Object.entries(questionDefs)) {
    if (def.type === 'choice') {
      out[name] = { type: 'choice', instructions: def.instructions ?? '', criteria: def.criteria };
    } else if (def.type === 'score') {
      out[name] = { type: 'score', instructions: def.instructions ?? '', legend: def.criteria };
    } else if (def.type === 'noul') {
      out[name] = { type: 'noul', instructions: def.instructions ?? '' };
    } else {
      throw new Error(`unknown question type '${def.type}' for '${name}'`);
    }
  }
  return out;
}
