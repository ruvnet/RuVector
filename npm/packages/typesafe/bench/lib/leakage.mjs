// Assertion B (ADR-007 §2): a model that ships `train-text-hashes.txt` must not
// have trained on any held-out text. The bench intersects the model's hashes
// with sha256Norm of the suite's held-out rows and refuses to score on a hit.
// Independent of the trainer — anyone can re-verify from the artifact alone.
//
// train-text-hashes.txt format (the contract the Rust trainer writes):
//   - one lowercase 64-hex sha256Norm(text) per line (lib/norm.mjs), one line
//     per training row (duplicates allowed), LF line endings, no CR, no blanks,
//     no comments;
//   - lines sorted ascending (byte order == lexical order for hex), so a
//     non-decreasing sequence;
//   - an optional single trailing LF. An empty file (zero rows) is rejected.
// Anything else is rejected (fail closed).
//
// Held-out set, mirroring the trainer's heldout-hashes.txt (plan Step 1.2):
// tickets → test + transfer + calibration; public suites → their test split.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { sha256Norm } from './norm.mjs';

const HEX64 = /^[0-9a-f]{64}$/;

/** Parse + validate a train-text-hashes.txt body. Returns the hash array. */
export function parseTrainHashes(body, { source = 'train-text-hashes.txt' } = {}) {
  if (typeof body !== 'string') throw new TypeError(`${source}: expected text`);
  if (body.includes('\r')) throw new Error(`${source}: CR line endings are not allowed (LF only)`);
  const text = body.endsWith('\n') ? body.slice(0, -1) : body;
  if (!text.length) throw new Error(`${source}: empty — a model must list at least one training row`);
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    if (!HEX64.test(lines[i])) {
      throw new Error(`${source}:${i + 1}: not a lowercase 64-hex sha256 (got ${JSON.stringify(lines[i].slice(0, 80))})`);
    }
    if (i > 0 && lines[i] < lines[i - 1]) {
      throw new Error(`${source}:${i + 1}: lines must be sorted ascending`);
    }
  }
  return lines;
}

/** Read + parse a train-hashes file; also returns the file's sha256. */
export function readTrainHashes(path) {
  const bytes = readFileSync(path);
  return {
    path,
    sha256: createHash('sha256').update(bytes).digest('hex'),
    hashes: parseTrainHashes(bytes.toString('utf8'), { source: path }),
  };
}

/**
 * Intersect train hashes with held-out rows grouped by split.
 * @param trainHashes string[] (from parseTrainHashes)
 * @param heldOut     { split: [{ id, text }] }
 * @returns report { ok, train_rows, train_unique, splits: { split: { heldout_rows,
 *          intersection, colliding_ids } }, intersection } — ids only, never text.
 */
export function checkLeakage(trainHashes, heldOut) {
  const train = new Set(trainHashes);
  const splits = {};
  let total = 0;
  for (const [split, items] of Object.entries(heldOut)) {
    const colliding = [];
    for (const it of items ?? []) if (train.has(sha256Norm(it.text))) colliding.push(it.id);
    colliding.sort();
    splits[split] = { heldout_rows: (items ?? []).length, intersection: colliding.length, colliding_ids: colliding };
    total += colliding.length;
  }
  return { ok: total === 0, train_rows: trainHashes.length, train_unique: train.size, intersection: total, splits };
}

/** Throw (refuse to score) when the report shows any held-out text in train. */
export function assertNoLeakage(report) {
  if (report.ok) return report;
  const where = Object.entries(report.splits)
    .filter(([, s]) => s.intersection > 0)
    .map(([split, s]) => `${split}: ${s.intersection} (${s.colliding_ids.slice(0, 5).join(', ')}${s.intersection > 5 ? ', …' : ''})`)
    .join('; ');
  const err = new Error(`leakage: ${report.intersection} held-out text hash(es) appear in the model's train-text-hashes — refusing to score [${where}]`);
  err.code = 'LEAKAGE';
  err.report = report;
  throw err;
}

/**
 * Run Assertion B for a suite and throw (code LEAKAGE) on any hit; returns the
 * receipt block { file, file_sha256, ...checkLeakage } or null without hashes.
 */
export function leakageGate(trainHashes, heldOut) {
  if (!trainHashes) return null;
  const report = { file: trainHashes.path, file_sha256: trainHashes.sha256, ...checkLeakage(trainHashes.hashes, heldOut) };
  return assertNoLeakage(report);
}
