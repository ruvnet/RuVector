// Shared dataset plumbing: cached, hash-verified download of a canonical public
// source (ADR-006 §Where it lives: fetched at bench time, nothing redistributed).
// Cache lives under bench/.cache (gitignored). No datasets are committed.

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
export const DEFAULT_CACHE = join(HERE, '..', '.cache');

export const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

/**
 * Fetch `url` to `<cacheDir>/<name>`, returning its bytes. Cached on disk. If a
 * known sha256 is given it is verified (throws on mismatch); if null, the
 * computed hash is returned so a maintainer can pin it. Network/HTTP failure
 * throws a loud error naming the URL (the harness turns that into
 * "skipped: unavailable").
 */
export async function fetchCached(url, name, { cacheDir = DEFAULT_CACHE, knownSha256 = null, gunzip = false } = {}) {
  mkdirSync(cacheDir, { recursive: true });
  const path = join(cacheDir, name);
  let bytes;
  const cached = existsSync(path);
  if (cached) {
    bytes = readFileSync(path);
  } else {
    if (typeof fetch !== 'function') throw new Error(`global fetch unavailable (Node < 18?) for ${url}`);
    let res;
    try {
      res = await fetch(url, { redirect: 'follow' });
    } catch (e) {
      throw new Error(`network error fetching ${url}: ${e && e.message ? e.message : e}`);
    }
    if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
    bytes = Buffer.from(await res.arrayBuffer());
    if (gunzip) bytes = gunzipSync(bytes);
    writeFileSync(path, bytes);
  }
  const digest = sha256(bytes);
  if (knownSha256 && digest !== knownSha256) {
    // Say where the bytes came from. A cached file failing its pin is a local
    // problem (tampered, truncated, or left by an older source), not upstream
    // drift; the old message cited only the URL and was misread as drift.
    const origin = cached ? `cached file ${path} (source ${url}); delete it to re-fetch` : `fetched from ${url}`;
    throw new Error(`sha256 mismatch for ${name}: expected ${knownSha256}, got ${digest} — ${origin}`);
  }
  return { bytes, path, sha256: digest, pinned: !!knownSha256 };
}

/** Parse a tab- (or space-) separated `head rel tail` triples file. */
export function parseTriplesTsv(text) {
  const out = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const parts = line.split(/\t/);
    const [s, r, o] = parts.length >= 3 ? parts : line.trim().split(/\s+/);
    if (s === undefined || r === undefined || o === undefined) continue;
    out.push({ s, r, o });
  }
  return out;
}

/**
 * Entity-consistent `--limit`: keep the top-N entities by train degree
 * (tie-broken by id string, deterministic), then keep every triple across ALL
 * splits whose head AND tail both survive. This preserves a connected subgraph
 * so filtered ranking stays meaningful (a random-triple slice would orphan the
 * filter sets). Returns { train, valid, test } filtered in place.
 */
export function limitSubgraph({ train, valid, test }, n) {
  if (typeof n !== 'number' || n <= 0) return { train, valid, test };
  const degree = new Map();
  for (const t of train) {
    degree.set(t.s, (degree.get(t.s) ?? 0) + 1);
    degree.set(t.o, (degree.get(t.o) ?? 0) + 1);
  }
  const keep = new Set(
    [...degree.entries()]
      .sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])))
      .slice(0, n)
      .map(([e]) => e),
  );
  const filt = (arr) => arr.filter((t) => keep.has(t.s) && keep.has(t.o));
  return { train: filt(train), valid: filt(valid), test: filt(test), entities: keep.size };
}

export function stableSplitHash(triples) {
  const keys = triples.map((t) => `${t.s} ${t.r} ${t.o}`).sort();
  return sha256(Buffer.from(keys.join('|')));
}

/** Print sha256 for any not-yet-pinned file so a maintainer can pin it. */
export function reportPins(name, files) {
  for (const [k, v] of Object.entries(files)) {
    if (v && !v.pinned) console.error(`[${name}] record sha256 for ${k}: ${v.sha256}`);
  }
}

/**
 * Carve a `transfer` split out of `valid` by a stable sha256 bucket (~`frac`),
 * so the loop-safety gate (ADR-004 gate 3) has a held-out transfer set.
 * Deterministic and documented; the remaining valid stays the validation set.
 */
export function carveTransfer(valid, frac = 0.3) {
  const cut = Math.round(frac * 100);
  const rest = [];
  const transfer = [];
  for (const t of valid) {
    const bucket = parseInt(sha256(`${t.s} ${t.r} ${t.o}`).slice(0, 8), 16) % 100;
    (bucket < cut ? transfer : rest).push(t);
  }
  return { valid: rest, transfer };
}

/**
 * Cache file name for one split. `tag` (the pinned upstream commit) keeps bytes
 * cached from an older source from colliding with the current pin.
 */
export const cacheFileName = (name, k, tag) => (tag ? `${name}@${tag}-${k}.txt` : `${name}-${k}.txt`);

/**
 * Fail closed unless the parsed, UNLIMITED dataset has exactly the canonical
 * split sizes and entity/relation counts (ADR-007 M0 identity assert). Counted
 * over the union of the raw splits; the transfer carve does not change it.
 * `expected` = { train, valid, test, entities, relations }.
 */
export function assertCanonicalCounts(name, { train, valid, test }, expected) {
  if (!expected) throw new Error(`${name}: no canonical counts declared — refusing to run`);
  const c = graphCounts({ train, valid, test });
  const bad = ['train', 'valid', 'test', 'entities', 'relations'].filter((k) => c[k] !== expected[k]);
  if (bad.length) {
    const d = bad.map((k) => `${k} ${c[k]} != ${expected[k]}`).join(', ');
    throw new Error(`${name}: not the canonical dataset (${d})`);
  }
  return c;
}

/**
 * Fetch+parse a standard 3-file triples dataset (train/valid/test.txt),
 * entity-consistently apply `--limit`, carve a transfer split, and package it in
 * the shape the harness consumes. `sources` maps train/valid/test -> URL;
 * `pins` maps the same keys -> sha256 (null until pinned). `expected` holds the
 * canonical counts, asserted on the full data before any `--limit` slicing;
 * `cacheTag` (upstream commit) namespaces the cache files.
 */
export async function loadStandardTriples({ name, sources, pins = {}, limit, cacheDir, licence, cacheTag, expected }) {
  const opts = { cacheDir };
  // Fail closed (ADR-007 M0): a benchmark suite never runs on unpinned bytes.
  // An unpinned file used to be fetched and only its hash printed, so a drifted
  // or swapped upstream would have been scored silently.
  const unpinned = ['train', 'valid', 'test'].filter((k) => !pins[k]);
  if (unpinned.length) {
    throw new Error(`${name}: no sha256 pin for ${unpinned.join(', ')} — refusing to run on unverified data`);
  }
  const got = {};
  for (const k of ['train', 'valid', 'test']) {
    got[k] = await fetchCached(sources[k], cacheFileName(name, k, cacheTag), { ...opts, knownSha256: pins[k] });
  }
  reportPins(name, got);
  let train = parseTriplesTsv(got.train.bytes.toString('utf8'));
  let valid = parseTriplesTsv(got.valid.bytes.toString('utf8'));
  let test = parseTriplesTsv(got.test.bytes.toString('utf8'));
  if (expected !== undefined) assertCanonicalCounts(name, { train, valid, test }, expected);
  if (typeof limit === 'number') ({ train, valid, test } = limitSubgraph({ train, valid, test }, limit));
  const carved = carveTransfer(valid);
  const splits = { train, valid: carved.valid, transfer: carved.transfer, test };
  return {
    name,
    licence: licence ?? null,
    sources: Object.fromEntries(Object.entries(sources)),
    fileHashes: Object.fromEntries(Object.entries(got).map(([k, v]) => [k, v.sha256])),
    splits,
    counts: graphCounts({ train, valid: carved.valid, transfer: carved.transfer, test }),
    splitsHash: stableSplitHash([...train, ...carved.valid, ...carved.transfer, ...test]),
  };
}

/**
 * Count distinct entities/relations across ALL split arrays, transfer included
 * (ADR-007 M0): the transfer split is carved out of valid, so an entity that
 * only appears there is still in the binding's entity table and in the tie-check
 * expectation (|E|+1)/2. FB15k-237 = 14541, WN18RR = 40943 (ConvE release; the
 * villmow `WN18RR/text` relabeling has 41105), CoDEx-M = 17050.
 */
export function graphCounts({ train = [], valid = [], transfer = [], test = [] }) {
  const ent = new Set();
  const rel = new Set();
  for (const arr of [train, valid, transfer, test]) {
    for (const t of arr) {
      ent.add(t.s);
      ent.add(t.o);
      rel.add(t.r);
    }
  }
  return {
    train: train.length,
    valid: valid.length,
    transfer: transfer.length,
    test: test.length,
    entities: ent.size,
    relations: rel.size,
  };
}
