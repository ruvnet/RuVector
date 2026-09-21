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
  if (existsSync(path)) {
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
    throw new Error(`sha256 mismatch for ${name}: expected ${knownSha256}, got ${digest} (source ${url})`);
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
 * Fetch+parse a standard 3-file triples dataset (train/valid/test.txt),
 * entity-consistently apply `--limit`, carve a transfer split, and package it in
 * the shape the harness consumes. `sources` maps train/valid/test -> URL;
 * `pins` maps the same keys -> sha256 (null until pinned).
 */
export async function loadStandardTriples({ name, sources, pins = {}, limit, cacheDir, licence }) {
  const opts = { cacheDir };
  const got = {};
  for (const k of ['train', 'valid', 'test']) {
    got[k] = await fetchCached(sources[k], `${name}-${k}.txt`, { ...opts, knownSha256: pins[k] ?? null });
  }
  reportPins(name, got);
  let train = parseTriplesTsv(got.train.bytes.toString('utf8'));
  let valid = parseTriplesTsv(got.valid.bytes.toString('utf8'));
  let test = parseTriplesTsv(got.test.bytes.toString('utf8'));
  if (typeof limit === 'number') ({ train, valid, test } = limitSubgraph({ train, valid, test }, limit));
  const carved = carveTransfer(valid);
  const splits = { train, valid: carved.valid, transfer: carved.transfer, test };
  return {
    name,
    licence: licence ?? null,
    sources: Object.fromEntries(Object.entries(sources)),
    fileHashes: Object.fromEntries(Object.entries(got).map(([k, v]) => [k, v.sha256])),
    splits,
    counts: { ...graphCounts({ train, valid: carved.valid, test }), transfer: carved.transfer.length },
    splitsHash: stableSplitHash([...train, ...carved.valid, ...carved.transfer, ...test]),
  };
}

/** Count distinct entities/relations across split arrays. */
export function graphCounts({ train = [], valid = [], test = [] }) {
  const ent = new Set();
  const rel = new Set();
  for (const arr of [train, valid, test]) {
    for (const t of arr) {
      ent.add(t.s);
      ent.add(t.o);
      rel.add(t.r);
    }
  }
  return { train: train.length, valid: valid.length, test: test.length, entities: ent.size, relations: rel.size };
}
