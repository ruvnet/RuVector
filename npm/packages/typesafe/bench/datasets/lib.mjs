// Shared dataset plumbing: cached, hash-verified download of a canonical public
// source (ADR-006 §Where it lives: "dataset fetchers download from the canonical
// sources at bench time and nothing is redistributed in the package"). Cache
// lives under bench/.cache (gitignored). No datasets or weights are committed.

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
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
export async function fetchCached(url, name, { cacheDir = DEFAULT_CACHE, knownSha256 = null } = {}) {
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
    writeFileSync(path, bytes);
  }
  const digest = sha256(bytes);
  if (knownSha256 && digest !== knownSha256) {
    throw new Error(`sha256 mismatch for ${name}: expected ${knownSha256}, got ${digest} (source ${url})`);
  }
  return { bytes, path, sha256: digest, pinned: !!knownSha256 };
}

/** Humanise an intent id (snake/dot case) into a plain-English criterion. */
export function humaniseLabel(label) {
  return String(label)
    .replace(/[._]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Build a gen0-style choice question from a label→description map. */
export function choiceQuestion(criteria, instructions) {
  return { intent: { type: 'choice', instructions, criteria } };
}

export function stableSplitHash(ids) {
  return sha256(Buffer.from([...ids].sort().join('|')));
}
