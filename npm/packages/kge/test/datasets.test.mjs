// Dataset integrity (ADR-007 M1): immutable sources, sha256 pins and the
// canonical-count identity assert. The offline tests need no network. The live
// tests load the real files and run only when they are already cached under
// bench/.cache or KGE_DATASET_FETCH=1 is set (they then download ~75 MB).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { assertCanonicalCounts, loadStandardTriples, fetchCached, DEFAULT_CACHE } from '../bench/datasets/lib.mjs';
import * as fb from '../bench/datasets/fb15k237.mjs';
import * as wn from '../bench/datasets/wn18rr.mjs';
import * as yago from '../bench/datasets/yago310.mjs';
import * as codex from '../bench/datasets/codexm.mjs';

// The standard published statistics. Changing a number here is a claim about
// the dataset, not a test fix.
const CANONICAL = {
  fb15k237: { mod: fb, counts: { train: 272115, valid: 17535, test: 20466, entities: 14541, relations: 237 } },
  wn18rr: { mod: wn, counts: { train: 86835, valid: 3034, test: 3134, entities: 40943, relations: 11 } },
  codexm: { mod: codex, counts: { train: 185584, valid: 10310, test: 10311, entities: 17050, relations: 51 } },
  yago310: { mod: yago, counts: { train: 1079040, valid: 5000, test: 5000, entities: 123182, relations: 37 } },
};

test('datasets: declared canonical counts are the standard ones', () => {
  for (const [name, { mod, counts }] of Object.entries(CANONICAL)) assert.deepEqual(mod.EXPECTED, counts, name);
  assert.deepEqual(codex.EXPECTED_NEGATIVES, { valid: 10310, test: 10311 });
});

test('datasets: every source is pinned to a git commit and every file to a sha256', () => {
  for (const [name, { mod }] of Object.entries(CANONICAL)) {
    assert.match(mod.COMMIT, /^[0-9a-f]{40}$/, name);
    for (const [k, url] of Object.entries(mod.SOURCES)) {
      assert.ok(url.includes(`/${mod.COMMIT}/`), `${name}.${k} is not commit-pinned: ${url}`);
      assert.doesNotMatch(url, /\/(master|main)\//, `${name}.${k} follows a branch`);
      assert.match(mod.PINS[k], /^[0-9a-f]{64}$/, `${name}.${k} unpinned`);
    }
  }
});

test('datasets: FB15k-237 is not the villmow LF copy and WN18RR is not the text variant', () => {
  // villmow FB15k-237 (same triples, LF): pins were valid for that mirror only.
  assert.notEqual(fb.PINS.train, '61099230e4439f90885ca9767739e31e8e32f54736fa1c35952b27997bc7c08a');
  // villmow WN18RR/text (41,105 entities).
  assert.notEqual(wn.PINS.train, 'a35aca3c963b71b8efa935c172bf07afe4472655552a0ad8d3b396b4154f1f29');
});

const t = (s, r, o) => ({ s, r, o });

test('assertCanonicalCounts: accepts exact counts, rejects a doctored entity set', () => {
  const splits = { train: [t('a', 'r', 'b'), t('b', 'q', 'c')], valid: [t('a', 'r', 'c')], test: [t('c', 'r', 'a')] };
  const exp = { train: 2, valid: 1, test: 1, entities: 3, relations: 2 };
  assert.equal(assertCanonicalCounts('x', splits, exp).entities, 3);
  // A WN18RR-text-style relabeling: same triple count, one id split into two names.
  const doctored = { ...splits, test: [t('c', 'r', 'a2')] };
  assert.throws(() => assertCanonicalCounts('x', doctored, exp), /not the canonical dataset \(entities 4 != 3\)/);
  assert.throws(() => assertCanonicalCounts('x', splits, undefined), /no canonical counts declared/);
});

test('loadStandardTriples: a pinned-but-wrong dataset fails the WN18RR identity assert', async () => {
  // Doctored cached split with valid pins for its own bytes: integrity passes,
  // identity must not (the villmow text split would reach exactly this path).
  const dir = mkdtempSync(join(tmpdir(), 'kge-ds-'));
  const body = { train: 'a\tr\tb\n', valid: 'a\tr\tc\n', test: 'c\tr\ta2\n' };
  const pins = {};
  for (const [k, v] of Object.entries(body)) {
    writeFileSync(join(dir, `doc-${k}.txt`), v);
    pins[k] = (await fetchCached('http://invalid/', `doc-${k}.txt`, { cacheDir: dir })).sha256;
  }
  const sources = { train: 'http://invalid/', valid: 'http://invalid/', test: 'http://invalid/' };
  await assert.rejects(
    loadStandardTriples({ name: 'doc', sources, pins, cacheDir: dir, expected: { ...wn.EXPECTED } }),
    /doc: not the canonical dataset \(train 1 != 86835/,
  );
});

test('fetchCached: a tampered cached file names the cache, not the URL, as origin', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'kge-ds-'));
  writeFileSync(join(dir, fb.cacheFile('train')), 'a\tr\tb\n');
  await assert.rejects(
    fetchCached(fb.SOURCES.train, fb.cacheFile('train'), { cacheDir: dir, knownSha256: fb.PINS.train }),
    /sha256 mismatch .*got b664af6c.* cached file .* delete it to re-fetch/,
  );
});

// ---------------------------------------------------------------------------
// live: the real files have the canonical counts
// ---------------------------------------------------------------------------

const FETCH = process.env.KGE_DATASET_FETCH === '1';
const cachedAll = (files) => files.every((f) => existsSync(join(DEFAULT_CACHE, f)));
const CACHE_FILES = {
  fb15k237: ['train', 'valid', 'test'].map(fb.cacheFile),
  wn18rr: ['train', 'valid', 'test'].map(wn.cacheFile),
  yago310: ['train', 'valid', 'test'].map(yago.cacheFile),
  codexm: Object.keys(codex.SOURCES).map((k) => `codexm@${codex.CACHE_TAG}-${k}`),
};

for (const [name, { mod, counts }] of Object.entries(CANONICAL)) {
  const skip = FETCH || cachedAll(CACHE_FILES[name]) ? false : 'not cached; set KGE_DATASET_FETCH=1';
  test(`live: ${name} loads with the canonical counts`, { skip, timeout: 600_000 }, async () => {
    const d = await mod.load({});
    const c = d.counts;
    assert.equal(c.train, counts.train);
    assert.equal(c.valid + c.transfer, counts.valid, 'transfer is carved out of valid');
    assert.equal(c.test, counts.test);
    assert.equal(c.entities, counts.entities);
    assert.equal(c.relations, counts.relations);
    if (name === 'codexm') assert.deepEqual(c.hard_negatives, codex.EXPECTED_NEGATIVES);
  });
}
