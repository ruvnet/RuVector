#!/usr/bin/env node
// WN18RR (Dettmers et al. 2018) — 93,003 triples, 40,943 entities, 11 relations;
// sparse/symmetric tie-break stress (ADR-006). Its symmetric relations produce
// large tied blocks, which is exactly where TOP tie-breaking inflates and
// RANDOM (this harness) does not. WordNet-derived: fetched at bench time,
// nothing redistributed. Baseline: LibKGE ComplEx filtered MRR 0.475
// (RotatE 0.478).
//
// Source (ADR-007 M1, verified 2026-09-27): the ID-based ConvE release the
// anchors use (ssl-RP "#Ent 40,943", `ComplEx(sizes=[40943,22,40943])`; kbc,
// IVR, DURA, LibKGE), read from the RotatE repo pinned by COMMIT. Byte-identical
// to train/valid/test.txt in TimDettmers/ConvE@f3c0eb28 WN18RR.tar.gz (tarball
// sha256 1eb9152f804c140d163462c0d49f65e5a0b30ab5fbb375c5e3353c11a6249060) and
// to villmow `WN18RR/original`.
//
// NOT used: villmow `WN18RR/text` (the previous source). Same 86,835/3,034/3,134
// triples in the same order with the same relations, but entities are WordNet
// synset names (`land_reform.n.01`) instead of 8-digit offsets. ConvE ids are
// bare offsets, which are unique only per part of speech, so 161 ConvE ids each
// stand for 2 or 3 synsets (e.g. 00594146 = lectureship.n.01 AND
// irregular.s.02); the text release splits them, giving 41,105 entities
// (MEASURED, aligned line by line). Different entity set and candidate count,
// so no anchor comparison is valid on it.

import { loadStandardTriples, cacheFileName } from './lib.mjs';

export const COMMIT = '2e440e0f9c687314d5ff67ead68ce985dc446e3a';
const BASE = `https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/${COMMIT}/data/wn18rr`;
export const SOURCES = { train: `${BASE}/train.txt`, valid: `${BASE}/valid.txt`, test: `${BASE}/test.txt` };
// Pinned 2026-09-27. Fails closed on drift.
export const PINS = {
  train: '038612e783c215ee5f3ca9fbfca27b8d0739be1028fe4ee7c174aecf0b83d5df',
  valid: '453ce7202afa58094a04d2b1560ee2b02660f1c260b32ce6651c8ccedd1028ab',
  test: '0383bceaaa1096cf3c03ec021ed0048068e2355dbfc0239b292cefdac821cec5',
};
// Canonical counts (MEASURED). The text variant (41,105 entities) fails this.
export const EXPECTED = { train: 86835, valid: 3034, test: 3134, entities: 40943, relations: 11 };
export const CACHE_TAG = COMMIT.slice(0, 8);
export const cacheFile = (k) => cacheFileName('wn18rr', k, CACHE_TAG);

export function load({ limit, cacheDir } = {}) {
  return loadStandardTriples({
    name: 'wn18rr',
    sources: SOURCES,
    pins: PINS,
    expected: EXPECTED,
    cacheTag: CACHE_TAG,
    limit,
    cacheDir,
    licence: 'WordNet-derived; follows source, not redistributed',
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const i = process.argv.indexOf('--limit');
  const limit = i >= 0 ? parseInt(process.argv[i + 1], 10) : undefined;
  load({ limit })
    .then((d) => console.log(`wn18rr OK — ${JSON.stringify(d.counts)}`))
    .catch((e) => {
      console.error(`wn18rr unavailable: ${e && e.message ? e.message : e}`);
      process.exit(1);
    });
}
