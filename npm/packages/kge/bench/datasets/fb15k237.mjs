#!/usr/bin/env node
// FB15k-237 (Toutanova & Chen 2015) — 310,116 triples, 14,541 entities, 237
// relations; dense-graph link prediction (ADR-006). Freebase-derived: fetched at
// bench time, nothing redistributed. Baseline: LibKGE ComplEx filtered MRR 0.348.
//
// Source (ADR-007 M1, verified 2026-09-27): the RotatE repo pinned by COMMIT,
// so the URL is immutable. Its three files are byte-identical (same sha256) to
// train/valid/test.txt inside TimDettmers/ConvE@f3c0eb28 FB15k-237.tar.gz
// (tarball sha256 311df7930f842a02979eaa7a802484a6afe62303c07f915f1e6edbc30cbe05e8),
// the release the anchors use. Those bytes have CRLF line endings.
//
// Provenance of the earlier pins: villmow/datasets_knowledge_embedding (branch
// master, single data commit 9e392453, 2019) holds the SAME triples in the same
// order with LF endings (train 61099230…, valid 749cbe9d…, test e2e35e8e…). The
// parsed splits and splitsHash are identical to this source. The "sha256
// mismatch … got b664af6c…" report was NOT upstream drift: b664af6c… is
// sha256("a\tr\tb\n"), the tampered cache file written by the fail-closed test
// in test/bench-m0.test.mjs. villmow master still serves 61099230….

import { loadStandardTriples, cacheFileName } from './lib.mjs';

export const COMMIT = '2e440e0f9c687314d5ff67ead68ce985dc446e3a';
const BASE = `https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/${COMMIT}/data/FB15k-237`;
export const SOURCES = { train: `${BASE}/train.txt`, valid: `${BASE}/valid.txt`, test: `${BASE}/test.txt` };
// Pinned 2026-09-27 (raw bytes, CRLF). Integrity fails closed on any change.
export const PINS = {
  train: '6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae',
  valid: 'cf6309010852f6a8d47a45df830a426415d1ee6f7a3970a8376ff1fb81db4a5c',
  test: '5711cf41623ceb4eacc50eb6108a3ca6565c7492e3caaf82a3e355cc660d1574',
};
// Canonical counts (MEASURED; ssl-RP README "#Ent 14,541"). Fails closed otherwise.
export const EXPECTED = { train: 272115, valid: 17535, test: 20466, entities: 14541, relations: 237 };
export const CACHE_TAG = COMMIT.slice(0, 8);
export const cacheFile = (k) => cacheFileName('fb15k237', k, CACHE_TAG);

export function load({ limit, cacheDir } = {}) {
  return loadStandardTriples({
    name: 'fb15k237',
    sources: SOURCES,
    pins: PINS,
    expected: EXPECTED,
    cacheTag: CACHE_TAG,
    limit,
    cacheDir,
    licence: 'Freebase-derived; follows source, not redistributed',
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const i = process.argv.indexOf('--limit');
  const limit = i >= 0 ? parseInt(process.argv[i + 1], 10) : undefined;
  load({ limit })
    .then((d) => console.log(`fb15k237 OK — ${JSON.stringify(d.counts)}`))
    .catch((e) => {
      console.error(`fb15k237 unavailable: ${e && e.message ? e.message : e}`);
      process.exit(1);
    });
}
