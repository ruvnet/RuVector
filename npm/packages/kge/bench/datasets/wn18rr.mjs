#!/usr/bin/env node
// WN18RR (Dettmers et al. 2018) — 93k triples, ~41k entities, 11 relations;
// sparse/symmetric tie-break stress (ADR-006). Its symmetric relations produce
// large tied blocks, which is exactly where TOP tie-breaking inflates and
// RANDOM (this harness) does not. Canonical splits from the standard mirror.
// WordNet-derived: fetched at bench time, nothing redistributed. Baseline:
// LibKGE ComplEx filtered MRR 0.475 (RotatE 0.478).

import { loadStandardTriples } from './lib.mjs';

const BASE = 'https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/WN18RR/text';
const SOURCES = { train: `${BASE}/train.txt`, valid: `${BASE}/valid.txt`, test: `${BASE}/test.txt` };
// Pinned after the first successful fetch (2026-09-21). Fails closed on drift.
const PINS = {
  train: 'a35aca3c963b71b8efa935c172bf07afe4472655552a0ad8d3b396b4154f1f29',
  valid: '2c71419fcd86a1720020a8709ab3e8bbb6e0d097a8abce41135fe4c43fec5f84',
  test: '80cfa61fe62ba084f58de42d8d0b384dc8b76bcd2ef7065b1a292a1e72b44c33',
};

export function load({ limit, cacheDir } = {}) {
  return loadStandardTriples({
    name: 'wn18rr',
    sources: SOURCES,
    pins: PINS,
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
