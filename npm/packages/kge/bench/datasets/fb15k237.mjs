#!/usr/bin/env node
// FB15k-237 (Toutanova & Chen 2015) — 310k triples, 14,541 entities, 237
// relations; dense-graph link prediction (ADR-006). Canonical splits from the
// standard mirror used by LibKGE/PyKEEN. Freebase-derived: fetched at bench
// time, nothing redistributed. Baseline: LibKGE ComplEx filtered MRR 0.348.

import { loadStandardTriples } from './lib.mjs';

const BASE = 'https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237';
const SOURCES = { train: `${BASE}/train.txt`, valid: `${BASE}/valid.txt`, test: `${BASE}/test.txt` };
// Pinned after the first successful fetch (2026-09-21). Integrity fails closed
// if the upstream file ever changes; re-pin deliberately if it does.
const PINS = {
  train: '61099230e4439f90885ca9767739e31e8e32f54736fa1c35952b27997bc7c08a',
  valid: '749cbe9d923bac7b9354da5614ecfed2e0220256d442c3e04a6b303db1f273d9',
  test: 'e2e35e8e6113de220140b6f44dc71a5207b0fc6872d575e874aefe13259b655b',
};

export function load({ limit, cacheDir } = {}) {
  return loadStandardTriples({
    name: 'fb15k237',
    sources: SOURCES,
    pins: PINS,
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
