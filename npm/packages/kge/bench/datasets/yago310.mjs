#!/usr/bin/env node
// YAGO3-10 (Dettmers et al. 2018) — ~1M triples, 123k entities, 37 relations;
// scale/throughput (ADR-006, informational). YAGO-derived: fetched at bench
// time, nothing redistributed. LibKGE ComplEx MRR 0.551 is INFORMATIONAL only;
// the latency/throughput gates are set after the ADR-002 spike. Fetched lazily —
// the harness only pulls this suite with `--suite yago310`.

import { loadStandardTriples } from './lib.mjs';

const BASE = 'https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/YAGO3-10';
const SOURCES = { train: `${BASE}/train.txt`, valid: `${BASE}/valid.txt`, test: `${BASE}/test.txt` };
const PINS = { train: null, valid: null, test: null };

export function load({ limit, cacheDir } = {}) {
  return loadStandardTriples({
    name: 'yago310',
    sources: SOURCES,
    pins: PINS,
    limit,
    cacheDir,
    licence: 'YAGO-derived; follows source, not redistributed',
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const i = process.argv.indexOf('--limit');
  const limit = i >= 0 ? parseInt(process.argv[i + 1], 10) : undefined;
  load({ limit })
    .then((d) => console.log(`yago310 OK — ${JSON.stringify(d.counts)}`))
    .catch((e) => {
      console.error(`yago310 unavailable: ${e && e.message ? e.message : e}`);
      process.exit(1);
    });
}
