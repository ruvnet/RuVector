#!/usr/bin/env node
// YAGO3-10 (Dettmers et al. 2018) — ~1M triples, 123k entities, 37 relations;
// scale/throughput (ADR-006, informational). YAGO-derived: fetched at bench
// time, nothing redistributed. LibKGE ComplEx MRR 0.551 is INFORMATIONAL only;
// the latency/throughput gates are set after the ADR-002 spike. Fetched lazily —
// the harness only pulls this suite with `--suite yago310`.
//
// Source (ADR-007 M0): the previous villmow mirror has no YAGO3-10 directory
// (HTTP 404), and the pins were all null, so the suite failed open. It now reads
// the plain-text copy in the RotatE repo (DeepGraphLearning/
// KnowledgeGraphEmbedding). Verified 2026-09-26: all three files are
// byte-identical (same sha256) to train/valid/test.txt inside the original
// TimDettmers/ConvE YAGO3-10.tar.gz — 1,079,040 / 5,000 / 5,000 lines.

import { loadStandardTriples } from './lib.mjs';

const BASE = 'https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10';
const SOURCES = { train: `${BASE}/train.txt`, valid: `${BASE}/valid.txt`, test: `${BASE}/test.txt` };
// Pinned 2026-09-26. Integrity fails closed if the upstream file ever changes.
export const PINS = {
  train: 'afb9b51c68d1c997e85655477045b4c5146cd2a6b50b6eea5373383f35dcb12a',
  valid: 'c9018b77ec77e99f8d48bc3258d404f10b2049a6c7b4ac6eb663697e799dc6f5',
  test: '003887ca8a34c90fcaf9b0250b1c30a4b5f617f80f8cf0aeab9723333061598a',
};

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
