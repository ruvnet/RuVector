#!/usr/bin/env node
// CoDEx-M (Safavi & Koutra 2020, arXiv:2009.07810) — 185k triples, 17,050
// entities, 51 relations; mid-scale gate + adversarial (ADR-006). Wikidata-
// derived; the repo states CC BY 4.0 — VERIFIED at fetch from its LICENSE file,
// not hardcoded. Ships human-curated HARD NEGATIVES (valid/test), used for the
// hard-negative confidence report. Baseline MRR: read from arXiv:2009.07810's
// table at harness build (ADR-006) — not invented here; the gate SKIPs until a
// maintainer records it with a source.

import { fetchCached, parseTriplesTsv, limitSubgraph, carveTransfer, graphCounts, stableSplitHash, reportPins } from './lib.mjs';

const REPO = 'https://raw.githubusercontent.com/tsafavi/codex/master';
const TRIP = `${REPO}/data/triples/codex-m`;
const SOURCES = {
  train: `${TRIP}/train.txt`,
  valid: `${TRIP}/valid.txt`,
  test: `${TRIP}/test.txt`,
  valid_neg: `${TRIP}/valid_negatives.txt`,
  test_neg: `${TRIP}/test_negatives.txt`,
  licence: `${REPO}/LICENSE`,
};
// Pinned after the first successful fetch (2026-09-21). Fails closed on drift.
const PINS = {
  train: 'd99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556',
  valid: '11c323096367354940846b9ed940c5dabddc1d20a70474222babe7e7fd28b28d',
  test: '0575ce05e4ce915e395bb4f708e8df9547989dd4c8c71407cbfd78c19daffcf7',
  valid_neg: '454a4f66ae85495e6e7df0a9f591ed3b8b6efbbcca515414cb641a28625cbcb2',
  test_neg: '7caaf71f7a85fbd915dce8199f5b1e1744ead58e0675549fcb01c60599944648',
  licence: '8a8d718d139b33e7f1938aaf5c2b724f3ae1930411ca9593339d87cd5740621e',
};

/** Best-effort SPDX detection from the LICENSE text. */
function detectLicence(text) {
  const t = text.toLowerCase();
  if (t.includes('creative commons') && t.includes('attribution') && t.includes('4.0')) {
    return t.includes('sharealike') ? 'CC-BY-SA-4.0' : 'CC-BY-4.0';
  }
  if (t.includes('mit license')) return 'MIT';
  if (t.includes('apache license') && t.includes('2.0')) return 'Apache-2.0';
  return 'UNKNOWN (see LICENSE bytes hash)';
}

export async function load({ limit, cacheDir } = {}) {
  const opts = { cacheDir };
  const got = {};
  for (const k of Object.keys(SOURCES)) {
    got[k] = await fetchCached(SOURCES[k], `codexm-${k}`, { ...opts, knownSha256: PINS[k] ?? null });
  }
  reportPins('codexm', got);

  let train = parseTriplesTsv(got.train.bytes.toString('utf8'));
  let valid = parseTriplesTsv(got.valid.bytes.toString('utf8'));
  let test = parseTriplesTsv(got.test.bytes.toString('utf8'));
  let validNeg = parseTriplesTsv(got.valid_neg.bytes.toString('utf8'));
  let testNeg = parseTriplesTsv(got.test_neg.bytes.toString('utf8'));

  if (typeof limit === 'number') {
    ({ train, valid, test } = limitSubgraph({ train, valid, test }, limit));
    const kept = new Set([...train, ...valid, ...test].flatMap((t) => [t.s, t.o]));
    const inKept = (t) => kept.has(t.s) && kept.has(t.o);
    validNeg = validNeg.filter(inKept);
    testNeg = testNeg.filter(inKept);
  }
  const carved = carveTransfer(valid);
  // The repo ROOT LICENSE covers the CODE (detected below — MIT as of the pin).
  // The TRIPLES are Wikidata-derived and licensed CC BY 4.0 per the CoDEx paper
  // (arXiv:2009.07810) and README — not re-verified from a data-dir LICENSE, and
  // moot for redistribution here since nothing is redistributed (fetched at
  // bench time). Both are recorded so the claim is auditable, not assumed.
  const codeLicence = detectLicence(got.licence.bytes.toString('utf8'));
  const dataLicence = 'CC-BY-4.0 (per CoDEx README/paper arXiv:2009.07810; Wikidata-derived)';

  return {
    name: 'codexm',
    licence: dataLicence,
    code_licence: codeLicence,
    data_licence: dataLicence,
    licence_hash: got.licence.sha256,
    sources: Object.fromEntries(Object.entries(SOURCES)),
    fileHashes: Object.fromEntries(Object.entries(got).map(([k, v]) => [k, v.sha256])),
    splits: { train, valid: carved.valid, transfer: carved.transfer, test },
    hardNegatives: { valid: validNeg, test: testNeg },
    counts: {
      ...graphCounts({ train, valid: carved.valid, test }),
      transfer: carved.transfer.length,
      hard_negatives: { valid: validNeg.length, test: testNeg.length },
    },
    splitsHash: stableSplitHash([...train, ...carved.valid, ...carved.transfer, ...test]),
    baselineMrr: null, // read from arXiv:2009.07810 at build; SKIP its gate until recorded
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const i = process.argv.indexOf('--limit');
  const limit = i >= 0 ? parseInt(process.argv[i + 1], 10) : undefined;
  load({ limit })
    .then((d) => console.log(`codexm OK — code=${d.code_licence} data=${d.data_licence} — ${JSON.stringify(d.counts)}`))
    .catch((e) => {
      console.error(`codexm unavailable: ${e && e.message ? e.message : e}`);
      process.exit(1);
    });
}
