#!/usr/bin/env node
// CLINC150 + OOS (Larson et al. 2019) — 150 in-scope intents plus an
// out-of-scope slice. Canonical source: clinc/oos-eval (GitHub raw,
// data/data_full.json: keys train/val/test + oos_train/oos_val/oos_test, each
// a list of [text, label] pairs). The OOS slice is the abstain/noul set used
// for the OOS AUROC gate (ADR-006).

import { fetchCached, humaniseLabel, choiceQuestion, stableSplitHash } from './lib.mjs';

const SOURCE = 'https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json';
const KNOWN_SHA256 = '36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0';

function toItems(pairs, prefix) {
  return (pairs ?? []).map(([text, label], i) => ({ id: `${prefix}-${i}`, text, label }));
}

export async function load({ limit, cacheDir } = {}) {
  const f = await fetchCached(SOURCE, 'clinc150-data_full.json', { cacheDir, knownSha256: KNOWN_SHA256 });
  if (!f.pinned) console.error(`[clinc150] record sha256 for data_full: ${f.sha256}`);
  const data = JSON.parse(f.bytes.toString('utf8'));

  const trainItems = toItems(data.train, 'clinc-tr');
  let inScope = toItems(data.test, 'clinc-te');
  let oos = toItems(data.oos_test, 'clinc-oos');
  if (typeof limit === 'number') {
    inScope = inScope.slice(0, limit);
    oos = oos.slice(0, Math.max(1, Math.round(limit * 0.1)));
  }
  // OOS items carry the sentinel label 'oos' so the choice arm can be scored
  // for in-scope accuracy while the OOS AUROC gate uses the abstain mass.
  const oosItems = oos.map((it) => ({ ...it, label: 'oos', oos: true }));

  const labels = [...new Set(trainItems.map((i) => i.label))].sort();
  const criteria = Object.fromEntries(labels.map((l) => [l, humaniseLabel(l)]));
  const testItems = [...inScope, ...oosItems];

  return {
    labels,
    questions: choiceQuestion(criteria, 'Which assistant intent does this utterance express'),
    trainItems,
    testItems,
    inScopeItems: inScope,
    oosItems,
    counts: { train: trainItems.length, test_in_scope: inScope.length, test_oos: oosItems.length, labels: labels.length },
    hasOos: true,
    splitsHash: stableSplitHash([...trainItems, ...testItems].map((i) => i.id)),
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const li = process.argv.indexOf('--limit');
  const limit = li >= 0 ? parseInt(process.argv[li + 1], 10) : undefined;
  load({ limit })
    .then((d) =>
      console.log(`clinc150 OK — ${d.counts.train} train / ${d.counts.test_in_scope} in-scope / ${d.counts.test_oos} oos / ${d.labels.length} labels`),
    )
    .catch((e) => {
      console.error(`clinc150 unavailable: ${e && e.message ? e.message : e}`);
      process.exit(1);
    });
}
