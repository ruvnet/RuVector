#!/usr/bin/env node
// HWU64 (Liu et al. 2019) — 64 intents across 21 domains. Canonical source:
// xliuhw/NLU-Evaluation-Data (GitHub raw), a semicolon-separated CSV whose
// columns include `scenario`, `intent`, and `answer` (the utterance). Research
// use. The label is `scenario_intent` so the 64 intents are distinct.

import { createHash } from 'node:crypto';
import { fetchCached, humaniseLabel, choiceQuestion, stableSplitHash } from './lib.mjs';

const SOURCE =
  'https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data/master/AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv';
const KNOWN_SHA256 = '5f6dbf6d38fc111217924945ac59c554e0b926d5aa836ecdd0d089d2ca48e1d9';

function parseSemicolon(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  const header = lines.shift().split(';').map((c) => c.trim().toLowerCase());
  const sIdx = header.indexOf('scenario');
  const iIdx = header.indexOf('intent');
  const aIdx = header.indexOf('answer');
  const rows = [];
  for (const line of lines) {
    const parts = line.split(';');
    if (parts.length <= Math.max(sIdx, iIdx, aIdx)) continue;
    const scenario = (parts[sIdx] || '').trim();
    const intent = (parts[iIdx] || '').trim();
    const answer = (parts[aIdx] || '').trim();
    if (!scenario || !intent || !answer) continue;
    rows.push({ text: answer, label: `${scenario}_${intent}` });
  }
  return rows;
}

// Deterministic 80/20 train/test by a stable hash of the utterance (not row
// order, so the split is stable if upstream row order shifts).
function splitTrainTest(rows) {
  const train = [];
  const test = [];
  rows.forEach((r, i) => {
    const h = createHash('sha256').update(r.text + '|' + r.label).digest();
    const bucket = h[0] % 100;
    (bucket < 20 ? test : train).push({ id: `hwu-${i}`, ...r });
  });
  return { train, test };
}

export async function load({ limit, cacheDir } = {}) {
  const f = await fetchCached(SOURCE, 'hwu64-all.csv', { cacheDir, knownSha256: KNOWN_SHA256 });
  if (!f.pinned) console.error(`[hwu64] record sha256 for all.csv: ${f.sha256}`);
  const rows = parseSemicolon(f.bytes.toString('utf8'));
  const { train, test } = splitTrainTest(rows);
  let testItems = test;
  if (typeof limit === 'number') testItems = testItems.slice(0, limit);

  const labels = [...new Set(rows.map((r) => r.label))].sort();
  const criteria = Object.fromEntries(labels.map((l) => [l, humaniseLabel(l)]));

  return {
    labels,
    questions: choiceQuestion(criteria, 'Which intent (scenario and action) does this utterance express'),
    trainItems: train,
    testItems,
    counts: { train: train.length, test: testItems.length, labels: labels.length },
    hasOos: false,
    splitsHash: stableSplitHash([...train, ...testItems].map((i) => i.id)),
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const li = process.argv.indexOf('--limit');
  const limit = li >= 0 ? parseInt(process.argv[li + 1], 10) : undefined;
  load({ limit })
    .then((d) => console.log(`hwu64 OK — ${d.counts.train} train / ${d.counts.test} test / ${d.labels.length} labels`))
    .catch((e) => {
      console.error(`hwu64 unavailable: ${e && e.message ? e.message : e}`);
      process.exit(1);
    });
}
