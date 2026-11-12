#!/usr/bin/env node
// Banking77 (PolyAI, arXiv:2003.04807) — 77 fine-grained banking intents.
// Canonical source: PolyAI-LDN/task-specific-datasets (GitHub raw). Research use.

import { fetchCached, humaniseLabel, choiceQuestion, stableSplitHash } from './lib.mjs';

const BASE = 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data';
const SOURCES = {
  train: `${BASE}/train.csv`,
  test: `${BASE}/test.csv`,
  categories: `${BASE}/categories.json`,
};
// Pin after the first successful fetch (printed by the fetcher when null).
const KNOWN_SHA256 = {
  train: 'b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b',
  test: 'd12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d',
  categories: '53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63',
};

function parseCsv(text) {
  // train/test.csv are "text,category" with a header; text may be quoted.
  const rows = [];
  const lines = text.split(/\r?\n/);
  const header = lines.shift();
  const cols = header.split(',').map((c) => c.trim().toLowerCase());
  const tIdx = cols.indexOf('text');
  const cIdx = cols.indexOf('category');
  for (const line of lines) {
    if (!line.trim()) continue;
    const { fields } = splitCsvLine(line);
    if (fields.length < 2) continue;
    rows.push({ text: fields[tIdx], label: fields[cIdx] });
  }
  return rows;
}

function splitCsvLine(line) {
  const fields = [];
  let cur = '';
  let q = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (q) {
      if (ch === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') q = false;
      else cur += ch;
    } else if (ch === '"') q = true;
    else if (ch === ',') {
      fields.push(cur);
      cur = '';
    } else cur += ch;
  }
  fields.push(cur);
  return { fields };
}

export async function load({ limit, cacheDir } = {}) {
  const opts = { cacheDir };
  const train = await fetchCached(SOURCES.train, 'banking77-train.csv', { ...opts, knownSha256: KNOWN_SHA256.train });
  const test = await fetchCached(SOURCES.test, 'banking77-test.csv', { ...opts, knownSha256: KNOWN_SHA256.test });
  const cats = await fetchCached(SOURCES.categories, 'banking77-categories.json', {
    ...opts,
    knownSha256: KNOWN_SHA256.categories,
  });
  reportPins('banking77', { train, test, categories: cats });

  const labels = JSON.parse(cats.bytes.toString('utf8'));
  const criteria = Object.fromEntries(labels.map((l) => [l, humaniseLabel(l)]));
  const trainItems = parseCsv(train.bytes.toString('utf8')).map((r, i) => ({ id: `b77-tr-${i}`, ...r }));
  let testItems = parseCsv(test.bytes.toString('utf8')).map((r, i) => ({ id: `b77-te-${i}`, ...r }));
  if (typeof limit === 'number') testItems = testItems.slice(0, limit);

  return {
    labels,
    questions: choiceQuestion(criteria, 'Which banking intent does this message express'),
    trainItems,
    testItems,
    counts: { train: trainItems.length, test: testItems.length, labels: labels.length },
    hasOos: false,
    splitsHash: stableSplitHash([...trainItems, ...testItems].map((i) => i.id)),
  };
}

function reportPins(name, files) {
  for (const [k, v] of Object.entries(files)) {
    if (!v.pinned) console.error(`[${name}] record sha256 for ${k}: ${v.sha256}`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const limitArg = process.argv.indexOf('--limit');
  const limit = limitArg >= 0 ? parseInt(process.argv[limitArg + 1], 10) : undefined;
  load({ limit })
    .then((d) => console.log(`banking77 OK — ${d.counts.train} train / ${d.counts.test} test / ${d.labels.length} labels`))
    .catch((e) => {
      console.error(`banking77 unavailable: ${e && e.message ? e.message : e}`);
      process.exit(1);
    });
}
