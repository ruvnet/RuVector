// Jev's continuous noul in the replay: AUROC is reported when a capture keeps
// `scores`, and stays null (as before) when it does not.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { replayJev } from '../lib/arms.mjs';
import { scoreRecords } from '../lib/receipt.mjs';

const BENCH = join(dirname(fileURLToPath(import.meta.url)), '..');
const readJson = (p) => JSON.parse(readFileSync(join(BENCH, p), 'utf8'));
const testItems = readJson('fixtures/tickets-decisions.json').split.test;
const departments = readJson('fixtures/tickets-decisions.json').departments;

test('the 2026-09-21 baseline has no continuous scores, so urgent AUROC stays null', () => {
  const r = replayJev(readJson('jev-baseline-2026-09-21.json'), testItems, { arm: 'baseline' });
  assert.ok(r.available);
  assert.ok(r.records.every((x) => x.urgent.score === null));
  assert.equal(scoreRecords(r.records, { departments }).urgent_auroc, null);
});

test('the live capture replays continuous noul and reports urgent AUROC', () => {
  const r = replayJev(readJson('jev-live-2026-09-25.json'), testItems, { arm: 'live' });
  assert.ok(r.available);
  assert.equal(r.records.length, 150);
  assert.ok(r.records.every((x) => typeof x.urgent.score === 'number'));
  const m = scoreRecords(r.records, { departments });
  assert.ok(Math.abs(m.choice_accuracy - 0.8467) < 1e-3, `accuracy ${m.choice_accuracy}`);
  // At the fixed 0.5 threshold Jev looks weak on urgency; ranked, it is strong.
  assert.ok(Math.abs(m.urgent_accuracy - 0.5867) < 1e-3, `urgent acc ${m.urgent_accuracy}`);
  assert.ok(m.urgent_auroc > 0.9, `urgent AUROC ${m.urgent_auroc}`);
});
