// OpenJev v1 harness additions: the explicit calibration slice (ADR-008 §4) and
// the public-suite validation view for --no-test selection runs.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { loadTickets, excludeHeldOutText } from '../lib/fixture.mjs';
import { trainTicketQuestions } from '../lib/arms.mjs';
import { isValBucket, validationView } from '../lib/public-val.mjs';
import { sha256Norm } from '../lib/norm.mjs';

function recorder() {
  const batches = [];
  return { batches, engine: { trainJson: (json) => (batches.push(JSON.parse(json)), '{}') } };
}

test('tickets calibration split is sent whole as `calibration`, never as training examples', () => {
  const fixture = loadTickets();
  const pool = excludeHeldOutText(fixture.bySplit);
  const calib = fixture.bySplit.calibration;
  const { batches, engine } = recorder();
  const result = trainTicketQuestions(engine, pool.trainItems, fixture.questions, { shots: 10000, calibrationItems: calib });
  const calibTexts = new Set(calib.map((it) => it.text));
  for (const b of batches) {
    assert.equal(b.calibration.length, calib.length, `${b.question}: every calibration row, no shot cap`);
    assert.ok(b.calibration.every((e) => calibTexts.has(e.text)));
    assert.ok(b.examples.every((e) => !calibTexts.has(e.text)), 'calibration text leaked into training examples');
    assert.equal(result.questions[b.question].calibrationUsed, calib.length);
  }
  const legend = fixture.questions.frustration.criteria;
  assert.deepEqual(new Set(batches[1].calibration.map((e) => e.label)), new Set(['yes', 'no']));
  assert.ok(batches[2].calibration.every((e) => legend.includes(e.label)));

  // Without calibration items the request shape is unchanged (engine carve).
  const plain = recorder();
  trainTicketQuestions(plain.engine, pool.trainItems, fixture.questions, { shots: 10000 });
  assert.ok(plain.batches.every((b) => !('calibration' in b)));
});

test('public validation view mirrors the exporter carve and drops held-out collisions', () => {
  const trainItems = Array.from({ length: 200 }, (_, i) => ({ id: `b77-tr-${i}`, text: `row ${i}`, label: `l${i % 3}` }));
  const valIds = trainItems.filter((it) => isValBucket(it.id));
  assert.ok(valIds.length > 5 && valIds.length < 40, 'about 10 % land in the bucket');
  const collide = valIds[0];
  const ds = { trainItems, testItems: [{ id: 'b77-te-0', text: `  ROW ${collide.text.split(' ')[1]}!`, label: 'l0' }] };
  const v = validationView('banking77', ds);
  assert.equal(v.trainItems.length + valIds.length, trainItems.length);
  assert.ok(v.trainItems.every((it) => !isValBucket(it.id)), 'validation rows never train the engine');
  assert.equal(v.counts.dropped_heldout_collisions, 1, 'normalised test-text collision dropped');
  assert.ok(v.evalItems.every((it) => sha256Norm(it.text) !== sha256Norm(collide.text)));

  const clinc = validationView('clinc150', {
    trainItems: trainItems.slice(0, 3),
    testItems: [],
    valItems: [{ id: 'clinc-va-0', text: 'x', label: 'a', oos: false }, { id: 'clinc-oosva-0', text: 'y', label: 'oos', oos: true }],
  });
  assert.equal(clinc.trainItems.length, 3, 'CLINC150 trains on the full official train');
  assert.deepEqual(clinc.evalItems.map((it) => it.oos), [false, true]);
  assert.throws(() => validationView('clinc150', { trainItems: [], testItems: [] }), /official val/);
});
