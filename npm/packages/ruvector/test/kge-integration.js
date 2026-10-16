'use strict';

// Real published native/WASM runtime tests. Install the optional peer explicitly:
// npm install --no-save @ruvector/kge@0.2.0
// npm run build && node test/kge-integration.js
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { createGraphReasoner, loadGraphReasoner } = require(
  process.env.KGE_INTEGRATION_ENTRY || '../dist/integrations/kge');
const rawEntry = require.resolve('@ruvector/kge');
assert.equal(require.cache[rawEntry], undefined, 'importing the adapter must not load its optional peer');
const raw = require('@ruvector/kge');
const facts = [
  { s: 'Ada', r: 'bornIn', o: 'London', split: 'train' },
  { s: 'London', r: 'locatedIn', o: 'England', split: 'train' },
  { s: 'Grace', r: 'bornIn', o: 'NewYork', split: 'train' },
  { s: 'NewYork', r: 'locatedIn', o: 'USA', split: 'train' },
];
const query = { s: 'Ada', r: 'bornIn', k: 3 };
const composition = { s: 'Ada', r1: 'bornIn', r2: 'locatedIn', k: 3 };
const isClosed = error => error.code === 'ERR_INTEGRATION_UNTRAINED';
const isInvalid = error => error.code === 'ERR_INTEGRATION_INVALID';
const hash = value => createHash('sha256').update(value).digest('hex');

for (const scorer of ['hole', 'rotate']) {
  test(`${scorer}: real training, exact save/load rankings, invalidation`, async () => {
    const graph = await createGraphReasoner({ scorer, dims: 8, seed: 5, trained: true });
    assert.ok(['native', 'wasm'].includes(graph.backend));
    assert.equal(graph.trained, false);
    graph.addTriples(facts);
    assert.throws(() => graph.predict(query), isClosed);
    assert.throws(() => graph.similarRelations({ r: 'bornIn' }), isClosed);
    assert.throws(() => graph.compose(composition), isClosed);
    assert.throws(() => graph.save(), isClosed);
    const pending = graph.train({ epochs: 3, lr: 0.1 });
    assert.throws(() => graph.predict(query), isClosed);
    assert.throws(() => graph.addTriples(facts), isInvalid);
    await assert.rejects(graph.train({ epochs: 1 }), isInvalid);
    const report = await pending;
    assert.equal(report.epochs, 3);
    assert.ok(Number.isFinite(report.loss));
    report.epochs = 0; // cannot alter the internal training receipt
    assert.equal(graph.trained, true);
    const saved = graph.save();
    const loaded = await loadGraphReasoner(saved);
    const upstream = raw.loadKge(JSON.parse(saved).model);
    assert.deepEqual(loaded.predict(query), graph.predict(query));
    assert.deepEqual(loaded.predict(query), upstream.predict(query));
    assert.deepEqual(loaded.similarRelations({ r: 'bornIn' }), upstream.similarRelations({ r: 'bornIn' }));
    if (scorer === 'rotate') assert.deepEqual(loaded.compose(composition), upstream.compose(composition));
    else assert.throws(() => loaded.compose(composition), error => error.kind === 'unsupported');
    assert.equal(loaded.save(), saved);
    assert.equal(loaded.stats().trained, true);
    loaded.addTriples([{ s: 'Alan', r: 'bornIn', o: 'London' }]);
    assert.throws(() => loaded.predict(query), isClosed);
    assert.throws(() => loaded.save(), isClosed);
    await graph.train({ epochs: 0 }).then(() => assert.fail('invalid epochs accepted'), () => {});
    assert.equal(graph.trained, false, 'failed retraining fails closed');
  });
}

test('raw upstream saves cannot claim training, even with initialized tables or trained:true', async () => {
  const model = raw.createKge({ dims: 8, seed: 1 });
  model.addTriples(facts);
  model.predict(query); // creates random tables; that does not mean trained
  await assert.rejects(loadGraphReasoner(model.save(), { trained: true }), isInvalid);
  const wrapped = { format: 'ruvector-graph-reasoner-v1', model: model.save(), trained: true };
  await assert.rejects(loadGraphReasoner(JSON.stringify(wrapped)), isInvalid);
  await model.train({ epochs: 1 });
  await assert.rejects(loadGraphReasoner(model.save()), isInvalid); // no persisted report upstream
});

test('model/report mismatch, malformed embeddings, signatures, and keys fail closed', async () => {
  const graph = await createGraphReasoner({ scorer: 'rotate', dims: 8, seed: 6 });
  graph.addTriples(facts);
  await graph.train({ epochs: 2 });
  const saved = graph.save();
  const mismatched = JSON.parse(saved);
  mismatched.training.report.entities += 1;
  await assert.rejects(loadGraphReasoner(JSON.stringify(mismatched)), isInvalid);
  const edited = JSON.parse(saved);
  edited.model += ' ';
  await assert.rejects(loadGraphReasoner(JSON.stringify(edited)), isInvalid);
  const malformed = JSON.parse(saved);
  const model = JSON.parse(malformed.model);
  model.model.tables.entities.pop();
  malformed.model = JSON.stringify(model);
  malformed.training.modelSha256 = hash(malformed.model);
  await assert.rejects(loadGraphReasoner(JSON.stringify(malformed)), isInvalid);
  await assert.rejects(loadGraphReasoner('{'), isInvalid);
  const key = 'graph-integration-secret-for-test';
  const signed = graph.save({ key });
  const loaded = await loadGraphReasoner(signed, { key });
  assert.deepEqual(loaded.predict(query), graph.predict(query));
  await assert.rejects(loadGraphReasoner(signed), isInvalid);
  await assert.rejects(loadGraphReasoner(signed, {}), isInvalid);
  await assert.rejects(loadGraphReasoner(saved, { key }), isInvalid);
  await assert.rejects(loadGraphReasoner(signed, { key: 'different-secret-for-this-test' }), isInvalid);
  const tampered = JSON.parse(signed);
  tampered.training.report.loss += 1;
  await assert.rejects(loadGraphReasoner(JSON.stringify(tampered), { key }), isInvalid);
  await assert.rejects(loadGraphReasoner(JSON.stringify(tampered)), isInvalid);
  tampered.hmacSha256 = null;
  await assert.rejects(loadGraphReasoner(JSON.stringify(tampered)), isInvalid);
  assert.throws(() => graph.save({ key: undefined }), isInvalid);
  await assert.rejects(loadGraphReasoner(signed, { key: undefined }), isInvalid);
});
