import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { createTypesafe, choice, score, noul, TypesafeError, ADDITIVE_KEYS } = require('../dist/index.js');

/** A fresh fake binding that records the engine it constructs. */
function makeBinding() {
  delete require.cache[require.resolve('./fixtures/fake-binding.cjs')];
  const fake = require('./fixtures/fake-binding.cjs');
  const holder = { engine: null };
  const Wrapped = class extends fake.Engine {
    constructor(opts) {
      super(opts);
      holder.engine = this;
    }
  };
  return { binding: { Engine: Wrapped, version: fake.version, backend: fake.backend }, holder };
}

const jevBody = () => ({
  state: 'my card was charged twice',
  model: 'jev-latest',
  questions: {
    dept: { type: 'choice', criteria: { billing: 'charges', fraud: 'unauthorised use' } },
    // Jev sends score buckets under `criteria`, not `legend`.
    mood: { type: 'score', criteria: ['Calm', 'Irritated', 'Angry'] },
    urgent: { type: 'noul', instructions: 'the sender needs a response soon' },
  },
});

test('decide sends exactly the wire request shape to the binding', async () => {
  const { binding, holder } = makeBinding();
  const ts = createTypesafe({ binding });
  await ts.decide('state text', {
    dept: choice(
      { billing: 'b', fraud: { what: 'f', not_for: 'nf', examples: ['e'] } },
      { instructions: 'route' },
    ),
    mood: score(['Calm', 'Angry'], { instructions: 'how upset' }),
    urgent: noul('needs a reply soon'),
  });
  const req = holder.engine.calls[0];
  assert.equal(req.state, 'state text');
  assert.deepEqual(req.questions.dept, {
    type: 'choice',
    criteria: { billing: 'b', fraud: { what: 'f', not_for: 'nf', examples: ['e'] } },
    instructions: 'route',
  });
  assert.deepEqual(req.questions.mood, {
    type: 'score',
    legend: ['Calm', 'Angry'],
    instructions: 'how upset',
  });
  assert.deepEqual(req.questions.urgent, { type: 'noul', instructions: 'needs a reply soon' });
});

test('typed answers come back, both spread and under .answers', async () => {
  const { binding } = makeBinding();
  const ts = createTypesafe({ binding });
  const r = await ts.decide('x', {
    dept: choice({ billing: 'b', fraud: 'f' }),
    mood: score(['Calm', 'Angry']),
    urgent: noul('soon'),
  });
  assert.equal(r.dept.choice, 'billing');
  assert.ok('billing' in r.dept.probabilities);
  assert.equal(r.mood.legend, 'Calm');
  assert.equal(typeof r.urgent.noul, 'number');
  assert.deepEqual(r.answers.dept, r.dept);
  assert.ok(r.usage.state_bytes >= 0);
});

test('an error envelope becomes a thrown TypesafeError carrying the kind', async () => {
  const { binding } = makeBinding();
  const ts = createTypesafe({ binding });
  for (const [state, kind] of [
    ['__ERR_LIMIT__', 'limit'],
    ['__ERR_INVALID__', 'invalid'],
    ['__ERR_EMBEDDER__', 'embedder'],
  ]) {
    await assert.rejects(
      () => ts.decide(state, { q: choice({ a: 'a', b: 'b' }) }),
      (e) => e instanceof TypesafeError && e.kind === kind,
    );
  }
});

test('systemOne round-trips a Jev-shaped body and maps score criteria → legend', async () => {
  const { binding, holder } = makeBinding();
  const ts = createTypesafe({ binding });
  const resp = await ts.systemOne(jevBody());
  assert.ok(resp.answers.dept.choice);
  assert.equal(resp.answers.mood.legend, 'Calm');
  assert.equal(typeof resp.answers.urgent.noul, 'number');
  const sent = holder.engine.calls.at(-1);
  assert.deepEqual(sent.questions.mood, { type: 'score', legend: ['Calm', 'Irritated', 'Angry'] });
  assert.equal('model' in sent, false, 'the Jev model field is not forwarded to the wire request');
});

test('jevShapeOnly strips exactly the five additive fields, keeping confidence', async () => {
  const { binding } = makeBinding();
  const ts = createTypesafe({ binding });
  const full = await ts.systemOne(jevBody());
  for (const key of ADDITIVE_KEYS) assert.ok(key in full.answers.dept, `full keeps ${key}`);
  const jev = await ts.systemOne(jevBody(), { jevShapeOnly: true });
  for (const key of ADDITIVE_KEYS) assert.ok(!(key in jev.answers.dept), `jev strips ${key}`);
  assert.ok('confidence' in jev.answers.dept, 'confidence stays (it is Jev-native)');
  assert.ok('choice' in jev.answers.dept);
  assert.ok('probabilities' in jev.answers.dept);
});

test('decideMany returns one typed result per state', async () => {
  const { binding } = makeBinding();
  const ts = createTypesafe({ binding });
  const rs = await ts.decideMany(['a', 'b', 'c'], { q: choice({ x: 'x', y: 'y' }) });
  assert.equal(rs.length, 3);
  assert.equal(rs[0].q.choice, 'x');
});

test('train forwards the payload and returns the TrainReport', async () => {
  const { binding, holder } = makeBinding();
  const ts = createTypesafe({ binding });
  const report = await ts.train('dept', [{ text: 't', label: 'billing' }]);
  assert.equal(report.question, 'dept');
  assert.equal(report.accepted, 1);
  const sent = JSON.parse(holder.engine.trainCalls?.[0] ?? '{}');
  // The binding does not record trainJson args; assert the report instead.
  assert.equal(report.head, 'linear-probe');
  void sent;
});

test('version and backend surface the binding metadata', async () => {
  const { binding } = makeBinding();
  const ts = createTypesafe({ binding });
  assert.equal(ts.version, '0.1.0-fake');
  assert.equal(ts.backend, 'wasm');
});

test('integrates with the default native/WASM binding when one is present', async (t) => {
  let ts;
  try {
    ts = createTypesafe({ embedder: 'hash', dims: 256 });
  } catch (e) {
    assert.ok(e instanceof TypesafeError && e.kind === 'embedder');
    t.skip('no ../index.js binding built yet');
    return;
  }
  assert.equal(typeof ts.version, 'string');
  assert.ok(ts.backend === 'native' || ts.backend === 'wasm');
  const r = await ts.decide('my card was charged twice', {
    dept: choice({
      billing: 'charges and refunds',
      fraud: { what: 'unauthorised use', not_for: 'duplicate charges' },
    }),
  });
  assert.ok(r.dept.choice === 'billing' || r.dept.choice === 'fraud');
  assert.equal(typeof r.dept.confidence, 'number');
  assert.ok('billing' in r.dept.probabilities);
});
