import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  LIMITS, WeightRegistry, createMlp, learnFromOutcomes, loadFann, meanSquaredError, trainWithinBudget, trainingWork,
} from '../dist/esm/index.js';
import { QueryReshaper } from '../dist/esm/reshape.js';

function rand(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

test('loadFann: real ruv-fann loads; a missing package gives ENOFANN', async () => {
  const rt = await loadFann();
  assert.match(rt.backend, /^wasm/);
  await assert.rejects(loadFann(async () => { throw new Error('Cannot find package'); }), (e) => e.code === 'ENOFANN');
  await assert.rejects(loadFann(async () => ({ LIMITS: { ...LIMITS, maxRows: 1 }, loadRuntime: async () => rt })), (e) => e.code === 'ELIMITS');
});

test('trainWithinBudget splits work that exceeds one ruv-fann call', async () => {
  const rt = await loadFann();
  const model = createMlp([64, 32, 64], { seed: 2 });
  const rows = 1000;
  assert.ok(trainingWork(model, rows, 1) > LIMITS.maxTrainingWork, 'premise: one full-batch epoch is over budget');
  const r = rand(1);
  const x = Float64Array.from({ length: rows * 64 }, () => r() - 0.5);
  const y = x.map((v) => v * 0.5);
  assert.throws(() => rt.train(model, { inputs: x, targets: y, rows, epochs: 1, learningRate: 0.01 }), /budget/);
  const res = trainWithinBudget(rt, model, x, y, rows, { epochs: 3, learningRate: 0.05 });
  assert.ok(res.calls > 3, `mini-batched into ${res.calls} calls`);
  assert.ok(res.batchRows < rows);
  assert.ok(meanSquaredError(rt, res.model, x, y, rows) < meanSquaredError(rt, model, x, y, rows));
});

test('learnFromOutcomes accepts an improving child and links it to the parent', async () => {
  const reg = await WeightRegistry.open({ dimensions: 2, store: 'flat' });
  const parent = await reg.save(createMlp([2, 8, 1], { seed: 3 }), [1, 0], { task: 'product' });
  const r = rand(5);
  const trajectories = [];
  for (let i = 0; i < 60; i++) {
    const x = [r() * 2 - 1, r() * 2 - 1];
    trajectories.push({ input: x, output: [x[0] * x[1]], score: 1 });
  }
  trajectories.push({ input: [0.5, 0.5], output: [9], score: 0 }); // failure without correction: skipped
  trajectories.push({ input: [0.5, -0.5], output: [9], score: 0, correction: [-0.25] });
  const res = await reg.learnFromOutcomes(parent.id, trajectories, { epochs: 300, learningRate: 0.1 });
  assert.equal(res.accepted, true, res.reason);
  assert.equal(res.rows.skipped, 1);
  assert.equal(res.rows.corrections, 1);
  assert.ok(res.holdoutMse.after < res.holdoutMse.before);
  assert.equal(res.child.parent, parent.id);
  assert.equal(res.child.op, 'finetune');
  assert.equal(res.child.version, 2);
  assert.equal(reg.size, 2);
});

test('learnFromOutcomes rejects when the holdout does not improve', async () => {
  const reg = await WeightRegistry.open({ dimensions: 2, store: 'flat' });
  const parent = await reg.save(createMlp([2, 4, 1], { seed: 3 }), [1, 0]);
  const r = rand(9);
  const traj = Array.from({ length: 40 }, () => ({ input: [r(), r()], output: [r() * 2 - 1], score: 1 }));
  const res = await learnFromOutcomes(reg, parent.id, traj, { epochs: 1, learningRate: 1e-6 });
  assert.equal(res.accepted, false);
  assert.match(res.reason, /did not improve/);
  assert.equal(reg.size, 1, 'no child saved');
  const few = await learnFromOutcomes(reg, parent.id, [{ input: [1, 1], output: [1], score: 1 }]);
  assert.equal(few.accepted, false);
  await assert.rejects(learnFromOutcomes(reg, parent.id, [{ input: [1], output: [1], score: 1 }, { input: [1], output: [1], score: 1 }]), (e) => e.code === 'ESHAPE');
});

test('QueryReshaper starts as identity, trains toward relevant items, and is allocation-free on the hot path', async () => {
  const d = 8;
  const rs = await QueryReshaper.create(d, { hidden: 4 });
  const q = [1, 2, 0, 0, 0, 0, 0, 0];
  const id = rs.reshape(q);
  assert.ok(Math.abs(id[0] - 1 / Math.sqrt(5)) < 1e-12 && Math.abs(id[1] - 2 / Math.sqrt(5)) < 1e-12);
  const target = [0, 0, 1, 0, 0, 0, 0, 0];
  const pairs = Array.from({ length: 16 }, (_, i) => ({ query: q.map((v, j) => v + (j === 3 ? i * 0.01 : 0)), relevant: target, irrelevant: [[0, 0, 0, 0, 1, 0, 0, 0]] }));
  const fit = rs.train(pairs, { epochs: 200, learningRate: 0.1 });
  assert.ok(fit.finalLoss < fit.initialLoss);
  const out = new Float64Array(d);
  rs.reshapeInto(q, out);
  assert.ok(out[2] > id[2] + 0.2, 'moved toward the relevant direction');
  assert.throws(() => rs.reshapeInto(q, new Float64Array(3)), /length/);
  const again = await QueryReshaper.fromModel(rs.model);
  assert.deepEqual(Array.from(again.reshape(q)), Array.from(rs.reshape(q)));
  rs.dispose();
});
