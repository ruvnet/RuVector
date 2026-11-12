import assert from 'node:assert/strict';
import { mkdtemp, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createClusteringEvaluator, tuneClustering } from '../integrations/metaharness.mjs';
import { evolveNumeric } from '@metaharness/darwin';

function options(extra = {}) {
  return {
    training: new Float32Array([0, 0, 0, 2, 10, 10, 10, 12]),
    holdout: new Float32Array([0, 1, 10, 11]), dimensions: 2,
    baseline: { clusters: 2, epochs: 2, batchSize: 2, peakStep: 0.5, seed: 42 },
    domain: { clusters: { min: 1, max: 2, type: 'int', scale: 'linear' } },
    budget: { maxEvaluations: 3, maxDistanceCoordinates: 100_000, timeoutMs: 30_000 },
    ...extra,
  };
}

test('real WASM measures a separate holdout and flags quality regressions', async () => {
  const evaluator = createClusteringEvaluator(options());
  const baseline = await evaluator.evaluate(evaluator.baselineGenome, 'baseline');
  assert.equal(baseline.raw.trainMse, 1);
  assert.equal(baseline.raw.holdoutMse, 0);
  assert.equal(baseline.primary, 1);
  assert.equal(baseline.regressed, false);
  const candidate = await evaluator.evaluate({ clusters: 1 }, 'candidate');
  assert.equal(candidate.raw.holdoutMse, 50);
  assert.equal(candidate.raw.baselineHoldoutMse, 0);
  assert.equal(candidate.primary, 1 / 51);
  assert.equal(candidate.regressed, true);
  assert.ok(Number.isFinite(candidate.raw.fitMs) && candidate.raw.fitMs >= 0);
  assert.ok(Number.isFinite(candidate.raw.holdoutMs) && candidate.raw.holdoutMs >= 0);
  assert.equal(evaluator.receipt().sourceApplied, false);
});

test('inputs/configuration snapshot before awaits and receipts cannot mutate evidence', async () => {
  const supplied = options();
  supplied.training.slice = () => supplied.training;
  supplied.holdout.slice = () => supplied.holdout;
  const evaluator = createClusteringEvaluator(supplied);
  const pending = evaluator.evaluate(evaluator.baselineGenome, 'baseline');
  supplied.training.fill(999); supplied.holdout.fill(999);
  supplied.baseline.clusters = 1; supplied.domain.clusters.min = 2;
  supplied.budget.maxEvaluations = 0;
  const score = await pending;
  assert.equal(score.raw.holdoutMse, 0);
  score.raw.holdoutMse = 999;
  const receipt = evaluator.receipt();
  receipt.trials[0].score.raw.holdoutMse = 999;
  receipt.domain.clusters.min = 2;
  receipt.baselineConfig.clusters = 99;
  assert.equal(evaluator.receipt().baseline.score.raw.holdoutMse, 0);
  assert.equal(evaluator.receipt().domain.clusters.min, 1);
  assert.equal(evaluator.receipt().baselineConfig.clusters, 2);
  assert.equal((await evaluator.evaluate({ clusters: 1 }, 'candidate')).raw.holdoutMse, 50);
});

test('admission rejects invalid splits, unbounded domains and oversized work before training', () => {
  const badCases = [
    options({ holdout: new Float32Array() }),
    options({ holdout: new Float32Array([NaN, 1]) }),
    options({ dimensions: 0 }),
    options({ domain: {} }),
    options({ domain: { clusters: { min: 1, max: Infinity, type: 'int', scale: 'linear' } } }),
    options({ domain: { clusters: { min: 1, max: 5, type: 'int', scale: 'linear' } } }),
    options({ domain: { batchSize: { min: 1, max: 4, type: 'float', scale: 'linear' } } }),
    options({ domain: { peakStep: { min: 0.1, max: 1, type: 'float', scale: 'log', default: 0.1 } } }),
    options({ baseline: { clusters: 2, epochs: 2, batchSize: 0 } }),
    options({ baseline: { clusters: 2, epochs: null } }),
    options({ baseline: { clusters: 2, seed: null } }),
    options({ budget: { maxEvaluations: 0, maxDistanceCoordinates: 10_000 } }),
    options({ budget: { maxEvaluations: 1, maxDistanceCoordinates: 1 } }),
    options({ budget: { maxEvaluations: 1, maxDistanceCoordinates: 1e10 } }),
    options({ baseline: { clusters: 2, epochs: 0xffff_ffff, batchSize: 1 } }),
    options({ qualityTolerance: -1 }),
    options({ qualityTolerance: null }),
  ];
  for (const value of badCases) assert.throws(() => createClusteringEvaluator(value));
  const overlapping = options(); overlapping.holdout = overlapping.training.subarray(0, 4);
  assert.throws(() => createClusteringEvaluator(overlapping), /nonoverlapping/);
  assert.throws(() => createClusteringEvaluator(options({ training: new Float32Array(16_777_217) })), /64 MiB/);
});

test('evaluation dispatch, domain, concurrency and cooperative deadline limits are enforced', async () => {
  const evaluator = createClusteringEvaluator(options({ budget: { maxEvaluations: 1, maxDistanceCoordinates: 100_000 } }));
  const pending = evaluator.evaluate(evaluator.baselineGenome, 'baseline');
  await assert.rejects(evaluator.evaluate(evaluator.baselineGenome, 'parallel'), /Concurrent/);
  await pending;
  await assert.rejects(evaluator.evaluate({ clusters: 1 }, 'second'), /budget exhausted/);
  const firstCandidate = createClusteringEvaluator(options());
  await assert.rejects(firstCandidate.evaluate({ clusters: 1 }, 'wrong-first'), /baselineGenome/);
  await assert.rejects(firstCandidate.evaluate({ clusters: 99 }, 'outside'), /clusters/);
  const expired = createClusteringEvaluator(options({ budget: { maxEvaluations: 1, maxDistanceCoordinates: 100_000, timeoutMs: 1 } }));
  await new Promise(resolve => setTimeout(resolve, 10));
  await assert.rejects(expired.evaluate(expired.baselineGenome, 'expired'), /deadline/);
});

test('published Darwin evolveNumeric accepts the real clustering evaluator', async () => {
  const evaluator = createClusteringEvaluator(options());
  const archive = await mkdtemp(join(tmpdir(), 'e-means-sdk-test-'));
  try {
    const result = await evolveNumeric({ genomeSpec: evaluator.domain, baselineGenome: evaluator.baselineGenome,
      evaluator, generations: 1, childrenPerGeneration: 2, seed: 7, concurrency: 1, workRoot: archive });
    assert.equal(result.records.length, 3);
    assert.equal(result.baseline.score.raw.holdoutMse, 0);
    assert.equal(evaluator.receipt().dispatchCount, 3);
    assert.ok(result.records.every(record => Number.isFinite(record.score.primary)));
  } finally { await rm(archive, { recursive: true, force: true }); }
});

test('bounded tuneClustering uses Darwin0.13.0, validates evidence and removes its archive', async () => {
  const before = new Set((await readdir(tmpdir())).filter(name => name.startsWith('e-means-darwin-')));
  const supplied = options({ generations: 1, childrenPerGeneration: 2, mutationSeed: 7 });
  const result = await tuneClustering(supplied);
  assert.deepEqual(result.engine, { package: '@metaharness/darwin', version: '0.13.0', algorithm: 'evolveNumeric' });
  assert.equal(result.schema, 'e-means.clustering-tuning/1');
  assert.equal(result.trials.length, 3);
  assert.equal(result.dispatchCount, 3);
  assert.equal(result.baseline.score.raw.holdoutMse, 0);
  assert.equal(result.sourceApplied, false);
  assert.equal(result.temporaryArchiveRemoved, true);
  assert.ok(result.consumedWork <= result.budget.maxDistanceCoordinates);
  assert.ok(result.winner.score.primary >= result.baseline.score.primary);
  assert.deepEqual(supplied.training, options().training);
  assert.equal((await readdir(tmpdir())).filter(name => name.startsWith('e-means-darwin-') && !before.has(name)).length, 0);
  await assert.rejects(tuneClustering(options({ generations: 8, childrenPerGeneration: 8 })), /worst-case dispatches/);
});
