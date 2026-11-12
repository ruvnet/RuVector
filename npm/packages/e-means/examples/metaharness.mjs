import { tuneClustering } from '@ruvector/e-means/metaharness';

const result = await tuneClustering({
  training: new Float32Array([0, 0, 0, 2, 10, 10, 10, 12]),
  holdout: new Float32Array([0, 1, 10, 11]),
  dimensions: 2,
  baseline: { clusters: 2, epochs: 2, batchSize: 2, peakStep: 0.5, seed: 42 },
  domain: { peakStep: { min: 0.1, max: 1, type: 'float', scale: 'log' } },
  budget: { maxEvaluations: 3, maxDistanceCoordinates: 100_000, timeoutMs: 30_000 },
  generations: 1,
  childrenPerGeneration: 2,
  mutationSeed: 7,
});
console.log(JSON.stringify({ baselineHoldoutMse: result.baseline.score.raw.holdoutMse,
  winnerHoldoutMse: result.winner.score.raw.holdoutMse, recommendation: result.recommendation,
  sourceApplied: result.sourceApplied }, null, 2));
