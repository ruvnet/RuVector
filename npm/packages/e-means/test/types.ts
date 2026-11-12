import { init, fit, predict, type EMeansModel, type EMeansOptions } from '@ruvector/e-means';
import { init as initRuntime } from '@ruvector/e-means/runtime';
import { createClusteringEvaluator, tuneClustering, type ClusteringEvaluationOptions } from '@ruvector/e-means/metaharness';

const options: EMeansOptions = { clusters: 2, epochs: 3, finalization: 'lastEpochMean' };
await init();
const model: EMeansModel = fit(new Float32Array([0, 0, 1, 1]), 2, options);
const assignments: Uint32Array = predict(new Float32Array([2, 2]), 2, model.centroids);
void assignments;
// @ts-expect-error missing required cluster count
fit(new Float32Array(), 1, { epochs: 3 });
// @ts-expect-error row storage must be Float32Array
fit([0, 1], 1, options);
// @ts-expect-error unsupported finalization policy
fit(new Float32Array(), 1, { clusters: 1, finalization: 'slowEma' });

await initRuntime(new WebAssembly.Module(new Uint8Array()));
// @ts-expect-error Node-free runtime requires an explicit source
await initRuntime();
const evaluation: ClusteringEvaluationOptions = {
  training: new Float32Array([0, 1]), holdout: new Float32Array([2]), dimensions: 1,
  baseline: { clusters: 1, epochs: 1 },
  domain: { epochs: { min: 1, max: 2, type: 'int', scale: 'linear' } },
  budget: { maxEvaluations: 2, maxDistanceCoordinates: 1000 },
};
const evaluator = createClusteringEvaluator(evaluation);
await evaluator.evaluate(evaluator.baselineGenome, 'baseline');
const tuned = await tuneClustering({ ...evaluation, generations: 1, childrenPerGeneration: 1 });
const neverApplied: false = tuned.sourceApplied;
void neverApplied;
// @ts-expect-error tuning requires a finite explicit evaluation budget
createClusteringEvaluator({ training: new Float32Array(), holdout: new Float32Array(), dimensions: 1, baseline: options, domain: {} });
