import assert from 'node:assert/strict';
import { cpus } from 'node:os';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { init, fit, predict } from '../e-means.mjs';
import { fitJson, fitTyped } from '../wasm/ruvector_e_means.js';

let baselinePath, repeats = 7;
for (let i = 2; i < process.argv.length; i++) {
  const argument = process.argv[i];
  if (argument === '--baseline') baselinePath = process.argv[++i];
  else if (argument === '--repeats') repeats = Number(process.argv[++i]);
  else throw new Error(`Unknown benchmark argument: ${argument}`);
}
assert.ok(Number.isInteger(repeats) && repeats >= 3 && repeats <= 31, '--repeats must be in 3..31');
await init();
let baseline;
if (baselinePath) {
  baseline = await import(pathToFileURL(resolve(baselinePath, 'e-means.mjs')).href);
  await baseline.init();
}

function materialize(handle) {
  try {
    return {
      centroids: handle.takeCentroids(), assignments: handle.takeAssignments(),
      dimensions: handle.dimensions, mse: handle.mse, batchSize: handle.batchSize,
      peakStep: handle.peakStep, updates: handle.updates, revivedClusters: handle.revivedClusters,
    };
  } finally {
    handle.free();
  }
}

function median(values) {
  return [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
}

// Interleave variants and rotate their order to reduce ordering/thermal bias.
// GC is outside the timed region; each result is fully materialized and consumed.
let checksum = 0;
function measure(variants) {
  const entries = Object.entries(variants), timings = new Map(entries.map(([name]) => [name, []]));
  for (const [, operation] of entries) for (let i = 0; i < 2; i++) operation();
  for (let round = 0; round < repeats; round++) {
    for (let i = 0; i < entries.length; i++) {
      const [name, operation] = entries[(i + round) % entries.length];
      globalThis.gc?.();
      const start = performance.now();
      const result = operation();
      timings.get(name).push(performance.now() - start);
      checksum += result instanceof Uint32Array ? result.length + (result[0] || 0) : result.mse + result.assignments.length;
    }
  }
  return Object.fromEntries([...timings].map(([name, samples]) => [name, median(samples)]));
}

function compareResults(a, b) {
  assert.equal(a.assignments.length, b.assignments.length);
  assert.equal(a.centroids.length, b.centroids.length);
  let labelDifferences = 0, centroidMaxAbsDifference = 0;
  for (let i = 0; i < a.assignments.length; i++) labelDifferences += Number(a.assignments[i] !== b.assignments[i]);
  for (let i = 0; i < a.centroids.length; i++) centroidMaxAbsDifference = Math.max(centroidMaxAbsDifference, Math.abs(a.centroids[i] - b.centroids[i]));
  return { labelDifferences, centroidMaxAbsDifference, mseAbsoluteDifference: Math.abs(a.mse - b.mse) };
}

const cases = [
  { name: 'large-assignment-result', samples: 1_000_000, dimensions: 1, clusters: 1, epochs: 1, batchSize: 4096 },
  { name: 'embedding-codebook', samples: 8192, dimensions: 64, clusters: 16, epochs: 3, batchSize: 256 },
];
const results = [];
for (const scenario of cases) {
  const { samples, dimensions, clusters, epochs, batchSize } = scenario;
  const data = new Float32Array(samples * dimensions);
  for (let row = 0; row < samples; row++) {
    for (let column = 0; column < dimensions; column++) {
      data[row * dimensions + column] = (row % clusters) * 4 + (column % 7) * 0.1 + ((row * 17 + column * 31) % 97) / 9700;
    }
  }
  const options = { clusters, epochs, batchSize, seed: 42 }, config = JSON.stringify(options);
  const optimizedModel = fit(data, dimensions, options);
  const jsonResult = JSON.parse(fitJson(data, dimensions, config));
  const jsonModel = { ...jsonResult, centroids: Float32Array.from(jsonResult.centroids), assignments: Uint32Array.from(jsonResult.assignments) };
  assert.deepEqual(optimizedModel, jsonModel, 'same optimized kernel must have identical JSON/typed results');
  const baselineModel = baseline?.fit(data, dimensions, options);
  if (baselineModel) assert.deepEqual(optimizedModel, baselineModel, 'optimized fit must preserve baseline outputs on the measured fixture');
  const fitVariants = { optimizedMs: () => fit(data, dimensions, options) };
  if (baseline) fitVariants.baselineMs = () => baseline.fit(data, dimensions, options);
  const fitTimes = measure(fitVariants);
  if (baseline) fitTimes.speedup = fitTimes.baselineMs / fitTimes.optimizedMs;

  const transport = measure({
    jsonMs: () => {
      const model = JSON.parse(fitJson(data, dimensions, config));
      return { ...model, centroids: Float32Array.from(model.centroids), assignments: Uint32Array.from(model.assignments) };
    },
    typedMs: () => materialize(fitTyped(data, dimensions, config)),
  });
  transport.speedup = transport.jsonMs / transport.typedMs;
  const predictVariants = { optimizedMs: () => predict(data, dimensions, optimizedModel.centroids) };
  if (baseline) predictVariants.baselineMs = () => baseline.predict(data, dimensions, baselineModel.centroids);
  const predictTimes = measure(predictVariants);
  if (baseline) predictTimes.speedup = predictTimes.baselineMs / predictTimes.optimizedMs;
  results.push({ ...scenario, fit: fitTimes, predict: predictTimes, sameKernelTransport: transport,
    baselineParity: baselineModel ? compareResults(optimizedModel, baselineModel) : undefined });
}
console.log(JSON.stringify({ runtime: { node: process.version, platform: process.platform, arch: process.arch,
  cpu: cpus()[0]?.model, logicalCpus: cpus().length }, repeats, warmupsPerVariant: 2,
  measurement: 'median wall-clock milliseconds, fully materialized results; explicit GC outside timing',
  sameKernelTransportMeasurement: 'full fit plus output materialization using one optimized WASM kernel; difference includes JSON serialization/parsing and array conversion, not a conversion-only timer',
  baselinePath: baselinePath ? resolve(baselinePath) : undefined, checksum, cases: results }, null, 2));
