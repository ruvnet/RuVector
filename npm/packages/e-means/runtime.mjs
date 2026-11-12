import initWasm, { fitTyped, predictFlat } from './wasm/ruvector_e_means.js';

let initialized = false;
let initialization;
const maxU32 = 0xffff_ffff;

/** Initialize from caller-supplied bytes or a precompiled module; no Node loader is reachable. */
export async function init(source) {
  if (initialized) return;
  if (!initialization) {
    initialization = (async () => {
      if (source === undefined) throw new TypeError('runtime init requires WASM bytes or a precompiled module');
      await initWasm({ module_or_path: source });
      initialized = true;
    })().catch((error) => {
      initialization = undefined;
      const message = typeof error?.message === 'string' ? error.message : String(error);
      throw new Error(`Could not initialize @ruvector/e-means: ${message}`, { cause: error });
    });
  }
  await initialization;
}

function ready() {
  if (!initialized) throw new Error('Call and await init() before using @ruvector/e-means');
}

function positiveInteger(value, name) {
  if (!Number.isInteger(value) || value <= 0 || value > maxU32) {
    throw new RangeError(`${name} must be an integer between 1 and ${maxU32}`);
  }
}

function vectors(data, dimensions, name, allowEmpty = false) {
  if (!(data instanceof Float32Array)) throw new TypeError(`${name} must be a Float32Array`);
  positiveInteger(dimensions, 'dimensions');
  if ((!allowEmpty && data.length === 0) || data.length % dimensions !== 0) {
    throw new RangeError(`${name} must contain complete ${dimensions}-dimensional vectors`);
  }
  for (let index = 0; index < data.length; index++) {
    if (!Number.isFinite(data[index])) throw new RangeError(`${name} must contain only finite values`);
  }
}

/** Cluster flat row-major f32 vectors using the e-means Rust implementation. */
export function fit(data, dimensions, options) {
  ready();
  vectors(data, dimensions, 'data');
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('options must be an object with clusters');
  }
  const allowed = ['clusters', 'epochs', 'batchSize', 'peakStep', 'seed', 'finalization'];
  for (const key of Object.keys(options)) {
    if (!allowed.includes(key)) throw new TypeError(`Unknown option: ${key}`);
  }
  positiveInteger(options.clusters, 'clusters');
  if (options.clusters > data.length / dimensions) throw new RangeError('clusters cannot exceed the number of vectors');
  if (options.epochs !== undefined) positiveInteger(options.epochs, 'epochs');
  if (options.batchSize !== undefined) positiveInteger(options.batchSize, 'batchSize');
  if (options.peakStep !== undefined &&
      (typeof options.peakStep !== 'number' || !Number.isFinite(options.peakStep) || options.peakStep <= 0 || options.peakStep > 1)) {
    throw new RangeError('peakStep must be a finite number in (0, 1]');
  }
  if (options.seed !== undefined && (!Number.isSafeInteger(options.seed) || options.seed < 0)) {
    throw new RangeError('seed must be a non-negative safe integer');
  }
  if (options.finalization !== undefined && !['lastEpochMean', 'ema'].includes(options.finalization)) {
    throw new RangeError('finalization must be lastEpochMean or ema');
  }
  // Copy supported options explicitly: inherited properties and toJSON do not become Rust configuration.
  const config = { clusters: options.clusters };
  for (const key of allowed.slice(1)) if (options[key] !== undefined) config[key] = options[key];
  const model = fitTyped(data, dimensions, JSON.stringify(config));
  try {
    return {
      centroids: model.takeCentroids(),
      assignments: model.takeAssignments(),
      dimensions: model.dimensions,
      mse: model.mse,
      batchSize: model.batchSize,
      peakStep: model.peakStep,
      updates: model.updates,
      revivedClusters: model.revivedClusters,
    };
  } finally {
    model.free();
  }
}

/** Assign vectors to their nearest centroid using squared Euclidean distance. */
export function predict(data, dimensions, centroids) {
  ready();
  vectors(data, dimensions, 'data', true);
  vectors(centroids, dimensions, 'centroids');
  return predictFlat(data, dimensions, centroids);
}
