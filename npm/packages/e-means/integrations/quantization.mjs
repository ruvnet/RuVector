import { fit, predict } from '../e-means.mjs';

const schema = 'ruvector-e-means-pq-v1';
const maxU32 = 0xffff_ffff;

function positiveInteger(value, name, maximum = maxU32) {
  if (!Number.isInteger(value) || value < 1 || value > maximum) {
    throw new RangeError(`${name} must be an integer in 1..${maximum}`);
  }
}

function shape(dimensions, subspaces, clusters) {
  positiveInteger(dimensions, 'dimensions');
  positiveInteger(subspaces, 'subspaces', dimensions);
  positiveInteger(clusters, 'clusters', 256);
  if (dimensions % subspaces !== 0) throw new RangeError('dimensions must be divisible by subspaces');
  return dimensions / subspaces;
}

function vectors(data, dimensions, allowEmpty = false) {
  if (!(data instanceof Float32Array)) throw new TypeError('data must be a Float32Array');
  if ((!allowEmpty && data.length === 0) || data.length % dimensions !== 0) {
    throw new RangeError('data must contain complete vectors');
  }
  for (const value of data) if (!Number.isFinite(value)) throw new RangeError('data must be finite');
}

function validated(model) {
  if (model === null || typeof model !== 'object' || model.schema !== schema) {
    throw new TypeError(`model must use schema ${schema}`);
  }
  const { dimensions, subspaces, clusters, centroids } = model;
  const subDimensions = shape(dimensions, subspaces, clusters);
  if (!(centroids instanceof Float32Array)) throw new TypeError('centroids must be a Float32Array');
  if (centroids.length !== dimensions * clusters) throw new RangeError('centroid count does not match the codebook shape');
  for (const value of centroids) if (!Number.isFinite(value)) throw new RangeError('centroids must be finite');
  return { dimensions, subspaces, clusters, centroids, subDimensions };
}

function gather(data, dimensions, subDimensions, subspace, target) {
  for (let row = 0; row < data.length / dimensions; row++) {
    const start = row * dimensions + subspace * subDimensions;
    target.set(data.subarray(start, start + subDimensions), row * subDimensions);
  }
}

function snapshotOptions(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('options must contain clusters');
  }
  // Read caller properties once so all subspaces use one stable configuration.
  const config = { ...options };
  const supported = ['clusters', 'epochs', 'batchSize', 'peakStep', 'seed', 'finalization'];
  for (const key of Object.keys(config)) if (!supported.includes(key)) throw new TypeError(`Unknown option: ${key}`);
  if (config.epochs !== undefined) positiveInteger(config.epochs, 'epochs');
  if (config.batchSize !== undefined) positiveInteger(config.batchSize, 'batchSize');
  if (config.peakStep !== undefined &&
      (typeof config.peakStep !== 'number' || !Number.isFinite(config.peakStep) || config.peakStep <= 0 || config.peakStep > 1)) {
    throw new RangeError('peakStep must be finite and in (0, 1]');
  }
  if (config.seed !== undefined && (!Number.isSafeInteger(config.seed) || config.seed < 0)) throw new RangeError('seed must be a non-negative safe integer');
  if (config.finalization !== undefined && !['lastEpochMean', 'ema'].includes(config.finalization)) throw new RangeError('unsupported finalization');
  return config;
}

/** Train independent subspace codebooks using the shared Rust/WASM fitter. Requires init(). */
export function trainProductQuantizer(data, dimensions, subspaces, options) {
  const config = snapshotOptions(options);
  const clusters = config.clusters;
  const subDimensions = shape(dimensions, subspaces, clusters);
  vectors(data, dimensions);
  const rows = data.length / dimensions;
  if (clusters > rows) throw new RangeError('clusters cannot exceed the number of vectors');
  const centroids = new Float32Array(clusters * dimensions);
  const subvector = new Float32Array(rows * subDimensions);
  for (let subspace = 0; subspace < subspaces; subspace++) {
    gather(data, dimensions, subDimensions, subspace, subvector);
    const model = fit(subvector, subDimensions, config);
    centroids.set(model.centroids, subspace * clusters * subDimensions);
  }
  return { schema, dimensions, subspaces, clusters, centroids };
}

/** Encode to row-major, one-byte-per-subspace codes using exact Rust/WASM assignment. */
export function encodeProductQuantizer(model, data) {
  const { dimensions, subspaces, clusters, centroids, subDimensions } = validated(model);
  vectors(data, dimensions, true);
  const rows = data.length / dimensions;
  const codes = new Uint8Array(rows * subspaces);
  const subvector = new Float32Array(rows * subDimensions);
  for (let subspace = 0; subspace < subspaces; subspace++) {
    gather(data, dimensions, subDimensions, subspace, subvector);
    const start = subspace * clusters * subDimensions;
    const assignments = predict(subvector, subDimensions, centroids.subarray(start, start + clusters * subDimensions));
    for (let row = 0; row < rows; row++) codes[row * subspaces + subspace] = assignments[row];
  }
  return codes;
}

/** Reconstruct row-major vectors from a validated codebook and byte codes. Does not require init(). */
export function decodeProductQuantizer(model, codes) {
  const { dimensions, subspaces, clusters, centroids, subDimensions } = validated(model);
  if (!(codes instanceof Uint8Array)) throw new TypeError('codes must be a Uint8Array');
  if (codes.length % subspaces !== 0) throw new RangeError('codes must contain complete subspace groups');
  for (const code of codes) if (code >= clusters) throw new RangeError('code exceeds the codebook cluster count');
  const rows = codes.length / subspaces;
  if (rows * dimensions > maxU32) throw new RangeError('decoded output exceeds the supported element count');
  const data = new Float32Array(rows * dimensions);
  for (let row = 0; row < rows; row++) {
    for (let subspace = 0; subspace < subspaces; subspace++) {
      const code = codes[row * subspaces + subspace];
      const start = (subspace * clusters + code) * subDimensions;
      data.set(centroids.subarray(start, start + subDimensions), row * dimensions + subspace * subDimensions);
    }
  }
  return data;
}

/** Return owned JSON-compatible codebook data. */
export function productQuantizerToJSON(model) {
  const { dimensions, subspaces, clusters, centroids } = validated(model);
  return { schema, dimensions, subspaces, clusters, centroids: Array.from(centroids) };
}

/** Restore a validated, owned typed codebook from parsed JSON. */
export function productQuantizerFromJSON(value) {
  if (value === null || typeof value !== 'object' || value.schema !== schema) {
    throw new TypeError(`model must use schema ${schema}`);
  }
  const { dimensions, subspaces, clusters, centroids } = value;
  shape(dimensions, subspaces, clusters);
  if (!Array.isArray(centroids) || centroids.length !== dimensions * clusters) {
    throw new RangeError('JSON centroids must be an array matching the codebook shape');
  }
  const owned = new Float32Array(centroids.length);
  for (let index = 0; index < centroids.length; index++) {
    const coordinate = centroids[index];
    if (typeof coordinate !== 'number' || !Number.isFinite(coordinate)) throw new RangeError('JSON centroids must be finite numbers');
    owned[index] = coordinate;
  }
  const model = { schema, dimensions, subspaces, clusters, centroids: owned };
  validated(model);
  return model;
}
