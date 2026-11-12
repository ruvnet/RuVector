export const LIMITS = Object.freeze({
  maxRequestBytes: 1024 * 1024,
  maxWork: 100_000_000,
  maxRows: 16_384,
  maxDimensions: 1536,
  maxClusters: 256,
  maxEpochs: 32,
  maxElements: 262_144,
  maxCodebookBytes: 8 * 1024 * 1024,
});

export class HTTPError extends Error {
  constructor(status, code, message) { super(message); this.status = status; this.code = code; }
}

export function plain(value, allowed, required, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value) ||
      ![Object.prototype, null].includes(Object.getPrototypeOf(value))) {
    throw new TypeError(`${name} must be a plain object`);
  }
  const output = Object.create(null);
  for (const key of Reflect.ownKeys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (typeof key !== 'string' || !allowed.includes(key) || !Object.hasOwn(descriptor, 'value')) {
      throw new TypeError(`unknown or accessor ${name} field`);
    }
    output[key] = descriptor.value;
  }
  if (required.some(key => !Object.hasOwn(output, key))) throw new TypeError(`missing ${name} field`);
  return output;
}

export function integer(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`${name} must be an integer in ${min}..${max}`);
  }
  return value;
}

export function safeText(value, name, maxBytes) {
  if (typeof value !== 'string' || value.length === 0 || /[\u0000-\u001f\u007f]/.test(value)) {
    throw new TypeError(`${name} must be nonempty text without control characters`);
  }
  for (const character of value) {
    const code = character.codePointAt(0);
    if (code >= 0xd800 && code <= 0xdfff) throw new TypeError(`${name} must contain valid Unicode`);
  }
  if (new TextEncoder().encode(value).byteLength > maxBytes) throw new RangeError(`${name} exceeds ${maxBytes} UTF-8 bytes`);
  return value;
}

export function floats(values, dimensions, name, allowEmpty = false) {
  if (!Array.isArray(values) || (!allowEmpty && values.length === 0) ||
      values.length > LIMITS.maxElements || values.length % dimensions !== 0) {
    throw new RangeError(`${name} must be a bounded array of complete vectors`);
  }
  integer(values.length / dimensions, allowEmpty ? 0 : 1, LIMITS.maxRows, `${name} rows`);
  for (const value of values) {
    if (!Number.isFinite(value) || Math.abs(value) > 3.4028234663852886e38) {
      throw new RangeError(`${name} must contain finite float32 values`);
    }
  }
  return new Float32Array(values);
}

export function fitOptions(value, rows) {
  const options = plain(value, ['clusters', 'epochs', 'batchSize', 'peakStep', 'seed', 'finalization'], ['clusters'], 'fit options');
  integer(options.clusters, 1, Math.min(rows, LIMITS.maxClusters), 'clusters');
  if (options.epochs !== undefined) integer(options.epochs, 1, LIMITS.maxEpochs, 'epochs');
  if (options.batchSize !== undefined) integer(options.batchSize, 1, LIMITS.maxRows, 'batchSize');
  if (options.peakStep !== undefined && (!Number.isFinite(options.peakStep) || options.peakStep <= 0 || options.peakStep > 1)) {
    throw new RangeError('peakStep must be in (0, 1]');
  }
  if (options.seed !== undefined) integer(options.seed, 0, Number.MAX_SAFE_INTEGER, 'seed');
  if (options.finalization !== undefined && !['lastEpochMean', 'ema'].includes(options.finalization)) {
    throw new TypeError('unsupported finalization');
  }
  return options;
}

export function checkedCodebook(value) {
  if (!value || typeof value !== 'object') throw new TypeError('codebook is required');
  const schemaDescriptor = Object.getOwnPropertyDescriptor(value, 'schema');
  if (schemaDescriptor && (!Object.hasOwn(schemaDescriptor, 'value') || schemaDescriptor.value !== 'ruvector.e-means-codebook/1')) {
    throw new TypeError('expected a single row-major codebook, not another model schema');
  }
  const dimensionsDescriptor = Object.getOwnPropertyDescriptor(value, 'dimensions');
  const centroidsDescriptor = Object.getOwnPropertyDescriptor(value, 'centroids');
  if (!dimensionsDescriptor || !centroidsDescriptor ||
      !Object.hasOwn(dimensionsDescriptor, 'value') || !Object.hasOwn(centroidsDescriptor, 'value')) {
    throw new TypeError('codebook dimensions and centroids must be own data properties');
  }
  const dimensions = integer(dimensionsDescriptor.value, 1, LIMITS.maxElements, 'dimensions');
  const centroids = centroidsDescriptor.value;
  if (!(centroids instanceof Float32Array) || centroids.length === 0 ||
      centroids.length > LIMITS.maxElements || centroids.length % dimensions !== 0) {
    throw new RangeError('codebook must contain complete finite centroid rows');
  }
  // Validate the owned snapshot, including when the caller supplies shared
  // storage or overrides its typed-array iterator.
  const owned = new Float32Array(centroids);
  for (let index = 0; index < owned.length; index++) {
    if (!Number.isFinite(owned[index])) throw new RangeError('centroids must be finite');
  }
  return { schema: 'ruvector.e-means-codebook/1', dimensions,
    clusters: owned.length / dimensions, centroids: owned };
}
