const PARAMS = ['clusters', 'epochs', 'batchSize', 'peakStep'];
const FIT_KEYS = [...PARAMS, 'seed', 'finalization'];
const MAX_U32 = 0xffff_ffff;
export const MAX_INPUT_BYTES = 64 * 1024 * 1024;
export const MAX_WORK = 1_000_000_000;

export function object(value, allowed, name) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new TypeError(`${name} must be an object`);
  for (const key of Object.keys(value)) if (!allowed.includes(key)) throw new TypeError(`Unsupported ${name} field: ${key}`);
  return value;
}

export function integer(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max) throw new RangeError(`${name} must be an integer in ${min}..${max}`);
  return value;
}

function finite(value, min, max, name) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) throw new RangeError(`${name} must be finite in ${min}..${max}`);
  return value;
}

function rows(data, dimensions, name) {
  if (!(data instanceof Float32Array)) throw new TypeError(`${name} must be a Float32Array`);
  if (!data.length || data.length % dimensions) throw new RangeError(`${name} must contain nonempty complete vectors`);
  for (let i = 0; i < data.length; i++) if (!Number.isFinite(data[i])) throw new RangeError(`${name} must contain only finite values`);
  return data.length / dimensions;
}

function config(value, samples) {
  object(value, FIT_KEYS, 'baseline');
  const out = { clusters: value.clusters, epochs: value.epochs === undefined ? 3 : value.epochs,
    seed: value.seed === undefined ? 42 : value.seed, finalization: value.finalization === undefined ? 'lastEpochMean' : value.finalization };
  integer(out.clusters, 1, samples, 'clusters');
  integer(out.epochs, 1, MAX_U32, 'epochs');
  integer(out.seed, 0, Number.MAX_SAFE_INTEGER, 'seed');
  if (!['lastEpochMean', 'ema'].includes(out.finalization)) throw new RangeError('Unsupported finalization');
  if (value.batchSize !== undefined) out.batchSize = integer(value.batchSize, 1, MAX_U32, 'batchSize');
  if (value.peakStep !== undefined) out.peakStep = finite(value.peakStep, Number.MIN_VALUE, 1, 'peakStep');
  return out;
}

function domain(value, baseline, samples) {
  object(value, PARAMS, 'domain');
  if (!Object.keys(value).length) throw new RangeError('domain must contain at least one tunable parameter');
  const result = {};
  for (const [name, supplied] of Object.entries(value)) {
    object(supplied, ['min', 'max', 'type', 'scale', 'default'], `${name} domain`);
    const min = finite(supplied.min, Number.MIN_VALUE, name === 'peakStep' ? 1 : MAX_U32, `${name}.min`);
    const max = finite(supplied.max, min, name === 'peakStep' ? 1 : MAX_U32, `${name}.max`);
    const type = name === 'peakStep' ? 'float' : 'int';
    if (supplied.type !== type || !['linear', 'log'].includes(supplied.scale)) throw new TypeError(`Invalid type or scale for ${name}`);
    if (type === 'int') { integer(min, 1, MAX_U32, `${name}.min`); integer(max, min, MAX_U32, `${name}.max`); }
    if (name === 'clusters' && max > samples) throw new RangeError('Cluster domain exceeds training rows');
    if (baseline[name] === undefined || baseline[name] < min || baseline[name] > max) throw new RangeError(`baseline ${name} must be explicitly inside its domain`);
    if (supplied.default !== undefined && supplied.default !== baseline[name]) throw new RangeError(`${name}.default must match the baseline`);
    result[name] = Object.freeze({ min, max, type, scale: supplied.scale, default: baseline[name] });
  }
  return Object.freeze(result);
}

// Coordinate-scale upper estimate, including short-batch statistics and O(k²)
// donor selection. It is an admission budget, not a measured instruction count.
export function workEstimate(config, n, holdout, d) {
  const k = config.clusters, epochs = config.epochs;
  // When automatic batch size is requested, B>=1 is a safe lower bound.
  const batch = Math.min(config.batchSize ?? 1, n);
  const updates = epochs * Math.ceil(n / batch);
  const work = n * k * d * epochs + (n + holdout) * k * d + holdout * d +
    n * d * (epochs + 1) + n * (epochs + 1) +
    updates * k * (8 * d + 6) + Math.floor(updates / 5) * (3 * k * k + 6 * k * d + 3 * k);
  if (!Number.isSafeInteger(work) || !Number.isSafeInteger(updates)) throw new RangeError('Clustering work estimate exceeds safe integer range');
  return work;
}

export function checkOptions(value) {
  object(value, ['training', 'holdout', 'dimensions', 'baseline', 'domain', 'budget', 'qualityTolerance'], 'evaluator options');
  const dimensions = integer(value.dimensions, 1, MAX_U32, 'dimensions');
  if (!(value.training instanceof Float32Array) || !(value.holdout instanceof Float32Array)) throw new TypeError('training and holdout must be Float32Array values');
  const bytes = value.training.byteLength + value.holdout.byteLength;
  if (bytes > MAX_INPUT_BYTES) throw new RangeError('Input snapshots exceed 64 MiB');
  const n = rows(value.training, dimensions, 'training'), h = rows(value.holdout, dimensions, 'holdout');
  if (value.training.buffer === value.holdout.buffer && value.training.byteOffset < value.holdout.byteOffset + value.holdout.byteLength &&
      value.holdout.byteOffset < value.training.byteOffset + value.training.byteLength) throw new RangeError('training and holdout must use separate, nonoverlapping rows');
  const baseline = Object.freeze(config(value.baseline, n));
  const space = domain(value.domain, baseline, n);
  object(value.budget, ['maxEvaluations', 'maxDistanceCoordinates', 'timeoutMs'], 'budget');
  const budget = Object.freeze({ maxEvaluations: integer(value.budget.maxEvaluations, 1, 32, 'maxEvaluations'),
    maxDistanceCoordinates: integer(value.budget.maxDistanceCoordinates, 1, MAX_WORK, 'maxDistanceCoordinates'),
    timeoutMs: integer(value.budget.timeoutMs === undefined ? 30_000 : value.budget.timeoutMs, 1, 120_000, 'timeoutMs') });
  const worst = { ...baseline };
  for (const [name, spec] of Object.entries(space)) worst[name] = name === 'batchSize' ? spec.min : spec.max;
  const worstCaseWorkPerTrial = workEstimate(worst, n, h, dimensions);
  if (worstCaseWorkPerTrial > budget.maxDistanceCoordinates) throw new RangeError(`Domain permits work ${worstCaseWorkPerTrial} above maxDistanceCoordinates`);
  const qualityTolerance = finite(value.qualityTolerance === undefined ? 0 : value.qualityTolerance, 0, 1, 'qualityTolerance');
  return { training: new Float32Array(value.training), holdout: new Float32Array(value.holdout), dimensions, n, h, baseline,
    domain: space, budget, qualityTolerance, numericSnapshotBytes: bytes, worstCaseWorkPerTrial };
}

export function checkGenome(value, space) {
  const keys = Object.keys(space);
  object(value, keys, 'genome');
  if (Object.keys(value).length !== keys.length) throw new RangeError('Genome must include every domain parameter');
  const out = {};
  for (const [name, spec] of Object.entries(space)) {
    out[name] = spec.type === 'int' ? integer(value[name], spec.min, spec.max, name) : finite(value[name], spec.min, spec.max, name);
  }
  return out;
}

export const clone = value => structuredClone(value);
