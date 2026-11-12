import { fit } from '../e-means.mjs';

const schema = 'ruvector-e-means-ivf-v1';

/** Train coarse centroids and construct stable posting lists from exact Rust/WASM assignments. */
export function partition(data, dimensions, options) {
  const model = fit(data, dimensions, options);
  const clusters = model.centroids.length / dimensions;
  const offsets = new Uint32Array(clusters + 1);
  for (const cluster of model.assignments) offsets[cluster + 1]++;
  for (let cluster = 0; cluster < clusters; cluster++) offsets[cluster + 1] += offsets[cluster];
  const cursors = offsets.slice(0, clusters);
  const indices = new Uint32Array(model.assignments.length);
  for (let row = 0; row < model.assignments.length; row++) {
    indices[cursors[model.assignments[row]]++] = row;
  }
  return { schema, model, offsets, indices };
}

function validated(value) {
  if (value === null || typeof value !== 'object' || value.schema !== schema) {
    throw new TypeError(`partition must use schema ${schema}`);
  }
  const { model, offsets, indices } = value;
  if (model === null || typeof model !== 'object') throw new TypeError('partition must contain a model');
  const { dimensions, centroids, assignments, mse, batchSize, peakStep, updates, revivedClusters } = model;
  if (!Number.isInteger(dimensions) || dimensions < 1 || dimensions > 0xffff_ffff) throw new RangeError('model dimensions must be positive');
  if (!(centroids instanceof Float32Array) || centroids.length === 0 || centroids.length % dimensions !== 0) {
    throw new RangeError('model centroids must be complete Float32Array vectors');
  }
  for (const coordinate of centroids) if (!Number.isFinite(coordinate)) throw new RangeError('model centroids must be finite');
  if (!Number.isFinite(mse) || mse < 0 || !Number.isFinite(peakStep) || peakStep <= 0 || peakStep > 1 ||
      !Number.isSafeInteger(batchSize) || batchSize < 1 || !Number.isSafeInteger(updates) || updates < 1 ||
      !Number.isSafeInteger(revivedClusters) || revivedClusters < 0) throw new RangeError('model metrics are invalid');
  if (!(assignments instanceof Uint32Array) || !(offsets instanceof Uint32Array) || !(indices instanceof Uint32Array)) {
    throw new TypeError('assignments, offsets and indices must be Uint32Array values');
  }
  const clusters = centroids.length / dimensions;
  if (offsets.length !== clusters + 1 || offsets[0] !== 0 || offsets[clusters] !== indices.length || indices.length !== assignments.length) {
    throw new RangeError('posting offsets must cover every model assignment exactly once');
  }
  for (let cluster = 0; cluster < clusters; cluster++) {
    const start = offsets[cluster];
    const end = offsets[cluster + 1];
    if (end < start || end > indices.length) throw new RangeError('posting offsets must be monotonic');
    let previous = -1;
    for (let cursor = start; cursor < end; cursor++) {
      const row = indices[cursor];
      if (row <= previous || row >= assignments.length || assignments[row] !== cluster) {
        throw new RangeError('posting indices must be unique, stable and consistent with model assignments');
      }
      previous = row;
    }
  }
  return { clusters, offsets, indices };
}

/** Return an owned copy of one posting list after validating the caller-owned partition. */
export function postingList(value, cluster) {
  const { clusters, offsets, indices } = validated(value);
  if (!Number.isInteger(cluster) || cluster < 0 || cluster >= clusters) throw new RangeError('cluster is outside the partition');
  return indices.slice(offsets[cluster], offsets[cluster + 1]);
}
