import { checkedCodebook, integer, LIMITS, plain, safeText } from './validation.mjs';
import { readBytes } from './http.mjs';

/** Persist centroids only, never input vectors or assignments. Binding is caller-owned. */
export async function persistCodebook(bucket, key, model) {
  if (typeof bucket?.put !== 'function') throw new TypeError('an R2 put binding is required');
  safeText(key, 'key', 1024);
  const checked = checkedCodebook(model);
  const body = JSON.stringify({ ...checked, centroids: Array.from(checked.centroids) });
  const bytes = new TextEncoder().encode(body).byteLength;
  if (bytes > LIMITS.maxCodebookBytes) throw new RangeError('codebook exceeds storage byte limit');
  const result = await bucket.put(key, body, { httpMetadata: { contentType: 'application/json' } });
  if (!result || typeof result !== 'object') throw new Error('R2 did not acknowledge the codebook write');
  return { key, dimensions: checked.dimensions, clusters: checked.clusters, bytes };
}

export async function loadCodebook(bucket, key) {
  if (typeof bucket?.get !== 'function') throw new TypeError('an R2 get binding is required');
  safeText(key, 'key', 1024);
  const stored = await bucket.get(key);
  if (stored === null) return null;
  if (!stored || !Number.isSafeInteger(stored.size) || stored.size < 0 || stored.size > LIMITS.maxCodebookBytes) {
    throw new RangeError('stored codebook exceeds storage byte limit');
  }
  const bytes = await readBytes(stored.body, LIMITS.maxCodebookBytes, 5000);
  let parsed;
  try { parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
  catch { throw new TypeError('stored codebook must be valid UTF-8 JSON'); }
  const record = plain(parsed, ['schema', 'dimensions', 'clusters', 'centroids'],
    ['schema', 'dimensions', 'clusters', 'centroids'], 'stored codebook');
  if (record.schema !== 'ruvector.e-means-codebook/1' || !Array.isArray(record.centroids) ||
      record.centroids.length > LIMITS.maxElements) throw new TypeError('unsupported stored codebook');
  for (const value of record.centroids) {
    if (!Number.isFinite(value) || Math.abs(value) > 3.4028234663852886e38) throw new RangeError('stored centroids must be finite float32 values');
  }
  const checked = checkedCodebook({ dimensions: record.dimensions, centroids: new Float32Array(record.centroids) });
  if (record.clusters !== checked.clusters) throw new TypeError('stored cluster count does not match centroid shape');
  return checked;
}

export class CentroidExportError extends Error {
  constructor(submitted, mutationIds, cause) {
    super('Vectorize export failed after a partially submitted batch sequence', { cause });
    this.name = 'CentroidExportError'; this.submitted = submitted; this.mutationIds = [...mutationIds];
  }
}

/** Submit centroids to Vectorize. Acknowledged mutations may not be query-visible yet. */
export async function exportCentroids(index, model, options) {
  if (typeof index?.upsert !== 'function') throw new TypeError('a Vectorize upsert binding is required');
  const config = plain(options, ['prefix', 'namespace', 'batchSize'], ['prefix'], 'Vectorize options');
  safeText(config.prefix, 'prefix', 64);
  if (config.namespace !== undefined) safeText(config.namespace, 'namespace', 64);
  const batchSize = integer(config.batchSize ?? 1000, 1, 1000, 'batchSize');
  const checked = checkedCodebook(model);
  integer(checked.dimensions, 1, 1536, 'Vectorize dimensions');
  // Every ID uses the same prefix and a decimal ordinal; the last is longest.
  safeText(`${config.prefix}:${checked.clusters - 1}`, 'vector id', 64);
  let submitted = 0;
  const mutationIds = [];
  while (submitted < checked.clusters) {
    const end = Math.min(checked.clusters, submitted + batchSize), vectors = [];
    for (let ordinal = submitted; ordinal < end; ordinal++) {
      const row = checked.centroids.subarray(ordinal * checked.dimensions, (ordinal + 1) * checked.dimensions);
      vectors.push({ id: `${config.prefix}:${ordinal}`, values: Array.from(row), metadata: { cluster: ordinal },
        ...(config.namespace === undefined ? {} : { namespace: config.namespace }) });
    }
    let result;
    try {
      result = await index.upsert(vectors);
      safeText(result?.mutationId, 'Vectorize mutation id', 256);
    }
    catch (cause) { throw new CentroidExportError(submitted, mutationIds, cause); }
    submitted = end;
    mutationIds.push(result.mutationId);
  }
  return { submitted, mutationIds };
}
