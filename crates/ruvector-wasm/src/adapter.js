/**
 * Normalizes raw WASM distances into similarity and supports legacy metadata getters.
 * Backend identity comes from the running VectorDB, never from the requested option.
 * HNSW is approximate; useHnsw:false selects exact flat search.
 * @module @ruvector/wasm/adapter
 */

/** This release implements portable HNSW. Inspect instance.indexType for its active backend. */
export const WASM_HNSW_AVAILABLE = true;

/**
 * Convert a raw distance score (lower is better) into a similarity where
 * higher is better, matching the contract the `.d.ts` advertises.
 *
 * Mirrors the conversion used by `@ruvector/router` so the whole ecosystem
 * agrees on what "score" means.
 *
 * @param {string} metric - 'cosine' | 'dot' | 'dotproduct' | 'euclidean' | 'manhattan'
 * @param {number} distance - Raw score returned by the WASM `search`.
 * @returns {number} Similarity, higher is better.
 */
export function distanceToSimilarity(metric, distance) {
  switch ((metric || 'cosine').toLowerCase()) {
    case 'cosine':
      // cosine distance = 1 - cosine_similarity  ⇒  similarity = 1 - distance
      return 1 - distance;
    case 'dot':
    case 'dotproduct':
      // dot "distance" is stored negated  ⇒  similarity = -distance
      return -distance;
    case 'euclidean':
    case 'manhattan':
    default:
      // unbounded distances: monotonic decreasing map into (0, 1]
      return 1 / (1 + distance);
  }
}

/**
 * @typedef {Object} AdapterSearchResult
 * @property {string} id - Vector id.
 * @property {number} similarity - Higher is better (see {@link distanceToSimilarity}).
 * @property {number} distance - Raw score from the WASM index (lower is better).
 * @property {number} score - Alias of `similarity`, so the documented
 *   "higher is better" score contract holds for callers reading `.score`.
 * @property {Float32Array=} vector - Vector data, when returned by the index.
 * @property {Record<string, any>=} metadata - Round-tripped metadata from the sidecar.
 */

/**
 * Correct wrapper around the generated WASM `VectorDB`.
 */
export class RuvectorWasmAdapter {
  /**
   * @param {any} db - A constructed WASM `VectorDB` instance (or a compatible
   *   test double exposing `insert`, `insertBatch`, `search`, `get`, `delete`,
   *   `len`/`isEmpty`).
   * @param {Object} [options]
   * @param {number} [options.dimensions] - Vector dimensions (informational).
   * @param {string} [options.metric='cosine'] - Distance metric the `db` was
   *   created with; controls the similarity conversion.
   * @param {boolean} [options.usesHnsw] - Deprecated index-type override;
   *   ignored: the running database is authoritative.
   */
  constructor(db, options = {}) {
    if (!db) {
      throw new Error('RuvectorWasmAdapter requires a VectorDB instance');
    }
    this._db = db;
    this._metric = (options.metric || 'cosine').toLowerCase();
    this._dimensions = options.dimensions;
    this._usesHnsw = db.indexType === 'hnsw';

    /**
     * Metadata sidecar: id -> metadata. Works around the WASM build not
     * round-tripping metadata through `search`/`get`.
     * @type {Map<string, Record<string, any>>}
     */
    this._metadata = new Map();
  }

  /**
   * Load the WASM module, construct a `VectorDB`, and wrap it.
   *
   * @param {Object} [options]
   * @param {number} options.dimensions - Vector dimensions (required).
   * @param {string} [options.metric='cosine'] - Distance metric.
   * @param {boolean} [options.useHnsw=true] - Requested at the WASM layer; note
   *   false selects an exact flat index.
   * @param {any} [options.module] - Pre-imported WASM module (exposing `default`
   *   init and `VectorDB`). If omitted, `@ruvector/wasm` is imported dynamically.
   * @returns {Promise<RuvectorWasmAdapter>}
   */
  static async create(options = {}) {
    const { dimensions, metric = 'cosine', useHnsw = true } = options;
    if (!dimensions || dimensions <= 0) {
      throw new Error('RuvectorWasmAdapter.create requires positive `dimensions`');
    }

    const mod = options.module ?? (await import('@ruvector/wasm'));
    // `web`/`bundler` targets export a default init() that must run once before
    // any class is constructed. `nodejs` targets have no default export.
    if (typeof mod.default === 'function') {
      await mod.default();
    }

    const VectorDB = mod.VectorDB;
    if (typeof VectorDB !== 'function') {
      throw new Error('@ruvector/wasm did not export a VectorDB constructor');
    }

    const db = new VectorDB(dimensions, metric, useHnsw);
    return new RuvectorWasmAdapter(db, { dimensions, metric });
  }

  /**
   * Whether the running database reports an active HNSW backend.
   * @returns {boolean}
   */
  get usesHnsw() {
    return this._usesHnsw;
  }

  /**
   * Index type, for callers that want to reason about search complexity.
   * @returns {'hnsw' | 'flat'}
   */
  get indexType() {
    return this._usesHnsw ? 'hnsw' : 'flat';
  }

  /**
   * Insert a single vector, recording its metadata in the sidecar.
   *
   * @param {Object} entry
   * @param {string} [entry.id] - Optional id (auto-generated by WASM if absent).
   * @param {Float32Array | number[]} entry.vector
   * @param {Record<string, any>} [entry.metadata]
   * @returns {string} The vector id (the WASM-assigned one when not supplied).
   */
  insert(entry) {
    const vector = toFloat32(entry.vector);
    // Still hand metadata to the WASM layer (forward-compat for when it
    // round-trips), but the sidecar is the source of truth on the way out.
    const id = this._db.insert(vector, entry.id, entry.metadata);
    if (entry.metadata !== undefined) {
      this._metadata.set(id, entry.metadata);
    } else {
      this._metadata.delete(id);
    }
    return id;
  }

  /**
   * Insert vectors in a batch, recording metadata in the sidecar.
   *
   * @param {Array<{ id?: string, vector: Float32Array | number[], metadata?: Record<string, any> }>} entries
   * @returns {string[]} Vector ids in the same order as `entries`.
   */
  insertBatch(entries) {
    const nativeEntries = entries.map((e) => ({
      id: e.id,
      vector: toFloat32(e.vector),
      metadata: e.metadata,
    }));
    const ids = this._db.insertBatch(nativeEntries);
    for (let i = 0; i < ids.length; i++) {
      const meta = entries[i] && entries[i].metadata;
      if (meta !== undefined) {
        this._metadata.set(ids[i], meta);
      } else {
        this._metadata.delete(ids[i]);
      }
    }
    return ids;
  }

  /**
   * Search for the `k` nearest vectors.
   *
   * Returns results ordered best-first by `similarity` (higher is better), with
   * the raw `distance` preserved and metadata re-attached from the sidecar.
   * When `filter` is supplied it is applied against the sidecar metadata (the
   * WASM filter relies on metadata that does not round-trip), over-fetching as
   * needed so `k` results survive the filter where possible.
   *
   * @param {Object} query
   * @param {Float32Array | number[]} query.vector
   * @param {number} query.k
   * @param {Record<string, any>} [query.filter] - Exact-match metadata filter.
   * @returns {AdapterSearchResult[]}
   */
  search(query) {
    const k = query.k;
    if (!Number.isSafeInteger(k) || k < 0) throw new Error('k must be a nonnegative safe integer');
    if (k === 0) return [];
    const vector = toFloat32(query.vector);
    const hasFilter = query.filter && Object.keys(query.filter).length > 0;

    // Legacy modules may not filter correctly. Fetch all rows for complete filtering.
    const fetch = hasFilter ? Math.max(this.len(), k) : k;
    const raw = this._db.search(vector, fetch, undefined) || [];

    let mapped = raw.map((r) => {
      try {
        const id = r.id;
        const distance = r.score;
        const metadata = this._metadata.has(id)
          ? this._metadata.get(id)
          : r.metadata;
        const similarity = distanceToSimilarity(this._metric, distance);
        // WASM getters return owned copies, so these remain valid after free().
        return {
          id,
          similarity,
          score: similarity,
          distance,
          vector: r.vector,
          metadata,
        };
      } finally {
        if (typeof r.free === 'function') r.free();
      }
    });

    if (hasFilter) {
      const entries = Object.entries(query.filter);
      mapped = mapped.filter((r) => {
        const md = r.metadata;
        if (!md) return false;
        return entries.every(([key, value]) => metadataEqual(md[key], value));
      });
    }

    // The index orders by ascending distance, but sort defensively
    // so a, b come before c regardless of the underlying index's guarantees.
    mapped.sort((a, b) => b.similarity - a.similarity);

    return mapped.slice(0, k);
  }

  /**
   * Get a vector by id, with metadata re-attached from the sidecar.
   *
   * @param {string} id
   * @returns {{ id: string, vector?: Float32Array, metadata?: Record<string, any> } | null}
   */
  get(id) {
    const entry = this._db.get(id);
    if (!entry) return null;
    try {
      return {
        id: entry.id ?? id,
        vector: entry.vector,
        metadata: this._metadata.has(id) ? this._metadata.get(id) : entry.metadata,
      };
    } finally {
      if (typeof entry.free === 'function') entry.free();
    }
  }

  /**
   * Delete a vector by id, dropping its sidecar metadata.
   * @param {string} id
   * @returns {boolean}
   */
  delete(id) {
    const deleted = this._db.delete(id);
    if (deleted) {
      this._metadata.delete(id);
    }
    return deleted;
  }

  /**
   * Number of vectors in the index.
   * @returns {number}
   */
  len() {
    if (typeof this._db.len === 'function') return this._db.len();
    return this._metadata.size;
  }

  /**
   * Whether the index is empty.
   * @returns {boolean}
   */
  isEmpty() {
    if (typeof this._db.isEmpty === 'function') return this._db.isEmpty();
    return this.len() === 0;
  }

  /**
   * Drop all sidecar metadata. Call this when you recreate the underlying db.
   */
  clearMetadata() {
    this._metadata.clear();
  }
}

/**
 * Coerce a vector into a `Float32Array` without copying when already one.
 * @param {Float32Array | number[]} v
 * @returns {Float32Array}
 */
function toFloat32(v) {
  return v instanceof Float32Array ? v : new Float32Array(v);
}

export default RuvectorWasmAdapter;

// Match the native JSON metadata equality, including nested objects and arrays.
function metadataEqual(a, b) {
  if (a === b) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every((key) =>
    Object.prototype.hasOwnProperty.call(b, key) && metadataEqual(a[key], b[key]));
}
