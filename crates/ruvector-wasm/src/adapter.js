/**
 * RuvectorWasmAdapter — a correct, ergonomic wrapper around the generated
 * `@ruvector/wasm` `VectorDB`.
 *
 * It exists to paper over three behaviours of the current WASM build that bite
 * callers who take the raw bindings (and the generated `.d.ts`) at face value:
 *
 *  1. **HNSW is not active in the WASM build.** The Rust crate compiles the
 *     `wasm32` target *without* the `hnsw` feature, so `VectorDB` silently falls
 *     back to a brute-force flat index (`vector_db.rs`:
 *     `"HNSW requested but not available (WASM build), using flat index"`).
 *     Results are still correct, but search is O(n), not O(log n). The win is
 *     latent until the upstream WASM HNSW lands. This adapter surfaces that fact
 *     via {@link RuvectorWasmAdapter#indexType} / {@link WASM_HNSW_AVAILABLE}
 *     instead of letting callers assume a logarithmic index.
 *
 *  2. **`result.score` is a cosine *distance*, not a similarity.** Lower is
 *     better and the ordering is correct (a, b before c), but that contradicts
 *     the generated `.d.ts` which advertises a "higher is better" score. This
 *     adapter exposes both the raw `distance` and a `similarity = 1 - distance`
 *     (generalised per metric) so "higher is better" actually holds.
 *
 *  3. **Metadata does not round-trip.** Inserted metadata comes back as `{}`
 *     (or `undefined`) from the WASM `search`/`get` getters. This adapter keeps
 *     an in-process **metadata sidecar** keyed by vector id and re-attaches it
 *     on the way out, so what you put in is what you get back.
 *
 * The adapter is dependency-injectable: pass a pre-constructed `VectorDB` (real
 * or a test double), or use {@link RuvectorWasmAdapter.create} to load and init
 * the WASM module for you.
 *
 * @module @ruvector/wasm/adapter
 */

/**
 * Whether the published WASM build ships an active HNSW index.
 *
 * The crate is compiled for `wasm32` without the `hnsw` cargo feature, so this
 * is `false` today: the WASM `VectorDB` uses a flat (brute-force) index. Flip
 * to `true` once the WASM build enables HNSW.
 *
 * @type {boolean}
 */
export const WASM_HNSW_AVAILABLE = false;

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
   * @param {boolean} [options.usesHnsw] - Override the index-type report. Defaults
   *   to {@link WASM_HNSW_AVAILABLE}.
   */
  constructor(db, options = {}) {
    if (!db) {
      throw new Error('RuvectorWasmAdapter requires a VectorDB instance');
    }
    this._db = db;
    this._metric = (options.metric || 'cosine').toLowerCase();
    this._dimensions = options.dimensions;
    this._usesHnsw = options.usesHnsw ?? WASM_HNSW_AVAILABLE;

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
   *   the WASM build falls back to flat regardless (see {@link WASM_HNSW_AVAILABLE}).
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
   * Whether this index is backed by HNSW. `false` for the current WASM build —
   * search is O(n) flat scan until upstream WASM HNSW lands.
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
    const vector = toFloat32(query.vector);
    const hasFilter = query.filter && Object.keys(query.filter).length > 0;

    // Over-fetch when filtering so post-filter results can still reach k.
    const fetch = hasFilter ? Math.max(k * 4, k) : k;
    const raw = this._db.search(vector, fetch, undefined) || [];

    let mapped = raw.map((r) => {
      const distance = r.score;
      const metadata = this._metadata.has(r.id)
        ? this._metadata.get(r.id)
        : r.metadata;
      const similarity = distanceToSimilarity(this._metric, distance);
      return {
        id: r.id,
        similarity,
        score: similarity,
        distance,
        vector: r.vector,
        metadata,
      };
    });

    if (hasFilter) {
      const entries = Object.entries(query.filter);
      mapped = mapped.filter((r) => {
        const md = r.metadata;
        if (!md) return false;
        return entries.every(([key, value]) => md[key] === value);
      });
    }

    // The flat index already orders by ascending distance, but sort defensively
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
    return {
      id: entry.id ?? id,
      vector: entry.vector,
      metadata: this._metadata.has(id) ? this._metadata.get(id) : entry.metadata,
    };
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
