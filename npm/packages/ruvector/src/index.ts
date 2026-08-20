/**
 * ruvector - High-performance vector database for Node.js
 *
 * This package automatically detects and uses the best available implementation:
 * 1. Native (Rust-based, fastest) - if available for your platform
 * 2. RVF (persistent store) - if @ruvector/rvf is installed
 * 3. Stub (testing fallback) - limited functionality
 *
 * Also provides safe wrappers for GNN and Attention modules that handle
 * array type conversions automatically.
 */

export * from './types';

// Export core wrappers (safe interfaces with automatic type conversion)
export * from './core';
export * from './services';
export * from './metaharness';

let implementation: any;
let implementationType: 'native' | 'rvf' | 'wasm' = 'wasm';

// Check for explicit --backend rvf flag or RUVECTOR_BACKEND env var
const rvfRequested = process.env.RUVECTOR_BACKEND === 'rvf' ||
  process.argv.includes('--backend') && process.argv[process.argv.indexOf('--backend') + 1] === 'rvf';

if (rvfRequested) {
  // Explicit rvf backend requested - fail hard if not available
  try {
    implementation = require('@ruvector/rvf');
    implementationType = 'rvf';
  } catch (e: any) {
    throw new Error(
      '@ruvector/rvf is not installed.\n' +
      '  Run: npm install @ruvector/rvf\n' +
      '  The --backend rvf flag requires this package.'
    );
  }
} else {
  try {
    // Try to load native module first
    implementation = require('@ruvector/core');
    implementationType = 'native';

    // Verify it's actually working (native module exports VectorDb, not VectorDB)
    if (typeof implementation.VectorDb !== 'function') {
      throw new Error('Native module loaded but VectorDb class not found');
    }
  } catch (e: any) {
    // Try rvf (persistent store) as second fallback
    try {
      implementation = require('@ruvector/rvf');
      implementationType = 'rvf';
    } catch (rvfErr: any) {
      // Graceful fallback - don't crash, just warn
      console.warn('[RuVector] Native module not available:', e.message);
      console.warn('[RuVector] RVF module not available:', rvfErr.message);
      console.warn('[RuVector] Vector operations will be limited. Install @ruvector/core or @ruvector/rvf for full functionality.');

      // Create a stub implementation that provides basic functionality
      implementation = {
        VectorDb: class StubVectorDb {
          constructor() {
            console.warn('[RuVector] Using stub VectorDb - install @ruvector/core for native performance');
          }
          async insert() { return 'stub-id-' + Date.now(); }
          async insertBatch(entries: any[]) { return entries.map(() => 'stub-id-' + Date.now()); }
          async search() { return []; }
          async delete() { return true; }
          async get() { return null; }
          async len() { return 0; }
          async isEmpty() { return true; }
        }
      };
      implementationType = 'wasm'; // Mark as fallback mode
    }
  }
}

/**
 * Get the current implementation type
 */
export function getImplementationType(): 'native' | 'rvf' | 'wasm' {
  return implementationType;
}

/**
 * Check if native implementation is being used
 */
export function isNative(): boolean {
  return implementationType === 'native';
}

/**
 * Check if RVF implementation is being used
 */
export function isRvf(): boolean {
  return implementationType === 'rvf';
}

/**
 * Check if stub/fallback implementation is being used
 */
export function isWasm(): boolean {
  return implementationType === 'wasm';
}

/**
 * Get version information
 */
export function getVersion(): { version: string; implementation: string } {
  const pkg = require('../package.json');
  return {
    version: pkg.version,
    implementation: implementationType
  };
}

export type VectorDistanceMetric = 'cosine' | 'euclidean' | 'dotproduct' | 'manhattan';

/** Canonical public spelling used by index info and the persisted wrapper contract. */
function canonicalizeMetric(metric: string | undefined): VectorDistanceMetric {
  const m = (metric ?? 'cosine').toLowerCase().replace(/[_\s-]/g, '');
  switch (m) {
    case 'cosine':
      return 'cosine';
    case 'euclidean':
    case 'l2':
      return 'euclidean';
    case 'dot':
    case 'dotproduct':
    case 'innerproduct':
      return 'dotproduct';
    case 'manhattan':
    case 'l1':
      return 'manhattan';
    default:
      throw new Error(
        `Invalid distance metric: expected cosine, euclidean, dotproduct, or manhattan; got "${metric}"`
      );
  }
}

/** Convert the canonical public spelling to the native N-API enum variant. */
function toNativeMetric(metric: VectorDistanceMetric): string {
  switch (metric) {
    case 'cosine': return 'Cosine';
    case 'euclidean': return 'Euclidean';
    case 'dotproduct': return 'DotProduct';
    case 'manhattan': return 'Manhattan';
  }
}

export type VectorIndexType = 'hnsw' | 'flat';

export interface VectorDBOptions {
  dimensions?: number;
  dimension?: number;
  storagePath?: string;
  distanceMetric?: string;
  metric?: string;
  hnswConfig?: any;
  indexType?: VectorIndexType;
}

export interface VectorIndexInfo {
  /** Requested index type; effective only when configurationVerified is true. */
  indexType: VectorIndexType;
  dimensions: number;
  /** Canonical lowercase requested metric; effective only when configurationVerified is true. */
  distanceMetric: VectorDistanceMetric;
  storagePath: string;
  mutationMode: 'in-place' | 'rebuild' | 'unverified';
  /** True only when native effective configuration is proven by the wrapper contract. */
  configurationVerified: boolean;
}

/**
 * Wrapper class that automatically handles metadata JSON conversion
 */
class VectorDBWrapper {
  private db: any;
  private readonly nativeOptions: any;
  private readonly requestedDimensions: number;
  private readonly usesImplicitStoragePath: boolean;
  private implicitSchemaVerified = false;
  private readonly idWriteTails = new Map<string, Promise<void>>();
  private hnswMutationTail: Promise<void> = Promise.resolve();
  readonly indexType: VectorIndexType;
  private readonly distanceMetric: VectorDistanceMetric;
  private readonly storageIdentity: string;
  private readonly configurationVerified: boolean;

  constructor(options: VectorDBOptions) {
    // Accept both `distanceMetric` (canonical) and `metric` (CLI shorthand).
    const distanceMetric = canonicalizeMetric(options.distanceMetric ?? (options as any).metric);
    const dimensions = options.dimensions ?? options.dimension;
    if (typeof dimensions !== 'number' || !Number.isInteger(dimensions) || dimensions <= 0) {
      throw new Error('Missing or invalid `dimensions` (the singular `dimension` alias is also accepted)');
    }
    this.requestedDimensions = dimensions;
    this.usesImplicitStoragePath = options.storagePath === undefined;
    const requestedIndexType = options.indexType?.toLowerCase();
    if (requestedIndexType !== undefined && requestedIndexType !== 'hnsw' && requestedIndexType !== 'flat') {
      throw new Error(`Invalid indexType: expected "hnsw" or "flat", got "${options.indexType}"`);
    }
    if (requestedIndexType === 'flat' && options.hnswConfig != null) {
      throw new Error('indexType "flat" cannot be combined with hnswConfig');
    }
    if (requestedIndexType === 'hnsw' && options.hnswConfig === null) {
      throw new Error('indexType "hnsw" cannot be combined with hnswConfig: null');
    }
    this.indexType = requestedIndexType === 'flat' || options.hnswConfig === null ? 'flat' : 'hnsw';
    this.distanceMetric = distanceMetric;
    const requestedStoragePath = options.storagePath ?? './ruvector.db';
    this.storageIdentity = /^[a-z][a-z0-9+.-]*:\/\//i.test(requestedStoragePath)
      ? requestedStoragePath
      : require('path').resolve(requestedStoragePath);
    const nativeOptions: any = {
      // The native N-API contract is plural even when callers use the public
      // singular alias. Keeping this mapping here prevents CLI/API drift.
      dimensions,
      // Pin even the implicit path at construction so a later process.chdir()
      // cannot make a containment rebuild open a different database.
      storagePath: this.storageIdentity,
    };
    // N-API selects FlatIndex only when hnswConfig is omitted. Passing null is
    // rejected by the binding, so do not add the property for a flat index.
    if (this.indexType === 'hnsw') {
      nativeOptions.hnswConfig = options.hnswConfig === undefined
        ? {
            m: 32,
            efConstruction: 200,
            efSearch: 100,
            maxElements: 10_000_000,
          }
        : { ...options.hnswConfig };
    }
    nativeOptions.distanceMetric = toNativeMetric(distanceMetric);
    this.nativeOptions = nativeOptions;
    this.db = new implementation.VectorDb(this.cloneNativeOptions());
    this.configurationVerified = this.verifyEffectiveNativeOptions(this.db);
  }

  /**
   * The native backend persists to `./ruvector.db` when storagePath is omitted.
   * On reopen it adopts the stored schema, so the mismatch is first observable
   * when a caller submits a vector. Add storage identity without mislabelling a
   * normal bad-vector error as a persisted-schema collision.
   */
  private rethrowWithStorageContext(error: unknown): never {
    const message = error instanceof Error ? error.message : String(error);
    const mismatch = message.match(/expected\s+(\d+)\s*,\s*(?:got|actual)\s+(\d+)/i);
    const storedDimensions = mismatch ? Number(mismatch[1]) : undefined;
    const submittedDimensions = mismatch ? Number(mismatch[2]) : undefined;

    if (
      this.usesImplicitStoragePath &&
      storedDimensions !== undefined &&
      submittedDimensions === this.requestedDimensions &&
      storedDimensions !== this.requestedDimensions
    ) {
      const diagnosed = new Error(
        `Vector schema collision at implicit persistent store "./ruvector.db": ` +
        `the caller requested ${this.requestedDimensions} dimensions, but the existing store ` +
        `expects ${storedDimensions}. Pass an explicit \`storagePath\` for each vector schema ` +
        `or reopen this store with ${storedDimensions} dimensions. Original error: ${message}`
      );
      (diagnosed as any).cause = error;
      throw diagnosed;
    }

    throw error;
  }

  private validateVector(vector: Float32Array): void {
    if (vector.length !== this.requestedDimensions) {
      throw new Error(
        `Vector dimension mismatch: expected ${this.requestedDimensions}, got ${vector.length}`
      );
    }
  }

  private cloneNativeOptions(): any {
    return {
      ...this.nativeOptions,
      ...(this.nativeOptions.hnswConfig === undefined
        ? {}
        : { hnswConfig: { ...this.nativeOptions.hnswConfig } }),
    };
  }

  private normalizedHnswConfig(config: any): {
    m: number;
    efConstruction: number;
    efSearch: number;
    maxElements: number;
  } {
    const normalized = {
      m: config?.m ?? 32,
      efConstruction: config?.efConstruction ?? 200,
      efSearch: config?.efSearch ?? 100,
      maxElements: config?.maxElements ?? 10_000_000,
    };
    for (const [name, value] of Object.entries(normalized)) {
      if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
        throw new Error(`Invalid effective HNSW configuration: ${name}=${String(value)}`);
      }
    }
    return normalized;
  }

  /**
   * Native core restores persisted options instead of constructor input. Only
   * its authoritative getter can certify the effective metric and index type.
   */
  private verifyEffectiveNativeOptions(candidate: any): boolean {
    if (implementationType !== 'native') return false;

    if (typeof candidate?.getOptions !== 'function') {
      if (this.indexType === 'flat') {
        throw new Error(
          'Cannot certify a flat index with this @ruvector/core build because it does not expose ' +
          'getOptions(). Upgrade to a coordinated core release before using mutable flat indexes.'
        );
      }
      return false;
    }

    let effective: any;
    try {
      effective = candidate.getOptions();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Cannot verify effective native vector configuration: ${message}`);
    }
    if (!effective || typeof effective !== 'object' || typeof effective.then === 'function') {
      throw new Error('Cannot verify effective native vector configuration: getOptions() returned an invalid value');
    }
    if (
      typeof effective.dimensions !== 'number' ||
      !Number.isInteger(effective.dimensions) ||
      effective.dimensions <= 0 ||
      typeof effective.distanceMetric !== 'string' ||
      typeof effective.storagePath !== 'string' ||
      (effective.indexType !== 'hnsw' && effective.indexType !== 'flat')
    ) {
      throw new Error('Cannot verify effective native vector configuration: getOptions() returned an invalid shape');
    }

    const effectiveMetric = canonicalizeMetric(effective.distanceMetric);
    const effectiveIndexType: VectorIndexType = effective.indexType;
    const hasHnswConfig = effective.hnswConfig !== undefined && effective.hnswConfig !== null;
    if ((effectiveIndexType === 'hnsw') !== hasHnswConfig) {
      throw new Error(
        'Cannot verify effective native vector configuration: indexType and hnswConfig disagree'
      );
    }

    const requestedHnsw = this.indexType === 'hnsw'
      ? this.normalizedHnswConfig(this.nativeOptions.hnswConfig)
      : undefined;
    const effectiveHnsw = effectiveIndexType === 'hnsw'
      ? this.normalizedHnswConfig(effective.hnswConfig)
      : undefined;
    const hnswMatches = requestedHnsw === undefined && effectiveHnsw === undefined ||
      requestedHnsw !== undefined && effectiveHnsw !== undefined &&
      requestedHnsw.m === effectiveHnsw.m &&
      requestedHnsw.efConstruction === effectiveHnsw.efConstruction &&
      requestedHnsw.efSearch === effectiveHnsw.efSearch &&
      requestedHnsw.maxElements === effectiveHnsw.maxElements;

    const mismatch = effective.dimensions !== this.requestedDimensions ||
      effectiveMetric !== this.distanceMetric ||
      effectiveIndexType !== this.indexType ||
      effective.storagePath !== this.storageIdentity ||
      !hnswMatches;
    if (mismatch) {
      const location = this.usesImplicitStoragePath
        ? 'implicit persistent store "./ruvector.db"'
        : `persistent store "${this.storageIdentity}"`;
      throw new Error(
        `Effective vector configuration mismatch at ${location}: native reports ` +
        `dimensions=${effective.dimensions}, distanceMetric=${effectiveMetric}, ` +
        `indexType=${effectiveIndexType}; requested dimensions=${this.requestedDimensions}, ` +
        `distanceMetric=${this.distanceMetric}, indexType=${this.indexType}. ` +
        'Persisted native configuration overrides constructor options; reopen with matching options ' +
        'or use a new storagePath.'
      );
    }

    return true;
  }

  /**
   * hnsw_rs cannot remove graph nodes. Reopen the same persistent store so the
   * native constructor rebuilds a clean graph from the unique storage records.
   */
  private assertHnswRebuildSupported(): void {
    if (implementationType !== 'native' || this.indexType !== 'hnsw') return;
    if (/^[a-z][a-z0-9+.-]*:\/\//i.test(this.storageIdentity)) {
      throw new Error(
        'Cannot safely rebuild a mutable HNSW index backed by a non-file storage URI. ' +
        'Use indexType "flat" for mutable explicit IDs.'
      );
    }
  }

  private rebuildNativeHnswIndex(): void {
    if (implementationType !== 'native' || this.indexType !== 'hnsw') return;
    this.assertHnswRebuildSupported();

    const previous = this.db;
    try {
      const replacement = new implementation.VectorDb(this.cloneNativeOptions());
      const replacementVerified = this.verifyEffectiveNativeOptions(replacement);
      if (replacementVerified !== this.configurationVerified) {
        throw new Error(
          'Replacement native index changed configuration-verification status; refusing to attach it'
        );
      }
      this.db = replacement;
    } catch (error) {
      this.db = previous;
      const message = error instanceof Error ? error.message : String(error);
      const diagnosed = new Error(
        'The vector mutation was persisted, but the HNSW containment rebuild failed. ' +
        `The previous wrapper remains attached; reopen this storage before further search. ${message}`
      );
      (diagnosed as any).cause = error;
      throw diagnosed;
    }
  }

  /** Detect an implicit persisted-schema mismatch before any upsert deletes data. */
  private async verifyImplicitSchema(vector: Float32Array): Promise<void> {
    if (!this.usesImplicitStoragePath || this.implicitSchemaVerified) return;
    try {
      await this.db.search({ vector, k: 1 });
    } catch (error) {
      this.rethrowWithStorageContext(error);
    }
    this.implicitSchemaVerified = true;
  }

  /** Serialize only writes whose explicit ID sets overlap. */
  private async withIdWriteLocks<T>(ids: string[], operation: () => Promise<T>): Promise<T> {
    const reservations = [...new Set(ids)].sort().map(id => {
      const previous = this.idWriteTails.get(id) ?? Promise.resolve();
      let release!: () => void;
      const gate = new Promise<void>(resolve => { release = resolve; });
      const tail = previous.then(() => gate);
      this.idWriteTails.set(id, tail);
      return { id, previous, release, tail };
    });

    await Promise.all(reservations.map(({ previous }) => previous));
    try {
      return await operation();
    } finally {
      for (const { id, release, tail } of reservations) {
        release();
        if (this.idWriteTails.get(id) === tail) this.idWriteTails.delete(id);
      }
    }
  }

  /** HNSW rebuilds replace the whole native index, so serialize every mutation. */
  private async withMutationLock<T>(ids: string[], operation: () => Promise<T>): Promise<T> {
    if (implementationType !== 'native' || this.indexType !== 'hnsw') {
      return this.withIdWriteLocks(ids, operation);
    }

    const previous = this.hnswMutationTail;
    let release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    const tail = previous.then(() => gate);
    this.hnswMutationTail = tail;
    await previous;
    try {
      return await operation();
    } finally {
      release();
      if (this.hnswMutationTail === tail) this.hnswMutationTail = Promise.resolve();
    }
  }

  getIndexInfo(): VectorIndexInfo {
    return {
      indexType: this.indexType,
      dimensions: this.requestedDimensions,
      distanceMetric: this.distanceMetric,
      storagePath: this.storageIdentity,
      mutationMode: implementationType === 'native' && this.indexType === 'hnsw'
        ? 'rebuild'
        : this.configurationVerified && this.indexType === 'flat'
          ? 'in-place'
          : 'unverified',
      configurationVerified: this.configurationVerified,
    };
  }

  /**
   * Insert a vector with optional metadata (objects are auto-converted to JSON)
   */
  async insert(entry: { id?: string; vector: Float32Array | number[]; metadata?: Record<string, any> }): Promise<string> {
    const vector = entry.vector instanceof Float32Array ? entry.vector : new Float32Array(entry.vector);
    this.validateVector(vector);
    const nativeEntry: any = {
      id: entry.id,
      vector,
    };

    // Auto-convert metadata object to JSON string
    if (entry.metadata) {
      nativeEntry.metadata = JSON.stringify(entry.metadata);
    }

    await this.verifyImplicitSchema(vector);

    return this.withMutationLock(entry.id === undefined ? [] : [entry.id], async () => {
      const replacesExisting = entry.id !== undefined && await this.db.get(entry.id) !== null;
      try {
        if (replacesExisting) this.assertHnswRebuildSupported();
        const id = await this.db.insert(nativeEntry);
        if (replacesExisting) this.rebuildNativeHnswIndex();
        return id;
      } catch (error) {
        this.rethrowWithStorageContext(error);
      }
    });
  }

  /**
   * Insert multiple vectors in batch
   */
  async insertBatch(entries: Array<{ id?: string; vector: Float32Array | number[]; metadata?: Record<string, any> }>): Promise<string[]> {
    if (entries.length === 0) return [];

    // Validate and serialize the complete batch before mutating an old value.
    // Repeated explicit IDs use deterministic last-write-wins semantics.
    const nativeEntries = entries.map(entry => {
      const vector = entry.vector instanceof Float32Array
        ? entry.vector
        : new Float32Array(entry.vector);
      this.validateVector(vector);
      return {
        id: entry.id,
        vector,
        metadata: entry.metadata ? JSON.stringify(entry.metadata) : undefined,
      };
    });
    const lastIndexById = new Map<string, number>();
    nativeEntries.forEach((entry, index) => {
      if (entry.id !== undefined) lastIndexById.set(entry.id, index);
    });

    await this.verifyImplicitSchema(nativeEntries[0].vector);

    const retainedIndexes = nativeEntries
      .map((entry, index) => ({ entry, index }))
      .filter(({ entry, index }) => entry.id === undefined || lastIndexById.get(entry.id) === index);

    return this.withMutationLock([...lastIndexById.keys()], async () => {
      let replacesExisting = false;
      for (const id of lastIndexById.keys()) {
        if (await this.db.get(id)) replacesExisting = true;
      }

      let insertedIds: string[];
      try {
        if (replacesExisting) this.assertHnswRebuildSupported();
        insertedIds = await this.db.insertBatch(retainedIndexes.map(({ entry }) => entry));
        if (replacesExisting) this.rebuildNativeHnswIndex();
      } catch (error) {
        this.rethrowWithStorageContext(error);
      }

      // Preserve the historical one-result-per-input return shape. Earlier
      // duplicates resolve to the same explicit ID as their winning final value.
      const ids = new Array<string>(entries.length);
      retainedIndexes.forEach(({ index }, retainedIndex) => {
        ids[index] = insertedIds[retainedIndex];
      });
      nativeEntries.forEach((entry, index) => {
        if (ids[index] === undefined && entry.id !== undefined) ids[index] = entry.id;
      });
      return ids;
    });
  }

  /**
   * Search for similar vectors (metadata is auto-parsed from JSON)
   */
  async search(query: { vector: Float32Array | number[]; k: number; filter?: Record<string, any>; efSearch?: number }): Promise<Array<{ id: string; score: number; vector?: Float32Array; metadata?: Record<string, any> }>> {
    const vector = query.vector instanceof Float32Array
      ? query.vector
      : new Float32Array(query.vector);
    this.validateVector(vector);
    const nativeQuery: any = {
      vector,
      k: query.k,
      efSearch: query.efSearch,
    };

    // Auto-convert filter object to JSON string
    if (query.filter) {
      nativeQuery.filter = JSON.stringify(query.filter);
    }

    let results: any[];
    try {
      results = await this.db.search(nativeQuery);
    } catch (error) {
      this.rethrowWithStorageContext(error);
    }

    // Auto-parse metadata JSON strings back to objects
    return results.map((r: any) => ({
      id: r.id,
      score: r.score,
      vector: r.vector,
      metadata: r.metadata ? JSON.parse(r.metadata) : undefined,
    }));
  }

  /**
   * Get a vector by ID (metadata is auto-parsed from JSON)
   */
  async get(id: string): Promise<{ id?: string; vector: Float32Array; metadata?: Record<string, any> } | null> {
    const entry = await this.db.get(id);
    if (!entry) return null;

    return {
      id: entry.id,
      vector: entry.vector,
      metadata: entry.metadata ? JSON.parse(entry.metadata) : undefined,
    };
  }

  /**
   * Delete a vector by ID
   */
  async delete(id: string): Promise<boolean> {
    return this.withMutationLock([id], async () => {
      if (await this.db.get(id)) this.assertHnswRebuildSupported();
      const deleted = await this.db.delete(id);
      if (deleted) this.rebuildNativeHnswIndex();
      return deleted;
    });
  }

  /**
   * Get the number of vectors in the database
   */
  async len(): Promise<number> {
    return this.db.len();
  }

  /**
   * Check if the database is empty
   */
  async isEmpty(): Promise<boolean> {
    return this.db.isEmpty();
  }
}

// Export the wrapper class (aliased as VectorDB for backwards compatibility)
export const VectorDb = VectorDBWrapper;
export const VectorDB = VectorDBWrapper;

// Also export the raw native implementation for advanced users
export const NativeVectorDb = implementation.VectorDb;

// ────────────────────────────────────────────────────────────────────────────
// Backwards-compat surface used by tests and older integrations
// ────────────────────────────────────────────────────────────────────────────

export interface VectorIndexOptions {
  dimension: number;
  metric?: string;
  indexType?: VectorIndexType;
  hnswConfig?: any;
}

/**
 * High-level index class compatible with the test-suite API.
 *
 * VectorIndex uses an isolated temporary store. Use VectorDb with an explicit
 * storagePath when the storage identity must remain durable across processes.
 */
export class VectorIndex {
  private db: VectorDBWrapper;
  private readonly _dimension: number;
  private readonly _metric?: string;
  private readonly _hnswConfig: any;
  private _storagePath: string;
  readonly indexType: VectorIndexType;

  constructor(opts: VectorIndexOptions) {
    if (opts.dimension <= 0) {
      throw new Error(`Invalid dimensions: must be positive, got ${opts.dimension}`);
    }
    const requestedIndexType = opts.indexType === undefined
      ? 'hnsw'
      : String(opts.indexType).toLowerCase();
    if (requestedIndexType !== 'hnsw' && requestedIndexType !== 'flat') {
      throw new Error(`Invalid indexType: expected "hnsw" or "flat", got "${opts.indexType}"`);
    }
    if (requestedIndexType === 'flat' && opts.hnswConfig != null) {
      throw new Error('indexType "flat" cannot be combined with hnswConfig');
    }
    if (requestedIndexType === 'hnsw' && opts.hnswConfig === null) {
      throw new Error('indexType "hnsw" cannot be combined with hnswConfig: null');
    }
    this._dimension = opts.dimension;
    this._metric = opts.metric;
    this.indexType = requestedIndexType;
    this._hnswConfig = this.indexType === 'flat'
      ? undefined
      : opts.hnswConfig === undefined
        ? undefined
        : { ...opts.hnswConfig };
    // Use a unique temp path per instance to avoid cross-instance dimension conflicts
    this._storagePath = this.createStoragePath();
    this.db = this.createDatabase(this._storagePath);
  }

  private createStoragePath(): string {
    return require('os').tmpdir() + `/ruvector-idx-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
  }

  private createDatabase(storagePath: string): VectorDBWrapper {
    return new VectorDBWrapper({
      dimensions: this._dimension,
      distanceMetric: this._metric,
      storagePath,
      hnswConfig: this._hnswConfig,
      indexType: this.indexType,
    });
  }

  async insert(entry: { id?: string; values: number[] }): Promise<string> {
    return this.db.insert({ id: entry.id, vector: new Float32Array(entry.values) });
  }

  async insertBatch(
    entries: Array<{ id?: string; values: number[] }>,
    _opts?: { batchSize?: number; progressCallback?: (p: number) => void }
  ): Promise<string[]> {
    const ids: string[] = [];
    const batchSize = _opts?.batchSize ?? entries.length;
    for (let i = 0; i < entries.length; i += batchSize) {
      const slice = entries.slice(i, i + batchSize);
      const batch = slice.map(e => ({ id: e.id, vector: new Float32Array(e.values) }));
      const batchIds = await this.db.insertBatch(batch);
      ids.push(...batchIds);
      _opts?.progressCallback?.(Math.min((i + batchSize) / entries.length, 1));
    }
    return ids;
  }

  async search(query: number[], opts: { k: number }): Promise<Array<{ id: string; score: number }>> {
    return this.db.search({ vector: new Float32Array(query), k: opts.k });
  }

  async get(id: string): Promise<{ id: string; values: number[] } | null> {
    const r = await this.db.get(id);
    if (!r) return null;
    return { id: r.id ?? id, values: Array.from(r.vector) };
  }

  async delete(id: string): Promise<boolean> {
    return this.db.delete(id);
  }

  async stats(): Promise<{ vectorCount: number; dimension: number }> {
    const count = await this.db.len();
    return { vectorCount: count, dimension: this._dimension };
  }

  getIndexInfo(): VectorIndexInfo {
    return this.db.getIndexInfo();
  }

  async clear(): Promise<void> {
    // Create a fresh isolated store while retaining metric and index settings.
    const newPath = this.createStoragePath();
    this._storagePath = newPath;
    this.db = this.createDatabase(newPath);
  }

  async optimize(): Promise<void> {
    // No-op: native HNSW self-optimises on insert
  }
}

/** Get backend info (compat with old getBackendInfo() call). */
export function getBackendInfo(): { type: 'native' | 'wasm'; version: string; features: string[] } {
  const type = implementationType === 'native' ? 'native' : 'wasm';
  const { version } = getVersion();
  const features: string[] = type === 'native'
    ? ['SIMD', 'Multi-threading', 'Rust-native']
    : ['Browser-compatible', 'Cross-platform'];
  return { type, version, features };
}

/** Check native availability (compat alias for isNative()). */
export function isNativeAvailable(): boolean {
  return implementationType === 'native';
}

/** Vector utility functions used by tests and downstream packages. */
export const Utils = {
  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) throw new Error('Vectors must have same dimension');
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom === 0 ? 0 : dot / denom;
  },
  euclideanDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) throw new Error('Vectors must have same dimension');
    return Math.sqrt(a.reduce((sum, v, i) => sum + (v - b[i]) ** 2, 0));
  },
  normalize(v: number[]): number[] {
    const mag = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    if (mag === 0) return v.slice();
    return v.map(x => x / mag);
  },
  randomVector(dimension: number): number[] {
    const v = Array.from({ length: dimension }, () => Math.random() * 2 - 1);
    return Utils.normalize(v);
  },
};

// Export everything from the implementation
export default implementation;
