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
    // @ruvector/rvf is a DIFFERENT module with a different surface: it exports
    // RvfDatabase, never VectorDb. Accepting it unchecked produced
    // "implementation.VectorDb is not a constructor" at first use, far from
    // the real cause. Fail here, with the reason, instead.
    if (typeof implementation?.VectorDb !== 'function') {
      throw new Error(
        '@ruvector/rvf does not provide a VectorDb class (it exports RvfDatabase).\n' +
        '  It cannot back the VectorDB API. Install @ruvector/core instead:\n' +
        '    npm install @ruvector/core'
      );
    }
    implementationType = 'rvf';
  } catch (e: any) {
    if (e instanceof Error && e.message.includes('does not provide a VectorDb')) throw e;
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
    // Try rvf (persistent store) as second fallback. It only qualifies if it
    // actually exposes VectorDb -- today it does not, so this correctly falls
    // through to the stub rather than handing back an unusable module.
    try {
      const rvf = require('@ruvector/rvf');
      if (typeof rvf?.VectorDb !== 'function') {
        throw new Error('@ruvector/rvf exports no VectorDb class (it provides RvfDatabase)');
      }
      implementation = rvf;
      implementationType = 'rvf';
    } catch (rvfErr: any) {
      // Graceful fallback - don't crash, just warn
      console.warn('[RuVector] Native module not available:', e.message);
      console.warn('[RuVector] RVF module not available:', rvfErr.message);
      console.warn('[RuVector] Vector operations will be limited. Install @ruvector/core or @ruvector/rvf for full functionality.');

      // Stub of last resort. It deliberately reports its own emptiness
      // truthfully (len 0, isEmpty true, search []) rather than pretending a
      // write succeeded: `insert` used to return a plausible "stub-id-..."
      // for data it silently discarded, so a caller storing vectors saw
      // success and lost every one of them. Writes now throw.
      const unavailable = () => new Error(
        '[RuVector] No vector backend is available: @ruvector/core failed to load ' +
        'and @ruvector/rvf cannot substitute for it.\n' +
        '  Install the native module: npm install @ruvector/core\n' +
        '  Reads return empty; writes are refused so data is not silently dropped.'
      );
      implementation = {
        VectorDb: class StubVectorDb {
          constructor() {
            console.warn('[RuVector] Using stub VectorDb - install @ruvector/core for native performance');
          }
          async insert(): Promise<never> { throw unavailable(); }
          async insertBatch(_entries: any[]): Promise<never> { throw unavailable(); }
          async delete(): Promise<never> { throw unavailable(); }
          async search() { return []; }
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

/**
 * Normalize a user-friendly distance metric string (`"cosine"`, `"euclidean"`,
 * etc.) to the PascalCase variant the native `JsDistanceMetric` enum accepts.
 * Native: { Euclidean, Cosine, DotProduct, Manhattan }.
 */
function normalizeMetric(metric: string | undefined): string | undefined {
  if (!metric) return metric;
  const m = metric.toLowerCase().replace(/[_\s-]/g, '');
  switch (m) {
    case 'cosine':
      return 'Cosine';
    case 'euclidean':
    case 'l2':
      return 'Euclidean';
    case 'dot':
    case 'dotproduct':
    case 'innerproduct':
      return 'DotProduct';
    case 'manhattan':
    case 'l1':
      return 'Manhattan';
    default:
      return metric; // pass through; native will error with the variant list.
  }
}

/**
 * Wrapper class that automatically handles metadata JSON conversion
 */
class VectorDBWrapper {
  private db: any;

  constructor(options: { dimensions?: number; dimension?: number; storagePath?: string; distanceMetric?: string; metric?: string; hnswConfig?: any }) {
    // Accept both `distanceMetric` (canonical) and `metric` (CLI shorthand).
    // Normalize to the PascalCase enum variant the native binding expects.
    const distanceMetric = normalizeMetric(options.distanceMetric ?? (options as any).metric);
    const dimensions = options.dimensions ?? options.dimension;
    if (typeof dimensions !== 'number' || !Number.isInteger(dimensions) || dimensions <= 0) {
      throw new Error('Missing or invalid `dimensions` (the singular `dimension` alias is also accepted)');
    }
    const nativeOptions: any = {
      // The native N-API contract is plural even when callers use the public
      // singular alias. Keeping this mapping here prevents CLI/API drift.
      dimensions,
      storagePath: options.storagePath,
      // The N-API binding maps an omitted hnswConfig to `None`, which selects
      // FlatIndex and unintentionally overrides ruvector-core's HNSW default.
      // Pass the documented defaults explicitly so the high-level VectorDB
      // remains an ANN database unless callers deliberately provide another
      // HNSW configuration.
      hnswConfig: options.hnswConfig === undefined
        ? {
            m: 32,
            efConstruction: 200,
            efSearch: 100,
            maxElements: 10_000_000,
          }
        : options.hnswConfig,
    };
    if (distanceMetric !== undefined) {
      nativeOptions.distanceMetric = distanceMetric;
    }
    this.db = new implementation.VectorDb(nativeOptions);
  }

  /**
   * Insert a vector with optional metadata (objects are auto-converted to JSON)
   */
  async insert(entry: { id?: string; vector: Float32Array | number[]; metadata?: Record<string, any> }): Promise<string> {
    const nativeEntry: any = {
      id: entry.id,
      vector: entry.vector instanceof Float32Array ? entry.vector : new Float32Array(entry.vector),
    };

    // Auto-convert metadata object to JSON string
    if (entry.metadata) {
      nativeEntry.metadata = JSON.stringify(entry.metadata);
    }

    return this.db.insert(nativeEntry);
  }

  /**
   * Insert multiple vectors in batch
   */
  async insertBatch(entries: Array<{ id?: string; vector: Float32Array | number[]; metadata?: Record<string, any> }>): Promise<string[]> {
    const nativeEntries = entries.map(entry => ({
      id: entry.id,
      vector: entry.vector instanceof Float32Array ? entry.vector : new Float32Array(entry.vector),
      metadata: entry.metadata ? JSON.stringify(entry.metadata) : undefined,
    }));

    return this.db.insertBatch(nativeEntries);
  }

  /**
   * Search for similar vectors (metadata is auto-parsed from JSON)
   */
  async search(query: { vector: Float32Array | number[]; k: number; filter?: Record<string, any>; efSearch?: number }): Promise<Array<{ id: string; score: number; vector?: Float32Array; metadata?: Record<string, any> }>> {
    const nativeQuery: any = {
      vector: query.vector instanceof Float32Array ? query.vector : new Float32Array(query.vector),
      k: query.k,
      efSearch: query.efSearch,
    };

    // Auto-convert filter object to JSON string
    if (query.filter) {
      nativeQuery.filter = JSON.stringify(query.filter);
    }

    const results = await this.db.search(nativeQuery);

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
    return this.db.delete(id);
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

/** High-level index class compatible with the test-suite API. */
export class VectorIndex {
  private db: VectorDBWrapper;
  private _dimension: number;
  private _storagePath: string;

  constructor(opts: { dimension: number; metric?: string; indexType?: string }) {
    if (opts.dimension <= 0) {
      throw new Error(`Invalid dimensions: must be positive, got ${opts.dimension}`);
    }
    this._dimension = opts.dimension;
    // Use a unique temp path per instance to avoid cross-instance dimension conflicts
    this._storagePath = require('os').tmpdir() + `/ruvector-idx-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    this.db = new VectorDBWrapper({ dimensions: opts.dimension, distanceMetric: opts.metric, storagePath: this._storagePath });
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

  async clear(): Promise<void> {
    // Create a fresh db at a new temp path to reset state
    const newPath = require('os').tmpdir() + `/ruvector-idx-${Date.now()}-${Math.random().toString(36).slice(2)}.db`;
    this._storagePath = newPath;
    this.db = new VectorDBWrapper({ dimensions: this._dimension, storagePath: newPath });
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
