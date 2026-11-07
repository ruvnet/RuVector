/**
 * RaBitQ Wrapper — 1-bit quantized vector index via @ruvector/rabitq-wasm
 *
 * Optional extra: install with `npx ruvector add rabitq`. Codes are 1 bit per
 * dimension (~32x smaller than f32) with an exact-L2² rerank of the top
 * `rerankFactor * k` candidates to restore recall. Same lazy-load pattern as
 * the other optional wrappers; nothing is loaded until first use.
 */

const RABITQ_NODE_ENTRY = '@ruvector/rabitq-wasm/node/ruvector_rabitq_wasm.js';

let rabitqModule: any = null;
let loadError: Error | null = null;

function getRabitqModule() {
  if (rabitqModule) return rabitqModule;
  if (loadError) throw loadError;

  try {
    // The package root is an ES module for browsers; the node/ build is CJS.
    rabitqModule = require(RABITQ_NODE_ENTRY);
    return rabitqModule;
  } catch (e: any) {
    loadError = new Error(
      `@ruvector/rabitq-wasm is not installed (${e.message.split('\n')[0]}).\n` +
      'Install with: npx ruvector add rabitq  (or: npm install @ruvector/rabitq-wasm)'
    );
    throw loadError;
  }
}

export function isRabitqAvailable(): boolean {
  try {
    getRabitqModule();
    return true;
  } catch {
    return false;
  }
}

export interface RabitqConfig {
  /** Vector dimensionality (required) */
  dim: number;
  /** Rotation seed; same (seed, dim, vectors) gives bit-identical codes. Default 42 */
  seed?: number | bigint;
  /** Exact-rerank pool multiplier on k. Default 20 */
  rerankFactor?: number;
}

export interface RabitqSearchResult {
  id: string;
  /** Approximate L2² distance after rerank */
  distance: number;
}

type VectorInput = Float32Array | number[][] | Float32Array[];

function flatten(vectors: VectorInput, dim: number, count: number): Float32Array {
  if (vectors instanceof Float32Array) {
    if (vectors.length !== count * dim) {
      throw new Error(`RabitqIndex: expected ${count * dim} floats (ids × dim), got ${vectors.length}`);
    }
    return vectors;
  }
  if (vectors.length !== count) {
    throw new Error(`RabitqIndex: ${count} ids but ${vectors.length} vectors`);
  }
  const flat = new Float32Array(count * dim);
  vectors.forEach((row, i) => {
    if (row.length !== dim) {
      throw new Error(`RabitqIndex: vector ${i} has length ${row.length}, expected dim ${dim}`);
    }
    flat.set(row as ArrayLike<number>, i * dim);
  });
  return flat;
}

/**
 * Immutable 1-bit quantized index. Build once from all vectors, then search.
 * Call free() when done to release WASM memory.
 */
export class RabitqIndex {
  private inner: any;
  private ids: string[];
  readonly dim: number;

  private constructor(inner: any, ids: string[], dim: number) {
    this.inner = inner;
    this.ids = ids;
    this.dim = dim;
  }

  /**
   * Build an index. `vectors` is either one row per id or a flat
   * Float32Array of length ids.length * dim.
   */
  static build(ids: string[], vectors: VectorInput, config: RabitqConfig): RabitqIndex {
    const dim = config.dim;
    if (!Number.isInteger(dim) || dim <= 0) {
      throw new Error(`RabitqIndex: dim must be a positive integer, got ${dim}`);
    }
    if (!Array.isArray(ids) || ids.length === 0) {
      throw new Error('RabitqIndex: cannot build an empty index (no ids)');
    }
    if (!Array.isArray(vectors) && !(vectors instanceof Float32Array)) {
      throw new Error('RabitqIndex: vectors must be an array of rows or a Float32Array');
    }
    if (Array.isArray(vectors) && vectors.length !== ids.length) {
      throw new Error(`RabitqIndex: ${ids.length} ids but ${vectors.length} vectors`);
    }
    const flat = flatten(vectors, dim, ids.length);
    const rerank = config.rerankFactor ?? 20;
    if (!Number.isInteger(rerank) || rerank <= 0) {
      throw new Error(`RabitqIndex: rerankFactor must be a positive integer, got ${rerank}`);
    }
    const seed = BigInt(config.seed ?? 42);

    const mod = getRabitqModule();
    const inner = mod.RabitqIndex.build(flat, dim, seed, rerank);
    return new RabitqIndex(inner, ids.map(String), dim);
  }

  /** Number of indexed vectors */
  get size(): number {
    return this.ids.length;
  }

  /** k nearest neighbours of `query`, ascending distance */
  search(query: Float32Array | number[], k: number = 10): RabitqSearchResult[] {
    if (!this.inner) throw new Error('RabitqIndex: index has been freed');
    if (query.length !== this.dim) {
      throw new Error(`RabitqIndex: query length ${query.length} does not match dim ${this.dim}`);
    }
    if (!Number.isInteger(k) || k <= 0) {
      throw new Error(`RabitqIndex: k must be a positive integer, got ${k}`);
    }
    const q = query instanceof Float32Array ? query : new Float32Array(query);
    const hits = this.inner.search(q, Math.min(k, this.ids.length));
    const out: RabitqSearchResult[] = [];
    for (const hit of hits) {
      out.push({ id: this.ids[hit.id], distance: hit.distance });
      if (typeof hit.free === 'function') hit.free();
    }
    return out;
  }

  /** Release the WASM-side index */
  free(): void {
    if (this.inner && typeof this.inner.free === 'function') this.inner.free();
    this.inner = null;
  }
}

export default RabitqIndex;
