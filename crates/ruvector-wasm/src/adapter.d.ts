/**
 * Type declarations for the RuvectorWasmAdapter.
 *
 * Unlike the generated `pkg/ruvector_wasm.d.ts`, the `score` documented here is
 * a real similarity (higher is better); the raw distance is exposed separately.
 *
 * @module @ruvector/wasm/adapter
 */

/**
 * Whether the published WASM build ships an active HNSW index.
 * `false` today: the WASM `VectorDB` falls back to a flat (brute-force) index.
 */
export const WASM_HNSW_AVAILABLE: boolean;

/**
 * Convert a raw distance (lower is better) into a similarity (higher is better).
 * @param metric 'cosine' | 'dot' | 'dotproduct' | 'euclidean' | 'manhattan'
 * @param distance Raw score returned by the WASM `search`.
 */
export function distanceToSimilarity(metric: string, distance: number): number;

/** A single search result, with similarity and metadata corrected. */
export interface AdapterSearchResult {
  /** Vector id. */
  id: string;
  /** Similarity score — higher is better. */
  similarity: number;
  /** Raw distance from the underlying index — lower is better. */
  distance: number;
  /** Alias of `similarity`, so a `.score` read honours "higher is better". */
  score: number;
  /** Vector data, when returned by the index. */
  vector?: Float32Array;
  /** Round-tripped metadata from the sidecar. */
  metadata?: Record<string, any>;
}

/** Minimal shape of the underlying WASM (or test-double) VectorDB. */
export interface WasmVectorDBLike {
  insert(
    vector: Float32Array,
    id?: string,
    metadata?: Record<string, any>
  ): string;
  insertBatch(
    entries: Array<{
      id?: string;
      vector: Float32Array;
      metadata?: Record<string, any>;
    }>
  ): string[];
  search(
    vector: Float32Array,
    k: number,
    filter?: Record<string, any>
  ): Array<{
    id: string;
    score: number;
    vector?: Float32Array;
    metadata?: Record<string, any>;
  }>;
  get(
    id: string
  ): { id?: string; vector?: Float32Array; metadata?: Record<string, any> } | null;
  delete(id: string): boolean;
  len?(): number;
  isEmpty?(): boolean;
}

export interface AdapterOptions {
  /** Vector dimensions (informational). */
  dimensions?: number;
  /** Distance metric the db was created with; controls similarity conversion. */
  metric?: string;
  /** Override the index-type report. Defaults to {@link WASM_HNSW_AVAILABLE}. */
  usesHnsw?: boolean;
}

export interface CreateOptions {
  /** Vector dimensions (required). */
  dimensions: number;
  /** Distance metric. Defaults to 'cosine'. */
  metric?: string;
  /** Requested at the WASM layer (the build falls back to flat regardless). */
  useHnsw?: boolean;
  /** Pre-imported WASM module; if omitted, `@ruvector/wasm` is imported. */
  module?: any;
}

/** Correct wrapper around the generated WASM `VectorDB`. */
export class RuvectorWasmAdapter {
  constructor(db: WasmVectorDBLike, options?: AdapterOptions);

  static create(options: CreateOptions): Promise<RuvectorWasmAdapter>;

  /** `false` for the current WASM build — flat O(n) search. */
  readonly usesHnsw: boolean;
  /** 'hnsw' | 'flat' — index type backing this adapter. */
  readonly indexType: 'hnsw' | 'flat';

  insert(entry: {
    id?: string;
    vector: Float32Array | number[];
    metadata?: Record<string, any>;
  }): string;

  insertBatch(
    entries: Array<{
      id?: string;
      vector: Float32Array | number[];
      metadata?: Record<string, any>;
    }>
  ): string[];

  search(query: {
    vector: Float32Array | number[];
    k: number;
    filter?: Record<string, any>;
  }): AdapterSearchResult[];

  get(
    id: string
  ): { id: string; vector?: Float32Array; metadata?: Record<string, any> } | null;

  delete(id: string): boolean;
  len(): number;
  isEmpty(): boolean;
  clearMetadata(): void;
}

export default RuvectorWasmAdapter;
