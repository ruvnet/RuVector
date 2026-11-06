export interface VectorEntry {
  id?: string;
  /** Must be a Float32Array; a plain number[] is rejected by the native binding. */
  vector: Float32Array;
  /** Optional metadata as a JSON string (use JSON.stringify / JSON.parse). */
  metadata?: string;
}

export interface SearchQuery {
  /** Must be a Float32Array; a plain number[] is rejected by the native binding. */
  vector: Float32Array;
  k: number;
  efSearch?: number;
}

export interface SearchResult {
  id: string;
  /** Distance score (lower is closer for distance metrics such as Cosine). */
  score: number;
  vector?: Float32Array;
  /** Metadata JSON string, if one was stored. */
  metadata?: string;
}

/** Distance metric names are case-sensitive. Default: 'Cosine'. */
export type DistanceMetric = 'Euclidean' | 'Cosine' | 'DotProduct' | 'Manhattan';

/** HNSW index parameters. Omit hnswConfig entirely to use an exact flat index. */
export interface HnswConfig {
  m?: number;
  efConstruction?: number;
  efSearch?: number;
  maxElements?: number;
}

export class VectorDb {
  /** storagePath defaults to './ruvector.db'; there is no in-memory mode. */
  constructor(options: { dimensions: number; storagePath?: string; distanceMetric?: DistanceMetric; hnswConfig?: HnswConfig });
  static withDimensions(dimensions: number): VectorDb;
  insert(entry: VectorEntry): Promise<string>;
  insertBatch(entries: VectorEntry[]): Promise<string[]>;
  search(query: SearchQuery): Promise<SearchResult[]>;
  delete(id: string): Promise<boolean>;
  get(id: string): Promise<VectorEntry | null>;
  len(): Promise<number>;
  isEmpty(): Promise<boolean>;
}

// Alias for backwards compatibility
export { VectorDb as VectorDB };

/** The installed @ruvector/core package version (e.g. "0.1.36"). */
export function version(): string;

/** The version of the bundled native engine (the Rust crate version). */
export function nativeVersion(): string;
