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

export interface VectorDbOptions {
  /** Vector dimensionality (required). */
  dimensions: number;
  /**
   * Where the database is stored. Semantics from `@ruvector/core` 0.2.0
   * (issue #1063):
   *
   * - Omitted: an in-memory database private to this instance. Nothing is
   *   written to disk and nothing is shared with other `VectorDb` instances;
   *   the data is lost when the instance is garbage-collected.
   * - A file path: a persistent database at that path. If a database already
   *   exists there, its stored `dimensions` (and `distanceMetric`, when one is
   *   passed) must equal the ones passed here, otherwise the constructor throws
   *   an error naming the path and both values. Use a separate path per logical
   *   index (for example per tenant); instances opened on the same path share
   *   one store.
   * - A value starting with `memory://`: explicitly in-memory, same as omitting it.
   *
   * Upgrading from 0.1.x: omitting it used to open a shared `./ruvector.db` in
   * the working directory. To keep using that store, pass
   * `storagePath: './ruvector.db'`.
   */
  storagePath?: string;
  /**
   * Distance metric. Defaults to `'Cosine'` for a new database. When omitted,
   * an existing database keeps its stored metric; when passed, it must match.
   */
  distanceMetric?: DistanceMetric;
  /** HNSW index configuration. */
  hnswConfig?: HnswConfig;
}

export class VectorDb {
  /**
   * Create a vector database. Throws if `storagePath` names an existing
   * database whose stored dimensions (or distance metric, when
   * `distanceMetric` is passed) differ from `options`.
   */
  constructor(options: VectorDbOptions);
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
