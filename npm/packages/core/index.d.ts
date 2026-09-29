export interface VectorEntry {
  id?: string;
  vector: Float32Array | number[];
}

export interface SearchQuery {
  vector: Float32Array | number[];
  k: number;
  efSearch?: number;
}

export interface SearchResult {
  id: string;
  score: number;
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
  distanceMetric?: string;
  /** HNSW index configuration. */
  hnswConfig?: any;
}

export class VectorDb {
  /**
   * Create a vector database. Throws if `storagePath` names an existing
   * database whose stored dimensions (or distance metric, when
   * `distanceMetric` is passed) differ from `options`.
   */
  constructor(options: VectorDbOptions);
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
