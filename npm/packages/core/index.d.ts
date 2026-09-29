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
   * Where the database is stored.
   *
   * - Omitted: an in-memory database private to this instance. Nothing is
   *   written to disk and nothing is shared with other `VectorDb` instances;
   *   the data is lost when the instance is garbage-collected.
   * - A file path: a persistent database at that path. If a database already
   *   exists there, its stored `dimensions` and `distanceMetric` must equal the
   *   ones passed here, otherwise the constructor throws an error naming the
   *   path and both values. Use a separate path per logical index (for example
   *   per tenant); instances opened on the same path share one store.
   * - A value starting with `memory://`: explicitly in-memory, same as omitting it.
   *
   * Before this was fixed (issue #1063), omitting it opened a shared
   * `./ruvector.db` in the working directory.
   */
  storagePath?: string;
  /** Distance metric. Defaults to `'Cosine'`. Checked against a stored database. */
  distanceMetric?: string;
  /** HNSW index configuration. */
  hnswConfig?: any;
}

export class VectorDb {
  /**
   * Create a vector database. Throws if `storagePath` names an existing
   * database whose stored dimensions or distance metric differ from `options`.
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
