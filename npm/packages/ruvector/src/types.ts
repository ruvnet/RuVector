/**
 * Vector entry representing a document with its embedding
 */
export interface VectorEntry {
  /** Unique identifier for the vector */
  id: string;
  /** Vector embedding (array of floats) */
  vector: number[];
  /** Optional metadata associated with the vector */
  metadata?: Record<string, any>;
}

/**
 * Search query parameters
 */
export interface SearchQuery {
  /** Query vector to search for */
  vector: number[];
  /** Number of results to return */
  k?: number;
  /** Optional metadata filters */
  filter?: Record<string, any>;
  /** Minimum similarity threshold (0-1) */
  threshold?: number;
}

/**
 * Search result containing matched vector and similarity score
 */
export interface SearchResult {
  /** ID of the matched vector */
  id: string;
  /** Similarity score (0-1, higher is better) */
  score: number;
  /** Vector data */
  vector: number[];
  /** Associated metadata */
  metadata?: Record<string, any>;
}

/**
 * Database configuration options
 */
export interface DbOptions {
  /**
   * Vector dimension size. `dimensions` is the canonical (native) name;
   * the singular `dimension` is accepted as an alias.
   */
  dimensions?: number;
  /** @deprecated Alias of `dimensions`. */
  dimension?: number;
  /** Distance metric to use (`distanceMetric` is accepted as an alias). */
  metric?: 'cosine' | 'euclidean' | 'dot' | 'manhattan';
  /** @deprecated Alias of `metric`. */
  distanceMetric?: string;
  /** Path to persist the database (`storagePath` is the canonical native name). */
  storagePath?: string;
  /** @deprecated Alias of `storagePath`. */
  path?: string;
  /**
   * HNSW index parameters — the canonical (native) name. Omit for the
   * defaults `{ m: 32, efConstruction: 200, efSearch: 100, maxElements: 1e7 }`.
   */
  hnswConfig?: {
    /** Maximum number of connections per layer */
    m?: number;
    /** Size of the dynamic candidate list during construction */
    efConstruction?: number;
    /** Size of the dynamic candidate list for search */
    efSearch?: number;
    /** Capacity hint for the index */
    maxElements?: number;
  };
  /**
   * @deprecated Alias of `hnswConfig`. Merged over the defaults, so a partial
   * object such as `{ efSearch: 200 }` changes only that setting.
   */
  hnsw?: {
    m?: number;
    efConstruction?: number;
    efSearch?: number;
    maxElements?: number;
  };
}

/**
 * Database statistics
 */
export interface DbStats {
  /** Total number of vectors */
  count: number;
  /** Vector dimension */
  dimension: number;
  /** Distance metric */
  metric: string;
  /** Memory usage in bytes */
  memoryUsage?: number;
  /** Index type */
  indexType?: string;
}

/**
 * Main VectorDB class interface
 */
export interface VectorDB {
  /**
   * Create a new vector database
   * @param options Database configuration
   */
  new(options: DbOptions): VectorDB;

  /**
   * Insert a single vector
   * @param entry Vector entry to insert
   */
  insert(entry: VectorEntry): void;

  /**
   * Insert multiple vectors in batch
   * @param entries Array of vector entries
   */
  insertBatch(entries: VectorEntry[]): void;

  /**
   * Search for similar vectors
   * @param query Search query parameters
   * @returns Array of search results
   */
  search(query: SearchQuery): SearchResult[];

  /**
   * Get vector by ID
   * @param id Vector ID
   * @returns Vector entry or null
   */
  get(id: string): VectorEntry | null;

  /**
   * Delete vector by ID
   * @param id Vector ID
   * @returns true if deleted, false if not found
   */
  delete(id: string): boolean;

  /**
   * Update vector metadata
   * @param id Vector ID
   * @param metadata New metadata
   */
  updateMetadata(id: string, metadata: Record<string, any>): void;

  /**
   * Get database statistics
   */
  stats(): DbStats;

  /**
   * Save database to disk
   * @param path Optional path (uses configured path if not provided)
   */
  save(path?: string): void;

  /**
   * Load database from disk
   * @param path Path to database file
   */
  load(path: string): void;

  /**
   * Clear all vectors from database
   */
  clear(): void;

  /**
   * Build HNSW index for faster search
   */
  buildIndex(): void;

  /**
   * Optimize database (rebuild indices, compact storage)
   */
  optimize(): void;
}
