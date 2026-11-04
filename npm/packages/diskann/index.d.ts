export interface DiskAnnOptions {
  /** Vector dimension (required) */
  dim: number;
  /** Maximum out-degree for Vamana graph (default: 64) */
  maxDegree?: number;
  /** Beam width during index construction (default: 128) */
  buildBeam?: number;
  /** Beam width during search (default: 64) */
  searchBeam?: number;
  /** Alpha for robust pruning, >= 1.0 (default: 1.2) */
  alpha?: number;
  /** PQ subspaces, 0 = no PQ (default: 0) */
  pqSubspaces?: number;
  /** PQ k-means iterations (default: 10) */
  pqIterations?: number;
  /** Directory for persistence (optional) */
  storagePath?: string;
}

export interface DiskAnnSearchResult {
  /** Vector ID */
  id: string;
  /** L2 squared distance */
  distance: number;
}

export class DiskAnn {
  constructor(options: DiskAnnOptions);

  /** Insert a vector with a string ID */
  insert(id: string, vector: Float32Array): void;

  /** Insert N vectors: `ids.length` ids and a flat Float32Array of length N * dim */
  insertBatch(ids: string[], vectors: Float32Array, dim: number): void;

  /** Build the Vamana graph index (required before search) */
  build(): void;

  /** Build index asynchronously */
  buildAsync(): Promise<void>;

  /** Search for k nearest neighbors */
  search(query: Float32Array, k: number): DiskAnnSearchResult[];

  /** Search asynchronously */
  searchAsync(query: Float32Array, k: number): Promise<DiskAnnSearchResult[]>;

  /** Delete a vector by ID */
  delete(id: string): boolean;

  /** Get the number of vectors */
  count(): number;

  /** Save index to directory */
  save(dir: string): void;

  /** Load index from directory */
  static load(dir: string): DiskAnn;
}
