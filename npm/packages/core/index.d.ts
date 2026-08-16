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

export type DistanceMetric = 'cosine' | 'euclidean' | 'dotproduct' | 'manhattan';
export type IndexType = 'hnsw' | 'flat';

export interface HnswConfig {
  m?: number;
  efConstruction?: number;
  efSearch?: number;
  maxElements?: number;
}

export interface EffectiveDbOptions {
  dimensions: number;
  distanceMetric: DistanceMetric;
  storagePath: string;
  hnswConfig?: HnswConfig;
  indexType: IndexType;
}

export class VectorDb {
  constructor(options: { dimensions: number; storagePath?: string; distanceMetric?: string; hnswConfig?: HnswConfig });
  getOptions(): EffectiveDbOptions;
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
