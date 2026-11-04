/**
 * DiskANN Wrapper — Vamana graph ANN for billion-scale vector search
 *
 * Wraps @ruvector/diskann for SSD-friendly approximate nearest neighbor search.
 * Provides the same lazy-load pattern as other native wrappers.
 */

let diskannModule: any = null;
let loadError: Error | null = null;

function getDiskAnnModule() {
  if (diskannModule) return diskannModule;
  if (loadError) throw loadError;

  try {
    diskannModule = require('@ruvector/diskann');
    return diskannModule;
  } catch (e: any) {
    loadError = new Error(
      `@ruvector/diskann not installed: ${e.message}\n` +
      `Install with: npx ruvector add diskann  (or: npm install @ruvector/diskann)`
    );
    throw loadError;
  }
}

export function isDiskAnnAvailable(): boolean {
  try {
    getDiskAnnModule();
    return true;
  } catch {
    return false;
  }
}

export interface DiskAnnConfig {
  dim: number;
  maxDegree?: number;
  buildBeam?: number;
  searchBeam?: number;
  alpha?: number;
  pqSubspaces?: number;
  pqIterations?: number;
  storagePath?: string;
}

export interface DiskAnnSearchResult {
  id: string;
  distance: number;
}

/**
 * DiskANN index for large-scale approximate nearest neighbor search.
 *
 * Uses the Vamana graph algorithm with optional Product Quantization.
 * Build after all inserts, then search.
 */
export class DiskAnnIndex {
  private inner: any;

  constructor(config: DiskAnnConfig) {
    const mod = getDiskAnnModule();
    this.inner = new mod.DiskAnn({
      dim: config.dim,
      maxDegree: config.maxDegree ?? 64,
      buildBeam: config.buildBeam ?? 128,
      searchBeam: config.searchBeam ?? 64,
      alpha: config.alpha ?? 1.2,
      pqSubspaces: config.pqSubspaces ?? 0,
      pqIterations: config.pqIterations ?? 10,
      storagePath: config.storagePath,
    });
  }

  /** Insert a vector with a string ID */
  insert(id: string, vector: Float32Array | number[]): void {
    const v = vector instanceof Float32Array ? vector : new Float32Array(vector);
    this.inner.insert(id, v);
  }

  /** Insert a batch of vectors (flat Float32Array: N * dim) */
  insertBatch(ids: string[], vectors: Float32Array, dim: number): void {
    this.inner.insertBatch(ids, vectors, dim);
  }

  /** Build the Vamana graph index (required before search) */
  build(): void {
    this.inner.build();
  }

  /** Build index asynchronously */
  async buildAsync(): Promise<void> {
    return this.inner.buildAsync();
  }

  /** Search for k nearest neighbors */
  search(query: Float32Array | number[], k: number = 10): DiskAnnSearchResult[] {
    const q = query instanceof Float32Array ? query : new Float32Array(query);
    return this.inner.search(q, k);
  }

  /** Search asynchronously */
  async searchAsync(query: Float32Array | number[], k: number = 10): Promise<DiskAnnSearchResult[]> {
    const q = query instanceof Float32Array ? query : new Float32Array(query);
    return this.inner.searchAsync(q, k);
  }

  /** Delete a vector by ID */
  delete(id: string): boolean {
    return this.inner.delete(id);
  }

  /** Get the number of vectors */
  count(): number {
    return this.inner.count();
  }

  /** Save index to directory */
  save(dir: string): void {
    this.inner.save(dir);
  }

  /** Load index from directory */
  static load(dir: string): DiskAnnIndex {
    const mod = getDiskAnnModule();
    const instance = new DiskAnnIndex({ dim: 1 }); // placeholder
    instance.inner = mod.DiskAnn.load(dir);
    return instance;
  }
}

export default DiskAnnIndex;
