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

export interface DiskAnnRecord {
  id: string;
  vector: Float32Array | number[];
}

export interface DiskAnnBuildOptions {
  maxDegree?: number;
  buildBeam?: number;
  searchBeam?: number;
  alpha?: number;
}

/** Files the native index reads from and writes to inside its directory. */
const INDEX_FILES = [
  'config.json', 'vectors.bin', 'graph.bin', 'ids.json',
  'tombstones.bin', 'pq.bin', 'pq_codes.bin',
];

/**
 * Refuse an index directory whose index files are symlinks or other
 * non-regular files. The native save/load opens these names directly and
 * would follow a planted link out of a confined directory (overwriting or
 * reading files elsewhere on disk).
 */
function assertRegularIndexFiles(dir: string): void {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const fs = require('fs');
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const path = require('path');
  for (const name of INDEX_FILES) {
    let st;
    try {
      st = fs.lstatSync(path.join(dir, name));
    } catch (e: any) {
      if (e && e.code === 'ENOENT') continue;
      throw e;
    }
    if (st.isSymbolicLink()) {
      throw new Error(`diskann: ${name} in ${dir} is a symbolic link; refusing to follow it`);
    }
    if (!st.isFile()) {
      throw new Error(`diskann: ${name} in ${dir} is not a regular file`);
    }
  }
}

/**
 * Build a Vamana index from records and persist it to `dir` (created if
 * missing). All vectors must share one dimension. Used by
 * `ruvector diskann build` and the `diskann_build` MCP tool.
 */
export function buildDiskAnnIndex(
  dir: string,
  records: DiskAnnRecord[],
  options: DiskAnnBuildOptions = {}
): { dir: string; count: number; dim: number } {
  if (!Array.isArray(records) || records.length === 0) {
    throw new Error('diskann: at least one {id, vector} record is required');
  }
  const dim = records[0].vector.length;
  if (!Number.isInteger(dim) || dim <= 0) throw new Error('diskann: vectors must be non-empty');
  const seen = new Set<string>();
  const flat = new Float32Array(records.length * dim);
  records.forEach((r, i) => {
    if (r.vector.length !== dim) {
      throw new Error(`diskann: record ${i} (${r.id}) has dimension ${r.vector.length}, expected ${dim}`);
    }
    if (seen.has(r.id)) throw new Error(`diskann: duplicate id "${r.id}"`);
    seen.add(r.id);
    flat.set(r.vector as ArrayLike<number>, i * dim);
  });
  const index = new DiskAnnIndex({ dim, ...options });
  index.insertBatch(records.map((r) => r.id), flat, dim);
  index.build();
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  require('fs').mkdirSync(dir, { recursive: true });
  assertRegularIndexFiles(dir);
  index.save(dir);
  return { dir, count: index.count(), dim };
}

/** Load a saved index from `dir` and return the k nearest neighbours (L2²). */
export function searchDiskAnnIndex(
  dir: string,
  query: Float32Array | number[],
  k: number = 10
): DiskAnnSearchResult[] {
  assertRegularIndexFiles(dir);
  const index = DiskAnnIndex.load(dir);
  const n = index.count();
  if (n === 0) return [];
  return index.search(query, Math.min(k, n));
}

export default DiskAnnIndex;
