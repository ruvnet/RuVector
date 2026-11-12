/* auto-generated: TypeScript declarations for @ruvector/rvf-node */

export interface RvfOptions {
  dimension: number;
  metric?: string;
  profile?: number;
  signing?: boolean;
  m?: number;
  efConstruction?: number;
}

export interface RvfQueryOptions {
  efSearch?: number;
  filter?: string;
  timeoutMs?: number;
  /** Opt in to the RaBitQ two-stage path (L2, cosine, inner product). */
  rabitq?: boolean;
  /**
   * Force the exact brute-force scan. Filtered queries whose filter keeps a
   * large share of the store otherwise use the approximate filtered HNSW walk.
   */
  forceExact?: boolean;
}

export interface RvfSearchResult {
  id: number;
  distance: number;
  /** The vector's committed metadata record; absent when it has none. */
  metadata?: RvfMetadataEntry[];
}

export interface RvfIngestResult {
  accepted: number;
  rejected: number;
  epoch: number;
}

export interface RvfDeleteResult {
  deleted: number;
  epoch: number;
}

export interface RvfCompactionResult {
  segmentsCompacted: number;
  bytesReclaimed: number;
  epoch: number;
}

export interface RvfStatus {
  totalVectors: number;
  totalSegments: number;
  fileSize: number;
  currentEpoch: number;
  profileId: number;
  compactionState: string;
  deadSpaceRatio: number;
  readOnly: boolean;
}

export interface RvfMetadataEntry {
  fieldId: number;
  /** "u64" | "i64" | "f64" | "string" | "bool" | "null" | "bytes" (hex). */
  valueType: string;
  value: string;
}

/** One vector's metadata record for `ingestBatchWithMetadata`. */
export interface RvfVectorMetadata {
  id: number;
  fields: RvfMetadataEntry[];
}

export interface RvfKernelData {
  header: Buffer;
  image: Buffer;
}

export interface RvfEbpfData {
  header: Buffer;
  payload: Buffer;
}

export interface RvfSegmentInfo {
  id: number;
  offset: number;
  payloadLength: number;
  segType: string;
}

/** Derivation types `derive()` accepts and a store can report. */
export type RvfDerivationType =
  | 'clone'
  | 'filter'
  | 'merge'
  | 'quantize'
  | 'reindex'
  | 'transform'
  | 'snapshot'
  | 'user_defined';

export class RvfDatabase {
  static create(path: string, options: RvfOptions): RvfDatabase;
  static open(path: string): RvfDatabase;
  static openReadonly(path: string): RvfDatabase;
  ingestBatch(vectors: Float32Array, ids: number[], metadata?: RvfMetadataEntry[]): RvfIngestResult;
  /** `ingestBatch` on the libuv threadpool (write lock). Rejects on error. */
  ingestBatchAsync(vectors: Float32Array, ids: number[], metadata?: RvfMetadataEntry[]): Promise<RvfIngestResult>;
  query(vector: Float32Array, k: number, options?: RvfQueryOptions): RvfSearchResult[];
  /** `query` on the libuv threadpool (read lock; concurrent calls run in parallel). Rejects on error. */
  queryAsync(vector: Float32Array, k: number, options?: RvfQueryOptions): Promise<RvfSearchResult[]>;
  /** Ingest vectors with per-vector metadata records (durable; returned on hits). */
  ingestBatchWithMetadata(vectors: Float32Array, ids: number[], records: RvfVectorMetadata[]): RvfIngestResult;
  /** `ingestBatchWithMetadata` on the libuv threadpool (write lock). */
  ingestBatchWithMetadataAsync(vectors: Float32Array, ids: number[], records: RvfVectorMetadata[]): Promise<RvfIngestResult>;
  /** A vector's committed metadata record, or null. */
  getMetadata(id: number): RvfMetadataEntry[] | null;
  /** A file-level metadata value, or null. */
  getFileMetadata(key: string): string | null;
  /** Durably set a file-level metadata value (`rvf.*` keys are reserved). */
  setFileMetadata(key: string, value: string): void;
  delete(ids: number[]): RvfDeleteResult;
  /** `delete` on the libuv threadpool (write lock). Rejects on error. */
  deleteAsync(ids: number[]): Promise<RvfDeleteResult>;
  deleteByFilter(filterJson: string): RvfDeleteResult;
  compact(): RvfCompactionResult;
  /** `compact` on the libuv threadpool (write lock); the event loop keeps running. */
  compactAsync(): Promise<RvfCompactionResult>;
  status(): RvfStatus;
  close(): void;
  fileId(): string;
  parentId(): string;
  lineageDepth(): number;
  /**
   * Create an empty child that records this store as its parent and how it
   * was derived (default `'filter'`). Unknown strings throw; `'merge'` is
   * recorded with this store as the only parent.
   */
  derive(childPath: string, options?: RvfOptions | null, derivationType?: RvfDerivationType | null): RvfDatabase;
  /** How this store was derived, or null (root store, or derived before rvf-node 0.4.7). */
  derivationType(): RvfDerivationType | null;
  embedKernel(arch: number, kernelType: number, flags: number, image: Buffer, apiPort: number, cmdline?: string): number;
  extractKernel(): RvfKernelData | null;
  embedEbpf(programType: number, attachType: number, maxDimension: number, bytecode: Buffer, btf?: Buffer): number;
  extractEbpf(): RvfEbpfData | null;
  segments(): RvfSegmentInfo[];
  dimension(): number;
}
