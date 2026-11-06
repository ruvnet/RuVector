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
  valueType: string;
  value: string;
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
  derive(childPath: string, options?: RvfOptions): RvfDatabase;
  embedKernel(arch: number, kernelType: number, flags: number, image: Buffer, apiPort: number, cmdline?: string): number;
  extractKernel(): RvfKernelData | null;
  embedEbpf(programType: number, attachType: number, maxDimension: number, bytecode: Buffer, btf?: Buffer): number;
  extractEbpf(): RvfEbpfData | null;
  segments(): RvfSegmentInfo[];
  dimension(): number;
}
