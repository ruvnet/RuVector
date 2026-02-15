import type {
  RvfOptions,
  RvfQueryOptions,
  RvfSearchResult,
  RvfIngestResult,
  RvfIngestEntry,
  RvfDeleteResult,
  RvfCompactionResult,
  RvfStatus,
  RvfFilterExpr,
  RvfKernelData,
  RvfEbpfData,
  RvfSegmentInfo,
  BackendType,
  CowStats,
} from './types';
import { RvfError, RvfErrorCode } from './errors';

// ---------------------------------------------------------------------------
// Backend interface — every backend (node, wasm) must implement this.
// ---------------------------------------------------------------------------

/**
 * Abstract backend that wraps either the native (N-API) or WASM build of
 * rvf-runtime.  The `RvfDatabase` class delegates all I/O to a backend
 * instance, keeping the public API identical regardless of runtime.
 */
export interface RvfBackend {
  /** Create a new store file at `path` with the given options. */
  create(path: string, options: RvfOptions): Promise<void>;
  /** Open an existing store at `path` for read-write access. */
  open(path: string): Promise<void>;
  /** Open an existing store at `path` for read-only access. */
  openReadonly(path: string): Promise<void>;
  /** Ingest a batch of vectors. */
  ingestBatch(entries: RvfIngestEntry[]): Promise<RvfIngestResult>;
  /** Query the k nearest neighbors. */
  query(vector: Float32Array, k: number, options?: RvfQueryOptions): Promise<RvfSearchResult[]>;
  /** Soft-delete vectors by ID. */
  delete(ids: string[]): Promise<RvfDeleteResult>;
  /** Soft-delete vectors matching a filter. */
  deleteByFilter(filter: RvfFilterExpr): Promise<RvfDeleteResult>;
  /** Run compaction to reclaim dead space. */
  compact(): Promise<RvfCompactionResult>;
  /** Get the current store status. */
  status(): Promise<RvfStatus>;
  /** Close the store, releasing locks. */
  close(): Promise<void>;
  // Lineage
  fileId(): Promise<string>;
  parentId(): Promise<string>;
  lineageDepth(): Promise<number>;
  derive(childPath: string, options?: RvfOptions): Promise<RvfBackend>;
  // Kernel / eBPF
  embedKernel(arch: number, kernelType: number, flags: number,
              image: Uint8Array, apiPort: number, cmdline?: string): Promise<number>;
  extractKernel(): Promise<RvfKernelData | null>;
  embedEbpf(programType: number, attachType: number, maxDimension: number,
            bytecode: Uint8Array, btf?: Uint8Array): Promise<number>;
  extractEbpf(): Promise<RvfEbpfData | null>;
  // Inspection
  segments(): Promise<RvfSegmentInfo[]>;
  dimension(): Promise<number>;
  // META_SEG KV
  putMeta(key: string, value: Uint8Array): Promise<void>;
  getMeta(key: string): Promise<Uint8Array | null>;
  deleteMeta(key: string): Promise<boolean>;
  listMetaKeys(): Promise<string[]>;
  // COW branching
  branch(childPath: string): Promise<RvfBackend>;
  freeze(): Promise<void>;
  cowStats(): Promise<CowStats | null>;
  isCowChild(): Promise<boolean>;
  parentStorePath(): Promise<string | null>;
  // Membership
  membershipContains(id: number): Promise<boolean>;
  membershipAdd(id: number): Promise<void>;
  membershipRemove(id: number): Promise<void>;
  membershipCount(): Promise<number | null>;
  // Witness
  lastWitnessHash(): Promise<Uint8Array>;
  queryAudited(vector: Float32Array, k: number, options?: RvfQueryOptions): Promise<RvfSearchResult[]>;
}

// ---------------------------------------------------------------------------
// NodeBackend — wraps @ruvector/rvf-node (N-API)
// ---------------------------------------------------------------------------

/**
 * Backend that delegates to the `@ruvector/rvf-node` native N-API addon.
 *
 * The native addon is loaded lazily on first use so that the SDK package can
 * be imported in environments where the native build is unavailable (e.g.
 * browsers) without throwing at import time.
 */
export class NodeBackend implements RvfBackend {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private native: any = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private handle: any = null;

  private async loadNative(): Promise<void> {
    if (this.native) return;
    try {
      // Dynamic import so the SDK can be bundled for browsers without
      // pulling in the native addon at compile time.
      this.native = await import('@ruvector/rvf-node');
    } catch {
      throw new RvfError(
        RvfErrorCode.BackendNotFound,
        'Could not load @ruvector/rvf-node — is it installed?',
      );
    }
  }

  private ensureHandle(): void {
    if (!this.handle) {
      throw new RvfError(RvfErrorCode.StoreClosed);
    }
  }

  async create(path: string, options: RvfOptions): Promise<void> {
    await this.loadNative();
    try {
      this.handle = await this.native.create(path, mapOptionsToNative(options));
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async open(path: string): Promise<void> {
    await this.loadNative();
    try {
      this.handle = await this.native.open(path);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async openReadonly(path: string): Promise<void> {
    await this.loadNative();
    try {
      this.handle = await this.native.openReadonly(path);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async ingestBatch(entries: RvfIngestEntry[]): Promise<RvfIngestResult> {
    this.ensureHandle();
    try {
      const ids = entries.map((e) => e.id);
      const vectors = entries.map((e) =>
        e.vector instanceof Float32Array ? e.vector : new Float32Array(e.vector),
      );
      const metadata = entries.some((e) => e.metadata)
        ? entries.map((e) => e.metadata ?? {})
        : undefined;
      const result = await this.native.ingestBatch(this.handle, ids, vectors, metadata);
      return {
        accepted: result.accepted,
        rejected: result.rejected,
        epoch: result.epoch,
      };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async query(
    vector: Float32Array,
    k: number,
    options?: RvfQueryOptions,
  ): Promise<RvfSearchResult[]> {
    this.ensureHandle();
    try {
      const nativeOpts = options ? mapQueryOptionsToNative(options) : undefined;
      const results = await this.native.query(this.handle, vector, k, nativeOpts);
      return (results as Array<{ id: string; distance: number }>).map((r) => ({
        id: r.id,
        distance: r.distance,
      }));
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async delete(ids: string[]): Promise<RvfDeleteResult> {
    this.ensureHandle();
    try {
      const result = await this.native.delete(this.handle, ids);
      return { deleted: result.deleted, epoch: result.epoch };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async deleteByFilter(filter: RvfFilterExpr): Promise<RvfDeleteResult> {
    this.ensureHandle();
    try {
      const result = await this.native.deleteByFilter(this.handle, filter);
      return { deleted: result.deleted, epoch: result.epoch };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async compact(): Promise<RvfCompactionResult> {
    this.ensureHandle();
    try {
      const result = await this.native.compact(this.handle);
      return {
        segmentsCompacted: result.segmentsCompacted ?? result.segments_compacted,
        bytesReclaimed: result.bytesReclaimed ?? result.bytes_reclaimed,
        epoch: result.epoch,
      };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async status(): Promise<RvfStatus> {
    this.ensureHandle();
    try {
      const s = await this.native.status(this.handle);
      return mapNativeStatus(s);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async close(): Promise<void> {
    if (!this.handle) return;
    try {
      await this.native.close(this.handle);
    } catch (err) {
      throw RvfError.fromNative(err);
    } finally {
      this.handle = null;
    }
  }

  async fileId(): Promise<string> {
    this.ensureHandle();
    try {
      return this.handle.fileId();
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async parentId(): Promise<string> {
    this.ensureHandle();
    try {
      return this.handle.parentId();
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async lineageDepth(): Promise<number> {
    this.ensureHandle();
    try {
      return this.handle.lineageDepth();
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async derive(childPath: string, options?: RvfOptions): Promise<RvfBackend> {
    this.ensureHandle();
    try {
      const nativeOpts = options ? mapOptionsToNative(options) : undefined;
      const childHandle = this.handle.derive(childPath, nativeOpts);
      const child = new NodeBackend();
      child.native = this.native;
      child.handle = childHandle;
      return child;
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async embedKernel(
    arch: number, kernelType: number, flags: number,
    image: Uint8Array, apiPort: number, cmdline?: string
  ): Promise<number> {
    this.ensureHandle();
    try {
      return this.handle.embedKernel(arch, kernelType, flags,
        Buffer.from(image), apiPort, cmdline);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async extractKernel(): Promise<RvfKernelData | null> {
    this.ensureHandle();
    try {
      const result = this.handle.extractKernel();
      if (!result) return null;
      return {
        header: new Uint8Array(result.header),
        image: new Uint8Array(result.image),
      };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async embedEbpf(
    programType: number, attachType: number, maxDimension: number,
    bytecode: Uint8Array, btf?: Uint8Array
  ): Promise<number> {
    this.ensureHandle();
    try {
      return this.handle.embedEbpf(programType, attachType, maxDimension,
        Buffer.from(bytecode), btf ? Buffer.from(btf) : undefined);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async extractEbpf(): Promise<RvfEbpfData | null> {
    this.ensureHandle();
    try {
      const result = this.handle.extractEbpf();
      if (!result) return null;
      return {
        header: new Uint8Array(result.header),
        payload: new Uint8Array(result.payload),
      };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async segments(): Promise<RvfSegmentInfo[]> {
    this.ensureHandle();
    try {
      const segs = this.handle.segments();
      return segs.map((s: any) => ({
        id: s.id,
        offset: s.offset,
        payloadLength: s.payloadLength ?? s.payload_length,
        segType: s.segType ?? s.seg_type,
      }));
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async dimension(): Promise<number> {
    this.ensureHandle();
    try {
      return this.handle.dimension();
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async putMeta(key: string, value: Uint8Array): Promise<void> {
    this.ensureHandle();
    try {
      this.handle.putMeta(key, Buffer.from(value));
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async getMeta(key: string): Promise<Uint8Array | null> {
    this.ensureHandle();
    try {
      const result = this.handle.getMeta(key);
      return result ? new Uint8Array(result) : null;
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async deleteMeta(key: string): Promise<boolean> {
    this.ensureHandle();
    try {
      return this.handle.deleteMeta(key);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async listMetaKeys(): Promise<string[]> {
    this.ensureHandle();
    try {
      return this.handle.listMetaKeys();
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async branch(childPath: string): Promise<RvfBackend> {
    this.ensureHandle();
    try {
      const childHandle = this.handle.branch(childPath);
      const child = new NodeBackend();
      child.native = this.native;
      child.handle = childHandle;
      return child;
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async freeze(): Promise<void> {
    this.ensureHandle();
    try {
      this.handle.freeze();
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async cowStats(): Promise<CowStats | null> {
    this.ensureHandle();
    try {
      return this.handle.cowStats() ?? null;
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async isCowChild(): Promise<boolean> {
    this.ensureHandle();
    try {
      return this.handle.isCowChild();
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async parentStorePath(): Promise<string | null> {
    this.ensureHandle();
    try {
      return this.handle.parentStorePath() ?? null;
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async membershipContains(id: number): Promise<boolean> {
    this.ensureHandle();
    try {
      return this.handle.membershipContains(id);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async membershipAdd(id: number): Promise<void> {
    this.ensureHandle();
    try {
      this.handle.membershipAdd(id);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async membershipRemove(id: number): Promise<void> {
    this.ensureHandle();
    try {
      this.handle.membershipRemove(id);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async membershipCount(): Promise<number | null> {
    this.ensureHandle();
    try {
      return this.handle.membershipCount() ?? null;
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async lastWitnessHash(): Promise<Uint8Array> {
    this.ensureHandle();
    try {
      return new Uint8Array(this.handle.lastWitnessHash());
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async queryAudited(vector: Float32Array, k: number, options?: RvfQueryOptions): Promise<RvfSearchResult[]> {
    this.ensureHandle();
    try {
      const f64Vec = Array.from(vector).map(v => v as number);
      const efSearch = options?.efSearch ?? 100;
      const results = this.handle.queryAudited(f64Vec, k, efSearch);
      return results.map((r: { id: number; distance: number }) => ({
        id: String(r.id),
        distance: r.distance,
      }));
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }
}

// ---------------------------------------------------------------------------
// WasmBackend — wraps @ruvector/rvf-wasm
// ---------------------------------------------------------------------------

/**
 * Backend that delegates to the `@ruvector/rvf-wasm` WASM build.
 *
 * Suitable for browser environments. The WASM module is loaded lazily.
 */
export class WasmBackend implements RvfBackend {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private wasm: any = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private handle: any = null;

  private async loadWasm(): Promise<void> {
    if (this.wasm) return;
    try {
      this.wasm = await import('@ruvector/rvf-wasm');
      if (typeof this.wasm.default === 'function') {
        await this.wasm.default();
      }
    } catch {
      throw new RvfError(
        RvfErrorCode.BackendNotFound,
        'Could not load @ruvector/rvf-wasm — is it installed?',
      );
    }
  }

  private ensureHandle(): void {
    if (!this.handle) {
      throw new RvfError(RvfErrorCode.StoreClosed);
    }
  }

  async create(path: string, options: RvfOptions): Promise<void> {
    await this.loadWasm();
    try {
      this.handle = this.wasm.create(path, mapOptionsToNative(options));
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async open(path: string): Promise<void> {
    await this.loadWasm();
    try {
      this.handle = this.wasm.open(path);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async openReadonly(path: string): Promise<void> {
    await this.loadWasm();
    try {
      this.handle = this.wasm.open_readonly(path);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async ingestBatch(entries: RvfIngestEntry[]): Promise<RvfIngestResult> {
    this.ensureHandle();
    try {
      const ids = entries.map((e) => e.id);
      const vectors = entries.map((e) =>
        e.vector instanceof Float32Array ? e.vector : new Float32Array(e.vector),
      );
      const metadata = entries.some((e) => e.metadata)
        ? entries.map((e) => e.metadata ?? {})
        : undefined;
      const result = this.wasm.ingest_batch(this.handle, ids, vectors, metadata);
      return {
        accepted: result.accepted,
        rejected: result.rejected,
        epoch: result.epoch,
      };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async query(
    vector: Float32Array,
    k: number,
    options?: RvfQueryOptions,
  ): Promise<RvfSearchResult[]> {
    this.ensureHandle();
    try {
      const nativeOpts = options ? mapQueryOptionsToNative(options) : undefined;
      const results = this.wasm.query(this.handle, vector, k, nativeOpts);
      return (results as Array<{ id: string; distance: number }>).map((r) => ({
        id: r.id,
        distance: r.distance,
      }));
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async delete(ids: string[]): Promise<RvfDeleteResult> {
    this.ensureHandle();
    try {
      const result = this.wasm.delete(this.handle, ids);
      return { deleted: result.deleted, epoch: result.epoch };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async deleteByFilter(filter: RvfFilterExpr): Promise<RvfDeleteResult> {
    this.ensureHandle();
    try {
      const result = this.wasm.delete_by_filter(this.handle, filter);
      return { deleted: result.deleted, epoch: result.epoch };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async compact(): Promise<RvfCompactionResult> {
    this.ensureHandle();
    try {
      const result = this.wasm.compact(this.handle);
      return {
        segmentsCompacted: result.segments_compacted ?? result.segmentsCompacted,
        bytesReclaimed: result.bytes_reclaimed ?? result.bytesReclaimed,
        epoch: result.epoch,
      };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async status(): Promise<RvfStatus> {
    this.ensureHandle();
    try {
      const s = this.wasm.status(this.handle);
      return mapNativeStatus(s);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async close(): Promise<void> {
    if (!this.handle) return;
    try {
      this.wasm.close(this.handle);
    } catch (err) {
      throw RvfError.fromNative(err);
    } finally {
      this.handle = null;
    }
  }

  async fileId(): Promise<string> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'fileId not supported in WASM backend');
  }
  async parentId(): Promise<string> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'parentId not supported in WASM backend');
  }
  async lineageDepth(): Promise<number> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'lineageDepth not supported in WASM backend');
  }
  async derive(_childPath: string, _options?: RvfOptions): Promise<RvfBackend> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'derive not supported in WASM backend');
  }
  async embedKernel(): Promise<number> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'embedKernel not supported in WASM backend');
  }
  async extractKernel(): Promise<RvfKernelData | null> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'extractKernel not supported in WASM backend');
  }
  async embedEbpf(): Promise<number> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'embedEbpf not supported in WASM backend');
  }
  async extractEbpf(): Promise<RvfEbpfData | null> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'extractEbpf not supported in WASM backend');
  }
  async segments(): Promise<RvfSegmentInfo[]> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'segments not supported in WASM backend');
  }
  async dimension(): Promise<number> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'dimension not supported in WASM backend');
  }
  async putMeta(): Promise<void> { throw new RvfError(RvfErrorCode.BackendNotFound, 'putMeta not supported in WASM backend'); }
  async getMeta(): Promise<Uint8Array | null> { throw new RvfError(RvfErrorCode.BackendNotFound, 'getMeta not supported in WASM backend'); }
  async deleteMeta(): Promise<boolean> { throw new RvfError(RvfErrorCode.BackendNotFound, 'deleteMeta not supported in WASM backend'); }
  async listMetaKeys(): Promise<string[]> { throw new RvfError(RvfErrorCode.BackendNotFound, 'listMetaKeys not supported in WASM backend'); }
  async branch(): Promise<RvfBackend> { throw new RvfError(RvfErrorCode.BackendNotFound, 'branch not supported in WASM backend'); }
  async freeze(): Promise<void> { throw new RvfError(RvfErrorCode.BackendNotFound, 'freeze not supported in WASM backend'); }
  async cowStats(): Promise<null> { throw new RvfError(RvfErrorCode.BackendNotFound, 'cowStats not supported in WASM backend'); }
  async isCowChild(): Promise<boolean> { throw new RvfError(RvfErrorCode.BackendNotFound, 'isCowChild not supported in WASM backend'); }
  async parentStorePath(): Promise<string | null> { throw new RvfError(RvfErrorCode.BackendNotFound, 'parentStorePath not supported in WASM backend'); }
  async membershipContains(): Promise<boolean> { throw new RvfError(RvfErrorCode.BackendNotFound, 'membershipContains not supported in WASM backend'); }
  async membershipAdd(): Promise<void> { throw new RvfError(RvfErrorCode.BackendNotFound, 'membershipAdd not supported in WASM backend'); }
  async membershipRemove(): Promise<void> { throw new RvfError(RvfErrorCode.BackendNotFound, 'membershipRemove not supported in WASM backend'); }
  async membershipCount(): Promise<number | null> { throw new RvfError(RvfErrorCode.BackendNotFound, 'membershipCount not supported in WASM backend'); }
  async lastWitnessHash(): Promise<Uint8Array> { throw new RvfError(RvfErrorCode.BackendNotFound, 'lastWitnessHash not supported in WASM backend'); }
  async queryAudited(): Promise<RvfSearchResult[]> { throw new RvfError(RvfErrorCode.BackendNotFound, 'queryAudited not supported in WASM backend'); }
}

// ---------------------------------------------------------------------------
// Backend resolution
// ---------------------------------------------------------------------------

/**
 * Resolve a `BackendType` to a concrete `RvfBackend` instance.
 *
 * - `'node'`  Always returns a `NodeBackend`.
 * - `'wasm'`  Always returns a `WasmBackend`.
 * - `'auto'`  Tries `node` first, falls back to `wasm`.
 */
export function resolveBackend(type: BackendType): RvfBackend {
  switch (type) {
    case 'node':
      return new NodeBackend();
    case 'wasm':
      return new WasmBackend();
    case 'auto': {
      // In Node.js environments, prefer native; in browsers, prefer WASM.
      const isNode =
        typeof process !== 'undefined' &&
        typeof process.versions !== 'undefined' &&
        typeof process.versions.node === 'string';
      return isNode ? new NodeBackend() : new WasmBackend();
    }
  }
}

// ---------------------------------------------------------------------------
// Mapping helpers (TS options -> native/wasm shapes)
// ---------------------------------------------------------------------------

function mapMetricToNative(metric: string | undefined): string {
  switch (metric) {
    case 'cosine':
      return 'Cosine';
    case 'dotproduct':
      return 'InnerProduct';
    case 'l2':
    default:
      return 'L2';
  }
}

function mapCompressionToNative(compression: string | undefined): string {
  switch (compression) {
    case 'scalar':
      return 'Scalar';
    case 'product':
      return 'Product';
    case 'none':
    default:
      return 'None';
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapOptionsToNative(options: RvfOptions): Record<string, any> {
  return {
    dimension: options.dimensions,
    metric: mapMetricToNative(options.metric),
    profile: options.profile ?? 0,
    compression: mapCompressionToNative(options.compression),
    signing: options.signing ?? false,
    m: options.m ?? 16,
    ef_construction: options.efConstruction ?? 200,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapQueryOptionsToNative(options: RvfQueryOptions): Record<string, any> {
  return {
    ef_search: options.efSearch ?? 100,
    filter: options.filter,
    timeout_ms: options.timeoutMs ?? 0,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapNativeStatus(s: any): RvfStatus {
  return {
    totalVectors: s.total_vectors ?? s.totalVectors ?? 0,
    totalSegments: s.total_segments ?? s.totalSegments ?? 0,
    fileSizeBytes: s.file_size ?? s.fileSizeBytes ?? 0,
    epoch: s.current_epoch ?? s.epoch ?? 0,
    profileId: s.profile_id ?? s.profileId ?? 0,
    compactionState: mapCompactionState(s.compaction_state ?? s.compactionState),
    deadSpaceRatio: s.dead_space_ratio ?? s.deadSpaceRatio ?? 0,
    readOnly: s.read_only ?? s.readOnly ?? false,
  };
}

function mapCompactionState(state: unknown): 'idle' | 'running' | 'emergency' {
  if (typeof state === 'string') {
    const lower = state.toLowerCase();
    if (lower === 'running') return 'running';
    if (lower === 'emergency') return 'emergency';
  }
  return 'idle';
}
