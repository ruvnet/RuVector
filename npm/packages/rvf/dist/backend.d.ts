import type { RvfOptions, RvfQueryOptions, RvfSearchResult, RvfIngestResult, RvfIngestEntry, RvfDeleteResult, RvfCompactionResult, RvfStatus, RvfFilterExpr, RvfKernelData, RvfEbpfData, RvfSegmentInfo, BackendType, DistanceMetric } from './types';
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
    fileId(): Promise<string>;
    parentId(): Promise<string>;
    lineageDepth(): Promise<number>;
    derive(childPath: string, options?: RvfOptions): Promise<RvfBackend>;
    branch(childPath: string): Promise<RvfBackend>;
    freeze(): Promise<number>;
    embedKernel(arch: number, kernelType: number, flags: number, image: Uint8Array, apiPort: number, cmdline?: string): Promise<number>;
    extractKernel(): Promise<RvfKernelData | null>;
    embedEbpf(programType: number, attachType: number, maxDimension: number, bytecode: Uint8Array, btf?: Uint8Array): Promise<number>;
    extractEbpf(): Promise<RvfEbpfData | null>;
    segments(): Promise<RvfSegmentInfo[]>;
    dimension(): Promise<number>;
    /**
     * Distance metric the store was created with. Optional so third-party
     * backends written against older versions still type-check.
     */
    metric?(): Promise<DistanceMetric>;
    /** Serialize the store to an in-memory `.rvf` byte buffer. */
    exportBytes(): Promise<Uint8Array>;
    /** Load a store from an in-memory `.rvf` byte buffer. */
    openBytes(bytes: Uint8Array): Promise<void>;
}
/**
 * Backend that delegates to the `@ruvector/rvf-node` native N-API addon.
 *
 * The native addon is loaded lazily on first use so that the SDK package can
 * be imported in environments where the native build is unavailable (e.g.
 * browsers) without throwing at import time.
 */
export declare class NodeBackend implements RvfBackend {
    private native;
    private handle;
    private idToLabel;
    private labelToId;
    private nextLabel;
    private storePath;
    private readOnly;
    private journal;
    /** Metadata field-name registry, persisted in the store's file metadata. */
    private fields;
    /** Byte size of the last snapshot written or loaded. */
    private snapshotBytes;
    private journalCompactMinBytes;
    private loadNative;
    private ensureHandle;
    /** Release a native handle without persisting mappings after open failed. */
    private discardHandle;
    create(path: string, options: RvfOptions): Promise<void>;
    open(path: string): Promise<void>;
    openReadonly(path: string): Promise<void>;
    ingestBatch(entries: RvfIngestEntry[]): Promise<RvfIngestResult>;
    query(vector: Float32Array, k: number, options?: RvfQueryOptions): Promise<RvfSearchResult[]>;
    delete(ids: string[]): Promise<RvfDeleteResult>;
    deleteByFilter(filter: RvfFilterExpr): Promise<RvfDeleteResult>;
    compact(): Promise<RvfCompactionResult>;
    status(): Promise<RvfStatus>;
    close(): Promise<void>;
    fileId(): Promise<string>;
    parentId(): Promise<string>;
    lineageDepth(): Promise<number>;
    derive(childPath: string, options?: RvfOptions): Promise<RvfBackend>;
    branch(childPath: string): Promise<RvfBackend>;
    freeze(): Promise<number>;
    embedKernel(arch: number, kernelType: number, flags: number, image: Uint8Array, apiPort: number, cmdline?: string): Promise<number>;
    extractKernel(): Promise<RvfKernelData | null>;
    embedEbpf(programType: number, attachType: number, maxDimension: number, bytecode: Uint8Array, btf?: Uint8Array): Promise<number>;
    extractEbpf(): Promise<RvfEbpfData | null>;
    segments(): Promise<RvfSegmentInfo[]>;
    dimension(): Promise<number>;
    metric(): Promise<DistanceMetric>;
    exportBytes(): Promise<Uint8Array>;
    openBytes(_bytes: Uint8Array): Promise<void>;
    /**
     * Get or allocate a numeric label for a string ID.
     * If the ID was already seen, returns the existing label.
     */
    private resolveLabel;
    /** Path to the sidecar mappings file. */
    private mappingsPath;
    /** Journal ops for labels allocated since `fromLabel` (labels are sequential). */
    private newLabelOps;
    /**
     * Append ops to the id-map journal. If the append fails, labels allocated
     * since `rollbackTo` are released so memory never runs ahead of disk.
     */
    private persistOps;
    /** Fold the journal into a fresh snapshot once it outgrows the snapshot. */
    private maybeCompactJournal;
    private closeJournal;
    /**
     * Persist the string↔label mapping to a sidecar JSON file.
     *
     * `delete()` resolves string ids through this map and silently filters out
     * anything unresolvable, so a lost or torn write turns every ingest since
     * the last good save into an undeletable-by-id vector. Persistence is
     * therefore NOT best-effort: the write is made atomic (temp file + rename,
     * so a crash/ENOSPC mid-write can never leave partial JSON at `mp`) and a
     * failure is surfaced rather than swallowed.
     */
    private saveMappings;
    private cleanupFailedChild;
    /**
     * Load the string↔label mapping from the sidecar JSON file if it exists.
     *
     * A corrupt sidecar must NOT degrade to empty maps: `nextLabel` would reset
     * to 1 and subsequent ingests would assign labels colliding with existing
     * vectors (silent data corruption), and the next `saveMappings()` would
     * overwrite the recoverable file. Instead the corrupt sidecar is quarantined
     * (renamed aside so it is not clobbered) and a `SidecarCorrupt` error is
     * raised so the caller learns string-id operations are unsafe.
     */
    private loadMappings;
    /**
     * Load the metadata field-name registry from the store's file metadata
     * (issue #704). Stores written before the registry existed, or by an
     * rvf-node without file-metadata support, start with an empty registry.
     */
    private loadFieldRegistry;
    /**
     * Encode entry metadata into native per-vector records, persisting any
     * newly registered field names before the vectors are committed.
     * Returns null when no entry carries metadata.
     */
    private encodeMetadataRecords;
    /** Decode a native hit's metadata entries through the field registry. */
    private decodeHitMetadata;
    /**
     * Replay `<sidecar>.journal` over the loaded snapshot. A torn last append
     * (crash mid-write) is dropped, and cut off on read-write opens so the next
     * append cannot fuse with it; any other damage fails loud like a corrupt
     * snapshot does.
     */
    private replayJournalFile;
    private loadSnapshot;
}
/**
 * Backend that delegates to the `@ruvector/rvf-wasm` WASM build.
 *
 * The WASM microkernel exposes C-ABI store functions (`rvf_store_create`,
 * `rvf_store_query`, etc.) operating on integer handles. This backend wraps
 * them behind the same `RvfBackend` interface.
 *
 * Suitable for browser environments. The WASM module is loaded lazily.
 */
export declare class WasmBackend implements RvfBackend {
    private wasm;
    /** Integer store handle returned by `rvf_store_create` / `rvf_store_open`. */
    private handle;
    private dim;
    /** Metric given to create(); unknown for stores loaded with openBytes(). */
    private storeMetric;
    private loadWasm;
    private ensureHandle;
    private metricCode;
    create(_path: string, options: RvfOptions): Promise<void>;
    open(_path: string): Promise<void>;
    openReadonly(_path: string): Promise<void>;
    ingestBatch(entries: RvfIngestEntry[]): Promise<RvfIngestResult>;
    query(vector: Float32Array, k: number, _options?: RvfQueryOptions): Promise<RvfSearchResult[]>;
    delete(ids: string[]): Promise<RvfDeleteResult>;
    deleteByFilter(_filter: RvfFilterExpr): Promise<RvfDeleteResult>;
    compact(): Promise<RvfCompactionResult>;
    status(): Promise<RvfStatus>;
    close(): Promise<void>;
    fileId(): Promise<string>;
    parentId(): Promise<string>;
    lineageDepth(): Promise<number>;
    derive(_childPath: string, _options?: RvfOptions): Promise<RvfBackend>;
    branch(_childPath: string): Promise<RvfBackend>;
    freeze(): Promise<number>;
    embedKernel(): Promise<number>;
    extractKernel(): Promise<RvfKernelData | null>;
    embedEbpf(): Promise<number>;
    extractEbpf(): Promise<RvfEbpfData | null>;
    segments(): Promise<RvfSegmentInfo[]>;
    dimension(): Promise<number>;
    metric(): Promise<DistanceMetric>;
    /**
     * Serialize the in-memory store to `.rvf` bytes via the `rvf_store_export`
     * C-ABI export. `rvf_store_export` follows a probe-then-write pattern:
     * called with a too-small (or zero-length) buffer it returns the negated
     * required size, so we probe first, allocate exactly that much, then
     * write for real.
     */
    exportBytes(): Promise<Uint8Array>;
    /** Load a store from `.rvf` bytes via the `rvf_store_open` C-ABI import. */
    openBytes(bytes: Uint8Array): Promise<void>;
}
/**
 * Resolve a `BackendType` to a concrete `RvfBackend` instance.
 *
 * - `'node'`  Always returns a `NodeBackend`.
 * - `'wasm'`  Always returns a `WasmBackend`.
 * - `'auto'`  Tries `node` first, falls back to `wasm`.
 */
export declare function resolveBackend(type: BackendType): RvfBackend;
