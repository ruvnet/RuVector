import type { RvfOptions, RvfQueryOptions, RvfSearchResult, RvfQueryOptionsWithCount, RvfIngestResult, RvfIngestEntry, RvfDeleteResult, RvfCompactionResult, RvfStatus, RvfFilterExpr, RvfKernelData, RvfEbpfData, RvfSegmentInfo, BackendType, DistanceMetric, DerivationType } from './types';
import type { RvfBackend } from './backend';
/**
 * Main user-facing RVF database class.
 *
 * Wraps a backend implementation (`NodeBackend` or `WasmBackend`) and exposes
 * an ergonomic async API that mirrors the Rust `RvfStore` surface.
 *
 * Use the static factory methods (`create`, `open`, `openReadonly`) to obtain
 * an instance. Do not construct directly.
 */
export declare class RvfDatabase {
    private backend;
    private closed;
    private constructor();
    /**
     * Create a new RVF store at `path`.
     *
     * @param path      File path for the new store.
     * @param options   Store creation options (dimensions is required).
     * @param backend   Backend to use. Default: `'auto'`.
     */
    static create(path: string, options: RvfOptions, backend?: BackendType): Promise<RvfDatabase>;
    /**
     * Open an existing RVF store for read-write access.
     *
     * @param path      File path to an existing `.rvf` file.
     * @param backend   Backend to use. Default: `'auto'`.
     */
    static open(path: string, backend?: BackendType): Promise<RvfDatabase>;
    /**
     * Open an existing RVF store for read-only access (no lock required).
     *
     * @param path      File path to an existing `.rvf` file.
     * @param backend   Backend to use. Default: `'auto'`.
     */
    static openReadonly(path: string, backend?: BackendType): Promise<RvfDatabase>;
    /**
     * Open a store from an in-memory `.rvf` byte buffer.
     *
     * Primarily for the WASM backend, which has no filesystem access — this is
     * the supported way to durably persist a browser-side store (e.g. to
     * IndexedDB/OPFS) and reload it later. The node backend does not support
     * this and will throw.
     *
     * @param bytes    A `.rvf` byte buffer, previously produced by `exportBytes()`.
     * @param backend  Backend to use. Default: `'auto'`.
     */
    static openBytes(bytes: Uint8Array, backend?: BackendType): Promise<RvfDatabase>;
    /**
     * Create an RvfDatabase from an already-initialized backend.
     *
     * Used internally (e.g. by `derive()`) to wrap a child backend that was
     * created by the native layer without going through the normal open/create
     * flow.
     */
    static fromBackend(backend: RvfBackend): RvfDatabase;
    /**
     * Ingest a batch of vectors into the store.
     *
     * @param entries  Array of `{ id, vector, metadata? }` entries.
     * @returns        Counts of accepted/rejected vectors and the new epoch.
     */
    ingestBatch(entries: RvfIngestEntry[]): Promise<RvfIngestResult>;
    /**
     * Soft-delete vectors by their IDs.
     *
     * @param ids  Vector IDs to delete.
     */
    delete(ids: string[]): Promise<RvfDeleteResult>;
    /**
     * Soft-delete all vectors matching a filter expression.
     *
     * @param filter  The filter to match against vector metadata.
     */
    deleteByFilter(filter: RvfFilterExpr): Promise<RvfDeleteResult>;
    /**
     * Query for the `k` nearest neighbors of a given vector.
     *
     * The count can be passed positionally (`query(vec, 10)`) or via an
     * options object (`query(vec, { k: 10, efSearch: 200 })`, with `topK`
     * and `limit` accepted as aliases for `k`). Conflicting aliases, an object
     * without a usable count, or a count outside the positive `u32` range throws a clear
     * {@link RvfErrorCode.InvalidArgument} rather than a low-level N-API error.
     *
     * @param vector   The query embedding.
     * @param k        Number of results, or an options object carrying it.
     * @param options  Optional query parameters (efSearch, filter, timeout).
     * @returns        Sorted search results (closest first).
     */
    query(vector: Float32Array | number[], k: number | RvfQueryOptionsWithCount, options?: RvfQueryOptions): Promise<RvfSearchResult[]>;
    /**
     * Run compaction to reclaim dead space from soft-deleted vectors.
     */
    compact(): Promise<RvfCompactionResult>;
    /**
     * Get the current store status (vector count, file size, epoch, etc.).
     */
    status(): Promise<RvfStatus>;
    /** Get this file's unique identifier as a hex string. */
    fileId(): Promise<string>;
    /** Get the parent file's identifier as a hex string (all zeros if root). */
    parentId(): Promise<string>;
    /** Get the lineage depth (0 for root files). */
    lineageDepth(): Promise<number>;
    /**
     * Derive a child store from this parent.
     *
     * Creates a new RVF file at `childPath` that records this store as its
     * parent for provenance tracking. Returns a new `RvfDatabase` wrapping
     * the child store.
     *
     * `derivationType` is recorded in the child (default `'filter'`); it
     * needs `@ruvector/rvf-node` >= 0.4.7. `'merge'` is recorded with this
     * store as the only parent.
     */
    derive(childPath: string, options?: RvfOptions, derivationType?: DerivationType): Promise<RvfDatabase>;
    /**
     * How this store was derived from its parent(s), or `null` for a root
     * store or one derived before `@ruvector/rvf-node` 0.4.7.
     */
    derivationType(): Promise<DerivationType | null>;
    /**
     * Create a durable copy-on-write branch that reads inherited vectors from
     * this store and persists only child edits.
     */
    branch(childPath: string): Promise<RvfDatabase>;
    /**
     * Freeze this generation before branching. Returns the frozen manifest
     * epoch; subsequent writes through this handle are rejected.
     */
    freeze(): Promise<number>;
    /** Embed a kernel image. Returns the segment ID. */
    embedKernel(arch: number, kernelType: number, flags: number, image: Uint8Array, apiPort: number, cmdline?: string): Promise<number>;
    /** Extract the kernel image. Returns null if not present. */
    extractKernel(): Promise<RvfKernelData | null>;
    /** Embed an eBPF program. Returns the segment ID. */
    embedEbpf(programType: number, attachType: number, maxDimension: number, bytecode: Uint8Array, btf?: Uint8Array): Promise<number>;
    /** Extract the eBPF program. Returns null if not present. */
    extractEbpf(): Promise<RvfEbpfData | null>;
    /** Get the list of segments in the store. */
    segments(): Promise<RvfSegmentInfo[]>;
    /** Get the vector dimensionality. */
    dimension(): Promise<number>;
    /** Get the distance metric the store was created with. */
    metric(): Promise<DistanceMetric>;
    /**
     * Serialize the store to an in-memory `.rvf` byte buffer.
     *
     * Use with `RvfDatabase.openBytes()` to durably persist a WASM-backed
     * (browser) store — e.g. writing the result to IndexedDB/OPFS. Not
     * supported by the node backend (which already persists to a file path).
     */
    exportBytes(): Promise<Uint8Array>;
    /**
     * Close the store, releasing the writer lock and flushing pending data.
     * On the Node backend, this may build and persist an HNSW index for a large
     * ingest-only store. Await close before opening the file read-only if the
     * first query must use the persisted index.
     *
     * After calling `close()`, all other methods will throw `RvfError` with
     * code `StoreClosed`.
     */
    close(): Promise<void>;
    /** True if the store has been closed. */
    get isClosed(): boolean;
    private ensureOpen;
}
