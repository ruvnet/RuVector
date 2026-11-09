"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RvfDatabase = void 0;
const backend_1 = require("./backend");
const errors_1 = require("./errors");
/**
 * Main user-facing RVF database class.
 *
 * Wraps a backend implementation (`NodeBackend` or `WasmBackend`) and exposes
 * an ergonomic async API that mirrors the Rust `RvfStore` surface.
 *
 * Use the static factory methods (`create`, `open`, `openReadonly`) to obtain
 * an instance. Do not construct directly.
 */
class RvfDatabase {
    constructor(backend) {
        this.closed = false;
        this.backend = backend;
    }
    // -----------------------------------------------------------------------
    // Factory methods
    // -----------------------------------------------------------------------
    /**
     * Create a new RVF store at `path`.
     *
     * @param path      File path for the new store.
     * @param options   Store creation options (dimensions is required).
     * @param backend   Backend to use. Default: `'auto'`.
     */
    static async create(path, options, backend = 'auto') {
        const impl = (0, backend_1.resolveBackend)(backend);
        await impl.create(path, options);
        return new RvfDatabase(impl);
    }
    /**
     * Open an existing RVF store for read-write access.
     *
     * @param path      File path to an existing `.rvf` file.
     * @param backend   Backend to use. Default: `'auto'`.
     */
    static async open(path, backend = 'auto') {
        const impl = (0, backend_1.resolveBackend)(backend);
        await impl.open(path);
        return new RvfDatabase(impl);
    }
    /**
     * Open an existing RVF store for read-only access (no lock required).
     *
     * @param path      File path to an existing `.rvf` file.
     * @param backend   Backend to use. Default: `'auto'`.
     */
    static async openReadonly(path, backend = 'auto') {
        const impl = (0, backend_1.resolveBackend)(backend);
        await impl.openReadonly(path);
        return new RvfDatabase(impl);
    }
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
    static async openBytes(bytes, backend = 'auto') {
        const impl = (0, backend_1.resolveBackend)(backend);
        await impl.openBytes(bytes);
        return new RvfDatabase(impl);
    }
    /**
     * Create an RvfDatabase from an already-initialized backend.
     *
     * Used internally (e.g. by `derive()`) to wrap a child backend that was
     * created by the native layer without going through the normal open/create
     * flow.
     */
    static fromBackend(backend) {
        return new RvfDatabase(backend);
    }
    // -----------------------------------------------------------------------
    // Write operations
    // -----------------------------------------------------------------------
    /**
     * Ingest a batch of vectors into the store.
     *
     * @param entries  Array of `{ id, vector, metadata? }` entries.
     * @returns        Counts of accepted/rejected vectors and the new epoch.
     */
    async ingestBatch(entries) {
        this.ensureOpen();
        return this.backend.ingestBatch(entries);
    }
    /**
     * Soft-delete vectors by their IDs.
     *
     * @param ids  Vector IDs to delete.
     */
    async delete(ids) {
        this.ensureOpen();
        return this.backend.delete(ids);
    }
    /**
     * Soft-delete all vectors matching a filter expression.
     *
     * @param filter  The filter to match against vector metadata.
     */
    async deleteByFilter(filter) {
        this.ensureOpen();
        return this.backend.deleteByFilter(filter);
    }
    // -----------------------------------------------------------------------
    // Read operations
    // -----------------------------------------------------------------------
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
    async query(vector, k, options) {
        this.ensureOpen();
        const { count, queryOptions } = normalizeQueryArgs(k, options);
        const f32 = vector instanceof Float32Array ? vector : new Float32Array(vector);
        return this.backend.query(f32, count, queryOptions);
    }
    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------
    /**
     * Run compaction to reclaim dead space from soft-deleted vectors.
     */
    async compact() {
        this.ensureOpen();
        return this.backend.compact();
    }
    /**
     * Get the current store status (vector count, file size, epoch, etc.).
     */
    async status() {
        this.ensureOpen();
        return this.backend.status();
    }
    // -----------------------------------------------------------------------
    // Lineage
    // -----------------------------------------------------------------------
    /** Get this file's unique identifier as a hex string. */
    async fileId() {
        this.ensureOpen();
        return this.backend.fileId();
    }
    /** Get the parent file's identifier as a hex string (all zeros if root). */
    async parentId() {
        this.ensureOpen();
        return this.backend.parentId();
    }
    /** Get the lineage depth (0 for root files). */
    async lineageDepth() {
        this.ensureOpen();
        return this.backend.lineageDepth();
    }
    /**
     * Derive a child store from this parent.
     *
     * Creates a new RVF file at `childPath` that records this store as its
     * parent for provenance tracking. Returns a new `RvfDatabase` wrapping
     * the child store.
     */
    async derive(childPath, options) {
        this.ensureOpen();
        const childBackend = await this.backend.derive(childPath, options);
        return RvfDatabase.fromBackend(childBackend);
    }
    /**
     * Create a durable copy-on-write branch that reads inherited vectors from
     * this store and persists only child edits.
     */
    async branch(childPath) {
        this.ensureOpen();
        const childBackend = await this.backend.branch(childPath);
        return RvfDatabase.fromBackend(childBackend);
    }
    /**
     * Freeze this generation before branching. Returns the frozen manifest
     * epoch; subsequent writes through this handle are rejected.
     */
    async freeze() {
        this.ensureOpen();
        return this.backend.freeze();
    }
    // -----------------------------------------------------------------------
    // Kernel / eBPF
    // -----------------------------------------------------------------------
    /** Embed a kernel image. Returns the segment ID. */
    async embedKernel(arch, kernelType, flags, image, apiPort, cmdline) {
        this.ensureOpen();
        return this.backend.embedKernel(arch, kernelType, flags, image, apiPort, cmdline);
    }
    /** Extract the kernel image. Returns null if not present. */
    async extractKernel() {
        this.ensureOpen();
        return this.backend.extractKernel();
    }
    /** Embed an eBPF program. Returns the segment ID. */
    async embedEbpf(programType, attachType, maxDimension, bytecode, btf) {
        this.ensureOpen();
        return this.backend.embedEbpf(programType, attachType, maxDimension, bytecode, btf);
    }
    /** Extract the eBPF program. Returns null if not present. */
    async extractEbpf() {
        this.ensureOpen();
        return this.backend.extractEbpf();
    }
    // -----------------------------------------------------------------------
    // Inspection
    // -----------------------------------------------------------------------
    /** Get the list of segments in the store. */
    async segments() {
        this.ensureOpen();
        return this.backend.segments();
    }
    /** Get the vector dimensionality. */
    async dimension() {
        this.ensureOpen();
        return this.backend.dimension();
    }
    /** Get the distance metric the store was created with. */
    async metric() {
        this.ensureOpen();
        if (!this.backend.metric) {
            throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'this backend does not report its metric');
        }
        return this.backend.metric();
    }
    // -----------------------------------------------------------------------
    // Byte-level persistence (WASM backend)
    // -----------------------------------------------------------------------
    /**
     * Serialize the store to an in-memory `.rvf` byte buffer.
     *
     * Use with `RvfDatabase.openBytes()` to durably persist a WASM-backed
     * (browser) store — e.g. writing the result to IndexedDB/OPFS. Not
     * supported by the node backend (which already persists to a file path).
     */
    async exportBytes() {
        this.ensureOpen();
        return this.backend.exportBytes();
    }
    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------
    /**
     * Close the store, releasing the writer lock and flushing pending data.
     * On the Node backend, this may build and persist an HNSW index for a large
     * ingest-only store. Await close before opening the file read-only if the
     * first query must use the persisted index.
     *
     * After calling `close()`, all other methods will throw `RvfError` with
     * code `StoreClosed`.
     */
    async close() {
        if (this.closed)
            return;
        this.closed = true;
        await this.backend.close();
    }
    /** True if the store has been closed. */
    get isClosed() {
        return this.closed;
    }
    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------
    ensureOpen() {
        if (this.closed) {
            throw new errors_1.RvfError(errors_1.RvfErrorCode.StoreClosed);
        }
    }
}
exports.RvfDatabase = RvfDatabase;
/**
 * Resolve the `(k, options)` arguments of `query()` into a validated result
 * count and query options, accepting both the positional (`k: number`) and
 * object (`{ k | topK | limit, ...options }`) call forms.
 *
 * Throws {@link RvfErrorCode.InvalidArgument} for an object with no usable
 * count or for a count that is not a positive integer — a clear SDK-level
 * error instead of the low-level N-API "Failed to convert napi value Object
 * into rust type `u32`" that leaks through otherwise.
 */
function normalizeQueryArgs(k, options) {
    let count;
    let queryOptions = options;
    if (typeof k === 'object' && k !== null) {
        const { k: kk, topK, limit, ...rest } = k;
        const aliases = [kk, topK, limit].filter((value) => value !== undefined);
        if (aliases.length === 0) {
            throw new errors_1.RvfError(errors_1.RvfErrorCode.InvalidArgument, 'query() options object must specify a result count as `k`, `topK`, or `limit`');
        }
        if (aliases.some((value) => value !== aliases[0])) {
            throw new errors_1.RvfError(errors_1.RvfErrorCode.InvalidArgument, '`k`, `topK`, and `limit` must agree when more than one is supplied');
        }
        count = aliases[0];
        // Merge object-form options with any explicit third argument (explicit wins).
        queryOptions = { ...rest, ...options };
    }
    else {
        count = k;
    }
    if (typeof count !== 'number' ||
        !Number.isSafeInteger(count) ||
        count <= 0 ||
        count > 4294967295) {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.InvalidArgument, `query() result count must be a positive u32 integer, got ${String(count)}`);
    }
    return { count, queryOptions };
}
//# sourceMappingURL=database.js.map