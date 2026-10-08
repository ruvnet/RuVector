"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.WasmBackend = exports.NodeBackend = void 0;
exports.resolveBackend = resolveBackend;
const errors_1 = require("./errors");
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
class NodeBackend {
    constructor() {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        this.native = null;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        this.handle = null;
        // String ID <-> Numeric Label mappings (N-API layer requires i64 labels)
        this.idToLabel = new Map();
        this.labelToId = new Map();
        this.nextLabel = 1; // RVF uses 1-based labels
        this.storePath = '';
    }
    async loadNative() {
        if (this.native)
            return;
        try {
            // Dynamic import so the SDK can be bundled for browsers without
            // pulling in the native addon at compile time.
            // The NAPI addon exports a `RvfDatabase` class with factory methods.
            const mod = await Promise.resolve().then(() => __importStar(require('@ruvector/rvf-node')));
            this.native = mod.RvfDatabase ?? mod.default?.RvfDatabase ?? mod;
        }
        catch {
            throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'Could not load @ruvector/rvf-node — is it installed?');
        }
    }
    ensureHandle() {
        if (!this.handle) {
            throw new errors_1.RvfError(errors_1.RvfErrorCode.StoreClosed);
        }
    }
    /** Release a native handle without persisting mappings after open failed. */
    discardHandle() {
        try {
            this.handle?.close();
        }
        catch {
            // Preserve the original open/sidecar error.
        }
        finally {
            this.handle = null;
            this.storePath = '';
            this.idToLabel.clear();
            this.labelToId.clear();
            this.nextLabel = 1;
        }
    }
    async create(path, options) {
        await this.loadNative();
        const fs = await Promise.resolve().then(() => __importStar(require('fs')));
        const sidecarPath = `${path}.idmap.json`;
        const existingPaths = [path, sidecarPath].filter((candidate) => fs.existsSync(candidate));
        const backups = [];
        try {
            // Precondition: refuse to clobber an existing file unless asked to.
            // The native layer surfaces this as a misleading FsyncFailed, so check
            // here and raise a clear, actionable error. An orphaned sidecar also
            // blocks creation because silently retaining it can desynchronize IDs.
            if (existingPaths.length > 0 && !options.overwrite) {
                throw new errors_1.RvfError(errors_1.RvfErrorCode.FileExists, `${existingPaths.join(', ')} already exists; use RvfDatabase.open() to reuse the store, or pass { overwrite: true } to replace it`);
            }
            if (options.overwrite) {
                const { randomUUID } = await Promise.resolve().then(() => __importStar(require('crypto')));
                for (const original of existingPaths) {
                    if (!fs.lstatSync(original).isFile()) {
                        throw new errors_1.RvfError(errors_1.RvfErrorCode.InvalidArgument, `refusing to overwrite non-file path ${original}`);
                    }
                    const backup = `${original}.overwrite-${process.pid}-${randomUUID()}.bak`;
                    fs.renameSync(original, backup);
                    backups.push({ original, backup });
                }
            }
            this.handle = await this.native.create(path, mapOptionsToNative(options));
            this.storePath = path;
            this.idToLabel.clear();
            this.labelToId.clear();
            this.nextLabel = 1;
            for (const { backup } of backups) {
                try {
                    fs.rmSync(backup, { force: true });
                }
                catch {
                    // The new store is valid; retain the recoverable backup if cleanup fails.
                }
            }
        }
        catch (err) {
            // If replacement fails after moving the old store aside, remove any
            // partially-created replacement and restore the original files.
            if (backups.length > 0) {
                try {
                    fs.rmSync(path, { force: true });
                    fs.rmSync(sidecarPath, { force: true });
                }
                catch {
                    // Continue restoring every backup that can be recovered.
                }
                for (const { original, backup } of backups.reverse()) {
                    try {
                        if (fs.existsSync(backup))
                            fs.renameSync(backup, original);
                    }
                    catch {
                        // The backup remains on disk with a unique, discoverable suffix.
                    }
                }
            }
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async open(path) {
        await this.loadNative();
        try {
            this.handle = await this.native.open(path);
            this.storePath = path;
            await this.loadMappings();
        }
        catch (err) {
            this.discardHandle();
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async openReadonly(path) {
        await this.loadNative();
        try {
            this.handle = await this.native.openReadonly(path);
            this.storePath = path;
            await this.loadMappings();
        }
        catch (err) {
            this.discardHandle();
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async ingestBatch(entries) {
        this.ensureHandle();
        try {
            rejectUnsupportedMetadata(entries);
            // NAPI signature: ingestBatch(vectors: Float32Array, ids: i64[], metadata?)
            // Flatten individual vectors into a single contiguous Float32Array.
            const n = entries.length;
            if (n === 0)
                return { accepted: 0, rejected: 0, epoch: 0 };
            const first = entries[0].vector;
            const dim = first instanceof Float32Array ? first.length : first.length;
            const flat = new Float32Array(n * dim);
            for (let i = 0; i < n; i++) {
                const v = entries[i].vector;
                const f32 = v instanceof Float32Array ? v : new Float32Array(v);
                flat.set(f32, i * dim);
            }
            // Map string IDs to numeric labels for the N-API layer.
            // The native Rust HNSW expects i64 labels — non-numeric strings cause
            // silent data loss (NaN → dropped).  We maintain a bidirectional
            // string↔label mapping and persist it as a sidecar JSON file.
            const ids = entries.map((e) => this.resolveLabel(e.id));
            const result = this.handle.ingestBatch(flat, ids);
            // Persist mappings after every ingest so they survive crashes.
            await this.saveMappings();
            return {
                accepted: Number(result.accepted),
                rejected: Number(result.rejected),
                epoch: result.epoch,
            };
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async query(vector, k, options) {
        this.ensureHandle();
        try {
            const nativeOpts = options ? mapQueryOptionsToNative(options) : undefined;
            const results = this.handle.query(vector, k, nativeOpts);
            // Map numeric labels back to original string IDs.
            return results.map((r) => ({
                id: this.labelToId.get(Number(r.id)) ?? String(r.id),
                distance: r.distance,
            }));
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async delete(ids) {
        this.ensureHandle();
        try {
            // Resolve string IDs to numeric labels for the N-API layer.
            const numIds = ids
                .map((id) => this.idToLabel.get(id))
                .filter((label) => label !== undefined);
            if (numIds.length === 0) {
                return { deleted: 0, epoch: 0 };
            }
            const result = this.handle.delete(numIds);
            // Remove deleted entries from the mapping.
            for (const id of ids) {
                const label = this.idToLabel.get(id);
                if (label !== undefined) {
                    this.idToLabel.delete(id);
                    this.labelToId.delete(label);
                }
            }
            await this.saveMappings();
            return { deleted: Number(result.deleted), epoch: result.epoch };
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async deleteByFilter(filter) {
        this.ensureHandle();
        try {
            // NAPI takes a JSON string for the filter expression.
            const result = this.handle.deleteByFilter(JSON.stringify(filter));
            return { deleted: Number(result.deleted), epoch: result.epoch };
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async compact() {
        this.ensureHandle();
        try {
            const result = this.handle.compact();
            return {
                segmentsCompacted: result.segmentsCompacted ?? result.segments_compacted,
                bytesReclaimed: Number(result.bytesReclaimed ?? result.bytes_reclaimed),
                epoch: result.epoch,
            };
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async status() {
        this.ensureHandle();
        try {
            const s = this.handle.status();
            return mapNativeStatus(s);
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async close() {
        if (!this.handle)
            return;
        let failure;
        try {
            await this.saveMappings();
        }
        catch (err) {
            failure = err;
        }
        try {
            this.handle.close();
        }
        catch (err) {
            failure ?? (failure = err);
        }
        finally {
            this.handle = null;
            this.idToLabel.clear();
            this.labelToId.clear();
            this.nextLabel = 1;
            this.storePath = '';
        }
        if (failure)
            throw errors_1.RvfError.fromNative(failure);
    }
    async fileId() {
        this.ensureHandle();
        try {
            return this.handle.fileId();
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async parentId() {
        this.ensureHandle();
        try {
            return this.handle.parentId();
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async lineageDepth() {
        this.ensureHandle();
        try {
            return this.handle.lineageDepth();
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async derive(childPath, options) {
        this.ensureHandle();
        let childHandle = null;
        try {
            const nativeOpts = options ? mapOptionsToNative(options) : undefined;
            childHandle = this.handle.derive(childPath, nativeOpts);
            const child = new NodeBackend();
            child.native = this.native;
            child.handle = childHandle;
            child.storePath = childPath;
            // Copy parent mappings to child (COW semantics)
            child.idToLabel = new Map(this.idToLabel);
            child.labelToId = new Map(this.labelToId);
            child.nextLabel = this.nextLabel;
            await child.saveMappings();
            return child;
        }
        catch (err) {
            await this.cleanupFailedChild(childHandle, childPath);
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async branch(childPath) {
        this.ensureHandle();
        let childHandle = null;
        try {
            childHandle = this.handle.branch(childPath);
            const child = new NodeBackend();
            child.native = this.native;
            child.handle = childHandle;
            child.storePath = childPath;
            child.idToLabel = new Map(this.idToLabel);
            child.labelToId = new Map(this.labelToId);
            child.nextLabel = this.nextLabel;
            await child.saveMappings();
            return child;
        }
        catch (err) {
            await this.cleanupFailedChild(childHandle, childPath);
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async freeze() {
        this.ensureHandle();
        try {
            return this.handle.freeze();
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async embedKernel(arch, kernelType, flags, image, apiPort, cmdline) {
        this.ensureHandle();
        try {
            return this.handle.embedKernel(arch, kernelType, flags, Buffer.from(image), apiPort, cmdline);
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async extractKernel() {
        this.ensureHandle();
        try {
            const result = this.handle.extractKernel();
            if (!result)
                return null;
            return {
                header: new Uint8Array(result.header),
                image: new Uint8Array(result.image),
            };
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async embedEbpf(programType, attachType, maxDimension, bytecode, btf) {
        this.ensureHandle();
        try {
            return this.handle.embedEbpf(programType, attachType, maxDimension, Buffer.from(bytecode), btf ? Buffer.from(btf) : undefined);
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async extractEbpf() {
        this.ensureHandle();
        try {
            const result = this.handle.extractEbpf();
            if (!result)
                return null;
            return {
                header: new Uint8Array(result.header),
                payload: new Uint8Array(result.payload),
            };
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async segments() {
        this.ensureHandle();
        try {
            const segs = this.handle.segments();
            return segs.map((s) => ({
                id: s.id,
                offset: s.offset,
                payloadLength: s.payloadLength ?? s.payload_length,
                segType: s.segType ?? s.seg_type,
            }));
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async dimension() {
        this.ensureHandle();
        try {
            return this.handle.dimension();
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async exportBytes() {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'exportBytes is not supported by the node backend — use a file path with create()/open() instead');
    }
    async openBytes(_bytes) {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'openBytes is not supported by the node backend — use a file path with open() instead');
    }
    // ─── String ID ↔ Numeric Label mapping helpers ───
    /**
     * Get or allocate a numeric label for a string ID.
     * If the ID was already seen, returns the existing label.
     */
    resolveLabel(id) {
        let label = this.idToLabel.get(id);
        if (label !== undefined)
            return label;
        label = this.nextLabel++;
        this.idToLabel.set(id, label);
        this.labelToId.set(label, id);
        return label;
    }
    /** Path to the sidecar mappings file. */
    mappingsPath() {
        return this.storePath ? this.storePath + '.idmap.json' : '';
    }
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
    async saveMappings() {
        const mp = this.mappingsPath();
        if (!mp)
            return;
        const fs = await Promise.resolve().then(() => __importStar(require('fs')));
        const data = JSON.stringify({
            idToLabel: Object.fromEntries(this.idToLabel),
            labelToId: Object.fromEntries(Array.from(this.labelToId.entries()).map(([k, v]) => [String(k), v])),
            nextLabel: this.nextLabel,
        });
        // A shared `${mp}.tmp` races across processes opening the same store.
        // Synchronous writes cannot interleave within one Node process; PID plus
        // timestamp also keeps independently-running writers on separate paths.
        const tmp = `${mp}.${process.pid}.${Date.now()}.tmp`;
        try {
            fs.writeFileSync(tmp, data, 'utf-8');
            fs.renameSync(tmp, mp);
        }
        catch (err) {
            try {
                fs.rmSync(tmp, { force: true });
            }
            catch {
                // best-effort cleanup of the temp file
            }
            throw new errors_1.RvfError(errors_1.RvfErrorCode.SidecarWriteFailed, `at ${mp}: ${err instanceof Error ? err.message : String(err)}`);
        }
    }
    async cleanupFailedChild(childHandle, childPath) {
        // If native creation failed before returning a handle (for example because
        // childPath already exists), the path is not ours to remove.
        if (!childHandle)
            return;
        try {
            childHandle?.close();
        }
        catch {
            // Preserve the original branch/sidecar error.
        }
        try {
            const fs = await Promise.resolve().then(() => __importStar(require('fs')));
            fs.rmSync(childPath, { force: true });
            fs.rmSync(`${childPath}.idmap.json`, { force: true });
        }
        catch {
            // Best-effort rollback; the original operation error remains primary.
        }
    }
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
    async loadMappings() {
        const mp = this.mappingsPath();
        if (!mp)
            return;
        const fs = await Promise.resolve().then(() => __importStar(require('fs')));
        if (!fs.existsSync(mp))
            return; // fresh store: no sidecar yet is legitimate
        let parsed;
        try {
            const candidate = JSON.parse(fs.readFileSync(mp, 'utf-8'));
            if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
                throw new TypeError('sidecar root must be an object');
            }
            const raw = candidate;
            if (!raw.idToLabel ||
                typeof raw.idToLabel !== 'object' ||
                Array.isArray(raw.idToLabel) ||
                !raw.labelToId ||
                typeof raw.labelToId !== 'object' ||
                Array.isArray(raw.labelToId) ||
                !Number.isSafeInteger(raw.nextLabel) ||
                raw.nextLabel < 1) {
                throw new TypeError('sidecar must contain idToLabel, labelToId, and a positive nextLabel');
            }
            const idToLabel = raw.idToLabel;
            const labelToId = raw.labelToId;
            let maxLabel = 0;
            for (const [id, label] of Object.entries(idToLabel)) {
                if (!Number.isSafeInteger(label) || label < 1) {
                    throw new TypeError(`invalid label for id ${JSON.stringify(id)}`);
                }
                if (labelToId[String(label)] !== id) {
                    throw new TypeError(`idToLabel/labelToId mismatch for id ${JSON.stringify(id)}`);
                }
                maxLabel = Math.max(maxLabel, label);
            }
            for (const [label, id] of Object.entries(labelToId)) {
                const numericLabel = Number(label);
                if (!Number.isSafeInteger(numericLabel) ||
                    numericLabel < 1 ||
                    String(numericLabel) !== label ||
                    typeof id !== 'string' ||
                    idToLabel[id] !== numericLabel) {
                    throw new TypeError(`invalid reverse mapping for label ${JSON.stringify(label)}`);
                }
            }
            if (raw.nextLabel <= maxLabel) {
                throw new TypeError('nextLabel must be greater than every allocated label');
            }
            parsed = {
                idToLabel: idToLabel,
                labelToId: labelToId,
                nextLabel: raw.nextLabel,
            };
        }
        catch (err) {
            const { randomUUID } = await Promise.resolve().then(() => __importStar(require('crypto')));
            const quarantine = `${mp}.corrupt-${Date.now()}-${randomUUID()}`;
            try {
                fs.renameSync(mp, quarantine);
            }
            catch {
                // if we cannot move it aside, leave it in place — still fail loud
            }
            throw new errors_1.RvfError(errors_1.RvfErrorCode.SidecarCorrupt, `at ${mp} (quarantined to ${quarantine}): string-id delete()/ingest would ` +
                `silently corrupt data — restore a valid sidecar or recreate the store; ` +
                `${err instanceof Error ? err.message : String(err)}`);
        }
        this.idToLabel = new Map(Object.entries(parsed.idToLabel));
        this.labelToId = new Map(Object.entries(parsed.labelToId).map(([k, v]) => [Number(k), v]));
        this.nextLabel = parsed.nextLabel;
    }
}
exports.NodeBackend = NodeBackend;
// ---------------------------------------------------------------------------
// WasmBackend — wraps @ruvector/rvf-wasm
// ---------------------------------------------------------------------------
/**
 * Backend that delegates to the `@ruvector/rvf-wasm` WASM build.
 *
 * The WASM microkernel exposes C-ABI store functions (`rvf_store_create`,
 * `rvf_store_query`, etc.) operating on integer handles. This backend wraps
 * them behind the same `RvfBackend` interface.
 *
 * Suitable for browser environments. The WASM module is loaded lazily.
 */
class WasmBackend {
    constructor() {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        this.wasm = null;
        /** Integer store handle returned by `rvf_store_create` / `rvf_store_open`. */
        this.handle = 0;
        this.dim = 0;
    }
    async loadWasm() {
        if (this.wasm)
            return;
        try {
            const mod = await Promise.resolve().then(() => __importStar(require('@ruvector/rvf-wasm')));
            // wasm-pack default export is the init function
            if (typeof mod.default === 'function') {
                this.wasm = await mod.default();
            }
            else {
                this.wasm = mod;
            }
        }
        catch {
            throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'Could not load @ruvector/rvf-wasm — is it installed?');
        }
    }
    ensureHandle() {
        if (!this.handle) {
            throw new errors_1.RvfError(errors_1.RvfErrorCode.StoreClosed);
        }
    }
    metricCode(metric) {
        switch (metric) {
            case 'Cosine': return 2;
            case 'InnerProduct': return 1;
            default: return 0; // L2
        }
    }
    async create(_path, options) {
        await this.loadWasm();
        try {
            const nativeOpts = mapOptionsToNative(options);
            const dim = nativeOpts.dimension;
            const metric = this.metricCode(nativeOpts.metric);
            const h = this.wasm.rvf_store_create(dim, metric);
            if (h <= 0)
                throw new Error('rvf_store_create returned ' + h);
            this.handle = h;
            this.dim = dim;
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async open(_path) {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'WASM backend does not support file-based open (in-memory only)');
    }
    async openReadonly(_path) {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'WASM backend does not support file-based openReadonly (in-memory only)');
    }
    async ingestBatch(entries) {
        this.ensureHandle();
        try {
            rejectUnsupportedMetadata(entries);
            const n = entries.length;
            if (n === 0)
                return { accepted: 0, rejected: 0, epoch: 0 };
            const dim = this.dim || (entries[0].vector instanceof Float32Array
                ? entries[0].vector.length : entries[0].vector.length);
            const flat = new Float32Array(n * dim);
            const ids = new BigUint64Array(n);
            for (let i = 0; i < n; i++) {
                const v = entries[i].vector;
                const f32 = v instanceof Float32Array ? v : new Float32Array(v);
                flat.set(f32, i * dim);
                ids[i] = BigInt(entries[i].id);
            }
            // Allocate in WASM memory and call
            const vecsPtr = this.wasm.rvf_alloc(flat.byteLength);
            const idsPtr = this.wasm.rvf_alloc(ids.byteLength);
            new Float32Array(this.wasm.memory.buffer, vecsPtr, flat.length).set(flat);
            new BigUint64Array(this.wasm.memory.buffer, idsPtr, ids.length).set(ids);
            const accepted = this.wasm.rvf_store_ingest(this.handle, vecsPtr, idsPtr, n);
            this.wasm.rvf_free(vecsPtr, flat.byteLength);
            this.wasm.rvf_free(idsPtr, ids.byteLength);
            return { accepted: accepted > 0 ? accepted : 0, rejected: accepted < 0 ? n : 0, epoch: 0 };
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async query(vector, k, _options) {
        this.ensureHandle();
        try {
            const queryPtr = this.wasm.rvf_alloc(vector.byteLength);
            new Float32Array(this.wasm.memory.buffer, queryPtr, vector.length).set(vector);
            // Each result = 8 bytes id + 4 bytes dist = 12 bytes
            const outSize = k * 12;
            const outPtr = this.wasm.rvf_alloc(outSize);
            const count = this.wasm.rvf_store_query(this.handle, queryPtr, k, 0, outPtr);
            const results = [];
            const view = new DataView(this.wasm.memory.buffer);
            for (let i = 0; i < count; i++) {
                const off = outPtr + i * 12;
                const id = view.getBigUint64(off, true);
                const dist = view.getFloat32(off + 8, true);
                results.push({ id: String(id), distance: dist });
            }
            this.wasm.rvf_free(queryPtr, vector.byteLength);
            this.wasm.rvf_free(outPtr, outSize);
            return results;
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async delete(ids) {
        this.ensureHandle();
        try {
            const arr = new BigUint64Array(ids.map((id) => BigInt(id)));
            const ptr = this.wasm.rvf_alloc(arr.byteLength);
            new BigUint64Array(this.wasm.memory.buffer, ptr, arr.length).set(arr);
            const deleted = this.wasm.rvf_store_delete(this.handle, ptr, ids.length);
            this.wasm.rvf_free(ptr, arr.byteLength);
            return { deleted: deleted > 0 ? deleted : 0, epoch: 0 };
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async deleteByFilter(_filter) {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'deleteByFilter not supported in WASM backend');
    }
    async compact() {
        // There is no compaction export in the WASM C-ABI: `crates/rvf/rvf-wasm/src/lib.rs`
        // runs `rvf_store_create` .. `rvf_store_close` with no `rvf_store_compact`. Returning
        // `{segmentsCompacted: 0, bytesReclaimed: 0}` made "not implemented here" indistinguishable
        // from "ran, and there was nothing to reclaim" — and the WASM store DOES retain
        // soft-deleted entries (`export()` filters them), so a caller has real reason to expect
        // reclamation. Throwing matches how this class reports every other unsupported operation.
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'compact not supported in WASM backend');
    }
    async status() {
        this.ensureHandle();
        try {
            const outPtr = this.wasm.rvf_alloc(20);
            this.wasm.rvf_store_status(this.handle, outPtr);
            const view = new DataView(this.wasm.memory.buffer);
            const totalVectors = view.getUint32(outPtr, true);
            const dim = view.getUint32(outPtr + 4, true);
            // Offsets 12 and 16 were being allocated, written by WASM, and then discarded.
            // `crates/rvf/rvf-wasm/src/store.rs` writes the 20-byte buffer as:
            //   0 live · 4 dimension · 8 metric · 12 total entries · 16 deleted entries
            // `deleted / total` IS the dead-space ratio, so reporting a constant 0 here left every
            // caller that gates on the documented policy (dead_space_ratio > 0.20) permanently
            // inert — silently, because the field was present and plausible.
            const totalEntries = view.getUint32(outPtr + 12, true);
            const deletedEntries = view.getUint32(outPtr + 16, true);
            this.wasm.rvf_free(outPtr, 20);
            return {
                totalVectors,
                // Still fixed: the status buffer carries no segment count, file size or epoch, and
                // offset 12 is total VECTOR ENTRIES rather than segments. The in-memory WASM store
                // exports as a single VEC_SEG, so 1 is the honest logical answer here; the other two
                // remain limitations of this buffer, not values that could be derived from it.
                totalSegments: 1,
                fileSizeBytes: 0,
                epoch: 0,
                profileId: 0,
                compactionState: 'idle',
                deadSpaceRatio: totalEntries > 0 ? deletedEntries / totalEntries : 0,
                readOnly: false,
            };
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    async close() {
        if (!this.handle)
            return;
        try {
            this.wasm.rvf_store_close(this.handle);
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
        finally {
            this.handle = 0;
        }
    }
    async fileId() {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'fileId not supported in WASM backend');
    }
    async parentId() {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'parentId not supported in WASM backend');
    }
    async lineageDepth() {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'lineageDepth not supported in WASM backend');
    }
    async derive(_childPath, _options) {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'derive not supported in WASM backend');
    }
    async branch(_childPath) {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'branch not supported in WASM backend');
    }
    async freeze() {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'freeze not supported in WASM backend');
    }
    async embedKernel() {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'embedKernel not supported in WASM backend');
    }
    async extractKernel() {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'extractKernel not supported in WASM backend');
    }
    async embedEbpf() {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'embedEbpf not supported in WASM backend');
    }
    async extractEbpf() {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'extractEbpf not supported in WASM backend');
    }
    async segments() {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, 'segments not supported in WASM backend');
    }
    async dimension() {
        this.ensureHandle();
        const d = this.wasm.rvf_store_dimension(this.handle);
        if (d < 0)
            throw new errors_1.RvfError(errors_1.RvfErrorCode.StoreClosed);
        return d;
    }
    /**
     * Serialize the in-memory store to `.rvf` bytes via the `rvf_store_export`
     * C-ABI export. `rvf_store_export` follows a probe-then-write pattern:
     * called with a too-small (or zero-length) buffer it returns the negated
     * required size, so we probe first, allocate exactly that much, then
     * write for real.
     */
    async exportBytes() {
        this.ensureHandle();
        try {
            const probe = this.wasm.rvf_store_export(this.handle, 0, 0);
            const size = probe < 0 ? -probe : probe;
            if (size <= 0)
                return new Uint8Array(0);
            const ptr = this.wasm.rvf_alloc(size);
            try {
                const written = this.wasm.rvf_store_export(this.handle, ptr, size);
                if (written < 0) {
                    throw new Error(`rvf_store_export failed after size probe (size=${size})`);
                }
                return new Uint8Array(this.wasm.memory.buffer, ptr, written).slice();
            }
            finally {
                this.wasm.rvf_free(ptr, size);
            }
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
    /** Load a store from `.rvf` bytes via the `rvf_store_open` C-ABI import. */
    async openBytes(bytes) {
        await this.loadWasm();
        try {
            const ptr = this.wasm.rvf_alloc(bytes.byteLength);
            try {
                new Uint8Array(this.wasm.memory.buffer, ptr, bytes.byteLength).set(bytes);
                const h = this.wasm.rvf_store_open(ptr, bytes.byteLength);
                if (h <= 0)
                    throw new Error('rvf_store_open returned ' + h);
                this.handle = h;
                this.dim = this.wasm.rvf_store_dimension(h);
            }
            finally {
                this.wasm.rvf_free(ptr, bytes.byteLength);
            }
        }
        catch (err) {
            throw errors_1.RvfError.fromNative(err);
        }
    }
}
exports.WasmBackend = WasmBackend;
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
function resolveBackend(type) {
    switch (type) {
        case 'node':
            return new NodeBackend();
        case 'wasm':
            return new WasmBackend();
        case 'auto': {
            // In Node.js environments, prefer native; in browsers, prefer WASM.
            const isNode = typeof process !== 'undefined' &&
                typeof process.versions !== 'undefined' &&
                typeof process.versions.node === 'string';
            return isNode ? new NodeBackend() : new WasmBackend();
        }
    }
}
// ---------------------------------------------------------------------------
// Mapping helpers (TS options -> native/wasm shapes)
// ---------------------------------------------------------------------------
function mapMetricToNative(metric) {
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
function mapCompressionToNative(compression) {
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
/**
 * Resolve the vector dimensionality from create options, accepting the
 * documented `dimensions` (plural) as well as the `dimension` (singular)
 * alias that mirrors the native field name. Throws a clear error naming
 * the public option instead of letting the native layer report the
 * internal `dimension` field (issue #641).
 */
function resolveDimensions(options) {
    const dims = options.dimensions ?? options.dimension;
    if (typeof dims !== 'number' || !Number.isInteger(dims) || dims <= 0) {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.InvalidOptions, `Missing or invalid \`dimensions\` option: expected a positive integer, got ${JSON.stringify(dims)}. ` +
            'Pass { dimensions: N } (plural) to RvfDatabase.create().');
    }
    return dims;
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapOptionsToNative(options) {
    return {
        dimension: resolveDimensions(options),
        metric: mapMetricToNative(options.metric),
        profile: options.profile ?? 0,
        compression: mapCompressionToNative(options.compression),
        signing: options.signing ?? false,
        m: options.m ?? 16,
        ef_construction: options.efConstruction ?? 200,
    };
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapQueryOptionsToNative(options) {
    return {
        ef_search: options.efSearch ?? 100,
        // NAPI accepts the filter as a JSON string, not an object.
        filter: options.filter ? JSON.stringify(filterToNativeJson(options.filter)) : undefined,
        timeout_ms: options.timeoutMs ?? 0,
    };
}
/**
 * Infer the native `valueType` ("u64" | "i64" | "f64" | "string" | "bool")
 * from a JS filter value and stringify it, matching what the N-API filter
 * parser requires (`crates/rvf/rvf-node/src/lib.rs::parse_filter_value`).
 * The public `RvfFilterExpr` type deliberately omits `valueType` — the SDK
 * infers it here so callers don't have to know the native wire format
 * (issue #704: the SDK previously omitted `valueType` entirely, which the
 * native parser requires and rejects).
 */
function filterValueToNative(value) {
    if (typeof value === 'boolean') {
        return { valueType: 'bool', value: value ? 'true' : 'false' };
    }
    if (typeof value === 'string') {
        return { valueType: 'string', value };
    }
    // number: integers map to u64/i64 (native has no single "number" type),
    // non-integers map to f64.
    if (!Number.isInteger(value)) {
        return { valueType: 'f64', value: String(value) };
    }
    return { valueType: value >= 0 ? 'u64' : 'i64', value: String(value) };
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function filterToNativeJson(expr) {
    switch (expr.op) {
        case 'eq':
        case 'ne':
        case 'lt':
        case 'le':
        case 'gt':
        case 'ge': {
            const { valueType, value } = filterValueToNative(expr.value);
            return { op: expr.op, fieldId: expr.fieldId, valueType, value };
        }
        case 'in': {
            // valueType must be uniform across all values for a single 'in' filter.
            const converted = expr.values.map(filterValueToNative);
            const valueType = converted[0]?.valueType ?? 'string';
            return {
                op: 'in',
                fieldId: expr.fieldId,
                valueType,
                values: converted.map((c) => c.value),
            };
        }
        case 'range': {
            const lo = filterValueToNative(expr.low);
            const hi = filterValueToNative(expr.high);
            return {
                op: 'range',
                fieldId: expr.fieldId,
                valueType: lo.valueType,
                low: lo.value,
                high: hi.value,
            };
        }
        case 'and':
            return { op: 'and', children: expr.exprs.map(filterToNativeJson) };
        case 'or':
            return { op: 'or', children: expr.exprs.map(filterToNativeJson) };
        case 'not':
            return { op: 'not', child: filterToNativeJson(expr.expr) };
    }
}
/**
 * Immediate safety measure for issue #704: the SDK does not yet have a
 * design for mapping arbitrary string metadata field names to the native
 * layer's numeric `fieldId` + typed `value`, so silently accepting
 * `RvfIngestEntry.metadata` would silently drop it (the original bug).
 * Reject loudly instead until metadata ingestion is implemented.
 */
function rejectUnsupportedMetadata(entries) {
    const hasMetadata = entries.some((e) => e.metadata && Object.keys(e.metadata).length > 0);
    if (hasMetadata) {
        throw new errors_1.RvfError(errors_1.RvfErrorCode.MetadataNotSupported);
    }
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapNativeStatus(s) {
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
function mapCompactionState(state) {
    if (typeof state === 'string') {
        const lower = state.toLowerCase();
        if (lower === 'running')
            return 'running';
        if (lower === 'emergency')
            return 'emergency';
    }
    return 'idle';
}
//# sourceMappingURL=backend.js.map