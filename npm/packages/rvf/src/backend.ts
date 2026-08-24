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
  RvfFilterValue,
  RvfKernelData,
  RvfEbpfData,
  RvfSegmentInfo,
  BackendType,
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
  branch(childPath: string): Promise<RvfBackend>;
  freeze(): Promise<number>;
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
  // Byte-level persistence (for backends with no file-based storage, e.g. WASM)
  /** Serialize the store to an in-memory `.rvf` byte buffer. */
  exportBytes(): Promise<Uint8Array>;
  /** Load a store from an in-memory `.rvf` byte buffer. */
  openBytes(bytes: Uint8Array): Promise<void>;
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

  // String ID <-> Numeric Label mappings (N-API layer requires i64 labels)
  private idToLabel: Map<string, number> = new Map();
  private labelToId: Map<number, string> = new Map();
  private nextLabel: number = 1; // RVF uses 1-based labels
  private storePath: string = '';

  private async loadNative(): Promise<void> {
    if (this.native) return;
    try {
      // Dynamic import so the SDK can be bundled for browsers without
      // pulling in the native addon at compile time.
      // The NAPI addon exports a `RvfDatabase` class with factory methods.
      const mod = await import('@ruvector/rvf-node');
      this.native = mod.RvfDatabase ?? mod.default?.RvfDatabase ?? mod;
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

  /** Release a native handle without persisting mappings after open failed. */
  private discardHandle(): void {
    try {
      this.handle?.close();
    } catch {
      // Preserve the original open/sidecar error.
    } finally {
      this.handle = null;
      this.storePath = '';
      this.idToLabel.clear();
      this.labelToId.clear();
      this.nextLabel = 1;
    }
  }

  async create(path: string, options: RvfOptions): Promise<void> {
    await this.loadNative();
    const fs = await import('fs');
    const sidecarPath = `${path}.idmap.json`;
    const existingPaths = [path, sidecarPath].filter((candidate) => fs.existsSync(candidate));
    const backups: Array<{ original: string; backup: string }> = [];
    try {
      // Precondition: refuse to clobber an existing file unless asked to.
      // The native layer surfaces this as a misleading FsyncFailed, so check
      // here and raise a clear, actionable error. An orphaned sidecar also
      // blocks creation because silently retaining it can desynchronize IDs.
      if (existingPaths.length > 0 && !options.overwrite) {
        throw new RvfError(
          RvfErrorCode.FileExists,
          `${existingPaths.join(', ')} already exists; use RvfDatabase.open() to reuse the store, or pass { overwrite: true } to replace it`,
        );
      }

      if (options.overwrite) {
        const { randomUUID } = await import('crypto');
        for (const original of existingPaths) {
          if (!fs.lstatSync(original).isFile()) {
            throw new RvfError(
              RvfErrorCode.InvalidArgument,
              `refusing to overwrite non-file path ${original}`,
            );
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
        } catch {
          // The new store is valid; retain the recoverable backup if cleanup fails.
        }
      }
    } catch (err) {
      // If replacement fails after moving the old store aside, remove any
      // partially-created replacement and restore the original files.
      if (backups.length > 0) {
        try {
          fs.rmSync(path, { force: true });
          fs.rmSync(sidecarPath, { force: true });
        } catch {
          // Continue restoring every backup that can be recovered.
        }
        for (const { original, backup } of backups.reverse()) {
          try {
            if (fs.existsSync(backup)) fs.renameSync(backup, original);
          } catch {
            // The backup remains on disk with a unique, discoverable suffix.
          }
        }
      }
      throw RvfError.fromNative(err);
    }
  }

  async open(path: string): Promise<void> {
    await this.loadNative();
    try {
      this.handle = await this.native.open(path);
      this.storePath = path;
      await this.loadMappings();
    } catch (err) {
      this.discardHandle();
      throw RvfError.fromNative(err);
    }
  }

  async openReadonly(path: string): Promise<void> {
    await this.loadNative();
    try {
      this.handle = await this.native.openReadonly(path);
      this.storePath = path;
      await this.loadMappings();
    } catch (err) {
      this.discardHandle();
      throw RvfError.fromNative(err);
    }
  }

  async ingestBatch(entries: RvfIngestEntry[]): Promise<RvfIngestResult> {
    this.ensureHandle();
    try {
      rejectUnsupportedMetadata(entries);
      // NAPI signature: ingestBatch(vectors: Float32Array, ids: i64[], metadata?)
      // Flatten individual vectors into a single contiguous Float32Array.
      const n = entries.length;
      if (n === 0) return { accepted: 0, rejected: 0, epoch: 0 };
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
      const results = this.handle.query(vector, k, nativeOpts);
      // Map numeric labels back to original string IDs.
      return (results as Array<{ id: number; distance: number }>).map((r) => ({
        id: this.labelToId.get(Number(r.id)) ?? String(r.id),
        distance: r.distance,
      }));
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async delete(ids: string[]): Promise<RvfDeleteResult> {
    this.ensureHandle();
    try {
      // Resolve string IDs to numeric labels for the N-API layer.
      const numIds = ids
        .map((id) => this.idToLabel.get(id))
        .filter((label): label is number => label !== undefined);
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
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async deleteByFilter(filter: RvfFilterExpr): Promise<RvfDeleteResult> {
    this.ensureHandle();
    try {
      // NAPI takes a JSON string for the filter expression.
      const result = this.handle.deleteByFilter(JSON.stringify(filter));
      return { deleted: Number(result.deleted), epoch: result.epoch };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async compact(): Promise<RvfCompactionResult> {
    this.ensureHandle();
    try {
      const result = this.handle.compact();
      return {
        segmentsCompacted: result.segmentsCompacted ?? result.segments_compacted,
        bytesReclaimed: Number(result.bytesReclaimed ?? result.bytes_reclaimed),
        epoch: result.epoch,
      };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async status(): Promise<RvfStatus> {
    this.ensureHandle();
    try {
      const s = this.handle.status();
      return mapNativeStatus(s);
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async close(): Promise<void> {
    if (!this.handle) return;
    let failure: unknown;
    try {
      await this.saveMappings();
    } catch (err) {
      failure = err;
    }
    try {
      this.handle.close();
    } catch (err) {
      failure ??= err;
    } finally {
      this.handle = null;
      this.idToLabel.clear();
      this.labelToId.clear();
      this.nextLabel = 1;
      this.storePath = '';
    }
    if (failure) throw RvfError.fromNative(failure);
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
    let childHandle: any = null;
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
    } catch (err) {
      await this.cleanupFailedChild(childHandle, childPath);
      throw RvfError.fromNative(err);
    }
  }

  async branch(childPath: string): Promise<RvfBackend> {
    this.ensureHandle();
    let childHandle: any = null;
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
    } catch (err) {
      await this.cleanupFailedChild(childHandle, childPath);
      throw RvfError.fromNative(err);
    }
  }

  async freeze(): Promise<number> {
    this.ensureHandle();
    try {
      return this.handle.freeze();
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

  async exportBytes(): Promise<Uint8Array> {
    throw new RvfError(
      RvfErrorCode.BackendNotFound,
      'exportBytes is not supported by the node backend — use a file path with create()/open() instead',
    );
  }

  async openBytes(_bytes: Uint8Array): Promise<void> {
    throw new RvfError(
      RvfErrorCode.BackendNotFound,
      'openBytes is not supported by the node backend — use a file path with open() instead',
    );
  }

  // ─── String ID ↔ Numeric Label mapping helpers ───

  /**
   * Get or allocate a numeric label for a string ID.
   * If the ID was already seen, returns the existing label.
   */
  private resolveLabel(id: string): number {
    let label = this.idToLabel.get(id);
    if (label !== undefined) return label;
    label = this.nextLabel++;
    this.idToLabel.set(id, label);
    this.labelToId.set(label, id);
    return label;
  }

  /** Path to the sidecar mappings file. */
  private mappingsPath(): string {
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
  private async saveMappings(): Promise<void> {
    const mp = this.mappingsPath();
    if (!mp) return;
    const fs = await import('fs');
    const data = JSON.stringify({
      idToLabel: Object.fromEntries(this.idToLabel),
      labelToId: Object.fromEntries(
        Array.from(this.labelToId.entries()).map(([k, v]) => [String(k), v]),
      ),
      nextLabel: this.nextLabel,
    });
    // A shared `${mp}.tmp` races across processes opening the same store.
    // Synchronous writes cannot interleave within one Node process; PID plus
    // timestamp also keeps independently-running writers on separate paths.
    const tmp = `${mp}.${process.pid}.${Date.now()}.tmp`;
    try {
      fs.writeFileSync(tmp, data, 'utf-8');
      fs.renameSync(tmp, mp);
    } catch (err) {
      try {
        fs.rmSync(tmp, { force: true });
      } catch {
        // best-effort cleanup of the temp file
      }
      throw new RvfError(
        RvfErrorCode.SidecarWriteFailed,
        `at ${mp}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  private async cleanupFailedChild(childHandle: any, childPath: string): Promise<void> {
    // If native creation failed before returning a handle (for example because
    // childPath already exists), the path is not ours to remove.
    if (!childHandle) return;
    try {
      childHandle?.close();
    } catch {
      // Preserve the original branch/sidecar error.
    }
    try {
      const fs = await import('fs');
      fs.rmSync(childPath, { force: true });
      fs.rmSync(`${childPath}.idmap.json`, { force: true });
    } catch {
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
  private async loadMappings(): Promise<void> {
    const mp = this.mappingsPath();
    if (!mp) return;
    const fs = await import('fs');
    if (!fs.existsSync(mp)) return; // fresh store: no sidecar yet is legitimate
    let parsed: {
      idToLabel: Record<string, number>;
      labelToId: Record<string, string>;
      nextLabel: number;
    };
    try {
      const candidate: unknown = JSON.parse(fs.readFileSync(mp, 'utf-8'));
      if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
        throw new TypeError('sidecar root must be an object');
      }
      const raw = candidate as Record<string, unknown>;
      if (
        !raw.idToLabel ||
        typeof raw.idToLabel !== 'object' ||
        Array.isArray(raw.idToLabel) ||
        !raw.labelToId ||
        typeof raw.labelToId !== 'object' ||
        Array.isArray(raw.labelToId) ||
        !Number.isSafeInteger(raw.nextLabel) ||
        (raw.nextLabel as number) < 1
      ) {
        throw new TypeError('sidecar must contain idToLabel, labelToId, and a positive nextLabel');
      }

      const idToLabel = raw.idToLabel as Record<string, unknown>;
      const labelToId = raw.labelToId as Record<string, unknown>;
      let maxLabel = 0;
      for (const [id, label] of Object.entries(idToLabel)) {
        if (!Number.isSafeInteger(label) || (label as number) < 1) {
          throw new TypeError(`invalid label for id ${JSON.stringify(id)}`);
        }
        if (labelToId[String(label)] !== id) {
          throw new TypeError(`idToLabel/labelToId mismatch for id ${JSON.stringify(id)}`);
        }
        maxLabel = Math.max(maxLabel, label as number);
      }
      for (const [label, id] of Object.entries(labelToId)) {
        const numericLabel = Number(label);
        if (
          !Number.isSafeInteger(numericLabel) ||
          numericLabel < 1 ||
          String(numericLabel) !== label ||
          typeof id !== 'string' ||
          idToLabel[id] !== numericLabel
        ) {
          throw new TypeError(`invalid reverse mapping for label ${JSON.stringify(label)}`);
        }
      }
      if ((raw.nextLabel as number) <= maxLabel) {
        throw new TypeError('nextLabel must be greater than every allocated label');
      }
      parsed = {
        idToLabel: idToLabel as Record<string, number>,
        labelToId: labelToId as Record<string, string>,
        nextLabel: raw.nextLabel as number,
      };
    } catch (err) {
      const { randomUUID } = await import('crypto');
      const quarantine = `${mp}.corrupt-${Date.now()}-${randomUUID()}`;
      try {
        fs.renameSync(mp, quarantine);
      } catch {
        // if we cannot move it aside, leave it in place — still fail loud
      }
      throw new RvfError(
        RvfErrorCode.SidecarCorrupt,
        `at ${mp} (quarantined to ${quarantine}): string-id delete()/ingest would ` +
          `silently corrupt data — restore a valid sidecar or recreate the store; ` +
          `${err instanceof Error ? err.message : String(err)}`,
      );
    }
    this.idToLabel = new Map(
      Object.entries(parsed.idToLabel),
    );
    this.labelToId = new Map(
      Object.entries(parsed.labelToId).map(([k, v]) => [Number(k), v]),
    );
    this.nextLabel = parsed.nextLabel;
  }
}

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
export class WasmBackend implements RvfBackend {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private wasm: any = null;
  /** Integer store handle returned by `rvf_store_create` / `rvf_store_open`. */
  private handle: number = 0;
  private dim: number = 0;

  private async loadWasm(): Promise<void> {
    if (this.wasm) return;
    try {
      const mod = await import('@ruvector/rvf-wasm');
      // wasm-pack default export is the init function
      if (typeof mod.default === 'function') {
        this.wasm = await mod.default();
      } else {
        this.wasm = mod;
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

  private metricCode(metric: string | undefined): number {
    switch (metric) {
      case 'Cosine': return 2;
      case 'InnerProduct': return 1;
      default: return 0; // L2
    }
  }

  async create(_path: string, options: RvfOptions): Promise<void> {
    await this.loadWasm();
    try {
      const nativeOpts = mapOptionsToNative(options);
      const dim = nativeOpts.dimension as number;
      const metric = this.metricCode(nativeOpts.metric as string);
      const h = this.wasm.rvf_store_create(dim, metric);
      if (h <= 0) throw new Error('rvf_store_create returned ' + h);
      this.handle = h;
      this.dim = dim;
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async open(_path: string): Promise<void> {
    throw new RvfError(
      RvfErrorCode.BackendNotFound,
      'WASM backend does not support file-based open (in-memory only)',
    );
  }

  async openReadonly(_path: string): Promise<void> {
    throw new RvfError(
      RvfErrorCode.BackendNotFound,
      'WASM backend does not support file-based openReadonly (in-memory only)',
    );
  }

  async ingestBatch(entries: RvfIngestEntry[]): Promise<RvfIngestResult> {
    this.ensureHandle();
    try {
      rejectUnsupportedMetadata(entries);
      const n = entries.length;
      if (n === 0) return { accepted: 0, rejected: 0, epoch: 0 };
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
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async query(
    vector: Float32Array,
    k: number,
    _options?: RvfQueryOptions,
  ): Promise<RvfSearchResult[]> {
    this.ensureHandle();
    try {
      const queryPtr = this.wasm.rvf_alloc(vector.byteLength);
      new Float32Array(this.wasm.memory.buffer, queryPtr, vector.length).set(vector);
      // Each result = 8 bytes id + 4 bytes dist = 12 bytes
      const outSize = k * 12;
      const outPtr = this.wasm.rvf_alloc(outSize);
      const count = this.wasm.rvf_store_query(this.handle, queryPtr, k, 0, outPtr);
      const results: RvfSearchResult[] = [];
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
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async delete(ids: string[]): Promise<RvfDeleteResult> {
    this.ensureHandle();
    try {
      const arr = new BigUint64Array(ids.map((id) => BigInt(id)));
      const ptr = this.wasm.rvf_alloc(arr.byteLength);
      new BigUint64Array(this.wasm.memory.buffer, ptr, arr.length).set(arr);
      const deleted = this.wasm.rvf_store_delete(this.handle, ptr, ids.length);
      this.wasm.rvf_free(ptr, arr.byteLength);
      return { deleted: deleted > 0 ? deleted : 0, epoch: 0 };
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async deleteByFilter(_filter: RvfFilterExpr): Promise<RvfDeleteResult> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'deleteByFilter not supported in WASM backend');
  }

  async compact(): Promise<RvfCompactionResult> {
    // There is no compaction export in the WASM C-ABI: `crates/rvf/rvf-wasm/src/lib.rs`
    // runs `rvf_store_create` .. `rvf_store_close` with no `rvf_store_compact`. Returning
    // `{segmentsCompacted: 0, bytesReclaimed: 0}` made "not implemented here" indistinguishable
    // from "ran, and there was nothing to reclaim" — and the WASM store DOES retain
    // soft-deleted entries (`export()` filters them), so a caller has real reason to expect
    // reclamation. Throwing matches how this class reports every other unsupported operation.
    throw new RvfError(RvfErrorCode.BackendNotFound, 'compact not supported in WASM backend');
  }

  async status(): Promise<RvfStatus> {
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
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  async close(): Promise<void> {
    if (!this.handle) return;
    try {
      this.wasm.rvf_store_close(this.handle);
    } catch (err) {
      throw RvfError.fromNative(err);
    } finally {
      this.handle = 0;
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
  async branch(_childPath: string): Promise<RvfBackend> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'branch not supported in WASM backend');
  }
  async freeze(): Promise<number> {
    throw new RvfError(RvfErrorCode.BackendNotFound, 'freeze not supported in WASM backend');
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
    this.ensureHandle();
    const d = this.wasm.rvf_store_dimension(this.handle);
    if (d < 0) throw new RvfError(RvfErrorCode.StoreClosed);
    return d;
  }

  /**
   * Serialize the in-memory store to `.rvf` bytes via the `rvf_store_export`
   * C-ABI export. `rvf_store_export` follows a probe-then-write pattern:
   * called with a too-small (or zero-length) buffer it returns the negated
   * required size, so we probe first, allocate exactly that much, then
   * write for real.
   */
  async exportBytes(): Promise<Uint8Array> {
    this.ensureHandle();
    try {
      const probe = this.wasm.rvf_store_export(this.handle, 0, 0);
      const size = probe < 0 ? -probe : probe;
      if (size <= 0) return new Uint8Array(0);
      const ptr = this.wasm.rvf_alloc(size);
      try {
        const written = this.wasm.rvf_store_export(this.handle, ptr, size);
        if (written < 0) {
          throw new Error(`rvf_store_export failed after size probe (size=${size})`);
        }
        return new Uint8Array(this.wasm.memory.buffer, ptr, written).slice();
      } finally {
        this.wasm.rvf_free(ptr, size);
      }
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }

  /** Load a store from `.rvf` bytes via the `rvf_store_open` C-ABI import. */
  async openBytes(bytes: Uint8Array): Promise<void> {
    await this.loadWasm();
    try {
      const ptr = this.wasm.rvf_alloc(bytes.byteLength);
      try {
        new Uint8Array(this.wasm.memory.buffer, ptr, bytes.byteLength).set(bytes);
        const h = this.wasm.rvf_store_open(ptr, bytes.byteLength);
        if (h <= 0) throw new Error('rvf_store_open returned ' + h);
        this.handle = h;
        this.dim = this.wasm.rvf_store_dimension(h);
      } finally {
        this.wasm.rvf_free(ptr, bytes.byteLength);
      }
    } catch (err) {
      throw RvfError.fromNative(err);
    }
  }
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

/**
 * Resolve the vector dimensionality from create options, accepting the
 * documented `dimensions` (plural) as well as the `dimension` (singular)
 * alias that mirrors the native field name. Throws a clear error naming
 * the public option instead of letting the native layer report the
 * internal `dimension` field (issue #641).
 */
function resolveDimensions(options: RvfOptions): number {
  const dims = options.dimensions ?? options.dimension;
  if (typeof dims !== 'number' || !Number.isInteger(dims) || dims <= 0) {
    throw new RvfError(
      RvfErrorCode.InvalidOptions,
      `Missing or invalid \`dimensions\` option: expected a positive integer, got ${JSON.stringify(dims)}. ` +
        'Pass { dimensions: N } (plural) to RvfDatabase.create().',
    );
  }
  return dims;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapOptionsToNative(options: RvfOptions): Record<string, any> {
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
function mapQueryOptionsToNative(options: RvfQueryOptions): Record<string, any> {
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
function filterValueToNative(value: RvfFilterValue): { valueType: string; value: string } {
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
function filterToNativeJson(expr: RvfFilterExpr): Record<string, any> {
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
function rejectUnsupportedMetadata(entries: RvfIngestEntry[]): void {
  const hasMetadata = entries.some(
    (e) => e.metadata && Object.keys(e.metadata).length > 0,
  );
  if (hasMetadata) {
    throw new RvfError(RvfErrorCode.MetadataNotSupported);
  }
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
