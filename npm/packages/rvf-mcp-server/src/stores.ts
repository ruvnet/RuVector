/**
 * On-disk store plumbing for the RVF MCP server: data-dir path confinement
 * and the per-store metadata log.
 *
 * Vectors live in a real `.rvf` file (via @ruvector/rvf). The native backend
 * does not store per-vector metadata, so the server keeps it beside the store
 * in `<store>.mcp-meta.jsonl`: a header line, then `["s",id,metadata]` /
 * `["d",id]` ops. It is rewritten (temp file + rename) on close/compact and
 * whenever the appended tail outgrows the last rewrite.
 */

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';

export type Metadata = Record<string, string | number | boolean>;

export const META_SUFFIX = '.mcp-meta.jsonl';
const META_VERSION = 1;
const META_COMPACT_MIN_BYTES = 1 << 20;

/** Default data directory: $RVF_MCP_DATA_DIR, else ~/.rvf-mcp-server. */
export function defaultDataDir(): string {
  return process.env.RVF_MCP_DATA_DIR || path.join(os.homedir(), '.rvf-mcp-server');
}

/**
 * Resolve a model-supplied store path inside `dataDir`.
 *
 * Tool arguments are untrusted: absolute paths, `..` segments, NUL bytes and
 * symlinks that lead outside the data dir are all rejected, and containment
 * is checked before anything is created. With `create`, missing
 * intermediate directories are then made inside the data dir.
 */
export function resolveStorePath(dataDir: string, storePath: string, create: boolean): string {
  if (typeof storePath !== 'string' || storePath.length === 0 || storePath.length > 1024) {
    throw new Error('store path must be a non-empty relative path');
  }
  if (storePath.includes('\0')) throw new Error('store path contains a NUL byte');
  if (path.isAbsolute(storePath) || path.win32.isAbsolute(storePath)) {
    throw new Error(`store path must be relative to the data dir: ${storePath}`);
  }
  if (storePath.split(/[\\/]+/).includes('..')) {
    throw new Error(`store path must not contain "..": ${storePath}`);
  }
  const root = path.resolve(dataDir);
  const target = path.resolve(root, storePath);
  if (!target.startsWith(root + path.sep)) {
    throw new Error(`store path escapes the data dir: ${storePath}`);
  }
  fs.mkdirSync(root, { recursive: true });
  const realRoot = fs.realpathSync(root);
  // Deepest ancestor that exists (lstat: a dangling symlink counts as existing).
  let probe = path.dirname(target);
  while (probe !== root && lstatOrNull(probe) === null) probe = path.dirname(probe);
  let realProbe: string;
  try {
    realProbe = fs.realpathSync(probe);
  } catch {
    throw new Error(`store path goes through a broken symlink: ${storePath}`);
  }
  if (realProbe !== realRoot && !realProbe.startsWith(realRoot + path.sep)) {
    throw new Error(`store path escapes the data dir via a symlink: ${storePath}`);
  }
  if (lstatOrNull(target)?.isSymbolicLink()) {
    throw new Error(`store path is a symlink: ${storePath}`);
  }
  if (create) fs.mkdirSync(path.dirname(target), { recursive: true });
  return target;
}

function lstatOrNull(p: string): fs.Stats | null {
  try {
    return fs.lstatSync(p);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return null;
    throw err;
  }
}

export interface MetaHeader {
  dimensions: number;
  metric: string;
}

/** Metadata for one store, persisted as an append-only log. */
export class MetadataLog {
  readonly entries = new Map<string, Metadata>();
  header: MetaHeader | null = null;
  private bytes = 0;
  private compactedBytes = 0;

  constructor(readonly file: string, private readonly readOnly: boolean) {}

  /** Load the log if present. A torn final line (crash mid-append) is ignored. */
  load(): void {
    let text: string;
    try {
      text = fs.readFileSync(this.file, 'utf-8');
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return;
      throw err;
    }
    const lines = text.split('\n');
    // The last element is either '' (clean end) or a torn, unterminated line.
    const complete = lines.length - 1;
    let valid = 0;
    for (let i = 0; i < complete; i++) {
      const parsed = this.parse(lines[i], i === 0);
      if (!parsed) {
        if (lines.slice(i + 1, complete).some((l) => l.length > 0)) {
          throw new Error(`corrupt metadata log ${this.file} at line ${i + 1}`);
        }
        break;
      }
      valid += Buffer.byteLength(lines[i], 'utf-8') + 1;
    }
    if (!this.readOnly && valid < Buffer.byteLength(text, 'utf-8')) {
      fs.truncateSync(this.file, valid);
    }
    this.bytes = valid;
    this.compactedBytes = valid;
  }

  private parse(line: string, first: boolean): boolean {
    let v: unknown;
    try {
      v = JSON.parse(line);
    } catch {
      return false;
    }
    if (first) {
      const h = v as { rvfMcpMeta?: number; dimensions?: number; metric?: string };
      if (!h || h.rvfMcpMeta !== META_VERSION) return false;
      this.header = { dimensions: Number(h.dimensions), metric: String(h.metric) };
      return true;
    }
    if (!Array.isArray(v) || typeof v[1] !== 'string') return false;
    if (v[0] === 's' && v[2] && typeof v[2] === 'object') {
      this.entries.set(v[1], v[2] as Metadata);
      return true;
    }
    if (v[0] === 'd') {
      this.entries.delete(v[1]);
      return true;
    }
    return false;
  }

  /** Record new metadata (or its removal) for ids, then persist. */
  apply(sets: Array<[string, Metadata]>, deletes: string[]): void {
    let out = '';
    for (const id of deletes) {
      if (this.entries.delete(id)) out += `${JSON.stringify(['d', id])}\n`;
    }
    for (const [id, meta] of sets) {
      this.entries.set(id, meta);
      out += `${JSON.stringify(['s', id, meta])}\n`;
    }
    if (!out || this.readOnly) return;
    if (this.bytes === 0) {
      this.rewrite();
      return;
    }
    fs.appendFileSync(this.file, out, 'utf-8');
    this.bytes += Buffer.byteLength(out, 'utf-8');
    if (this.bytes - this.compactedBytes > Math.max(META_COMPACT_MIN_BYTES, this.compactedBytes)) {
      this.rewrite();
    }
  }

  /** Rewrite the log as header + current entries (atomic). */
  rewrite(): void {
    if (this.readOnly || !this.header) return;
    let out = `${JSON.stringify({ rvfMcpMeta: META_VERSION, ...this.header })}\n`;
    for (const [id, meta] of this.entries) out += `${JSON.stringify(['s', id, meta])}\n`;
    const tmp = `${this.file}.${process.pid}.${Date.now()}.tmp`;
    fs.writeFileSync(tmp, out, 'utf-8');
    fs.renameSync(tmp, this.file);
    this.bytes = Buffer.byteLength(out, 'utf-8');
    this.compactedBytes = this.bytes;
  }
}

/** True when `meta` matches every field of `filter` exactly. */
export function matchesFilter(meta: Metadata | undefined, filter: Metadata): boolean {
  if (!meta) return false;
  for (const [key, val] of Object.entries(filter)) {
    if (meta[key] !== val) return false;
  }
  return true;
}

const NATIVE_METRICS: Record<string, string> = { l2: 'l2', cosine: 'cosine', inner_product: 'dotproduct' };

/**
 * The distance metric an open `.rvf` file was created with, read from the
 * native store. @ruvector/rvf has no public accessor for it yet, so this asks
 * the native handle directly and returns undefined if that is unavailable.
 */
export function storedMetric(db: unknown): string | undefined {
  const handle = (db as { backend?: { handle?: { metric?: () => unknown } } })?.backend?.handle;
  if (typeof handle?.metric !== 'function') return undefined;
  try {
    const raw = handle.metric();
    return typeof raw === 'string' ? NATIVE_METRICS[raw] : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Native per-vector metadata field holding the vector's string id. The
 * @ruvector/rvf string-id map lives in a `<store>.idmap.json` sidecar; with
 * the id also inside the `.rvf`, a store copied without that sidecar still
 * answers with (and filters by) the original ids.
 */
export const ID_FIELD = 'rvf_mcp_id';

/** True when neither the id-map snapshot nor its journal is beside `file`. */
export function idMapMissing(file: string): boolean {
  return !fs.existsSync(`${file}.idmap.json`) && !fs.existsSync(`${file}.idmap.json.journal`);
}

export function missingIdMapError(storePath: string): Error {
  return new Error(
    `${storePath}.idmap.json is missing, so string ids cannot be resolved for vectors ` +
      `written without an embedded id (rvf-mcp-server <= 0.2.3). Copy every ${storePath}* ` +
      `file together, or re-ingest the vectors with rvf-mcp-server >= 0.2.4.`,
  );
}
