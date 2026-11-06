import * as fs from 'fs';

/**
 * Append-only journal for the NodeBackend string-id ↔ label map.
 *
 * The `.idmap.json` snapshot keeps its original format (older SDK versions
 * read it unchanged). Mutations between snapshots are appended to
 * `<snapshot>.journal` as newline-delimited JSON, so persisting one ingest
 * costs one small append instead of a full O(n) snapshot rewrite.
 *
 * Line format:
 *   {"rvfIdmapJournal":1}      header (first line)
 *   ["a","<id>",<label>]       id was assigned label
 *   ["d","<id>"]               id was deleted
 *
 * Replay is last-writer-wins per id, so replaying a journal over a snapshot
 * that already contains it (a crash between snapshot rename and journal
 * removal) is a no-op. Labels are never reused, so a label owned by two
 * different ids is corruption.
 */

export const JOURNAL_SUFFIX = '.journal';
export const JOURNAL_VERSION = 1;
const HEADER = `${JSON.stringify({ rvfIdmapJournal: JOURNAL_VERSION })}\n`;

export type JournalOp =
  | { op: 'add'; id: string; label: number }
  | { op: 'del'; id: string };

/** In-memory id map that the journal replays into. */
export interface IdMapState {
  idToLabel: Map<string, number>;
  labelToId: Map<number, string>;
  nextLabel: number;
}

/** A journal line that is not a torn tail failed to parse or validate. */
export class JournalCorruptError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'JournalCorruptError';
  }
}

export interface ReplayResult {
  /** Byte length of the valid prefix (header + complete, valid ops). */
  validBytes: number;
  /** Bytes dropped from the end because the last append was torn. */
  tornBytes: number;
  /** Number of ops applied. */
  ops: number;
}

/** Encode ops as journal lines (no header). */
export function encodeOps(ops: JournalOp[]): string {
  let out = '';
  for (const o of ops) {
    out += o.op === 'add'
      ? `${JSON.stringify(['a', o.id, o.label])}\n`
      : `${JSON.stringify(['d', o.id])}\n`;
  }
  return out;
}

function parseOp(line: string): JournalOp | null {
  let v: unknown;
  try {
    v = JSON.parse(line);
  } catch {
    return null;
  }
  if (!Array.isArray(v) || typeof v[1] !== 'string') return null;
  if (v[0] === 'a' && v.length === 3 && Number.isSafeInteger(v[2]) && v[2] >= 1) {
    return { op: 'add', id: v[1], label: v[2] };
  }
  if (v[0] === 'd' && v.length === 2) return { op: 'del', id: v[1] };
  return null;
}

function isHeader(line: string): boolean {
  try {
    const v = JSON.parse(line);
    return !!v && typeof v === 'object' && v.rvfIdmapJournal === JOURNAL_VERSION;
  } catch {
    return false;
  }
}

/** Apply one op to the state. Throws JournalCorruptError on a label clash. */
export function applyOp(state: IdMapState, o: JournalOp): void {
  if (o.op === 'del') {
    const label = state.idToLabel.get(o.id);
    if (label !== undefined) {
      state.idToLabel.delete(o.id);
      state.labelToId.delete(label);
    }
    return;
  }
  const owner = state.labelToId.get(o.label);
  if (owner !== undefined && owner !== o.id) {
    throw new JournalCorruptError(
      `label ${o.label} assigned to ${JSON.stringify(o.id)} but already owned by ${JSON.stringify(owner)}`,
    );
  }
  const prev = state.idToLabel.get(o.id);
  if (prev !== undefined && prev !== o.label) state.labelToId.delete(prev);
  state.idToLabel.set(o.id, o.label);
  state.labelToId.set(o.label, o.id);
  if (o.label >= state.nextLabel) state.nextLabel = o.label + 1;
}

/**
 * Replay a journal buffer into `state`.
 *
 * Invalid or unterminated lines are tolerated only at the very end (a torn
 * last append); anything invalid followed by a valid line is corruption.
 */
export function replayJournal(buf: Buffer, state: IdMapState): ReplayResult {
  let pos = 0;
  let validBytes = 0;
  let ops = 0;
  let sawHeader = false;
  let badAt = -1;
  let lines = 0;
  while (pos < buf.length) {
    const nl = buf.indexOf(0x0a, pos);
    if (nl === -1) break; // unterminated tail: torn append
    const line = buf.toString('utf-8', pos, nl);
    const next = nl + 1;
    lines++;
    const op = sawHeader ? parseOp(line) : null;
    const ok = sawHeader ? op !== null : isHeader(line);
    if (!ok) {
      if (badAt === -1) badAt = pos;
    } else {
      if (badAt !== -1) {
        throw new JournalCorruptError(`invalid journal line at byte ${badAt}`);
      }
      if (!sawHeader) sawHeader = true;
      else {
        applyOp(state, op as JournalOp);
        ops++;
      }
      validBytes = next;
    }
    pos = next;
  }
  // The header and the first ops go out in one write, so a damaged header
  // followed by further complete lines is not a torn append.
  if (!sawHeader && lines > 1) {
    throw new JournalCorruptError('invalid journal header');
  }
  return { validBytes, tornBytes: buf.length - validBytes, ops };
}

/**
 * Writer side of the journal. Opens the file lazily and writes each batch of
 * ops with a single positional write.
 */
export class IdMapJournalWriter {
  private fd: number | null = null;
  /** Current journal size in bytes (including header). */
  bytes: number;

  constructor(readonly path: string, existingBytes = 0) {
    this.bytes = existingBytes;
  }

  /** Append ops. On failure the partial tail is cut off before rethrowing. */
  append(ops: JournalOp[]): void {
    if (ops.length === 0) return;
    const start = this.bytes;
    let data = encodeOps(ops);
    if (start === 0) data = HEADER + data;
    const buf = Buffer.from(data, 'utf-8');
    try {
      if (this.fd === null) this.fd = fs.openSync(this.path, 'a');
      let off = 0;
      while (off < buf.length) {
        off += fs.writeSync(this.fd, buf, off, buf.length - off, start + off);
      }
      this.bytes = start + buf.length;
    } catch (err) {
      // Never leave a fragment that a later append would fuse with.
      try {
        if (start === 0) fs.rmSync(this.path, { force: true });
        else fs.truncateSync(this.path, start);
      } catch {
        // Replay treats a trailing fragment as a torn append anyway.
      }
      this.close();
      throw err;
    }
  }

  close(): void {
    if (this.fd === null) return;
    try {
      fs.closeSync(this.fd);
    } finally {
      this.fd = null;
    }
  }

  /** Drop the journal after its contents were folded into a snapshot. */
  reset(): void {
    this.close();
    fs.rmSync(this.path, { force: true });
    this.bytes = 0;
  }
}
