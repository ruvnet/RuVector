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
export declare const JOURNAL_SUFFIX = ".journal";
export declare const JOURNAL_VERSION = 1;
export type JournalOp = {
    op: 'add';
    id: string;
    label: number;
} | {
    op: 'del';
    id: string;
};
/** In-memory id map that the journal replays into. */
export interface IdMapState {
    idToLabel: Map<string, number>;
    labelToId: Map<number, string>;
    nextLabel: number;
}
/** A journal line that is not a torn tail failed to parse or validate. */
export declare class JournalCorruptError extends Error {
    constructor(message: string);
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
export declare function encodeOps(ops: JournalOp[]): string;
/** Apply one op to the state. Throws JournalCorruptError on a label clash. */
export declare function applyOp(state: IdMapState, o: JournalOp): void;
/**
 * Replay a journal buffer into `state`.
 *
 * Invalid or unterminated lines are tolerated only at the very end (a torn
 * last append); anything invalid followed by a valid line is corruption.
 */
export declare function replayJournal(buf: Buffer, state: IdMapState): ReplayResult;
/**
 * Writer side of the journal. Opens the file lazily and writes each batch of
 * ops with a single positional write.
 */
export declare class IdMapJournalWriter {
    readonly path: string;
    private fd;
    /** Current journal size in bytes (including header). */
    bytes: number;
    constructor(path: string, existingBytes?: number);
    /** Append ops. On failure the partial tail is cut off before rethrowing. */
    append(ops: JournalOp[]): void;
    close(): void;
    /** Drop the journal after its contents were folded into a snapshot. */
    reset(): void;
}
