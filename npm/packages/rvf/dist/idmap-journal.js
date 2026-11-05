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
exports.IdMapJournalWriter = exports.JournalCorruptError = exports.JOURNAL_VERSION = exports.JOURNAL_SUFFIX = void 0;
exports.encodeOps = encodeOps;
exports.applyOp = applyOp;
exports.replayJournal = replayJournal;
const fs = __importStar(require("fs"));
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
exports.JOURNAL_SUFFIX = '.journal';
exports.JOURNAL_VERSION = 1;
const HEADER = `${JSON.stringify({ rvfIdmapJournal: exports.JOURNAL_VERSION })}\n`;
/** A journal line that is not a torn tail failed to parse or validate. */
class JournalCorruptError extends Error {
    constructor(message) {
        super(message);
        this.name = 'JournalCorruptError';
    }
}
exports.JournalCorruptError = JournalCorruptError;
/** Encode ops as journal lines (no header). */
function encodeOps(ops) {
    let out = '';
    for (const o of ops) {
        out += o.op === 'add'
            ? `${JSON.stringify(['a', o.id, o.label])}\n`
            : `${JSON.stringify(['d', o.id])}\n`;
    }
    return out;
}
function parseOp(line) {
    let v;
    try {
        v = JSON.parse(line);
    }
    catch {
        return null;
    }
    if (!Array.isArray(v) || typeof v[1] !== 'string')
        return null;
    if (v[0] === 'a' && v.length === 3 && Number.isSafeInteger(v[2]) && v[2] >= 1) {
        return { op: 'add', id: v[1], label: v[2] };
    }
    if (v[0] === 'd' && v.length === 2)
        return { op: 'del', id: v[1] };
    return null;
}
function isHeader(line) {
    try {
        const v = JSON.parse(line);
        return !!v && typeof v === 'object' && v.rvfIdmapJournal === exports.JOURNAL_VERSION;
    }
    catch {
        return false;
    }
}
/** Apply one op to the state. Throws JournalCorruptError on a label clash. */
function applyOp(state, o) {
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
        throw new JournalCorruptError(`label ${o.label} assigned to ${JSON.stringify(o.id)} but already owned by ${JSON.stringify(owner)}`);
    }
    const prev = state.idToLabel.get(o.id);
    if (prev !== undefined && prev !== o.label)
        state.labelToId.delete(prev);
    state.idToLabel.set(o.id, o.label);
    state.labelToId.set(o.label, o.id);
    if (o.label >= state.nextLabel)
        state.nextLabel = o.label + 1;
}
/**
 * Replay a journal buffer into `state`.
 *
 * Invalid or unterminated lines are tolerated only at the very end (a torn
 * last append); anything invalid followed by a valid line is corruption.
 */
function replayJournal(buf, state) {
    let pos = 0;
    let validBytes = 0;
    let ops = 0;
    let sawHeader = false;
    let badAt = -1;
    let lines = 0;
    while (pos < buf.length) {
        const nl = buf.indexOf(0x0a, pos);
        if (nl === -1)
            break; // unterminated tail: torn append
        const line = buf.toString('utf-8', pos, nl);
        const next = nl + 1;
        lines++;
        const op = sawHeader ? parseOp(line) : null;
        const ok = sawHeader ? op !== null : isHeader(line);
        if (!ok) {
            if (badAt === -1)
                badAt = pos;
        }
        else {
            if (badAt !== -1) {
                throw new JournalCorruptError(`invalid journal line at byte ${badAt}`);
            }
            if (!sawHeader)
                sawHeader = true;
            else {
                applyOp(state, op);
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
class IdMapJournalWriter {
    constructor(path, existingBytes = 0) {
        this.path = path;
        this.fd = null;
        this.bytes = existingBytes;
    }
    /** Append ops. On failure the partial tail is cut off before rethrowing. */
    append(ops) {
        if (ops.length === 0)
            return;
        const start = this.bytes;
        let data = encodeOps(ops);
        if (start === 0)
            data = HEADER + data;
        const buf = Buffer.from(data, 'utf-8');
        try {
            if (this.fd === null)
                this.fd = fs.openSync(this.path, 'a');
            let off = 0;
            while (off < buf.length) {
                off += fs.writeSync(this.fd, buf, off, buf.length - off, start + off);
            }
            this.bytes = start + buf.length;
        }
        catch (err) {
            // Never leave a fragment that a later append would fuse with.
            try {
                if (start === 0)
                    fs.rmSync(this.path, { force: true });
                else
                    fs.truncateSync(this.path, start);
            }
            catch {
                // Replay treats a trailing fragment as a torn append anyway.
            }
            this.close();
            throw err;
        }
    }
    close() {
        if (this.fd === null)
            return;
        try {
            fs.closeSync(this.fd);
        }
        finally {
            this.fd = null;
        }
    }
    /** Drop the journal after its contents were folded into a snapshot. */
    reset() {
        this.close();
        fs.rmSync(this.path, { force: true });
        this.bytes = 0;
    }
}
exports.IdMapJournalWriter = IdMapJournalWriter;
//# sourceMappingURL=idmap-journal.js.map