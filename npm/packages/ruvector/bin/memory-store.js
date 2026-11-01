'use strict';

/**
 * Memory records in intelligence.json, shared by the CLI hooks and the MCP
 * server. The two writers historically used different shapes
 * ({id, memory_type, timestamp} vs {type, created}, the latter without an id),
 * so this module gives every memory a stable id and a normalized view without
 * rewriting the stored fields other readers (Rust CLI, @ruvector/cli) expect.
 */

const crypto = require('crypto');
const { validateRecordId, validateMetadata } = require('./record-validate');

const MAX_CONTENT = 64 * 1024;
const MAX_LIST = 1000;

function fail(message, code = 'ERR_INVALID_MEMORY') {
  const err = new Error(message);
  err.code = code;
  return err;
}

/** A new collision-free memory id (two writes in the same millisecond used to collide). */
function newMemoryId() {
  return `mem_${Date.now().toString(36)}_${crypto.randomBytes(5).toString('hex')}`;
}

/**
 * Give every memory an id. Missing ids are derived from the memory's content
 * and timestamp, so the CLI and the MCP server (separate processes) derive the
 * same id for the same legacy entry and re-running is a no-op. Duplicate ids
 * (the old `mem_<seconds>` scheme) are disambiguated deterministically.
 * Returns the number of memories that changed.
 */
function ensureMemoryIds(data) {
  if (!data || !Array.isArray(data.memories)) return 0;
  const seen = new Set();
  let changed = 0;
  data.memories.forEach((m, index) => {
    if (!m || typeof m !== 'object') return;
    let id = typeof m.id === 'string' && m.id.length > 0 ? m.id : null;
    if (!id) {
      const basis = `${m.content || ''}\u0000${m.created || m.timestamp || ''}\u0000${m.memory_type || m.type || ''}`;
      id = `mem_${crypto.createHash('sha256').update(basis).digest('hex').slice(0, 16)}`;
    }
    if (seen.has(id)) id = `${id}_${index}`;
    if (id !== m.id) { m.id = id; changed++; }
    seen.add(id);
  });
  return changed;
}

/** Normalized, embedding-free view of a stored memory. */
function memoryView(m) {
  const created = m.created || (typeof m.timestamp === 'number'
    ? new Date(m.timestamp * (m.timestamp < 1e12 ? 1000 : 1)).toISOString()
    : m.timestamp) || null;
  const view = { id: m.id, type: m.memory_type || m.type || 'general', content: m.content || '', created };
  if (m.updated) view.updated = m.updated;
  if (m.metadata && Object.keys(m.metadata).length > 0) view.metadata = m.metadata;
  return view;
}

function findMemory(data, id) {
  validateRecordId(id, 'memory id');
  const memories = Array.isArray(data && data.memories) ? data.memories : [];
  const index = memories.findIndex(m => m && m.id === id);
  return index >= 0 ? { index, memory: memories[index] } : null;
}

function getMemory(data, id) {
  const hit = findMemory(data, id);
  return hit ? memoryView(hit.memory) : null;
}

/** Newest first; optional type filter; bounded page size. */
function listMemories(data, { type, limit = 50, offset = 0 } = {}) {
  const memories = Array.isArray(data && data.memories) ? data.memories : [];
  const lim = Math.min(Math.max(parseInt(limit, 10) || 50, 1), MAX_LIST);
  const off = Math.max(parseInt(offset, 10) || 0, 0);
  const filtered = [];
  for (let i = memories.length - 1; i >= 0; i--) {
    const m = memories[i];
    if (!m) continue;
    if (type && (m.memory_type || m.type || 'general') !== type) continue;
    filtered.push(m);
  }
  return { total: filtered.length, offset: off, items: filtered.slice(off, off + lim).map(memoryView) };
}

/** Remove memories by id. Returns { deleted, missing }. */
function deleteMemories(data, ids) {
  const wanted = new Set(ids.map(id => validateRecordId(id, 'memory id')));
  const memories = Array.isArray(data && data.memories) ? data.memories : [];
  const deleted = [];
  data.memories = memories.filter(m => {
    if (m && wanted.has(m.id)) { deleted.push(m.id); return false; }
    return true;
  });
  if (data.stats) data.stats.total_memories = data.memories.length;
  const gone = new Set(deleted);
  return { deleted, missing: [...wanted].filter(id => !gone.has(id)) };
}

/** Validate an update patch: content, type and/or metadata. */
function validateMemoryPatch(patch) {
  const out = {};
  if (patch.content !== undefined) {
    if (typeof patch.content !== 'string' || patch.content.trim().length === 0) throw fail('content must be a non-empty string');
    if (patch.content.length > MAX_CONTENT) throw fail(`content exceeds ${MAX_CONTENT} characters`);
    out.content = patch.content;
  }
  if (patch.type !== undefined) out.type = validateMemoryType(patch.type);
  if (patch.metadata !== undefined) out.metadata = validateMetadata(patch.metadata) || {};
  if (Object.keys(out).length === 0) throw fail('nothing to update: pass content, type and/or metadata');
  return out;
}

function validateMemoryType(type) {
  if (typeof type !== 'string' || !/^[A-Za-z0-9_.:-]{1,64}$/.test(type)) {
    throw fail('type must be 1-64 characters of letters, digits, _ . : -');
  }
  return type;
}

/**
 * Apply a validated patch in place. `embedding` must be supplied when content
 * changed (computed by the caller through the same embedder and provenance
 * gate as a new memory). The stored field names are preserved per shape.
 */
function applyMemoryPatch(memory, patch, embedding) {
  if (patch.content !== undefined) {
    if (!embedding) throw fail('internal: content changed without a new embedding');
    memory.content = patch.content;
    memory.embedding = embedding;
  }
  if (patch.type !== undefined) {
    if ('memory_type' in memory || !('type' in memory)) memory.memory_type = patch.type;
    if ('type' in memory) memory.type = patch.type;
  }
  if (patch.metadata !== undefined) memory.metadata = patch.metadata;
  memory.updated = new Date().toISOString();
  return memoryView(memory);
}

module.exports = {
  MAX_CONTENT,
  newMemoryId,
  ensureMemoryIds,
  memoryView,
  findMemory,
  getMemory,
  listMemories,
  deleteMemories,
  validateMemoryPatch,
  validateMemoryType,
  applyMemoryPatch,
};
