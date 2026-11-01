'use strict';

/**
 * Validation for vector records arriving from untrusted callers (MCP tool
 * arguments, CLI flags, JSON files). Shared by the CLI and the MCP server so
 * both enforce the same limits.
 */

const LIMITS = Object.freeze({
  idLength: 256,
  maxIds: 10000,
  maxEntries: 10000,
  maxDimension: 65536,
  maxK: 1000,
  metadataBytes: 64 * 1024,
  metadataDepth: 8,
});

const UNSAFE_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

function fail(message) {
  const err = new Error(message);
  err.code = 'ERR_INVALID_RECORD';
  return err;
}

/** A record id: non-empty string (numbers are accepted and stringified), no control characters. */
function validateRecordId(id, what = 'id') {
  if (typeof id === 'number' && Number.isSafeInteger(id) && id >= 0) id = String(id);
  if (typeof id !== 'string' || id.length === 0) throw fail(`${what} must be a non-empty string`);
  if (id.length > LIMITS.idLength) throw fail(`${what} exceeds ${LIMITS.idLength} characters`);
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f\u007f]/.test(id)) throw fail(`${what} contains control characters`);
  return id;
}

function validateIdList(ids, what = 'ids') {
  if (!Array.isArray(ids) || ids.length === 0) throw fail(`${what} must be a non-empty array`);
  if (ids.length > LIMITS.maxIds) throw fail(`${what} exceeds ${LIMITS.maxIds} entries`);
  return ids.map((id, i) => validateRecordId(id, `${what}[${i}]`));
}

/** A vector: array (or typed array) of finite numbers, optionally of an exact dimension. */
function validateVector(vector, dimension, what = 'vector') {
  if (!(Array.isArray(vector) || vector instanceof Float32Array || vector instanceof Float64Array)) {
    throw fail(`${what} must be an array of numbers`);
  }
  if (vector.length === 0 || vector.length > LIMITS.maxDimension) {
    throw fail(`${what} length must be between 1 and ${LIMITS.maxDimension}`);
  }
  if (dimension !== undefined && dimension !== null && vector.length !== dimension) {
    throw fail(`${what} has dimension ${vector.length}, store expects ${dimension}`);
  }
  for (let i = 0; i < vector.length; i++) {
    if (typeof vector[i] !== 'number' || !Number.isFinite(vector[i])) {
      throw fail(`${what}[${i}] is not a finite number`);
    }
  }
  return vector instanceof Float32Array ? vector : Float32Array.from(vector);
}

function checkMetadataNode(value, depth, path) {
  if (depth > LIMITS.metadataDepth) throw fail(`metadata nests deeper than ${LIMITS.metadataDepth} levels`);
  if (Array.isArray(value)) {
    value.forEach((v, i) => checkMetadataNode(v, depth + 1, `${path}[${i}]`));
    return;
  }
  if (value !== null && typeof value === 'object') {
    for (const key of Object.keys(value)) {
      if (UNSAFE_KEYS.has(key)) throw fail(`metadata key ${JSON.stringify(key)} is not allowed (${path})`);
      checkMetadataNode(value[key], depth + 1, `${path}.${key}`);
    }
    return;
  }
  if (typeof value === 'number' && !Number.isFinite(value)) throw fail(`metadata value at ${path} is not finite`);
  if (!['string', 'number', 'boolean'].includes(typeof value) && value !== null) {
    throw fail(`metadata value at ${path} has unsupported type ${typeof value}`);
  }
}

/**
 * Metadata: a plain JSON object with no prototype-polluting keys at any depth,
 * bounded in size and nesting. JSON strings are parsed first. Returns a clean
 * copy (re-parsed, so getters and non-JSON values cannot survive).
 */
function validateMetadata(metadata, what = 'metadata') {
  if (metadata === undefined || metadata === null) return undefined;
  if (typeof metadata === 'string') {
    if (metadata.length > LIMITS.metadataBytes) throw fail(`${what} exceeds ${LIMITS.metadataBytes} bytes`);
    try { metadata = JSON.parse(metadata); } catch { throw fail(`${what} is not valid JSON`); }
  }
  if (typeof metadata !== 'object' || Array.isArray(metadata)) throw fail(`${what} must be a JSON object`);
  checkMetadataNode(metadata, 0, what);
  const json = JSON.stringify(metadata);
  if (Buffer.byteLength(json) > LIMITS.metadataBytes) throw fail(`${what} exceeds ${LIMITS.metadataBytes} bytes`);
  return JSON.parse(json);
}

/** Validate an array of {id, vector, metadata?} entries for insert/ingest. */
function validateEntries(entries, dimension, what = 'entries') {
  if (!Array.isArray(entries) || entries.length === 0) throw fail(`${what} must be a non-empty array`);
  if (entries.length > LIMITS.maxEntries) throw fail(`${what} exceeds ${LIMITS.maxEntries} entries`);
  const dim = dimension || (entries[0] && Array.isArray(entries[0].vector) ? entries[0].vector.length : undefined);
  return entries.map((e, i) => {
    if (!e || typeof e !== 'object') throw fail(`${what}[${i}] must be an object`);
    const out = {
      id: validateRecordId(e.id, `${what}[${i}].id`),
      vector: validateVector(e.vector, dim, `${what}[${i}].vector`),
    };
    const metadata = validateMetadata(e.metadata, `${what}[${i}].metadata`);
    if (metadata !== undefined) out.metadata = metadata;
    return out;
  });
}

function validateK(k, fallback = 10) {
  if (k === undefined || k === null) return fallback;
  const n = typeof k === 'number' ? k : parseInt(k, 10);
  if (!Number.isInteger(n) || n < 1) throw fail(`k must be a positive integer`);
  return Math.min(n, LIMITS.maxK);
}

module.exports = {
  LIMITS,
  validateRecordId,
  validateIdList,
  validateVector,
  validateMetadata,
  validateEntries,
  validateK,
};
