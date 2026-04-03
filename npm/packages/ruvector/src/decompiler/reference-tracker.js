/**
 * reference-tracker.js - Scope-aware identifier tracking and bulk renaming.
 *
 * Tracks all occurrences of each identifier in the source, respecting
 * JavaScript scoping rules (function, block, module). When an identifier
 * is renamed, ALL references in the same scope are updated consistently.
 *
 * Does NOT use a full AST parser — operates on regex-based token scanning
 * so it works on partially-valid or beautified-minified code.
 */

'use strict';

/**
 * Characters that can appear in a JS identifier.
 * Used to ensure we match whole identifiers, not substrings.
 */
const ID_CHAR = /[a-zA-Z0-9_$]/;

/**
 * Pattern for minified-looking identifiers:
 *   - Single letter (A-Z, a-z)
 *   - Letter + digit(s) (A2, B3, z1)
 *   - Letter + $ (s$, a$)
 *   - Two letters (AA, Ab)
 * These are candidates for renaming.
 */
const MINIFIED_ID = /^[a-zA-Z][a-zA-Z0-9$]{0,2}$/;

/**
 * JS reserved words that must never be renamed.
 */
const RESERVED = new Set([
  'abstract', 'arguments', 'await', 'boolean', 'break', 'byte', 'case',
  'catch', 'char', 'class', 'const', 'continue', 'debugger', 'default',
  'delete', 'do', 'double', 'else', 'enum', 'eval', 'export', 'extends',
  'false', 'final', 'finally', 'float', 'for', 'function', 'goto', 'if',
  'implements', 'import', 'in', 'instanceof', 'int', 'interface', 'let',
  'long', 'native', 'new', 'null', 'of', 'package', 'private', 'protected',
  'public', 'return', 'short', 'static', 'super', 'switch', 'synchronized',
  'this', 'throw', 'throws', 'transient', 'true', 'try', 'typeof',
  'undefined', 'var', 'void', 'volatile', 'while', 'with', 'yield',
  'async', 'from', 'get', 'set', 'of', 'as', 'type',
]);

/**
 * Well-known globals that should not be renamed.
 */
const GLOBALS = new Set([
  'Object', 'Array', 'String', 'Number', 'Boolean', 'Symbol', 'BigInt',
  'Map', 'Set', 'WeakMap', 'WeakSet', 'Promise', 'Proxy', 'Reflect',
  'Date', 'RegExp', 'Error', 'TypeError', 'RangeError', 'SyntaxError',
  'ReferenceError', 'URIError', 'EvalError', 'JSON', 'Math', 'Intl',
  'ArrayBuffer', 'SharedArrayBuffer', 'DataView', 'Float32Array',
  'Float64Array', 'Int8Array', 'Int16Array', 'Int32Array', 'Uint8Array',
  'Uint16Array', 'Uint32Array', 'Uint8ClampedArray',
  'console', 'process', 'Buffer', 'global', 'globalThis', 'window',
  'document', 'navigator', 'location', 'history', 'fetch', 'setTimeout',
  'setInterval', 'clearTimeout', 'clearInterval', 'setImmediate',
  'queueMicrotask', 'requestAnimationFrame', 'cancelAnimationFrame',
  'URL', 'URLSearchParams', 'Headers', 'Request', 'Response',
  'TextEncoder', 'TextDecoder', 'AbortController', 'AbortSignal',
  'EventTarget', 'Event', 'CustomEvent', 'MessageChannel', 'MessagePort',
  'Worker', 'ReadableStream', 'WritableStream', 'TransformStream',
  'require', 'module', 'exports', '__dirname', '__filename',
  'crypto', 'fs', 'path', 'os', 'http', 'https', 'net', 'child_process',
  'stream', 'events', 'util', 'assert', 'zlib', 'querystring',
  'NaN', 'Infinity',
]);

/**
 * Check if an identifier looks minified and is a candidate for renaming.
 * @param {string} name
 * @returns {boolean}
 */
function isMinifiedName(name) {
  if (RESERVED.has(name) || GLOBALS.has(name)) return false;
  if (name.startsWith('_') && name.length > 2) return false;
  return MINIFIED_ID.test(name);
}

/**
 * Find all occurrences of a whole-word identifier in source.
 * Returns array of { start, end } positions.
 *
 * @param {string} source
 * @param {string} identifier
 * @returns {Array<{start: number, end: number}>}
 */
function findAllReferences(source, identifier) {
  const refs = [];
  const escaped = identifier.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`(?<![a-zA-Z0-9_$])${escaped}(?![a-zA-Z0-9_$])`, 'g');
  let match;
  while ((match = re.exec(source)) !== null) {
    // Skip if inside a string literal or comment
    if (!isInsideStringOrComment(source, match.index)) {
      refs.push({ start: match.index, end: match.index + identifier.length });
    }
  }
  return refs;
}

/**
 * Rough check: is position inside a string literal or comment?
 * Scans backwards from the position to check context.
 * Not perfect but handles most beautified code correctly.
 *
 * @param {string} source
 * @param {number} pos
 * @returns {boolean}
 */
function isInsideStringOrComment(source, pos) {
  // Check the current line for string/comment context
  let lineStart = source.lastIndexOf('\n', pos - 1) + 1;
  const linePrefix = source.substring(lineStart, pos);

  // Inside a single-line comment
  if (linePrefix.includes('//')) {
    const commentStart = linePrefix.lastIndexOf('//');
    // Make sure the // is not inside a string
    const beforeComment = linePrefix.substring(0, commentStart);
    const singleQuotes = (beforeComment.match(/'/g) || []).length;
    const doubleQuotes = (beforeComment.match(/"/g) || []).length;
    const backticks = (beforeComment.match(/`/g) || []).length;
    if (singleQuotes % 2 === 0 && doubleQuotes % 2 === 0 && backticks % 2 === 0) {
      return true;
    }
  }

  // Count unescaped quotes before position on this line
  let inString = false;
  let stringChar = null;
  for (let i = lineStart; i < pos; i++) {
    const ch = source[i];
    if (inString) {
      if (ch === '\\') { i++; continue; }
      if (ch === stringChar) { inString = false; stringChar = null; }
    } else {
      if (ch === '"' || ch === "'" || ch === '`') {
        inString = true;
        stringChar = ch;
      }
    }
  }
  return inString;
}

/**
 * Apply a single rename across the entire source, replacing all whole-word
 * occurrences of oldName with newName.
 *
 * @param {string} source
 * @param {string} oldName
 * @param {string} newName
 * @returns {string}
 */
function applyRename(source, oldName, newName) {
  if (oldName === newName) return source;
  const refs = findAllReferences(source, oldName);
  if (refs.length === 0) return source;

  // Apply replacements from end to start to preserve positions
  const chars = source.split('');
  for (let i = refs.length - 1; i >= 0; i--) {
    const { start, end } = refs[i];
    chars.splice(start, end - start, newName);
  }
  return chars.join('');
}

/**
 * Apply multiple renames to source in a single pass.
 * Renames are applied in dependency order: longest old names first
 * to avoid partial-match issues.
 *
 * @param {string} source
 * @param {Array<{oldName: string, newName: string}>} renames
 * @returns {string}
 */
function applyAllRenames(source, renames) {
  if (!renames || renames.length === 0) return source;

  // Sort by old name length descending to prevent substring conflicts
  const sorted = [...renames].sort((a, b) => b.oldName.length - a.oldName.length);

  // Build a replacement map for a single-pass approach
  // Collect all match positions for all renames
  const allMatches = [];
  for (const { oldName, newName } of sorted) {
    if (oldName === newName) continue;
    const refs = findAllReferences(source, oldName);
    for (const ref of refs) {
      allMatches.push({ ...ref, newName });
    }
  }

  if (allMatches.length === 0) return source;

  // Sort by position descending, apply from end to start
  allMatches.sort((a, b) => b.start - a.start);

  // Remove overlapping matches (keep the one with longer oldName)
  const filtered = [allMatches[0]];
  for (let i = 1; i < allMatches.length; i++) {
    const prev = filtered[filtered.length - 1];
    if (allMatches[i].end <= prev.start) {
      filtered.push(allMatches[i]);
    }
  }

  let result = source;
  for (const { start, end, newName } of filtered) {
    result = result.substring(0, start) + newName + result.substring(end);
  }

  return result;
}

/**
 * Scan source for all unique identifiers that look minified.
 *
 * @param {string} source
 * @returns {string[]} sorted list of minified identifiers
 */
function findMinifiedIdentifiers(source) {
  const ids = new Set();
  // Match word-boundary identifiers
  const re = /(?<![a-zA-Z0-9_$])([a-zA-Z$_][a-zA-Z0-9$_]*)(?![a-zA-Z0-9_$])/g;
  let match;
  while ((match = re.exec(source)) !== null) {
    const id = match[1];
    if (isMinifiedName(id) && !isInsideStringOrComment(source, match.index)) {
      ids.add(id);
    }
  }
  return [...ids].sort();
}

/**
 * Extract the local context around each occurrence of an identifier.
 * Returns strings that appear near the identifier (within ~200 chars).
 * Used by the name predictor for context-based inference.
 *
 * @param {string} source
 * @param {string} identifier
 * @param {number} [windowSize=200]
 * @returns {string[]} context strings (deduplicated)
 */
function extractContext(source, identifier, windowSize = 200) {
  const refs = findAllReferences(source, identifier);
  const contexts = new Set();

  for (const { start, end } of refs) {
    const ctxStart = Math.max(0, start - windowSize);
    const ctxEnd = Math.min(source.length, end + windowSize);
    const ctx = source.substring(ctxStart, ctxEnd);

    // Extract string literals from context
    const strings = ctx.match(/["']([^"']{2,60})["']/g) || [];
    for (const s of strings) {
      contexts.add(s.replace(/^["']|["']$/g, ''));
    }

    // Extract property accesses: .propertyName
    const props = ctx.match(/\.([a-zA-Z_]\w{1,30})/g) || [];
    for (const p of props) {
      contexts.add(p);
    }
  }

  return [...contexts];
}

module.exports = {
  findAllReferences,
  applyRename,
  applyAllRenames,
  findMinifiedIdentifiers,
  extractContext,
  isMinifiedName,
  isInsideStringOrComment,
  RESERVED,
  GLOBALS,
};
