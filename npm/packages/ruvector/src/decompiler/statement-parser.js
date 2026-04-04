/**
 * statement-parser.js - Parse JavaScript source into top-level statements.
 *
 * Tracks brace/paren/bracket depth and string/template/regex contexts
 * to split at true statement boundaries. Never splits a statement
 * across modules -- a statement is atomic.
 */

'use strict';

/**
 * Parse source into top-level statements by tracking brace/paren/bracket depth.
 *
 * A "top-level statement" ends when:
 *   - We encounter a semicolon at depth 0, OR
 *   - We encounter a closing brace that brings depth to 0 AND the next
 *     non-whitespace token does not continue the expression (like `=`, `.`,
 *     `,`, `(`, etc.) -- this avoids splitting `var { x } = obj;` or
 *     `obj.method()` into two statements.
 *
 * String literals, template literals, regex literals, and comments are
 * tracked so delimiters inside them are not counted.
 *
 * @param {string} source
 * @returns {Array<{code: string, start: number, end: number}>}
 */
function parseTopLevelStatements(source) {
  const statements = [];
  let depth = 0;
  let start = 0;
  let i = 0;
  const len = source.length;

  while (i < len) {
    const ch = source[i];
    const next = i + 1 < len ? source[i + 1] : '';

    // ── Skip single-line comments ──
    if (ch === '/' && next === '/') {
      const eol = source.indexOf('\n', i + 2);
      i = eol === -1 ? len : eol + 1;
      continue;
    }

    // ── Skip multi-line comments ──
    if (ch === '/' && next === '*') {
      const end = source.indexOf('*/', i + 2);
      i = end === -1 ? len : end + 2;
      continue;
    }

    // ── Skip string literals ──
    if (ch === '"' || ch === "'") {
      i = skipString(source, i, ch);
      continue;
    }

    // ── Skip template literals ──
    if (ch === '`') {
      i = skipTemplateLiteral(source, i);
      continue;
    }

    // ── Skip regex literals ──
    if (ch === '/' && isRegexStart(source, i)) {
      i = skipRegex(source, i);
      continue;
    }

    // ── Track depth ──
    if (ch === '{' || ch === '(' || ch === '[') {
      depth++;
      i++;
      continue;
    }

    if (ch === '}' || ch === ')' || ch === ']') {
      depth = Math.max(0, depth - 1);

      // Closing brace at depth 0 MAY be a statement boundary
      if (depth === 0 && ch === '}') {
        if (!isStatementBoundaryAfterBrace(source, i + 1)) {
          i++;
          continue;
        }

        const code = source.substring(start, i + 1).trim();
        if (code.length > 0) {
          statements.push({ code, start, end: i + 1 });
        }
        start = i + 1;
        i++;
        continue;
      }

      i++;
      continue;
    }

    // ── Semicolon at depth 0 is a statement boundary ──
    if (ch === ';' && depth === 0) {
      const code = source.substring(start, i + 1).trim();
      if (code.length > 0) {
        statements.push({ code, start, end: i + 1 });
      }
      start = i + 1;
      i++;
      continue;
    }

    i++;
  }

  // Remaining code (unterminated statement)
  const remaining = source.substring(start).trim();
  if (remaining.length > 0) {
    statements.push({ code: remaining, start, end: len });
  }

  return statements;
}

/**
 * After a `}` at depth 0, decide whether this is truly a statement boundary.
 * Returns true if it IS a boundary (next token starts a new statement).
 * Returns false if the expression continues (e.g. `}.method()`, `} = obj`, etc.)
 *
 * @param {string} source
 * @param {number} afterPos - position right after the `}`
 * @returns {boolean}
 */
function isStatementBoundaryAfterBrace(source, afterPos) {
  const len = source.length;
  let j = afterPos;

  // Skip whitespace and comments to find the next meaningful token
  while (j < len) {
    const c = source[j];

    if (c === ' ' || c === '\t' || c === '\r' || c === '\n') {
      j++;
      continue;
    }

    if (c === '/' && j + 1 < len && source[j + 1] === '/') {
      const eol = source.indexOf('\n', j + 2);
      j = eol === -1 ? len : eol + 1;
      continue;
    }

    if (c === '/' && j + 1 < len && source[j + 1] === '*') {
      const end = source.indexOf('*/', j + 2);
      j = end === -1 ? len : end + 2;
      continue;
    }

    break;
  }

  if (j >= len) return true;

  const nextChar = source[j];

  // These tokens CONTINUE the expression -- NOT a statement boundary
  const continuationChars = '.=,([?:&|+\\-*/%<>^~!;)';
  if (continuationChars.includes(nextChar)) {
    return false;
  }

  // Check for multi-char continuation tokens
  const ahead = source.substring(j, j + 15);
  if (/^(?:instanceof|in|of|from)\s/.test(ahead)) return false;
  if (/^as\s/.test(ahead)) return false;

  return true;
}

/**
 * Skip a string literal starting at position i (where source[i] is the quote).
 * @param {string} source
 * @param {number} i
 * @param {string} quote - the quote character
 * @returns {number}
 */
function skipString(source, i, quote) {
  const len = source.length;
  i++;
  while (i < len) {
    if (source[i] === '\\') { i += 2; continue; }
    if (source[i] === quote) return i + 1;
    i++;
  }
  return len;
}

/**
 * Skip a template literal starting at position i (where source[i] is backtick).
 * @param {string} source
 * @param {number} i
 * @returns {number}
 */
function skipTemplateLiteral(source, i) {
  const len = source.length;
  i++;
  while (i < len) {
    if (source[i] === '\\') { i += 2; continue; }
    if (source[i] === '`') return i + 1;
    if (source[i] === '$' && i + 1 < len && source[i + 1] === '{') {
      i = skipTemplateExpression(source, i + 2);
      continue;
    }
    i++;
  }
  return len;
}

/**
 * Skip a template expression (inside ${...}) starting after the opening ${.
 * @param {string} source
 * @param {number} i
 * @returns {number}
 */
function skipTemplateExpression(source, i) {
  const len = source.length;
  let exprDepth = 1;
  while (i < len && exprDepth > 0) {
    const ch = source[i];
    if (ch === '\\') { i += 2; continue; }
    if (ch === '{') { exprDepth++; i++; continue; }
    if (ch === '}') { exprDepth--; i++; continue; }
    if (ch === '`') { i = skipTemplateLiteral(source, i); continue; }
    if (ch === '"' || ch === "'") { i = skipString(source, i, ch); continue; }
    i++;
  }
  return i;
}

/**
 * Heuristic: is source[i] the start of a regex literal?
 * @param {string} source
 * @param {number} i
 * @returns {boolean}
 */
function isRegexStart(source, i) {
  let j = i - 1;
  while (j >= 0 && (source[j] === ' ' || source[j] === '\t' || source[j] === '\n' || source[j] === '\r')) {
    j--;
  }
  if (j < 0) return true;
  const prev = source[j];
  if (/[\w$)\].]/.test(prev)) return false;
  return true;
}

/**
 * Skip a regex literal starting at position i.
 * @param {string} source
 * @param {number} i
 * @returns {number}
 */
function skipRegex(source, i) {
  const len = source.length;
  i++;
  while (i < len) {
    if (source[i] === '\\') { i += 2; continue; }
    if (source[i] === '[') {
      i++;
      while (i < len && source[i] !== ']') {
        if (source[i] === '\\') { i += 2; continue; }
        i++;
      }
      i++;
      continue;
    }
    if (source[i] === '/') {
      i++;
      while (i < len && /[gimsuy]/.test(source[i])) i++;
      return i;
    }
    i++;
  }
  return len;
}

module.exports = { parseTopLevelStatements };
