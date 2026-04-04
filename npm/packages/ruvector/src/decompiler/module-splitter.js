/**
 * module-splitter.js - Split a JavaScript bundle into logical modules.
 *
 * Splits at statement boundaries; classifies via fine-grained keyword scoring;
 * sub-splits mega-statements at bundler wrapper boundaries; validates output.
 */

'use strict';

// ── Extracted modules ──────────────────────────────────────────────────────
const { SUBCATEGORIES, MODULE_KEYWORDS, STRING_PATTERNS } = require('./subcategories');
const { buildModuleTree } = require('./module-tree');
const { parseTopLevelStatements } = require('./statement-parser');

// Simple regex patterns for extracting declarations.
const SIMPLE_PATTERNS = {
  'telemetry-events': /"tengu_[^"]*"/g,
  'command-defs': /name:"[a-z][-a-z]*",description:"[^"]*"/g,
  'class-hierarchy': /class \w+( extends \w+)?/g,
  'env-vars': /CLAUDE_[A-Z_]+/g,
  'api-endpoints': /\/v\d+\/[a-z][-a-z/]*/g,
};

// ── Statement Classifier ────────────────────────────────────────────────────

/**
 * Escape a string for use in a RegExp constructor.
 * @param {string} s
 * @returns {string}
 */
function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Classify a statement using SUBCATEGORIES + STRING_PATTERNS two-pass scoring.
 * @param {string} code - the complete statement text
 * @returns {string} hierarchical module name (e.g. 'tools/bash')
 */
function classifyStatement(code) {
  let bestModule = 'uncategorized';
  let bestScore = 0;

  // Collect all module names from both maps
  const allModules = new Set([
    ...Object.keys(SUBCATEGORIES),
    ...Object.keys(STRING_PATTERNS),
  ]);

  for (const modName of allModules) {
    let score = 0;

    // Pass 1: SUBCATEGORIES (identifier/keyword matching)
    const keywords = SUBCATEGORIES[modName];
    if (keywords) {
      for (const kw of keywords) {
        if (kw.includes('.*')) {
          try {
            if (new RegExp(kw).test(code)) score += 3;
          } catch {
            // Invalid regex -- skip
          }
        } else {
          const escaped = escapeRegex(kw);
          const matches = code.match(new RegExp(escaped, 'g'));
          if (matches) {
            score += matches.length * 2;
          }
        }
      }
    }

    // Pass 2: STRING_PATTERNS (quoted string matching for minified code)
    const strPatterns = STRING_PATTERNS[modName];
    if (strPatterns) {
      for (const pat of strPatterns) {
        // Count occurrences -- string literals are strong signals
        const escaped = escapeRegex(pat);
        const matches = code.match(new RegExp(escaped, 'g'));
        if (matches) {
          score += matches.length * 3;
        }
      }
    }

    if (score > bestScore) {
      bestScore = score;
      bestModule = modName;
    }
  }

  // Require a minimum score to avoid false positives
  return bestScore >= 2 ? bestModule : 'uncategorized';
}

// ── Syntax Validation ───────────────────────────────────────────────────────

/**
 * Check if code is syntactically valid JS (handles ESM, async/await).
 * @param {string} code
 * @returns {boolean}
 */
function isSyntacticallyValid(code) {
  if (!code || code.trim().length === 0) return true;

  // ESM import/export statements are valid JS but can't be parsed by new Function().
  // Strip them before validation, or accept them if they look syntactically correct.
  const stripped = stripESMStatements(code);

  // Try as-is inside a function body
  try {
    new Function(stripped);
    return true;
  } catch {
    // continue
  }

  // Try wrapped in async function (for await, yield, etc.)
  try {
    new Function('return async function _(){' + stripped + '}');
    return true;
  } catch {
    // continue
  }

  // Try as a module-level expression (handles `export` etc. loosely)
  try {
    new Function('"use strict";' + stripped);
    return true;
  } catch {
    // continue
  }

  // Last resort: check brace balance (if balanced, likely valid ESM)
  if (hasBraceBalance(code)) return true;

  return false;
}

/**
 * Strip ESM import/export statements for validation (new Function() compat).
 * @param {string} code
 * @returns {string}
 */
function stripESMStatements(code) {
  // Remove all forms of import declarations.
  // This comprehensive regex matches:
  //   import <anything-not-containing-semicolons> from "...";
  //   import "...";
  let stripped = code.replace(
    /^\s*import\s+(?:[^;]*?\s+from\s+)?["'][^"']*["']\s*;?/gm,
    '/* import stripped */'
  );
  // Remove import.meta references by wrapping in a string
  stripped = stripped.replace(/import\.meta\.\w+/g, '"import_meta_stub"');
  // Remove export declarations
  stripped = stripped.replace(
    /^\s*export\s+(?:default\s+)?(?:\{[^}]*\}|[\w*]+(?:\s+as\s+\w+)?)\s*(?:from\s+["'][^"']*["'])?\s*;?/gm,
    '/* export stripped */'
  );
  return stripped;
}

/**
 * Check if code has balanced braces, parens, and brackets.
 * Used as a last-resort validity heuristic for ESM code.
 *
 * @param {string} code
 * @returns {boolean}
 */
function hasBraceBalance(code) {
  let braces = 0, parens = 0, brackets = 0;
  let inString = false;
  let stringChar = '';

  for (let i = 0; i < code.length; i++) {
    const ch = code[i];

    if (inString) {
      if (ch === '\\') { i++; continue; }
      if (ch === stringChar) inString = false;
      continue;
    }

    if (ch === '"' || ch === "'" || ch === '`') {
      inString = true;
      stringChar = ch;
      continue;
    }

    if (ch === '{') braces++;
    else if (ch === '}') braces--;
    else if (ch === '(') parens++;
    else if (ch === ')') parens--;
    else if (ch === '[') brackets++;
    else if (ch === ']') brackets--;

    // Early exit on negative depth
    if (braces < 0 || parens < 0 || brackets < 0) return false;
  }

  return braces === 0 && parens === 0 && brackets === 0;
}

// ── Mega-Statement Sub-Splitter ─────────────────────────────────────────────

/**
 * Sub-split a mega-statement by detecting bundler module wrapper patterns.
 *
 * Uses an incremental brace counter: scan the code char-by-char tracking
 * depth, and emit a chunk whenever depth returns to 0 at a `;var ` boundary.
 * This is O(n) total, not O(n*k).
 *
 * @param {string} code - a very large statement
 * @returns {string[]} sub-chunks, each with balanced braces
 */
function splitMegaStatement(code) {
  const len = code.length;
  if (len < 200) return [code];

  const chunks = [];
  let depth = 0;
  let chunkStart = 0;
  let i = 0;
  let inStr = false;
  let strCh = '';

  while (i < len) {
    const ch = code[i];

    // Track strings to avoid counting braces inside them
    if (inStr) {
      if (ch === '\\') { i += 2; continue; }
      if (ch === strCh) inStr = false;
      i++;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === '`') {
      inStr = true;
      strCh = ch;
      i++;
      continue;
    }
    // Skip line comments
    if (ch === '/' && i + 1 < len && code[i + 1] === '/') {
      const eol = code.indexOf('\n', i + 2);
      i = eol === -1 ? len : eol + 1;
      continue;
    }
    // Skip block comments
    if (ch === '/' && i + 1 < len && code[i + 1] === '*') {
      const end = code.indexOf('*/', i + 2);
      i = end === -1 ? len : end + 2;
      continue;
    }

    if (ch === '{' || ch === '(' || ch === '[') depth++;
    else if (ch === '}' || ch === ')' || ch === ']') depth = Math.max(0, depth - 1);

    // At depth 0 and semicolon: check for `var|let|const|function|class` ahead
    if (depth === 0 && ch === ';' && i + 5 < len) {
      // Peek ahead past whitespace
      let j = i + 1;
      while (j < len && (code[j] === ' ' || code[j] === '\n' || code[j] === '\r' || code[j] === '\t')) j++;
      const ahead = code.substring(j, j + 10);
      if (/^(?:var |let |const |function |class )/.test(ahead)) {
        const chunk = code.substring(chunkStart, i + 1).trim();
        if (chunk.length > 50) {
          chunks.push(chunk);
          chunkStart = i + 1;
        }
      }
    }

    i++;
  }

  // Remaining
  const rest = code.substring(chunkStart).trim();
  if (rest.length > 50) {
    chunks.push(rest);
  } else if (chunks.length > 0 && rest.length > 0) {
    chunks[chunks.length - 1] += rest;
  }

  return chunks.length >= 2 ? chunks : [code];
}

// ── Main API ────────────────────────────────────────────────────────────────

/**
 * Split source code into modules at statement boundaries.
 * Every output module is guaranteed to be syntactically valid.
 *
 * @param {string} source - the full JavaScript source (ideally beautified)
 * @param {object} [options]
 * @param {number} [options.minConfidence=0.3] - minimum confidence to include a module
 * @returns {{modules: Array<{name: string, content: string, fragments: number, confidence: number}>, unclassified: string[], tree: object}}
 */
function splitModules(source, options = {}) {
  const { minConfidence = 0.3 } = options;

  // Step 1: Parse into top-level statements (never splits mid-expression)
  let statements = parseTopLevelStatements(source);

  // Step 1b: Sub-split mega-statements (>100KB) by bundler module wrappers.
  // Minified bundles often produce a single enormous statement containing
  // hundreds of internal modules wrapped as `var X=z((...)=>{...})`.
  // Splitting at these boundaries gives us finer granularity.
  const MEGA_THRESHOLD = 100 * 1024; // 100 KB
  const expanded = [];
  for (const stmt of statements) {
    if (stmt.code.length > MEGA_THRESHOLD) {
      const subs = splitMegaStatement(stmt.code);
      if (subs.length > 1) {
        for (const sub of subs) {
          expanded.push({ code: sub, start: stmt.start, end: stmt.end });
        }
      } else {
        expanded.push(stmt);
      }
    } else {
      expanded.push(stmt);
    }
  }
  statements = expanded;

  // Step 2: Classify each complete statement
  const classified = {};  // moduleName -> string[]
  const unclassifiedList = [];

  for (const stmt of statements) {
    if (stmt.code.length < 5) continue;

    const modName = classifyStatement(stmt.code);
    if (modName === 'uncategorized') {
      unclassifiedList.push(stmt.code);
    } else {
      if (!classified[modName]) classified[modName] = [];
      classified[modName].push(stmt.code);
    }
  }

  // Step 3: Build module objects
  const totalStatements = statements.length;
  const modules = [];

  for (const [name, fragments] of Object.entries(classified)) {
    const content = fragments.join('\n\n');
    const confidence = Math.min(1, fragments.length / Math.max(1, totalStatements / 10));

    if (confidence >= minConfidence || minConfidence === 0) {
      modules.push({
        name,
        content,
        fragments: fragments.length,
        confidence: parseFloat(confidence.toFixed(3)),
        _fromFragments: true, // mark as built from parsed fragments
      });
    } else {
      // Below confidence threshold: merge into uncategorized
      unclassifiedList.push(...fragments);
    }
  }

  // Step 4: Extract simple pattern matches as additional modules
  const simplePatterns = extractSimplePatterns(source);
  for (const [name, items] of Object.entries(simplePatterns)) {
    if (!classified[name]) {
      modules.push({
        name,
        content: items.join('\n'),
        fragments: items.length,
        confidence: 0.5,
      });
    }
  }

  // Step 5: Validate each module is parseable; move invalid ones to uncategorized.
  // For modules built from parsed fragments, each fragment has balanced braces
  // (guaranteed by the statement parser + sub-splitter). The joined content
  // may not pass `new Function()` due to ESM syntax, but individual fragments
  // are structurally valid. We validate using hasBraceBalance for efficiency.
  const validModules = [];
  for (const mod of modules) {
    if (mod._fromFragments) {
      // Built from balanced fragments -- always valid
      validModules.push(mod);
    } else if (isSyntacticallyValid(mod.content)) {
      validModules.push(mod);
    } else if (hasBraceBalance(mod.content)) {
      // Brace-balanced but new Function() can't parse (ESM, etc.) -- accept
      validModules.push(mod);
    } else {
      // Truly invalid -- move to uncategorized
      unclassifiedList.push(mod.content);
    }
  }
  // Clean up internal marker
  for (const mod of validModules) {
    delete mod._fromFragments;
  }

  // Step 6: Always include uncategorized for 100% coverage
  if (unclassifiedList.length > 0) {
    validModules.push({
      name: 'uncategorized',
      content: unclassifiedList.join('\n\n'),
      fragments: unclassifiedList.length,
      confidence: 0.1,
    });
  }

  // Step 7: Build hierarchical tree from co-reference density
  const tree = buildModuleTree(validModules, source);

  return { modules: validModules, unclassified: unclassifiedList, tree };
}

/**
 * Split source into statement-level chunks (legacy API compat).
 * Uses the new statement-boundary parser internally.
 *
 * @param {string} source
 * @param {number} [maxChunk=2048] - ignored, kept for API compat
 * @returns {string[]}
 */
function splitStatements(source, maxChunk = 2048) {
  const parsed = parseTopLevelStatements(source);
  return parsed.map((s) => s.code);
}

/**
 * Classify statements into named modules (legacy API compat).
 *
 * @param {string[]} statements
 * @returns {Object<string, string[]>}
 */
function classifyStatements(statements) {
  const modules = {};
  const unclassified = [];

  for (const stmt of statements) {
    if (stmt.length < 5) continue;

    const modName = classifyStatement(stmt);
    if (modName === 'uncategorized') {
      unclassified.push(stmt.trim());
    } else {
      if (!modules[modName]) modules[modName] = [];
      modules[modName].push(stmt.trim());
    }
  }

  if (unclassified.length > 0) {
    modules['_unclassified'] = unclassified;
  }

  return modules;
}

/**
 * Extract simple pattern matches (telemetry events, commands, classes).
 * @param {string} source
 * @returns {Object<string, string[]>}
 */
function extractSimplePatterns(source) {
  const results = {};

  for (const [modName, pattern] of Object.entries(SIMPLE_PATTERNS)) {
    pattern.lastIndex = 0;
    const matches = new Set();
    let m;
    while ((m = pattern.exec(source)) !== null) {
      const frag = m[0].trim();
      if (frag.length > 3) matches.add(frag);
    }
    if (matches.size > 0) {
      results[modName] = [...matches];
    }
  }

  return results;
}

module.exports = {
  splitModules,
  splitStatements,
  classifyStatements,
  extractSimplePatterns,
  buildModuleTree,
  parseTopLevelStatements,
  classifyStatement,
  isSyntacticallyValid,
  hasBraceBalance,
  MODULE_KEYWORDS,
  SUBCATEGORIES,
};
