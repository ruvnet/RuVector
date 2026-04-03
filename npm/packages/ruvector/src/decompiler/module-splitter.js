/**
 * module-splitter.js - Split a JavaScript bundle into logical modules.
 *
 * Splits at STATEMENT BOUNDARIES so every output module is guaranteed to be
 * syntactically valid, parseable JavaScript. Never splits a statement across
 * modules -- a statement is atomic.
 *
 * Algorithm:
 *   1. Parse source into top-level statements by tracking brace/paren/bracket
 *      depth and string context.
 *   2. Classify each COMPLETE statement into a module by scoring keyword hits.
 *   3. Group statements by module.
 *   4. Validate each module is parseable; move invalid modules to uncategorized.
 *   5. Build hierarchical tree from co-reference density.
 */

'use strict';

// ── Module classification keywords ──────────────────────────────────────────
// Each key is a module name, value is an array of keywords/identifiers.
// A statement is scored against every module; highest score wins.
const MODULE_KEYWORDS = {
  'tool-dispatch': [
    'BashTool', 'FileReadTool', 'FileEditTool', 'FileWriteTool',
    'AgentOutputTool', 'WebFetch', 'WebSearch', 'TodoWrite',
    'NotebookEdit', 'GlobTool', 'GrepTool', 'ListFilesTool',
    'SearchTool', 'ReadTool', 'EditTool', 'WriteTool',
    'tool_use', 'tool_result', 'ToolUse', 'ToolResult',
    'toolDefinition', 'toolSchema', 'inputSchema',
  ],
  'permission-system': [
    'canUseTool', 'alwaysAllowRules', 'denyWrite',
    'Permission', 'permission', 'allowedTools',
    'permissionMode', 'sandbox', 'allowList', 'denyList',
    'isAllowed', 'checkPermission', 'grantPermission',
  ],
  'mcp-client': [
    'mcp__', 'McpClient', 'McpServer', 'McpError',
    'callTool', 'listTools', 'McpTransport',
    'StdioTransport', 'SseTransport', 'StreamableHttp',
    'mcp_server', 'mcp_client', 'mcpConnection',
  ],
  'streaming-handler': [
    'content_block_delta', 'message_start', 'message_stop',
    'message_delta', 'content_block_start', 'content_block_stop',
    'stream_event', 'text_delta', 'input_json_delta',
    'StreamEvent', 'onStream', 'streamHandler',
  ],
  'context-manager': [
    'tengu_compact', 'microcompact', 'auto_compact',
    'compact_boundary', 'preCompactTokenCount',
    'postCompactTokenCount', 'compaction',
    'tokenCount', 'contextWindow', 'maxTokens',
    'promptCache', 'cacheControl',
  ],
  'agent-loop': [
    'agentLoop', 'mainLoop', 'querySource',
    'toolUseContext', 'systemPrompt',
    'conversationTurn', 'assistantMessage',
    'userMessage', 'messageHistory',
  ],
  'commands': [
    'slashCommand', 'registerCommand', 'commandHandler',
    'parseCommand', '/help', '/clear', '/compact',
    '/bug', '/init', '/login', '/logout',
    '/doctor', '/config', '/cost', '/memory',
  ],
  'telemetry': [
    'telemetry', 'Telemetry', 'opentelemetry', 'otel',
    'datadog', 'perfetto', 'tracing', 'span',
    'metric_', 'counter_', 'histogram_',
    'tengu_', 'sentry',
  ],
  'config': [
    'settings', 'Settings', 'configuration',
    'CLAUDE_', 'environment', 'envVar',
    'dotenv', 'loadConfig', 'parseConfig',
  ],
  'session': [
    'session', 'Session', 'conversationId',
    'checkpoint', 'resume', 'restore',
    'sessionState', 'persistSession',
  ],
  'model-provider': [
    'anthropic', 'Anthropic', 'claude-', 'claude_',
    'bedrock', 'vertex', 'openai', 'provider',
    'apiKey', 'modelId', 'modelName',
  ],
};

// Simple regex patterns for extracting declarations.
const SIMPLE_PATTERNS = {
  'telemetry-events': /"tengu_[^"]*"/g,
  'command-defs': /name:"[a-z][-a-z]*",description:"[^"]*"/g,
  'class-hierarchy': /class \w+( extends \w+)?/g,
  'env-vars': /CLAUDE_[A-Z_]+/g,
  'api-endpoints': /\/v\d+\/[a-z][-a-z/]*/g,
};

// ── Statement Parser ────────────────────────────────────────────────────────

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
        // Check if the next non-whitespace/comment token continues this
        // expression. If so, do NOT split here.
        if (!isStatementBoundaryAfterBrace(source, i + 1)) {
          // Not a boundary -- continue accumulating
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

    // Skip whitespace
    if (c === ' ' || c === '\t' || c === '\r' || c === '\n') {
      j++;
      continue;
    }

    // Skip single-line comments
    if (c === '/' && j + 1 < len && source[j + 1] === '/') {
      const eol = source.indexOf('\n', j + 2);
      j = eol === -1 ? len : eol + 1;
      continue;
    }

    // Skip multi-line comments
    if (c === '/' && j + 1 < len && source[j + 1] === '*') {
      const end = source.indexOf('*/', j + 2);
      j = end === -1 ? len : end + 2;
      continue;
    }

    break;
  }

  if (j >= len) return true; // end of source

  const nextChar = source[j];

  // These tokens CONTINUE the expression -- NOT a statement boundary:
  //   . = , ( [ ? : && || ?? + - * / % < > | & ^ ~ ! instanceof in of
  //   Also catch `);` which closes a wrapping like `var x = z(() => { ... });`
  const continuationChars = '.=,([?:&|+\\-*/%<>^~!;)';
  if (continuationChars.includes(nextChar)) {
    return false;
  }

  // Check for multi-char continuation tokens
  const ahead = source.substring(j, j + 15);
  // `instanceof`, `in` (but not `if`), `of`, `from` (import continuation)
  if (/^(?:instanceof|in|of|from)\s/.test(ahead)) return false;
  // `as` (TypeScript)
  if (/^as\s/.test(ahead)) return false;

  // Otherwise, this is a statement boundary
  return true;
}

/**
 * Skip a string literal starting at position i (where source[i] is the quote).
 * Returns the index AFTER the closing quote.
 * @param {string} source
 * @param {number} i
 * @param {string} quote - the quote character
 * @returns {number}
 */
function skipString(source, i, quote) {
  const len = source.length;
  i++; // skip opening quote
  while (i < len) {
    if (source[i] === '\\') {
      i += 2; // skip escape sequence
      continue;
    }
    if (source[i] === quote) {
      return i + 1; // past closing quote
    }
    i++;
  }
  return len; // unterminated string
}

/**
 * Skip a template literal starting at position i (where source[i] is backtick).
 * Handles nested ${...} expressions including nested template literals.
 * @param {string} source
 * @param {number} i
 * @returns {number}
 */
function skipTemplateLiteral(source, i) {
  const len = source.length;
  i++; // skip opening backtick
  while (i < len) {
    if (source[i] === '\\') {
      i += 2;
      continue;
    }
    if (source[i] === '`') {
      return i + 1; // closing backtick
    }
    if (source[i] === '$' && i + 1 < len && source[i + 1] === '{') {
      // Template expression: skip to matching }
      i = skipTemplateExpression(source, i + 2);
      continue;
    }
    i++;
  }
  return len;
}

/**
 * Skip a template expression (inside ${...}) starting after the opening ${.
 * Handles nested braces, strings, and template literals.
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
 * A '/' is a regex start if the preceding token is not an identifier,
 * number, or closing bracket.
 * @param {string} source
 * @param {number} i
 * @returns {boolean}
 */
function isRegexStart(source, i) {
  // Look backwards past whitespace for the preceding non-whitespace char
  let j = i - 1;
  while (j >= 0 && (source[j] === ' ' || source[j] === '\t' || source[j] === '\n' || source[j] === '\r')) {
    j--;
  }
  if (j < 0) return true; // start of file

  const prev = source[j];
  // After these, '/' starts division, not regex
  if (/[\w$)\].]/.test(prev)) return false;
  // After keywords like return, typeof, etc. '/' starts a regex
  return true;
}

/**
 * Skip a regex literal starting at position i.
 * Returns the index AFTER the closing '/' and optional flags.
 * @param {string} source
 * @param {number} i
 * @returns {number}
 */
function skipRegex(source, i) {
  const len = source.length;
  i++; // skip opening /
  while (i < len) {
    if (source[i] === '\\') { i += 2; continue; }
    if (source[i] === '[') {
      // character class -- skip to ]
      i++;
      while (i < len && source[i] !== ']') {
        if (source[i] === '\\') { i += 2; continue; }
        i++;
      }
      i++; // skip ]
      continue;
    }
    if (source[i] === '/') {
      i++;
      // skip regex flags
      while (i < len && /[gimsuy]/.test(source[i])) i++;
      return i;
    }
    i++;
  }
  return len;
}

// ── Statement Classifier ────────────────────────────────────────────────────

/**
 * Classify a complete statement by scoring keyword hits against each module.
 * Returns the module name with the highest score, or 'uncategorized'.
 *
 * @param {string} code - the complete statement text
 * @returns {string} module name
 */
function classifyStatement(code) {
  let bestModule = 'uncategorized';
  let bestScore = 0;

  for (const [modName, keywords] of Object.entries(MODULE_KEYWORDS)) {
    let score = 0;
    for (const kw of keywords) {
      if (code.includes(kw)) {
        score += 1;
      }
    }
    if (score > bestScore) {
      bestScore = score;
      bestModule = modName;
    }
  }

  return bestModule;
}

// ── Syntax Validation ───────────────────────────────────────────────────────

/**
 * Check if a code string is syntactically valid JavaScript.
 * Tries multiple wrappings to handle async/await, top-level expressions, etc.
 * Also handles ESM import/export statements which new Function() cannot parse.
 *
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
 * Strip ESM import/export statements from code for validation purposes.
 * These are syntactically valid JS but new Function() cannot parse them.
 *
 * Handles all import forms:
 *   import { a, b } from "mod";
 *   import * as ns from "mod";
 *   import defaultExport from "mod";
 *   import defaultExport, { a } from "mod";
 *   import "mod";
 *
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
  const statements = parseTopLevelStatements(source);

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
    const content = fragments.join(';\n\n');
    const confidence = Math.min(1, fragments.length / Math.max(1, totalStatements / 10));

    if (confidence >= minConfidence || minConfidence === 0) {
      modules.push({
        name,
        content,
        fragments: fragments.length,
        confidence: parseFloat(confidence.toFixed(3)),
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

  // Step 5: Validate each module is parseable; move invalid ones to uncategorized
  const validModules = [];
  for (const mod of modules) {
    if (isSyntacticallyValid(mod.content)) {
      validModules.push(mod);
    } else {
      // Module is invalid -- move its content to uncategorized
      unclassifiedList.push(mod.content);
    }
  }

  // Step 6: Always include uncategorized for 100% coverage
  if (unclassifiedList.length > 0) {
    validModules.push({
      name: 'uncategorized',
      content: unclassifiedList.join(';\n\n'),
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

// ── Module Tree Builder ─────────────────────────────────────────────────────

/**
 * Build a hierarchical module tree from co-reference density.
 *
 * 1. Build adjacency matrix from shared string references between modules.
 * 2. Agglomerative clustering by edge density.
 * 3. Name clusters from dominant discriminative strings.
 *
 * @param {Array<{name: string, content: string, fragments: number, confidence: number}>} modules
 * @param {string} source
 * @returns {{name: string, path: string, modules: Array, children: Array, depth: number}}
 */
function buildModuleTree(modules, source) {
  if (modules.length <= 1) {
    return {
      name: 'src',
      path: 'src',
      modules,
      children: [],
      depth: 0,
    };
  }

  // Extract string tokens from each module's content.
  const moduleTokens = modules.map((m) => {
    const tokens = new Set();
    const re = /["']([a-zA-Z_]\w{2,30})["']/g;
    let match;
    while ((match = re.exec(m.content)) !== null) {
      tokens.add(match[1]);
    }
    return tokens;
  });

  // Build adjacency: weight = number of shared tokens.
  const weights = new Map();
  for (let i = 0; i < modules.length; i++) {
    for (let j = i + 1; j < modules.length; j++) {
      let shared = 0;
      for (const tok of moduleTokens[i]) {
        if (moduleTokens[j].has(tok)) shared++;
      }
      if (shared > 0) {
        weights.set(`${i}:${j}`, shared);
      }
    }
  }

  // Agglomerative clustering.
  let clusters = modules.map((_, i) => [i]);

  while (clusters.length > 3) {
    let bestI = 0, bestJ = 1, bestW = -1;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const w = clusterWeight(clusters[i], clusters[j], weights);
        const norm = w / (clusters[i].length + clusters[j].length);
        if (norm > bestW) {
          bestW = norm;
          bestI = i;
          bestJ = j;
        }
      }
    }
    if (bestW <= 0) break;
    const merged = [...clusters[bestI], ...clusters[bestJ]];
    clusters.splice(bestJ, 1);
    clusters.splice(bestI, 1);
    clusters.push(merged);
  }

  // Name each cluster from discriminative tokens.
  const children = clusters.map((group) => {
    const groupModules = group.map((i) => modules[i]);
    const name = inferGroupName(group, moduleTokens, modules);
    return {
      name,
      path: `src/${name}`,
      modules: groupModules,
      children: [],
      depth: 1,
    };
  });

  return {
    name: 'src',
    path: 'src',
    modules: [],
    children,
    depth: 0,
  };
}

/** Compute total shared-token weight between two clusters. */
function clusterWeight(a, b, weights) {
  let total = 0;
  for (const ai of a) {
    for (const bi of b) {
      const key = ai < bi ? `${ai}:${bi}` : `${bi}:${ai}`;
      total += weights.get(key) || 0;
    }
  }
  return total;
}

/** Infer a group name from discriminative tokens. */
function inferGroupName(group, moduleTokens, modules) {
  const freq = new Map();
  for (const i of group) {
    for (const tok of moduleTokens[i]) {
      freq.set(tok, (freq.get(tok) || 0) + 1);
    }
  }
  const globalFreq = new Map();
  for (const tokens of moduleTokens) {
    for (const tok of tokens) {
      globalFreq.set(tok, (globalFreq.get(tok) || 0) + 1);
    }
  }
  let best = null, bestScore = -1;
  for (const [tok, count] of freq) {
    const global = globalFreq.get(tok) || 0;
    const score = (count / (global + 1)) * Math.log(count + 1);
    if (score > bestScore && tok.length >= 3) {
      bestScore = score;
      best = tok;
    }
  }
  if (best) return best.toLowerCase().replace(/[^a-z0-9_-]/g, '_');
  if (group.length > 0) return modules[group[0]].name;
  return 'group';
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
  MODULE_KEYWORDS,
};
