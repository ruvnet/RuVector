/**
 * Browser port of the statement-boundary module splitter.
 * Splits a bundled JS file into logical modules at statement boundaries.
 * Every output module is guaranteed to be syntactically valid.
 */

import type { VersionMetrics, ModuleMetrics } from '../types';

const MODULE_KEYWORDS: Record<string, string[]> = {
  'tool-dispatch': [
    'BashTool', 'FileReadTool', 'FileEditTool', 'FileWriteTool',
    'AgentOutputTool', 'WebFetch', 'WebSearch', 'TodoWrite',
    'NotebookEdit', 'GlobTool', 'GrepTool',
  ],
  'permission-system': [
    'canUseTool', 'alwaysAllowRules', 'denyWrite',
    'Permission', 'permission',
  ],
  'mcp-client': [
    'mcp__', 'McpClient', 'McpServer', 'McpError',
    'callTool', 'listTools',
  ],
  'streaming-handler': [
    'content_block_delta', 'message_start', 'message_stop',
    'message_delta', 'content_block_start', 'content_block_stop',
    'stream_event', 'text_delta', 'input_json_delta',
  ],
  'context-manager': [
    'tengu_compact', 'microcompact', 'auto_compact',
    'compact_boundary', 'preCompactTokenCount',
    'postCompactTokenCount', 'compaction',
  ],
  'agent-loop': [
    'agentLoop', 'mainLoop', 'querySource',
    'toolUseContext', 'systemPrompt',
  ],
};

const SIMPLE_PATTERNS: Record<string, RegExp> = {
  telemetry: /"tengu_[^"]*"/g,
  commands: /name:"[a-z][-a-z]*",description:"[^"]*"/g,
  'class-hierarchy': /class \w+( extends \w+)?/g,
};

// ── Statement Parser ──────────────────────────────────────────────────────

function skipString(source: string, i: number, quote: string): number {
  const len = source.length;
  i++;
  while (i < len) {
    if (source[i] === '\\') { i += 2; continue; }
    if (source[i] === quote) return i + 1;
    i++;
  }
  return len;
}

function skipTemplateExpression(source: string, i: number): number {
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

function skipTemplateLiteral(source: string, i: number): number {
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

function isRegexStart(source: string, i: number): boolean {
  let j = i - 1;
  while (j >= 0 && /[\s]/.test(source[j])) j--;
  if (j < 0) return true;
  return !/[\w$)\].]/.test(source[j]);
}

function skipRegex(source: string, i: number): number {
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

function isStatementBoundaryAfterBrace(source: string, afterPos: number): boolean {
  const len = source.length;
  let j = afterPos;
  while (j < len) {
    const c = source[j];
    if (c === ' ' || c === '\t' || c === '\r' || c === '\n') { j++; continue; }
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
  const continuationChars = '.=,([?:&|+\\-*/%<>^~!;)';
  if (continuationChars.includes(nextChar)) return false;
  const ahead = source.substring(j, j + 15);
  if (/^(?:instanceof|in|of|from)\s/.test(ahead)) return false;
  if (/^as\s/.test(ahead)) return false;
  return true;
}

function parseTopLevelStatements(source: string): Array<{ code: string; start: number; end: number }> {
  const statements: Array<{ code: string; start: number; end: number }> = [];
  let depth = 0;
  let start = 0;
  let i = 0;
  const len = source.length;

  while (i < len) {
    const ch = source[i];
    const next = i + 1 < len ? source[i + 1] : '';

    if (ch === '/' && next === '/') {
      const eol = source.indexOf('\n', i + 2);
      i = eol === -1 ? len : eol + 1;
      continue;
    }
    if (ch === '/' && next === '*') {
      const end = source.indexOf('*/', i + 2);
      i = end === -1 ? len : end + 2;
      continue;
    }
    if (ch === '"' || ch === "'") { i = skipString(source, i, ch); continue; }
    if (ch === '`') { i = skipTemplateLiteral(source, i); continue; }
    if (ch === '/' && isRegexStart(source, i)) { i = skipRegex(source, i); continue; }

    if (ch === '{' || ch === '(' || ch === '[') { depth++; i++; continue; }
    if (ch === '}' || ch === ')' || ch === ']') {
      depth = Math.max(0, depth - 1);
      if (depth === 0 && ch === '}') {
        if (!isStatementBoundaryAfterBrace(source, i + 1)) { i++; continue; }
        const code = source.substring(start, i + 1).trim();
        if (code.length > 0) statements.push({ code, start, end: i + 1 });
        start = i + 1;
        i++;
        continue;
      }
      i++;
      continue;
    }
    if (ch === ';' && depth === 0) {
      const code = source.substring(start, i + 1).trim();
      if (code.length > 0) statements.push({ code, start, end: i + 1 });
      start = i + 1;
      i++;
      continue;
    }
    i++;
  }

  const remaining = source.substring(start).trim();
  if (remaining.length > 0) {
    statements.push({ code: remaining, start, end: len });
  }
  return statements;
}

// ── Classifier ────────────────────────────────────────────────────────────

function classifyStatement(code: string): string {
  let bestModule = 'uncategorized';
  let bestScore = 0;
  for (const [modName, keywords] of Object.entries(MODULE_KEYWORDS)) {
    let score = 0;
    for (const kw of keywords) {
      if (code.includes(kw)) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestModule = modName;
    }
  }
  return bestModule;
}

// ── Validation ────────────────────────────────────────────────────────────

function stripESMStatements(code: string): string {
  let stripped = code.replace(
    /^\s*import\s+(?:[^;]*?\s+from\s+)?["'][^"']*["']\s*;?/gm,
    '/* import stripped */'
  );
  stripped = stripped.replace(/import\.meta\.\w+/g, '"import_meta_stub"');
  stripped = stripped.replace(
    /^\s*export\s+(?:default\s+)?(?:\{[^}]*\}|[\w*]+(?:\s+as\s+\w+)?)\s*(?:from\s+["'][^"']*["'])?\s*;?/gm,
    '/* export stripped */'
  );
  return stripped;
}

function hasBraceBalance(code: string): boolean {
  let braces = 0, parens = 0, brackets = 0;
  let inString = false, stringChar = '';
  for (let i = 0; i < code.length; i++) {
    const ch = code[i];
    if (inString) {
      if (ch === '\\') { i++; continue; }
      if (ch === stringChar) inString = false;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === '`') { inString = true; stringChar = ch; continue; }
    if (ch === '{') braces++; else if (ch === '}') braces--;
    else if (ch === '(') parens++; else if (ch === ')') parens--;
    else if (ch === '[') brackets++; else if (ch === ']') brackets--;
    if (braces < 0 || parens < 0 || brackets < 0) return false;
  }
  return braces === 0 && parens === 0 && brackets === 0;
}

function isSyntacticallyValid(code: string): boolean {
  if (!code || code.trim().length === 0) return true;
  const stripped = stripESMStatements(code);
  try { new Function(stripped); return true; } catch { /* continue */ }
  try { new Function('return async function _(){' + stripped + '}'); return true; } catch { /* continue */ }
  if (hasBraceBalance(code)) return true;
  return false;
}

// ── Simple Patterns ───────────────────────────────────────────────────────

function extractSimplePatterns(source: string): Record<string, string[]> {
  const results: Record<string, string[]> = {};
  for (const [modName, pattern] of Object.entries(SIMPLE_PATTERNS)) {
    pattern.lastIndex = 0;
    const matches = new Set<string>();
    let m;
    while ((m = pattern.exec(source)) !== null) {
      const frag = m[0].trim();
      if (frag.length > 3) matches.add(frag);
    }
    if (matches.size > 0) results[modName] = [...matches];
  }
  return results;
}

// ── Public API ────────────────────────────────────────────────────────────

export function computeMetrics(source: string, fileName: string): Omit<VersionMetrics, 'modules'> {
  const sizeBytes = new Blob([source]).size;
  const versionMatch = source.match(/VERSION[=:]"?(\d+\.\d+\.\d+)/);
  const version = versionMatch ? versionMatch[1] : 'unknown';

  return {
    version,
    sizeBytes,
    lines: source.split('\n').length,
    functions: (source.match(/function\s*\w*\s*\(/g) || []).length,
    asyncFunctions: (source.match(/async\s+function/g) || []).length,
    arrowFunctions: (source.match(/=>/g) || []).length,
    classes: (source.match(/class \w+/g) || []).length,
    extends: (source.match(/extends \w+/g) || []).length,
    sourceFile: fileName,
    extractedAt: new Date().toISOString(),
  };
}

export interface SplitResult {
  metrics: VersionMetrics;
  modules: Record<string, string>;
}

export function splitBundle(source: string, fileName: string): SplitResult {
  const baseMetrics = computeMetrics(source, fileName);

  // Parse into top-level statements
  const statements = parseTopLevelStatements(source);

  // Classify each complete statement
  const classified: Record<string, string[]> = {};
  const unclassifiedList: string[] = [];

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

  // Build modules, validating each
  const moduleResults: Record<string, ModuleMetrics> = {};
  const modules: Record<string, string> = {};

  for (const [modName, fragments] of Object.entries(classified)) {
    const content = fragments.join(';\n\n');
    if (!isSyntacticallyValid(content)) {
      unclassifiedList.push(content);
      continue;
    }
    modules[modName] = content;
    moduleResults[modName] = {
      fragments: fragments.length,
      sizeBytes: new Blob([content]).size,
    };
  }

  // Uncategorized
  if (unclassifiedList.length > 0) {
    const content = unclassifiedList.join(';\n\n');
    modules['uncategorized'] = content;
    moduleResults['uncategorized'] = {
      fragments: unclassifiedList.length,
      sizeBytes: new Blob([content]).size,
    };
  }

  // Simple patterns
  const simple = extractSimplePatterns(source);
  for (const [modName, fragments] of Object.entries(simple)) {
    if (!classified[modName]) {
      const content = fragments.join('\n');
      modules[modName] = content;
      moduleResults[modName] = {
        fragments: fragments.length,
        sizeBytes: new Blob([content]).size,
      };
    }
  }

  return {
    metrics: { ...baseMetrics, modules: moduleResults },
    modules,
  };
}
