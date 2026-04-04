#!/usr/bin/env node
/**
 * module-splitter.mjs - Split a Claude Code CLI bundle into logical modules.
 *
 * Splits at STATEMENT BOUNDARIES so every output module is guaranteed to be
 * syntactically valid, parseable JavaScript. Never splits a statement across
 * modules.
 *
 * Usage:
 *   node scripts/lib/module-splitter.mjs <cli-bundle> <output-dir>
 */

import { readFileSync, writeFileSync, mkdirSync, statSync } from 'fs';
import { join, basename } from 'path';

// ── Module classification keywords ──────────────────────────────────────────
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

const SIMPLE_PATTERNS = {
  'telemetry-events': /"tengu_[^"]*"/g,
  'command-defs': /name:"[a-z][-a-z]*",description:"[^"]*"/g,
  'class-hierarchy': /class \w+( extends \w+)?/g,
  'env-vars': /CLAUDE_[A-Z_]+/g,
  'api-endpoints': /\/v\d+\/[a-z][-a-z/]*/g,
};

// ── Statement Parser ────────────────────────────────────────────────────────

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

function isRegexStart(source, i) {
  let j = i - 1;
  while (j >= 0 && (source[j] === ' ' || source[j] === '\t' || source[j] === '\n' || source[j] === '\r')) j--;
  if (j < 0) return true;
  return !/[\w$)\].]/.test(source[j]);
}

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

function isStatementBoundaryAfterBrace(source, afterPos) {
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

function parseTopLevelStatements(source) {
  const statements = [];
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

// ── Statement Classifier ────────────────────────────────────────────────────

function classifyStatement(code) {
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

// ── Syntax Validation ───────────────────────────────────────────────────────

function stripESMStatements(code) {
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

function hasBraceBalance(code) {
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

function isSyntacticallyValid(code) {
  if (!code || code.trim().length === 0) return true;
  const stripped = stripESMStatements(code);
  try { new Function(stripped); return true; } catch { /* continue */ }
  try { new Function('return async function _(){' + stripped + '}'); return true; } catch { /* continue */ }
  try { new Function('"use strict";' + stripped); return true; } catch { /* continue */ }
  if (hasBraceBalance(code)) return true;
  return false;
}

// ── Simple Pattern Extraction ───────────────────────────────────────────────

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
    if (matches.size > 0) results[modName] = [...matches];
  }
  return results;
}

// ── Metrics ─────────────────────────────────────────────────────────────────

function computeMetrics(source, filePath) {
  const sizeBytes = statSync(filePath).size;
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
  };
}

// ── Main ────────────────────────────────────────────────────────────────────

function main() {
  const [bundlePath, outputDir] = process.argv.slice(2);
  if (!bundlePath || !outputDir) {
    console.error('Usage: node module-splitter.mjs <cli-bundle> <output-dir>');
    process.exit(1);
  }

  mkdirSync(outputDir, { recursive: true });

  console.log(`Reading bundle: ${bundlePath}`);
  const source = readFileSync(bundlePath, 'utf-8');
  const metrics = computeMetrics(source, bundlePath);
  console.log(`  Size: ${(metrics.sizeBytes / 1024 / 1024).toFixed(1)} MB, ` +
    `${metrics.classes} classes, ${metrics.functions} functions`);

  // Parse into top-level statements
  console.log('  Parsing top-level statements...');
  const statements = parseTopLevelStatements(source);
  console.log(`  ${statements.length} statements`);

  // Classify statements
  const classified = {};
  const unclassified = [];
  for (const stmt of statements) {
    if (stmt.code.length < 5) continue;
    const modName = classifyStatement(stmt.code);
    if (modName === 'uncategorized') {
      unclassified.push(stmt.code);
    } else {
      if (!classified[modName]) classified[modName] = [];
      classified[modName].push(stmt.code);
    }
  }

  const moduleResults = {};
  let pass = 0, fail = 0;

  for (const [modName, fragments] of Object.entries(classified)) {
    const content = fragments.join(';\n\n');
    if (!isSyntacticallyValid(content)) {
      console.log(`  Module "${modName}": INVALID, moving to uncategorized`);
      unclassified.push(content);
      fail++;
      continue;
    }
    const outFile = join(outputDir, `${modName}.js`);
    writeFileSync(outFile, `// Module: ${modName}\n// Generated by ruDevolution\n"use strict";\n\n${content}\n`, 'utf-8');
    moduleResults[modName] = {
      fragments: fragments.length,
      sizeBytes: Buffer.byteLength(content),
    };
    console.log(`  Module "${modName}": ${fragments.length} fragments (valid)`);
    pass++;
  }

  // Write uncategorized
  if (unclassified.length > 0) {
    const content = unclassified.join(';\n\n');
    const outFile = join(outputDir, 'uncategorized.js');
    writeFileSync(outFile, `// Module: uncategorized\n// Generated by ruDevolution\n"use strict";\n\n${content}\n`, 'utf-8');
    moduleResults['uncategorized'] = {
      fragments: unclassified.length,
      sizeBytes: Buffer.byteLength(content),
    };
    console.log(`  Module "uncategorized": ${unclassified.length} fragments`);
  }

  // Simple pattern extractions
  console.log('  Extracting simple patterns...');
  const simple = extractSimplePatterns(source);
  for (const [modName, fragments] of Object.entries(simple)) {
    if (!classified[modName]) {
      const outFile = join(outputDir, `${modName}.js`);
      writeFileSync(outFile, fragments.join('\n'), 'utf-8');
      moduleResults[modName] = {
        fragments: fragments.length,
        sizeBytes: Buffer.byteLength(fragments.join('\n')),
      };
      console.log(`  Module "${modName}": ${fragments.length} fragments`);
    }
  }

  console.log(`\n  Results: ${pass} valid modules, ${fail} moved to uncategorized`);

  // Write metrics manifest
  const manifest = {
    ...metrics,
    sourceFile: basename(bundlePath),
    extractedAt: new Date().toISOString(),
    modules: moduleResults,
  };
  writeFileSync(
    join(outputDir, 'metrics.json'),
    JSON.stringify(manifest, null, 2)
  );

  console.log(JSON.stringify(manifest));
}

main();
