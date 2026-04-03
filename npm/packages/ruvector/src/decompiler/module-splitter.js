/**
 * module-splitter.js - Split a JavaScript bundle into logical modules.
 *
 * Ported from scripts/lib/module-splitter.mjs for use as a library.
 * Takes source code and returns an array of detected modules with their
 * content and classification metadata.
 */

'use strict';

// Module extraction: keyword -> module name.
// A line containing the keyword is assigned to that module.
// Order matters: first match wins for each line.
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
// These provide additional extraction on top of keyword classification.
const SIMPLE_PATTERNS = {
  'telemetry-events': /"tengu_[^"]*"/g,
  'command-defs': /name:"[a-z][-a-z]*",description:"[^"]*"/g,
  'class-hierarchy': /class \w+( extends \w+)?/g,
  'env-vars': /CLAUDE_[A-Z_]+/g,
  'api-endpoints': /\/v\d+\/[a-z][-a-z/]*/g,
};

/**
 * Split source into statement-level chunks.
 * For minified bundles, semicolon splitting gives logical units.
 * @param {string} source
 * @param {number} [maxChunk=2048] - max chunk size in characters
 * @returns {string[]}
 */
function splitStatements(source, maxChunk = 2048) {
  const raw = source.split(';');
  const chunks = [];
  let buffer = '';

  for (const part of raw) {
    if (buffer.length + part.length > maxChunk && buffer.length > 0) {
      chunks.push(buffer);
      buffer = part;
    } else {
      buffer += (buffer ? ';' : '') + part;
    }
  }
  if (buffer.length > 0) chunks.push(buffer);
  return chunks;
}

/**
 * Classify statements into named modules based on keyword matching.
 * @param {string[]} statements
 * @returns {Object<string, string[]>} moduleName -> array of statement strings
 */
function classifyStatements(statements) {
  const modules = {};
  const unclassified = [];

  for (const stmt of statements) {
    if (stmt.length < 10) continue;

    let matched = false;
    for (const [modName, keywords] of Object.entries(MODULE_KEYWORDS)) {
      if (keywords.some((kw) => stmt.includes(kw))) {
        if (!modules[modName]) modules[modName] = [];
        modules[modName].push(stmt.trim());
        matched = true;
        break;
      }
    }

    if (!matched) {
      unclassified.push(stmt.trim());
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

/**
 * Split source code into modules.
 *
 * @param {string} source - the full JavaScript source
 * @param {object} [options]
 * @param {number} [options.minConfidence=0.3] - minimum confidence to include a module
 * @returns {{modules: Array<{name: string, content: string, fragments: number, confidence: number}>, unclassified: string[]}}
 */
function splitModules(source, options = {}) {
  const { minConfidence = 0.3 } = options;
  const statements = splitStatements(source);
  const classified = classifyStatements(statements);
  const simplePatterns = extractSimplePatterns(source);

  const totalStatements = statements.length;
  const modules = [];

  for (const [name, fragments] of Object.entries(classified)) {
    if (name === '_unclassified') continue;
    const confidence = Math.min(1, fragments.length / Math.max(1, totalStatements / 10));
    if (confidence >= minConfidence) {
      modules.push({
        name,
        content: fragments.join(';\n\n'),
        fragments: fragments.length,
        confidence: parseFloat(confidence.toFixed(3)),
      });
    }
  }

  // Merge simple patterns as additional modules
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

  const unclassified = classified['_unclassified'] || [];

  // Always include unclassified as a module for 100% coverage
  if (unclassified.length > 0) {
    modules.push({
      name: 'uncategorized',
      content: unclassified.join(';\n\n'),
      fragments: unclassified.length,
      confidence: 0.1,
    });
  }

  // Build hierarchical tree from co-reference density.
  const tree = buildModuleTree(modules, source);

  return { modules, unclassified, tree };
}

/**
 * Build a hierarchical module tree from co-reference density.
 *
 * The tree emerges from the actual code's dependency graph:
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
  // Collect token frequency in this group.
  const freq = new Map();
  for (const i of group) {
    for (const tok of moduleTokens[i]) {
      freq.set(tok, (freq.get(tok) || 0) + 1);
    }
  }
  // Collect global frequency.
  const globalFreq = new Map();
  for (const tokens of moduleTokens) {
    for (const tok of tokens) {
      globalFreq.set(tok, (globalFreq.get(tok) || 0) + 1);
    }
  }
  // Score by discriminativeness.
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
  // Fallback: use first module name.
  if (group.length > 0) return modules[group[0]].name;
  return 'group';
}

module.exports = {
  splitModules,
  splitStatements,
  classifyStatements,
  extractSimplePatterns,
  buildModuleTree,
  MODULE_KEYWORDS,
};
