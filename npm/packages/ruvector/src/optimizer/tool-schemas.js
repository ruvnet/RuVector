/**
 * Decompiled tool schemas from Claude Code v2.1.91.
 *
 * These schemas mirror the exact inputSchema definitions found in the
 * decompiled binary. They enable Tier-1 (WASM booster) validation of
 * tool calls without invoking the LLM, and provide a known-pattern
 * cache for deterministic tool responses.
 */

'use strict';

/**
 * Tool input schemas extracted from decompiled Claude Code v2.1.91.
 * A trailing '?' on the type string means the field is optional.
 */
const TOOL_SCHEMAS = {
  Bash: {
    command: 'string',
    description: 'string?',
    timeout: 'number?',
    run_in_background: 'boolean?',
  },
  Read: {
    file_path: 'string',
    offset: 'number?',
    limit: 'number?',
    pages: 'string?',
  },
  Edit: {
    file_path: 'string',
    old_string: 'string',
    new_string: 'string',
    replace_all: 'boolean?',
  },
  Write: {
    file_path: 'string',
    content: 'string',
  },
  Glob: {
    pattern: 'string',
    path: 'string?',
  },
  Grep: {
    pattern: 'string',
    path: 'string?',
    glob: 'string?',
    type: 'string?',
    output_mode: 'string?',
    '-i': 'boolean?',
    '-n': 'boolean?',
    '-A': 'number?',
    '-B': 'number?',
    '-C': 'number?',
    context: 'number?',
    head_limit: 'number?',
    offset: 'number?',
    multiline: 'boolean?',
  },
  Agent: {
    prompt: 'string',
    description: 'string?',
  },
  WebFetch: {
    url: 'string',
  },
  WebSearch: {
    query: 'string',
  },
  TodoWrite: {
    todos: 'array',
  },
  NotebookEdit: {
    notebook_path: 'string',
    cell_number: 'number',
    new_source: 'string',
    cell_type: 'string?',
  },
  ToolSearch: {
    query: 'string',
    max_results: 'number?',
  },
  Skill: {
    skill: 'string',
    args: 'string?',
  },
  EnterWorktree: {
    name: 'string?',
    branch: 'string?',
    path: 'string?',
  },
  ExitWorktree: {
    name: 'string?',
  },
};

/** All known tool names. */
const TOOL_NAMES = Object.keys(TOOL_SCHEMAS);

/** Base type checkers for schema validation. */
const TYPE_CHECKS = {
  string: (v) => typeof v === 'string',
  number: (v) => typeof v === 'number' && !Number.isNaN(v),
  boolean: (v) => typeof v === 'boolean',
  array: (v) => Array.isArray(v),
  object: (v) => v !== null && typeof v === 'object' && !Array.isArray(v),
};

/**
 * Validate a tool call's arguments against the known schema.
 * Returns { valid: true } or { valid: false, errors: string[] }.
 *
 * This is intended as a fast Tier-1 check — it validates structure
 * without calling any LLM. Unknown tool names pass validation
 * (we assume they are custom MCP tools).
 *
 * @param {string} toolName
 * @param {object} args
 * @returns {{ valid: boolean, errors?: string[] }}
 */
function validateToolCall(toolName, args) {
  const schema = TOOL_SCHEMAS[toolName];
  if (!schema) {
    // Unknown tool — let it through (may be an MCP extension)
    return { valid: true };
  }

  if (!args || typeof args !== 'object') {
    return { valid: false, errors: ['args must be a non-null object'] };
  }

  const errors = [];

  for (const [field, typeSpec] of Object.entries(schema)) {
    const isOptional = typeSpec.endsWith('?');
    const baseType = isOptional ? typeSpec.slice(0, -1) : typeSpec;
    const value = args[field];

    if (value === undefined || value === null) {
      if (!isOptional) {
        errors.push(`Missing required field: ${field}`);
      }
      continue;
    }

    const checker = TYPE_CHECKS[baseType];
    if (checker && !checker(value)) {
      errors.push(`Field '${field}' expected ${baseType}, got ${typeof value}`);
    }
  }

  return errors.length === 0
    ? { valid: true }
    : { valid: false, errors };
}

/**
 * Cached patterns for common tool calls that produce deterministic
 * or near-deterministic results. The booster can skip LLM calls
 * entirely for these patterns.
 */
const CACHED_PATTERNS = {
  'Glob:**/*.ts': { pattern: '**/*.ts' },
  'Glob:**/*.js': { pattern: '**/*.js' },
  'Glob:**/*.py': { pattern: '**/*.py' },
  'Glob:**/*.rs': { pattern: '**/*.rs' },
  'Glob:**/*.json': { pattern: '**/*.json' },
  'Glob:package.json': { pattern: '**/package.json' },
  'Glob:Cargo.toml': { pattern: '**/Cargo.toml' },
  'Grep:import.*from': { pattern: 'import.*from', type: 'js' },
  'Grep:use ': { pattern: '^use ', type: 'rust' },
  'Grep:TODO|FIXME': { pattern: 'TODO|FIXME' },
};

/**
 * Look up a cached pattern. Returns the pattern args or null.
 * @param {string} toolName
 * @param {string} shortKey - a short identifier for the pattern
 * @returns {object|null}
 */
function getCachedPattern(toolName, shortKey) {
  return CACHED_PATTERNS[`${toolName}:${shortKey}`] || null;
}

module.exports = {
  TOOL_SCHEMAS,
  TOOL_NAMES,
  validateToolCall,
  CACHED_PATTERNS,
  getCachedPattern,
};
