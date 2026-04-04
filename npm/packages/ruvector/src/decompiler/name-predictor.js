/**
 * name-predictor.js - Pattern-based name prediction for minified identifiers.
 *
 * Uses the 210+ training patterns from claude-code-patterns.json to infer
 * meaningful names based on context strings, property accesses, and
 * structural patterns found near each identifier.
 *
 * Falls back to structural heuristics when no pattern matches.
 */

'use strict';

const fs = require('fs');
const path = require('path');

/** Cached patterns array — loaded once. */
let _cachedPatterns = null;

/**
 * Load patterns from the JSON training data.
 * @param {string} [patternPath] - override path to patterns JSON
 * @returns {Array<{context_strings: string[], property_names: string[], inferred_name: string, module_hint: string, confidence: number}>}
 */
function loadPatterns(patternPath) {
  if (_cachedPatterns && !patternPath) return _cachedPatterns;

  const defaultPath = path.resolve(
    __dirname,
    '../../../../../crates/ruvector-decompiler/data/claude-code-patterns.json',
  );
  const resolved = patternPath || defaultPath;

  try {
    const raw = fs.readFileSync(resolved, 'utf-8');
    _cachedPatterns = JSON.parse(raw);
    return _cachedPatterns;
  } catch {
    // Pattern file not found — return empty, rely on structural rules
    _cachedPatterns = [];
    return _cachedPatterns;
  }
}

/**
 * Structural rename rules — deterministic, no model needed.
 * Ordered by specificity (most specific first).
 */
const STRUCTURAL_RULES = [
  // Async generators
  {
    test: (decl) => /async\s+function\s*\*/.test(decl),
    nameFrom: (decl, ctx) => {
      if (ctx.some((s) => s.includes('agent')) && ctx.some((s) => s.includes('loop'))) return 'agentLoop';
      if (ctx.some((s) => s.includes('stream'))) return 'streamGenerator';
      return 'asyncGenerator';
    },
    confidence: 0.7,
  },
  // Error subclasses
  {
    test: (decl) => /class\s+\w+\s+extends\s+Error/.test(decl),
    nameFrom: (decl) => {
      const match = decl.match(/class\s+(\w+)\s+extends\s+(\w*Error)/);
      if (match && !isMinifiedLike(match[1])) return match[1];
      return 'CustomError';
    },
    confidence: 0.85,
  },
  // Class extending another class
  {
    test: (decl) => /class\s+\w+\s+extends\s+\w+/.test(decl),
    nameFrom: (decl) => {
      const match = decl.match(/class\s+(\w+)\s+extends\s+(\w+)/);
      if (match && !isMinifiedLike(match[1])) return match[1];
      if (match) return `${match[2]}Subclass`;
      return null;
    },
    confidence: 0.75,
  },
  // Regular class
  {
    test: (decl) => /class\s+\w+/.test(decl),
    nameFrom: (decl) => {
      const match = decl.match(/class\s+(\w+)/);
      if (match && !isMinifiedLike(match[1])) return match[1];
      return null;
    },
    confidence: 0.7,
  },
  // Export default function
  {
    test: (decl) => /export\s+default\s+function/.test(decl),
    nameFrom: () => 'defaultExport',
    confidence: 0.5,
  },
  // Named function
  {
    test: (decl) => /function\s+([a-zA-Z_$]\w+)/.test(decl),
    nameFrom: (decl) => {
      const match = decl.match(/function\s+([a-zA-Z_$]\w+)/);
      if (match && !isMinifiedLike(match[1])) return match[1];
      return null;
    },
    confidence: 0.8,
  },
];

/**
 * Parameter naming rules based on usage context.
 */
const PARAM_RULES = [
  { context: ['.messages', 'systemPrompt', 'canUseTool'], name: 'params', type: 'AgentLoopParams' },
  { context: ['.method', '.url', '.headers'], name: 'request', type: 'Request' },
  { context: ['.status', '.json', '.send'], name: 'response', type: 'Response' },
  { context: ['.next', '.done', '.value'], name: 'iterator', type: 'Iterator' },
  { context: ['.emit', '.on', '.removeListener'], name: 'emitter', type: 'EventEmitter' },
  { context: ['.pipe', '.write', '.end'], name: 'stream', type: 'Stream' },
  { context: ['.query', '.params', '.body'], name: 'req', type: 'Request' },
  { context: ['.resolve', '.reject'], name: 'promise', type: 'Promise' },
  { context: ['.name', '.version', '.description'], name: 'packageInfo', type: 'PackageInfo' },
  { context: ['.key', '.value', '.ttl'], name: 'cacheEntry', type: 'CacheEntry' },
  { context: ['.token', '.user', '.role'], name: 'session', type: 'Session' },
  { context: ['.width', '.height', '.x', '.y'], name: 'rect', type: 'Rect' },
  { context: ['.type', '.data', '.target'], name: 'event', type: 'Event' },
  { context: ['.path', '.content', '.encoding'], name: 'file', type: 'FileInfo' },
  { context: ['.host', '.port', '.protocol'], name: 'connectionInfo', type: 'ConnectionInfo' },
];

/**
 * Check if a name looks minified (short, no semantic meaning).
 * @param {string} name
 * @returns {boolean}
 */
function isMinifiedLike(name) {
  if (!name) return true;
  return /^[a-zA-Z][a-zA-Z0-9$]{0,2}$/.test(name);
}

/**
 * Score how well a pattern matches the given context.
 *
 * @param {object} pattern - from claude-code-patterns.json
 * @param {string[]} contextStrings - strings found near the identifier
 * @param {string[]} propertyNames - property accesses found near the identifier
 * @returns {number} 0-1 match score
 */
function scorePatternMatch(pattern, contextStrings, propertyNames) {
  let score = 0;
  let totalChecks = 0;

  // Check context string matches
  if (pattern.context_strings && pattern.context_strings.length > 0) {
    for (const ctx of pattern.context_strings) {
      totalChecks++;
      if (contextStrings.some((s) => s.includes(ctx) || ctx.includes(s))) {
        score++;
      }
    }
  }

  // Check property name matches
  if (pattern.property_names && pattern.property_names.length > 0) {
    for (const prop of pattern.property_names) {
      totalChecks++;
      const propAccess = `.${prop}`;
      if (propertyNames.some((p) => p === propAccess || p === prop)) {
        score++;
      }
    }
  }

  if (totalChecks === 0) return 0;
  return score / totalChecks;
}

/**
 * Predict a meaningful name for a minified identifier.
 *
 * @param {string} minifiedName - the original short name (e.g. "s$")
 * @param {string[]} contextStrings - strings/properties near the identifier
 * @param {object} [options]
 * @param {string} [options.declaration] - the declaration statement
 * @param {string} [options.patternPath] - path to patterns JSON
 * @param {number} [options.minConfidence=0.3] - minimum confidence to accept
 * @returns {{name: string, confidence: number, source: string}|null}
 */
function predictName(minifiedName, contextStrings, options = {}) {
  const { declaration = '', patternPath, minConfidence = 0.3 } = options;

  // Separate context strings from property accesses
  const props = contextStrings.filter((s) => s.startsWith('.'));
  const strings = contextStrings.filter((s) => !s.startsWith('.'));

  // 1. Try direct-assignment analysis (highest precision)
  // If X = Y.propertyName, then X should be named "propertyName"
  const directName = inferFromDirectAssignment(minifiedName, declaration);
  if (directName) {
    return { name: directName.name, confidence: directName.confidence, source: 'direct-assign' };
  }

  // 2. Try structural rules (high specificity)
  for (const rule of STRUCTURAL_RULES) {
    if (rule.test(declaration)) {
      const name = rule.nameFrom(declaration, contextStrings);
      if (name && rule.confidence >= minConfidence) {
        return { name, confidence: rule.confidence, source: 'structural' };
      }
    }
  }

  // 3. Try pattern matching against training data
  const patterns = loadPatterns(patternPath);
  let bestMatch = null;
  let bestScore = 0;

  for (const pattern of patterns) {
    const matchScore = scorePatternMatch(pattern, strings, props);
    const adjustedScore = matchScore * pattern.confidence;

    if (adjustedScore > bestScore && adjustedScore >= minConfidence) {
      bestScore = adjustedScore;
      bestMatch = pattern;
    }
  }

  if (bestMatch) {
    return {
      name: toCamelCase(bestMatch.inferred_name),
      confidence: bestScore,
      source: 'pattern',
    };
  }

  // 3. Try parameter naming rules
  for (const rule of PARAM_RULES) {
    const matches = rule.context.filter((c) =>
      contextStrings.some((s) => s.includes(c)),
    );
    if (matches.length >= 2) {
      return {
        name: rule.name,
        confidence: 0.6,
        source: 'param-rule',
        type: rule.type,
      };
    }
  }

  // 4. Heuristic fallbacks based on usage patterns
  const heuristic = heuristicName(minifiedName, contextStrings, declaration);
  if (heuristic && heuristic.confidence >= minConfidence) {
    return heuristic;
  }

  return null;
}

/**
 * Infer a name from direct assignment patterns.
 *   let B = A.messages  -> B should be "messages"
 *   let Z = await Y2(B, G, ...) -> Z might be "result" or inferred from Y2
 *   var s$ = async function*(...) -> handled by structural rules
 *
 * @param {string} minifiedName
 * @param {string} declaration
 * @returns {{name: string, confidence: number}|null}
 */
function inferFromDirectAssignment(minifiedName, declaration) {
  if (!declaration) return null;
  const escaped = minifiedName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  // Pattern: X = Y.propertyName  (property extraction)
  const propMatch = declaration.match(
    new RegExp(`${escaped}\\s*=\\s*\\w+\\.(\\w{2,30})`)
  );
  if (propMatch && propMatch[1].length > 2) {
    return { name: propMatch[1], confidence: 0.85 };
  }

  // Pattern: X = await Y(...)  (function call result)
  const awaitCallMatch = declaration.match(
    new RegExp(`${escaped}\\s*=\\s*await\\s+(\\w+)\\(`)
  );
  if (awaitCallMatch) {
    const fnName = awaitCallMatch[1];
    if (!isMinifiedLike(fnName)) {
      // Infer from function name: createApiRequest -> apiRequest
      const resultName = fnName.replace(/^(create|get|fetch|make|build|load)/, '').replace(/^./, (c) => c.toLowerCase());
      return { name: resultName || 'result', confidence: 0.7 };
    }
    return { name: 'result', confidence: 0.5 };
  }

  // Pattern: for await (let J of Z)  -> J is an item from the iterable
  const forOfMatch = declaration.match(
    new RegExp(`for\\s+await\\s*\\(\\s*(?:let|const|var)\\s+${escaped}\\s+of\\s+(\\w+)`)
  );
  if (forOfMatch) {
    return { name: 'item', confidence: 0.5 };
  }

  // Pattern: function parameter  (first param of function*)
  const funcParamMatch = declaration.match(
    new RegExp(`function\\s*\\*?\\s*\\w*\\s*\\(\\s*${escaped}(?:\\s*,|\\s*\\))`)
  );
  if (funcParamMatch) {
    // First parameter of a function -> "params" or "options"
    return { name: 'params', confidence: 0.5 };
  }

  return null;
}

/**
 * Heuristic name inference from usage patterns.
 *
 * @param {string} minifiedName
 * @param {string[]} contextStrings
 * @param {string} declaration
 * @returns {{name: string, confidence: number, source: string}|null}
 */
function heuristicName(minifiedName, contextStrings, declaration) {
  const ctx = contextStrings.join(' ').toLowerCase();

  // Callback / handler patterns
  if (ctx.includes('callback') || ctx.includes('handler') || ctx.includes('listener')) {
    return { name: 'handler', confidence: 0.4, source: 'heuristic' };
  }
  // Error variable
  if (/catch\s*\(\s*$/.test(declaration) || ctx.includes('error') && ctx.includes('catch')) {
    return { name: 'error', confidence: 0.5, source: 'heuristic' };
  }
  // Iterator / loop variable used with .length or [i]
  if (ctx.includes('.length') && ctx.includes('for')) {
    return { name: 'items', confidence: 0.4, source: 'heuristic' };
  }
  // Result of await
  if (declaration.includes('await')) {
    if (ctx.includes('fetch') || ctx.includes('request')) {
      return { name: 'response', confidence: 0.5, source: 'heuristic' };
    }
    return { name: 'result', confidence: 0.35, source: 'heuristic' };
  }
  // Boolean-looking (used in conditions)
  if (ctx.includes('if') && (ctx.includes('===true') || ctx.includes('===false') || ctx.includes('!!'))) {
    return { name: 'isEnabled', confidence: 0.35, source: 'heuristic' };
  }

  return null;
}

/**
 * Convert a PascalCase or kebab-case name to camelCase.
 * @param {string} name
 * @returns {string}
 */
function toCamelCase(name) {
  if (!name) return name;
  // If already camelCase, return as-is
  if (/^[a-z]/.test(name) && !name.includes('-') && !name.includes('_')) return name;
  // PascalCase -> camelCase
  if (/^[A-Z]/.test(name) && !name.includes('-') && !name.includes('_')) {
    return name[0].toLowerCase() + name.slice(1);
  }
  // kebab-case or snake_case -> camelCase
  return name.replace(/[-_](.)/g, (_, c) => c.toUpperCase()).replace(/^./, (c) => c.toLowerCase());
}

/**
 * Infer a parameter name from its position and usage.
 * @param {number} index - parameter position (0-based)
 * @param {string[]} contextStrings
 * @returns {string}
 */
function inferParamName(index, contextStrings) {
  // Try param rules
  for (const rule of PARAM_RULES) {
    const matches = rule.context.filter((c) =>
      contextStrings.some((s) => s.includes(c)),
    );
    if (matches.length >= 1) {
      return rule.name;
    }
  }

  // Generic fallback
  const fallbacks = ['param', 'value', 'data', 'input', 'arg', 'item'];
  return fallbacks[index] || `param${index}`;
}

module.exports = {
  predictName,
  loadPatterns,
  scorePatternMatch,
  inferParamName,
  toCamelCase,
  isMinifiedLike,
  STRUCTURAL_RULES,
  PARAM_RULES,
};
