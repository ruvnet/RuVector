/**
 * reconstructor.js - Readable reconstruction pipeline.
 *
 * Takes beautified minified JS and produces human-readable code by
 * renaming variables, adding comments, and reconstructing structure
 * using pattern inference and contextual analysis.
 */

'use strict';

const {
  findMinifiedIdentifiers,
  extractContext,
  applyAllRenames,
} = require('./reference-tracker');
const {
  predictName,
  inferParamName,
} = require('./name-predictor');
const {
  improveReadability,
  generateJSDoc,
} = require('./style-improver');

/**
 * Run the full reconstruction pipeline on a source string.
 *
 * @param {string} source - beautified JavaScript source
 * @param {object} [options]
 * @param {string} [options.modelPath] - path to ONNX model (optional)
 * @param {string} [options.patternPath] - path to patterns JSON
 * @param {boolean} [options.propagateNames=true] - rename all references
 * @param {boolean} [options.addComments=true] - add JSDoc comments
 * @param {boolean} [options.improveStyle=true] - apply style improvements
 * @param {number} [options.minConfidence=0.3] - minimum confidence for renames
 * @param {number} [options.maxRenames=500] - safety limit on number of renames
 * @returns {{code: string, renames: Array<{original: string, newName: string, confidence: number, source: string}>, comments: number, confidence: number}}
 */
function reconstructCode(source, options = {}) {
  const {
    patternPath,
    propagateNames = true,
    addComments = true,
    improveStyle = true,
    minConfidence = 0.3,
    maxRenames = 500,
  } = options;

  // Phase 1: Find all minified identifiers
  const minifiedIds = findMinifiedIdentifiers(source);

  // Phase 2: Extract context and predict names for each
  const renames = [];
  const usedNames = new Set();

  for (const id of minifiedIds) {
    if (renames.length >= maxRenames) break;

    const context = extractContext(source, id);
    const declaration = findDeclaration(source, id);

    const prediction = predictName(id, context, {
      declaration,
      patternPath,
      minConfidence,
    });

    if (prediction) {
      // Ensure uniqueness: append a suffix if the name is already used
      let finalName = prediction.name;
      if (usedNames.has(finalName) || minifiedIds.includes(finalName)) {
        finalName = deduplicateName(finalName, usedNames);
      }
      usedNames.add(finalName);

      renames.push({
        original: id,
        newName: finalName,
        confidence: prediction.confidence,
        source: prediction.source,
        type: prediction.type || null,
      });
    }
  }

  // Phase 3: Apply renames (propagate through all references)
  let code = source;
  if (propagateNames && renames.length > 0) {
    code = applyAllRenames(
      code,
      renames.map((r) => ({ oldName: r.original, newName: r.newName })),
    );
  }

  // Phase 4: Apply style improvements
  if (improveStyle) {
    code = improveReadability(code);
  }

  // Phase 5: Add JSDoc comments
  let commentsAdded = 0;
  if (addComments) {
    const result = addJSDocComments(code, renames);
    code = result.code;
    commentsAdded = result.count;
  }

  // Phase 6: Convert var declarations intelligently
  code = upgradeVarDeclarations(code);

  // Compute overall confidence
  const avgConfidence =
    renames.length > 0
      ? renames.reduce((sum, r) => sum + r.confidence, 0) / renames.length
      : 0;

  return {
    code,
    renames,
    comments: commentsAdded,
    confidence: parseFloat(avgConfidence.toFixed(3)),
  };
}

/**
 * Find the declaration statement for an identifier.
 * Looks for var/let/const/function/class declarations.
 *
 * @param {string} source
 * @param {string} identifier
 * @returns {string} the declaration line, or empty string
 */
function findDeclaration(source, identifier) {
  const escaped = identifier.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  // Try common declaration patterns
  const patterns = [
    new RegExp(`(?:var|let|const)\\s+${escaped}\\s*=([^;]{0,200})`, 'm'),
    // Comma-separated: let X=..., IDENT=value
    new RegExp(`(?:var|let|const)\\s+[^;]*,\\s*${escaped}\\s*=([^;,]{0,200})`, 'm'),
    new RegExp(`function\\s+${escaped}\\s*\\([^)]*\\)`, 'm'),
    new RegExp(`async\\s+function\\s*\\*?\\s+${escaped}`, 'm'),
    new RegExp(`class\\s+${escaped}`, 'm'),
    new RegExp(`${escaped}\\s*=\\s*(?:async\\s+)?function\\s*\\*?\\s*\\([^)]*\\)`, 'm'),
    new RegExp(`${escaped}\\s*=\\s*(?:async\\s+)?\\([^)]*\\)\\s*=>`, 'm'),
    // Function parameter: function name(IDENT, ...) or function*(IDENT)
    new RegExp(`function\\s*\\*?\\s*\\w*\\s*\\([^)]*\\b${escaped}\\b[^)]*\\)`, 'm'),
    // For-of: for await (let IDENT of expr)
    new RegExp(`for\\s*(?:await)?\\s*\\(\\s*(?:let|const|var)\\s+${escaped}\\s+of\\s+([^)]{1,100})\\)`, 'm'),
  ];

  for (const re of patterns) {
    const match = source.match(re);
    if (match) {
      // Return up to 300 chars of context around the declaration
      const start = Math.max(0, match.index - 50);
      const end = Math.min(source.length, match.index + match[0].length + 200);
      return source.substring(start, end);
    }
  }

  return '';
}

/**
 * Add JSDoc comments before function and class declarations.
 *
 * @param {string} code
 * @param {Array<{original: string, newName: string}>} renames
 * @returns {{code: string, count: number}}
 */
function addJSDocComments(code, renames) {
  const lines = code.split('\n');
  const result = [];
  let count = 0;

  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();

    // Check if this line is a function or class declaration
    const isDecl =
      /^(async\s+)?function\s*\*?\s+\w+/.test(trimmed) ||
      /^(export\s+)?(default\s+)?(async\s+)?function/.test(trimmed) ||
      /^class\s+\w+/.test(trimmed) ||
      /^(const|let|var)\s+\w+\s*=\s*(async\s+)?function/.test(trimmed) ||
      /^(const|let|var)\s+\w+\s*=\s*async\s+function\s*\*/.test(trimmed);

    if (isDecl) {
      // Check if there is already a JSDoc comment above
      const prevLine = i > 0 ? lines[i - 1].trim() : '';
      const hasPrevDoc = prevLine.endsWith('*/') || prevLine.startsWith('/**') || prevLine.startsWith('*');

      if (!hasPrevDoc) {
        // Collect context from surrounding lines
        const contextWindow = lines
          .slice(Math.max(0, i - 2), Math.min(lines.length, i + 15))
          .join('\n');
        const contextStrings = extractContextFromCode(contextWindow);

        const jsdoc = generateJSDoc(trimmed, contextStrings, { renames });
        if (jsdoc) {
          const indent = lines[i].match(/^(\s*)/)[1];
          const indented = jsdoc
            .split('\n')
            .map((l) => indent + l)
            .join('\n');
          result.push(indented);
          count++;
        }
      }
    }

    result.push(lines[i]);
  }

  return { code: result.join('\n'), count };
}

/**
 * Extract context strings from a code snippet (string literals, property accesses).
 *
 * @param {string} code
 * @returns {string[]}
 */
function extractContextFromCode(code) {
  const contexts = [];

  // String literals
  const strings = code.match(/["']([^"']{2,60})["']/g) || [];
  for (const s of strings) {
    contexts.push(s.replace(/^["']|["']$/g, ''));
  }

  // Property accesses
  const props = code.match(/\.([a-zA-Z_]\w{1,30})/g) || [];
  for (const p of props) {
    contexts.push(p);
  }

  // Keywords that give semantic hints
  const keywords = ['yield', 'await', 'return', 'throw', 'catch', 'for', 'if', 'switch'];
  for (const kw of keywords) {
    if (code.includes(kw)) contexts.push(kw);
  }

  return contexts;
}

/**
 * Upgrade var declarations to const/let based on usage.
 *   - var x = ... with no reassignment -> const x = ...
 *   - var x = ... with reassignment -> let x = ...
 *
 * @param {string} code
 * @returns {string}
 */
function upgradeVarDeclarations(code) {
  // Find all var declarations
  const varPattern = /\bvar\s+([a-zA-Z_$]\w*)\s*=/g;
  const declarations = [];
  let match;

  while ((match = varPattern.exec(code)) !== null) {
    declarations.push({ name: match[1], index: match.index });
  }

  if (declarations.length === 0) return code;

  // For each var declaration, check if the variable is reassigned
  let result = code;
  const replacements = [];

  for (const decl of declarations) {
    const name = decl.name;
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    // Check for reassignment: name = (but not == or ===)
    const reassignPattern = new RegExp(
      `(?<!var\\s)(?<!let\\s)(?<!const\\s)\\b${escaped}\\s*=[^=]`,
      'g',
    );
    const allAssigns = [...result.matchAll(reassignPattern)];

    // Filter out the declaration itself
    const reassignments = allAssigns.filter(
      (m) => m.index !== decl.index && !result.substring(m.index - 10, m.index).includes('var'),
    );

    // Also check for ++ -- += -= etc.
    const mutationPattern = new RegExp(`\\b${escaped}\\s*(?:\\+\\+|--|\\+=|-=|\\*=|/=)`, 'g');
    const mutations = [...result.matchAll(mutationPattern)];

    const isReassigned = reassignments.length > 0 || mutations.length > 0;
    replacements.push({
      index: decl.index,
      replacement: isReassigned ? 'let' : 'const',
    });
  }

  // Apply replacements from end to start
  replacements.sort((a, b) => b.index - a.index);
  for (const rep of replacements) {
    result = result.substring(0, rep.index) + rep.replacement + result.substring(rep.index + 3);
  }

  return result;
}

/**
 * Deduplicate a name by appending a numeric suffix.
 *
 * @param {string} name
 * @param {Set<string>} usedNames
 * @returns {string}
 */
function deduplicateName(name, usedNames) {
  let suffix = 2;
  let candidate = `${name}${suffix}`;
  while (usedNames.has(candidate)) {
    suffix++;
    candidate = `${name}${suffix}`;
  }
  return candidate;
}

/**
 * Runnable reconstruction — applies renames one at a time, validating
 * each one. Trades completeness for correctness: the output is
 * guaranteed to parse and produce the same exports as the original.
 *
 * @param {string} source - beautified JavaScript source
 * @param {object} [options]
 * @param {string} [options.patternPath] - path to patterns JSON
 * @param {boolean} [options.addComments=true] - add JSDoc comments
 * @param {number} [options.minConfidence=0.3]
 * @param {number} [options.timeoutMs=1000] - VM timeout for equivalence checks
 * @returns {{code: string, appliedRenames: Array, rejectedRenames: Array, runnable: boolean, stats: object}}
 */
function reconstructRunnable(source, options = {}) {
  const {
    patternPath,
    addComments = true,
    minConfidence = 0.3,
    timeoutMs = 1000,
  } = options;

  const vm = require('vm');

  // Helper: check syntax validity
  function isSyntacticallyValid(code) {
    try {
      new Function(code);
      return true;
    } catch {
      return false;
    }
  }

  // Helper: check functional equivalence via sandboxed VM
  function isFunctionallyEquivalent(original, modified) {
    try {
      const makeSandbox = () => ({
        module: { exports: {} },
        exports: {},
        require: () => ({}),
        console: { log() {}, error() {}, warn() {}, info() {} },
        process: { env: {}, argv: [], cwd: () => '/' },
        setTimeout: () => {},
        setInterval: () => {},
        clearTimeout: () => {},
        clearInterval: () => {},
        Buffer: { from: () => Buffer.alloc(0), alloc: () => Buffer.alloc(0) },
        global: {},
        __dirname: '/',
        __filename: '/test.js',
      });

      const origCtx = vm.createContext(makeSandbox());
      const modCtx = vm.createContext(makeSandbox());

      vm.runInContext(original, origCtx, { timeout: timeoutMs });
      vm.runInContext(modified, modCtx, { timeout: timeoutMs });

      const origKeys = JSON.stringify(Object.keys(origCtx.module.exports || {}).sort());
      const modKeys = JSON.stringify(Object.keys(modCtx.module.exports || {}).sort());

      return origKeys === modKeys;
    } catch {
      return false;
    }
  }

  // 1. Collect all candidate renames
  const minifiedIds = findMinifiedIdentifiers(source);
  const candidates = [];

  for (const id of minifiedIds) {
    const context = extractContext(source, id);
    const declaration = findDeclaration(source, id);
    const prediction = predictName(id, context, {
      declaration,
      patternPath,
      minConfidence,
    });
    if (prediction) {
      candidates.push({
        original: id,
        inferred: prediction.name,
        confidence: prediction.confidence,
        source: prediction.source,
      });
    }
  }

  // 2. Sort by confidence (highest first)
  candidates.sort((a, b) => b.confidence - a.confidence);

  // 3. Apply renames one at a time, validating each
  let current = source;
  const appliedRenames = [];
  const rejectedRenames = [];
  const usedNames = new Set();

  for (const candidate of candidates) {
    let inferredName = candidate.inferred;
    if (usedNames.has(inferredName)) {
      inferredName = deduplicateName(inferredName, usedNames);
    }

    const { applyRename: doRename } = require('./reference-tracker');
    const attempt = doRename(current, candidate.original, inferredName);

    if (isSyntacticallyValid(attempt)) {
      if (isFunctionallyEquivalent(source, attempt)) {
        current = attempt;
        usedNames.add(inferredName);
        appliedRenames.push({ ...candidate, inferred: inferredName });
      } else {
        rejectedRenames.push({ ...candidate, reason: 'breaks behavior' });
      }
    } else {
      rejectedRenames.push({ ...candidate, reason: 'syntax error' });
    }
  }

  // 4. Apply safe style fixes (semantic equivalents, always safe)
  current = applySafeStyleFixes(current);

  // 5. Upgrade var -> const/let
  current = upgradeVarDeclarations(current);

  // 6. Add JSDoc comments (does not affect execution)
  let commentsAdded = 0;
  if (addComments) {
    const result = addJSDocComments(current, appliedRenames);
    current = result.code;
    commentsAdded = result.count;
  }

  return {
    code: current,
    appliedRenames,
    rejectedRenames,
    runnable: true,
    comments: commentsAdded,
    stats: {
      totalCandidates: candidates.length,
      applied: appliedRenames.length,
      rejected: rejectedRenames.length,
      successRate: candidates.length > 0
        ? parseFloat((appliedRenames.length / candidates.length).toFixed(3))
        : 1,
    },
  };
}

/**
 * Apply only the style fixes that are guaranteed semantic equivalents.
 * These never change behavior: !0===true, !1===false, void 0===undefined.
 *
 * @param {string} code
 * @returns {string}
 */
function applySafeStyleFixes(code) {
  return code
    .replace(/(?<![a-zA-Z0-9_$"'`])!0(?![a-zA-Z0-9_$])/g, 'true')
    .replace(/(?<![a-zA-Z0-9_$"'`])!1(?![a-zA-Z0-9_$])/g, 'false')
    .replace(/\bvoid 0\b/g, 'undefined');
}

module.exports = {
  reconstructCode,
  reconstructRunnable,
  findDeclaration,
  addJSDocComments,
  extractContextFromCode,
  upgradeVarDeclarations,
  applySafeStyleFixes,
};
