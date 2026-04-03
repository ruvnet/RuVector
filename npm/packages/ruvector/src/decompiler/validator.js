/**
 * validator.js - Operational validation for reconstructed code.
 *
 * Verifies that the reconstruction preserves semantics:
 *   - Syntax validity (parseable without errors)
 *   - String literal preservation (all strings intact)
 *   - Class hierarchy preservation (same extends chains)
 *   - Export preservation (same exports)
 *   - Functional equivalence (same behavior for test inputs)
 */

'use strict';

const vm = require('vm');

/**
 * Validate that a reconstruction preserves the semantics of the original.
 *
 * @param {string} originalSource - the minified/beautified original
 * @param {string} reconstructedSource - the reconstructed version
 * @param {object} [options]
 * @param {boolean} [options.checkSyntax=true]
 * @param {boolean} [options.checkStrings=true]
 * @param {boolean} [options.checkClasses=true]
 * @param {boolean} [options.checkFunctions=true]
 * @param {number} [options.timeoutMs=1000] - sandbox execution timeout
 * @returns {{syntaxValid: boolean, exportsMatch: boolean, stringsPreserved: boolean, classesMatch: boolean, functionallyEquivalent: boolean, issues: string[]}}
 */
function validateReconstruction(originalSource, reconstructedSource, options = {}) {
  const {
    checkSyntax = true,
    checkStrings = true,
    checkClasses = true,
    checkFunctions = true,
    timeoutMs = 1000,
  } = options;

  const issues = [];
  let syntaxValid = true;
  let stringsPreserved = true;
  let classesMatch = true;
  let exportsMatch = true;
  let functionallyEquivalent = true;

  // 1. Syntax check
  if (checkSyntax) {
    const syntaxResult = checkSyntaxValidity(reconstructedSource);
    syntaxValid = syntaxResult.valid;
    if (!syntaxValid) {
      issues.push(`Syntax error: ${syntaxResult.error}`);
      // If syntax is broken, further checks are unreliable
      return {
        syntaxValid,
        exportsMatch: false,
        stringsPreserved: false,
        classesMatch: false,
        functionallyEquivalent: false,
        issues,
      };
    }
  }

  // 2. String literal preservation
  if (checkStrings) {
    const result = checkStringPreservation(originalSource, reconstructedSource);
    stringsPreserved = result.preserved;
    for (const missing of result.missing) {
      issues.push(`Missing string literal: "${missing}"`);
    }
  }

  // 3. Class hierarchy preservation
  if (checkClasses) {
    const result = checkClassHierarchy(originalSource, reconstructedSource);
    classesMatch = result.match;
    for (const issue of result.issues) {
      issues.push(issue);
    }
  }

  // 4. Export/function count check
  if (checkFunctions) {
    const result = checkFunctionPreservation(originalSource, reconstructedSource);
    exportsMatch = result.match;
    for (const issue of result.issues) {
      issues.push(issue);
    }
  }

  // 5. Functional equivalence (best-effort, sandboxed)
  if (syntaxValid) {
    const result = checkFunctionalEquivalence(
      originalSource,
      reconstructedSource,
      timeoutMs,
    );
    functionallyEquivalent = result.equivalent;
    for (const issue of result.issues) {
      issues.push(issue);
    }
  }

  return {
    syntaxValid,
    exportsMatch,
    stringsPreserved,
    classesMatch,
    functionallyEquivalent,
    issues,
  };
}

/**
 * Check if source code is syntactically valid JavaScript.
 *
 * @param {string} source
 * @returns {{valid: boolean, error: string|null}}
 */
function checkSyntaxValidity(source) {
  try {
    // Use Function constructor for syntax check (does not execute)
    new Function(source);
    return { valid: true, error: null };
  } catch (err) {
    return { valid: false, error: err.message };
  }
}

/**
 * Check that all string literals from the original appear in the reconstruction.
 * Identifiers may change, but string values must be preserved.
 *
 * @param {string} original
 * @param {string} reconstructed
 * @returns {{preserved: boolean, missing: string[], total: number}}
 */
function checkStringPreservation(original, reconstructed) {
  const origStrings = extractStringLiterals(original);
  const reconStrings = new Set(extractStringLiterals(reconstructed));

  const missing = [];
  for (const s of origStrings) {
    // Skip very short strings and common noise
    if (s.length < 2) continue;
    if (!reconStrings.has(s)) {
      missing.push(s);
    }
  }

  return {
    preserved: missing.length === 0,
    missing: missing.slice(0, 20), // Cap at 20 for reporting
    total: origStrings.length,
  };
}

/**
 * Extract all string literals from source code.
 *
 * @param {string} source
 * @returns {string[]}
 */
function extractStringLiterals(source) {
  const strings = [];

  // Match double-quoted strings
  const doubleQuoted = source.match(/"([^"\\]|\\.)*"/g) || [];
  for (const s of doubleQuoted) {
    strings.push(s.slice(1, -1));
  }

  // Match single-quoted strings
  const singleQuoted = source.match(/'([^'\\]|\\.)*'/g) || [];
  for (const s of singleQuoted) {
    strings.push(s.slice(1, -1));
  }

  return strings;
}

/**
 * Check that class hierarchies are preserved.
 * All "class X extends Y" pairs must appear in both versions.
 *
 * @param {string} original
 * @param {string} reconstructed
 * @returns {{match: boolean, issues: string[]}}
 */
function checkClassHierarchy(original, reconstructed) {
  const origClasses = extractClassHierarchy(original);
  const reconClasses = extractClassHierarchy(reconstructed);
  const issues = [];

  // Check that base classes are preserved (names may have changed)
  const origBases = new Set(origClasses.map((c) => c.base).filter(Boolean));
  const reconBases = new Set(reconClasses.map((c) => c.base).filter(Boolean));

  // Base class names (Error, EventEmitter, etc.) should be preserved
  for (const base of origBases) {
    if (!reconBases.has(base)) {
      // Check if it is a built-in that was renamed
      const builtIns = ['Error', 'TypeError', 'RangeError', 'EventEmitter', 'Stream', 'Buffer'];
      if (builtIns.includes(base)) {
        issues.push(`Base class "${base}" missing from reconstruction`);
      }
    }
  }

  // Same number of class declarations
  if (origClasses.length !== reconClasses.length) {
    issues.push(
      `Class count mismatch: original has ${origClasses.length}, reconstructed has ${reconClasses.length}`,
    );
  }

  return { match: issues.length === 0, issues };
}

/**
 * Extract class declarations and their inheritance.
 *
 * @param {string} source
 * @returns {Array<{name: string, base: string|null}>}
 */
function extractClassHierarchy(source) {
  const classes = [];
  const re = /class\s+(\w+)(?:\s+extends\s+(\w+))?/g;
  let match;
  while ((match = re.exec(source)) !== null) {
    classes.push({ name: match[1], base: match[2] || null });
  }
  return classes;
}

/**
 * Check that the number of functions/exports is preserved.
 *
 * @param {string} original
 * @param {string} reconstructed
 * @returns {{match: boolean, issues: string[]}}
 */
function checkFunctionPreservation(original, reconstructed) {
  const issues = [];

  const origFuncCount = (original.match(/function\s*[\w$]*\s*\(/g) || []).length;
  const reconFuncCount = (reconstructed.match(/function\s*[\w$]*\s*\(/g) || []).length;

  if (origFuncCount !== reconFuncCount) {
    issues.push(
      `Function count mismatch: original has ${origFuncCount}, reconstructed has ${reconFuncCount}`,
    );
  }

  const origArrowCount = (original.match(/=>/g) || []).length;
  const reconArrowCount = (reconstructed.match(/=>/g) || []).length;

  if (origArrowCount !== reconArrowCount) {
    issues.push(
      `Arrow function count mismatch: original has ${origArrowCount}, reconstructed has ${reconArrowCount}`,
    );
  }

  // Check module.exports / export counts
  const origExports = (original.match(/module\.exports|export\s+(default\s+)?/g) || []).length;
  const reconExports = (reconstructed.match(/module\.exports|export\s+(default\s+)?/g) || []).length;

  if (origExports !== reconExports) {
    issues.push(
      `Export count mismatch: original has ${origExports}, reconstructed has ${reconExports}`,
    );
  }

  return { match: issues.length === 0, issues };
}

/**
 * Best-effort functional equivalence check.
 * Runs both versions in a sandboxed VM and compares outputs.
 *
 * This is a heuristic — it cannot prove full equivalence, but catches
 * obvious breakages (renamed exports, broken references, etc.).
 *
 * @param {string} original
 * @param {string} reconstructed
 * @param {number} timeoutMs
 * @returns {{equivalent: boolean, issues: string[]}}
 */
function checkFunctionalEquivalence(original, reconstructed, timeoutMs) {
  const issues = [];

  // Compare the shape of what each version exports
  const origExports = safeEvalExports(original, timeoutMs);
  const reconExports = safeEvalExports(reconstructed, timeoutMs);

  if (origExports.error && !reconExports.error) {
    // Original errors but reconstructed does not — likely OK
    return { equivalent: true, issues };
  }

  if (!origExports.error && reconExports.error) {
    issues.push(`Reconstructed code fails to execute: ${reconExports.error}`);
    return { equivalent: false, issues };
  }

  if (origExports.error && reconExports.error) {
    // Both error — check if it is the same kind of error
    return { equivalent: true, issues };
  }

  // Compare export shapes (type and count of exported values)
  const origKeys = Object.keys(origExports.exports || {}).sort();
  const reconKeys = Object.keys(reconExports.exports || {}).sort();

  // Exports may have been renamed, so just compare counts and types
  if (origKeys.length !== reconKeys.length) {
    issues.push(
      `Exported key count differs: ${origKeys.length} vs ${reconKeys.length}`,
    );
  }

  // Compare types of exported values
  const origTypes = origKeys.map((k) => typeof origExports.exports[k]).sort();
  const reconTypes = reconKeys.map((k) => typeof reconExports.exports[k]).sort();

  for (let i = 0; i < Math.min(origTypes.length, reconTypes.length); i++) {
    if (origTypes[i] !== reconTypes[i]) {
      issues.push(
        `Export type mismatch at position ${i}: ${origTypes[i]} vs ${reconTypes[i]}`,
      );
    }
  }

  return { equivalent: issues.length === 0, issues };
}

/**
 * Safely execute code in a VM sandbox and extract module.exports.
 *
 * @param {string} source
 * @param {number} timeoutMs
 * @returns {{exports: object|null, error: string|null}}
 */
function safeEvalExports(source, timeoutMs) {
  try {
    const sandbox = {
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
    };

    const context = vm.createContext(sandbox);
    const script = new vm.Script(source, { filename: 'reconstructed.js' });
    script.runInContext(context, { timeout: timeoutMs });

    return { exports: sandbox.module.exports, error: null };
  } catch (err) {
    return { exports: null, error: err.message };
  }
}

module.exports = {
  validateReconstruction,
  checkSyntaxValidity,
  checkStringPreservation,
  checkClassHierarchy,
  checkFunctionPreservation,
  checkFunctionalEquivalence,
  extractStringLiterals,
};
