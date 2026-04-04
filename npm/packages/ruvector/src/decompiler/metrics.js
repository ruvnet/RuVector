/**
 * metrics.js - Code metrics extraction from JavaScript source.
 *
 * Computes structural metrics: function count, class count,
 * declaration count, line count, async patterns, etc.
 */

'use strict';

/**
 * Compute code metrics for a JavaScript source string.
 *
 * @param {string} source - JavaScript source code
 * @returns {{
 *   lines: number,
 *   sizeBytes: number,
 *   functions: number,
 *   asyncFunctions: number,
 *   arrowFunctions: number,
 *   classes: number,
 *   classExtensions: number,
 *   constDeclarations: number,
 *   letDeclarations: number,
 *   varDeclarations: number,
 *   imports: number,
 *   exports: number,
 *   requires: number,
 *   awaitExpressions: number,
 *   promiseUsages: number,
 *   tryBlocks: number,
 *   throwStatements: number,
 *   regexLiterals: number
 * }}
 */
function computeMetrics(source) {
  const count = (pattern) => (source.match(pattern) || []).length;

  return {
    lines: source.split('\n').length,
    sizeBytes: Buffer.byteLength(source, 'utf-8'),
    functions: count(/function\s*\w*\s*\(/g),
    asyncFunctions: count(/async\s+function/g),
    arrowFunctions: count(/=>/g),
    classes: count(/class\s+\w+/g),
    classExtensions: count(/extends\s+\w+/g),
    constDeclarations: count(/\bconst\s+/g),
    letDeclarations: count(/\blet\s+/g),
    varDeclarations: count(/\bvar\s+/g),
    imports: count(/\bimport\s+/g),
    exports: count(/\bexport\s+/g),
    requires: count(/\brequire\s*\(/g),
    awaitExpressions: count(/\bawait\s+/g),
    promiseUsages: count(/\bPromise\b/g),
    tryBlocks: count(/\btry\s*\{/g),
    throwStatements: count(/\bthrow\s+/g),
    regexLiterals: count(/\/[^/\n]+\/[gimsuy]*/g),
  };
}

/**
 * Compute a summary of metrics across multiple modules.
 *
 * @param {Array<{name: string, content: string}>} modules
 * @returns {{moduleCount: number, totalLines: number, totalBytes: number, perModule: object[]}}
 */
function computeModuleMetrics(modules) {
  const perModule = modules.map((mod) => ({
    name: mod.name,
    ...computeMetrics(mod.content),
  }));

  const totalLines = perModule.reduce((sum, m) => sum + m.lines, 0);
  const totalBytes = perModule.reduce((sum, m) => sum + m.sizeBytes, 0);

  return {
    moduleCount: modules.length,
    totalLines,
    totalBytes,
    perModule,
  };
}

module.exports = {
  computeMetrics,
  computeModuleMetrics,
};
