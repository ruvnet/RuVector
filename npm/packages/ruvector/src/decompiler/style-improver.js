/**
 * style-improver.js - Readability improvements and JSDoc generation.
 *
 * Transforms beautified-but-minified code into human-readable form:
 *   - Converts minification artifacts (!0, !1, void 0)
 *   - Adds blank lines between declarations
 *   - Converts optional chaining candidates
 *   - Generates JSDoc comments from context
 */

'use strict';

/**
 * Apply all readability improvements to source code.
 *
 * @param {string} source
 * @param {object} [options]
 * @param {boolean} [options.convertBooleans=true]
 * @param {boolean} [options.convertVoid=true]
 * @param {boolean} [options.optionalChaining=true]
 * @param {boolean} [options.addSpacing=true]
 * @param {boolean} [options.expandCommaExpressions=true]
 * @returns {string}
 */
function improveReadability(source, options = {}) {
  const {
    convertBooleans = true,
    convertVoid = true,
    optionalChaining = true,
    addSpacing = true,
    expandCommaExpressions = true,
  } = options;

  let result = source;

  if (convertBooleans) {
    result = convertMinifiedBooleans(result);
  }
  if (convertVoid) {
    result = convertVoidZero(result);
  }
  if (optionalChaining) {
    result = convertToOptionalChaining(result);
  }
  if (expandCommaExpressions) {
    result = expandCommaExprs(result);
  }
  if (addSpacing) {
    result = addBlankLines(result);
  }

  return result;
}

/**
 * Convert !0 -> true, !1 -> false.
 * Must not match inside strings or comments.
 *
 * @param {string} source
 * @returns {string}
 */
function convertMinifiedBooleans(source) {
  // Replace !0 with true (not inside strings)
  let result = source.replace(/(?<![a-zA-Z0-9_$"'`])!0(?![a-zA-Z0-9_$])/g, 'true');
  // Replace !1 with false
  result = result.replace(/(?<![a-zA-Z0-9_$"'`])!1(?![a-zA-Z0-9_$])/g, 'false');
  return result;
}

/**
 * Convert void 0 -> undefined.
 *
 * @param {string} source
 * @returns {string}
 */
function convertVoidZero(source) {
  return source.replace(/\bvoid 0\b/g, 'undefined');
}

/**
 * Convert guard chains to optional chaining.
 *   a && a.b && a.b.c  ->  a?.b?.c
 *   a && a.b           ->  a?.b
 *
 * Only applies to simple property access chains (no method calls).
 *
 * @param {string} source
 * @returns {string}
 */
function convertToOptionalChaining(source) {
  // Pattern: a && a.b && a.b.c
  // Match the longest chains first
  let result = source;

  // 3-level chain: a && a.b && a.b.c
  result = result.replace(
    /\b([a-zA-Z_$]\w*)(?:\s*&&\s*\1\.([a-zA-Z_$]\w*))(?:\s*&&\s*\1\.\2\.([a-zA-Z_$]\w*))/g,
    '$1?.$2?.$3',
  );

  // 2-level chain: a && a.b
  result = result.replace(
    /\b([a-zA-Z_$]\w*)\s*&&\s*\1\.([a-zA-Z_$]\w*)/g,
    '$1?.$2',
  );

  return result;
}

/**
 * Expand simple comma expressions into separate statements.
 * Only expands top-level comma expressions (not inside for-loops, function args, etc.).
 *
 *   a = 1, b = 2, c = 3  ->  a = 1;\nb = 2;\nc = 3
 *
 * @param {string} source
 * @returns {string}
 */
function expandCommaExprs(source) {
  const lines = source.split('\n');
  const result = [];

  for (const line of lines) {
    const trimmed = line.trim();

    // Skip lines that are clearly not comma expressions
    if (trimmed.startsWith('for') || trimmed.startsWith('//') || trimmed.startsWith('/*')) {
      result.push(line);
      continue;
    }

    // Skip function calls, array literals, object literals
    if (trimmed.includes('(') || trimmed.startsWith('{') || trimmed.startsWith('[')) {
      result.push(line);
      continue;
    }

    // Check for assignment comma expressions: a=1,b=2,c=3
    if (/^\s*[a-zA-Z_$]\w*\s*=.*,.*[a-zA-Z_$]\w*\s*=/.test(trimmed)) {
      const indent = line.match(/^(\s*)/)[1];
      // Split on comma but only between assignments
      const parts = splitCommaExpression(trimmed);
      if (parts.length > 1) {
        for (const part of parts) {
          result.push(indent + part.trim() + ';');
        }
        continue;
      }
    }

    result.push(line);
  }

  return result.join('\n');
}

/**
 * Split a comma expression into parts, respecting nested parens/brackets.
 * @param {string} expr
 * @returns {string[]}
 */
function splitCommaExpression(expr) {
  const parts = [];
  let depth = 0;
  let current = '';

  for (let i = 0; i < expr.length; i++) {
    const ch = expr[i];
    if (ch === '(' || ch === '[' || ch === '{') {
      depth++;
      current += ch;
    } else if (ch === ')' || ch === ']' || ch === '}') {
      depth--;
      current += ch;
    } else if (ch === ',' && depth === 0) {
      parts.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  if (current.trim()) parts.push(current);

  // Only return split if each part looks like an assignment
  const allAssignments = parts.every((p) => /\s*[a-zA-Z_$]\w*\s*=/.test(p.trim()));
  if (allAssignments && parts.length > 1) return parts;
  return [expr];
}

/**
 * Add blank lines between top-level declarations for readability.
 *
 * @param {string} source
 * @returns {string}
 */
function addBlankLines(source) {
  const lines = source.split('\n');
  const result = [];
  let prevWasDecl = false;

  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();
    const isDecl =
      /^(function|async function|class|const|let|var|export)/.test(trimmed) &&
      !trimmed.startsWith('const {') &&
      trimmed.length > 20;

    // Add blank line before a new declaration block
    if (isDecl && prevWasDecl && result.length > 0 && result[result.length - 1].trim() !== '') {
      // Only add if there is not already a blank line
      result.push('');
    }

    result.push(lines[i]);

    // Track if we just left a closing brace (end of function/class)
    if (trimmed === '}' || trimmed === '};') {
      prevWasDecl = true;
    } else if (trimmed === '') {
      prevWasDecl = false;
    } else {
      prevWasDecl = isDecl;
    }
  }

  return result.join('\n');
}

/**
 * Generate a JSDoc comment for a function/class declaration.
 *
 * @param {string} declaration - the declaration line(s)
 * @param {string[]} contextStrings - strings found near the declaration
 * @param {object} [options]
 * @param {Array<{oldName: string, newName: string}>} [options.renames] - applied renames
 * @returns {string|null} JSDoc comment string, or null if not applicable
 */
function generateJSDoc(declaration, contextStrings, options = {}) {
  const { renames = [] } = options;

  // Determine declaration type
  const isAsyncGen = /async\s+function\s*\*/.test(declaration);
  const isGenerator = /function\s*\*/.test(declaration);
  const isAsync = /async/.test(declaration);
  const isClass = /class\s+/.test(declaration);
  const isFunction =
    /function[\s*]/.test(declaration) || /=>\s*{/.test(declaration) || /=>\s*[^{]/.test(declaration);

  if (!isFunction && !isClass) return null;

  const lines = ['/**'];

  // Infer purpose from context
  const purpose = inferPurpose(declaration, contextStrings, renames);
  if (purpose) {
    lines.push(` * ${purpose}`);
  }

  if (isClass) {
    const extendsMatch = declaration.match(/extends\s+(\w+)/);
    if (extendsMatch) {
      lines.push(` * @extends ${extendsMatch[1]}`);
    }
    lines.push(' */');
    return lines.join('\n');
  }

  // Extract parameters
  const params = extractParams(declaration);
  if (params.length > 0) {
    for (const param of params) {
      const type = inferParamType(param, contextStrings);
      lines.push(` * @param {${type}} ${param}`);
    }
  }

  // Yields for generators
  if (isAsyncGen || isGenerator) {
    const yieldType = inferYieldType(declaration, contextStrings);
    lines.push(` * @yields {${yieldType}}`);
  }

  // Returns
  if (!isAsyncGen && !isGenerator) {
    const returnType = inferReturnType(declaration, contextStrings, isAsync);
    if (returnType) {
      lines.push(` * @returns {${returnType}}`);
    }
  }

  lines.push(' */');
  return lines.join('\n');
}

/**
 * Infer the purpose of a function from its context.
 * @param {string} declaration
 * @param {string[]} context
 * @param {Array<{oldName: string, newName: string}>} renames
 * @returns {string|null}
 */
function inferPurpose(declaration, context, renames) {
  const ctx = context.join(' ').toLowerCase();
  const decl = declaration.toLowerCase();

  // Use the renamed function name as a hint
  const funcNameMatch = declaration.match(/function\s+(\w+)|(\w+)\s*[=:]\s*(async\s+)?function/);
  const funcName = funcNameMatch ? (funcNameMatch[1] || funcNameMatch[2]) : null;

  if (funcName) {
    // Convert camelCase to sentence
    const words = funcName.replace(/([A-Z])/g, ' $1').toLowerCase().trim();
    if (words.length > 3 && !/^[a-z]$/.test(words)) {
      return capitalizeFirst(words) + '.';
    }
  }

  // Context-based purpose
  if (ctx.includes('stream') && ctx.includes('event')) return 'Process streaming events from the API.';
  if (ctx.includes('permission') && ctx.includes('check')) return 'Check if the operation is permitted.';
  if (ctx.includes('compact') && ctx.includes('token')) return 'Compact context to fit within token budget.';
  if (ctx.includes('tool') && ctx.includes('dispatch')) return 'Dispatch tool invocations to handlers.';
  if (ctx.includes('mcp') && ctx.includes('connect')) return 'Establish MCP server connection.';
  if (ctx.includes('fetch') && ctx.includes('api')) return 'Make an API request.';
  if (ctx.includes('parse') && ctx.includes('json')) return 'Parse JSON response data.';
  if (ctx.includes('validate') && ctx.includes('input')) return 'Validate input parameters.';
  if (ctx.includes('error') && ctx.includes('handle')) return 'Handle and format error responses.';

  return null;
}

/**
 * Extract parameter names from a function declaration.
 * @param {string} declaration
 * @returns {string[]}
 */
function extractParams(declaration) {
  // Match function parameters: function name(a, b, c) or (a, b) =>
  const match = declaration.match(/\(([^)]*)\)/);
  if (!match || !match[1].trim()) return [];

  return match[1]
    .split(',')
    .map((p) => p.trim())
    .filter((p) => p.length > 0)
    .map((p) => {
      // Handle destructuring: { a, b } -> "options"
      if (p.startsWith('{')) return 'options';
      if (p.startsWith('[')) return 'items';
      // Handle defaults: a = 1 -> a
      return p.split('=')[0].trim();
    });
}

/**
 * Infer a TypeScript-style type for a parameter from context.
 * @param {string} paramName
 * @param {string[]} context
 * @returns {string}
 */
function inferParamType(paramName, context) {
  const ctx = context.join(' ').toLowerCase();

  if (paramName === 'options' || paramName === 'config') return 'Object';
  if (paramName === 'items' || paramName === 'list') return 'Array';
  if (ctx.includes('string') || ctx.includes('name') || ctx.includes('path')) return 'string';
  if (ctx.includes('number') || ctx.includes('count') || ctx.includes('index')) return 'number';
  if (ctx.includes('boolean') || ctx.includes('flag') || ctx.includes('enabled')) return 'boolean';
  if (ctx.includes('callback') || ctx.includes('handler')) return 'Function';
  if (ctx.includes('promise') || ctx.includes('async')) return 'Promise';
  if (ctx.includes('array') || ctx.includes('list')) return 'Array';
  if (ctx.includes('message')) return 'Message';
  if (ctx.includes('request')) return 'Request';
  if (ctx.includes('response')) return 'Response';

  return '*';
}

/**
 * Infer what a generator yields.
 * @param {string} declaration
 * @param {string[]} context
 * @returns {string}
 */
function inferYieldType(declaration, context) {
  const ctx = context.join(' ');
  if (ctx.includes('stream_event') || ctx.includes('StreamEvent')) return 'StreamEvent';
  if (ctx.includes('message') || ctx.includes('Message')) return 'Message';
  if (ctx.includes('chunk') || ctx.includes('Chunk')) return 'Chunk';
  return 'Object';
}

/**
 * Infer the return type of a function.
 * @param {string} declaration
 * @param {string[]} context
 * @param {boolean} isAsync
 * @returns {string|null}
 */
function inferReturnType(declaration, context, isAsync) {
  const ctx = context.join(' ');

  let baseType = null;
  if (ctx.includes('boolean') || ctx.includes('true') || ctx.includes('false')) {
    baseType = 'boolean';
  } else if (ctx.includes('string') || ctx.includes('"')) {
    baseType = 'string';
  } else if (ctx.includes('number') || ctx.includes('.length')) {
    baseType = 'number';
  } else if (ctx.includes('array') || ctx.includes('[]')) {
    baseType = 'Array';
  }

  if (isAsync && baseType) return `Promise<${baseType}>`;
  if (isAsync) return 'Promise<*>';
  return baseType;
}

/**
 * Capitalize the first letter of a string.
 * @param {string} s
 * @returns {string}
 */
function capitalizeFirst(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

module.exports = {
  improveReadability,
  convertMinifiedBooleans,
  convertVoidZero,
  convertToOptionalChaining,
  expandCommaExprs,
  addBlankLines,
  generateJSDoc,
  extractParams,
  inferPurpose,
};
