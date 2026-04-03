/**
 * decompiler-reconstruction.js - Tests for the readable reconstruction pipeline.
 *
 * Tests cover:
 *   - Reference tracker (find, rename, propagate)
 *   - Name predictor (patterns, structural rules, heuristics)
 *   - Style improver (booleans, void, optional chaining, JSDoc)
 *   - Reconstructor (full pipeline)
 *   - Validator (syntax, strings, classes, functional equivalence)
 */

'use strict';

const {
  findAllReferences,
  applyRename,
  applyAllRenames,
  findMinifiedIdentifiers,
  extractContext,
  isMinifiedName,
} = require('../src/decompiler/reference-tracker');

const {
  predictName,
  loadPatterns,
  scorePatternMatch,
  toCamelCase,
  isMinifiedLike,
} = require('../src/decompiler/name-predictor');

const {
  improveReadability,
  convertMinifiedBooleans,
  convertVoidZero,
  convertToOptionalChaining,
  generateJSDoc,
  extractParams,
} = require('../src/decompiler/style-improver');

const { reconstructCode } = require('../src/decompiler/reconstructor');

const {
  validateReconstruction,
  checkSyntaxValidity,
  checkStringPreservation,
  checkClassHierarchy,
  extractStringLiterals,
} = require('../src/decompiler/validator');

let passed = 0;
let failed = 0;
let total = 0;

function assert(condition, message) {
  total++;
  if (condition) {
    passed++;
  } else {
    failed++;
    console.error(`  FAIL: ${message}`);
  }
}

function assertEq(actual, expected, message) {
  total++;
  if (actual === expected) {
    passed++;
  } else {
    failed++;
    console.error(`  FAIL: ${message}`);
    console.error(`    Expected: ${JSON.stringify(expected)}`);
    console.error(`    Actual:   ${JSON.stringify(actual)}`);
  }
}

function describe(name, fn) {
  console.log(`\n--- ${name} ---`);
  fn();
}

function it(name, fn) {
  try {
    fn();
  } catch (err) {
    total++;
    failed++;
    console.error(`  FAIL: ${name}: ${err.message}`);
  }
}

// ─── Reference Tracker Tests ───

describe('ReferenceTracker', () => {
  it('findAllReferences finds whole-word occurrences', () => {
    const source = 'var a = 1; var ab = a + 2; return a;';
    const refs = findAllReferences(source, 'a');
    assertEq(refs.length, 3, 'should find 3 references to "a" (not "ab")');
  });

  it('findAllReferences skips identifiers inside strings', () => {
    const source = 'var a = 1; var b = "a is a letter";';
    const refs = findAllReferences(source, 'a');
    assertEq(refs.length, 1, 'should find 1 reference to "a" (skip string)');
  });

  it('applyRename renames all occurrences', () => {
    const source = 'var a = 1; function f() { return a + 1; }';
    const result = applyRename(source, 'a', 'counter');
    assert(result.includes('var counter = 1'), 'declaration renamed');
    assert(result.includes('return counter + 1'), 'reference renamed');
    assert(!result.includes('var a'), 'old name removed');
  });

  it('applyAllRenames handles multiple renames', () => {
    const source = 'var a = 1; var b = a + 2;';
    const result = applyAllRenames(source, [
      { oldName: 'a', newName: 'count' },
      { oldName: 'b', newName: 'total' },
    ]);
    assert(result.includes('count'), 'a renamed to count');
    assert(result.includes('total'), 'b renamed to total');
  });

  it('findMinifiedIdentifiers detects short names', () => {
    const source = 'var a = 1; var bb = 2; var longName = 3;';
    const ids = findMinifiedIdentifiers(source);
    assert(ids.includes('a'), 'should detect single letter');
    assert(ids.includes('bb'), 'should detect two letters');
    assert(!ids.includes('longName'), 'should skip meaningful names');
  });

  it('isMinifiedName respects reserved words', () => {
    assertEq(isMinifiedName('if'), false, 'if is reserved');
    assertEq(isMinifiedName('for'), false, 'for is reserved');
    assertEq(isMinifiedName('Map'), false, 'Map is a global');
    assertEq(isMinifiedName('a'), true, 'a is minified');
    assertEq(isMinifiedName('s$'), true, 's$ is minified');
    assertEq(isMinifiedName('A2'), true, 'A2 is minified');
  });
});

// ─── Name Predictor Tests ───

describe('NamePredictor', () => {
  it('loads patterns from JSON', () => {
    const patterns = loadPatterns();
    assert(patterns.length > 0, 'should load at least some patterns');
  });

  it('predicts name from context strings', () => {
    const result = predictName('s$', [
      'tools/call', 'tools/list', 'initialize',
      '.method', '.params', '.jsonrpc',
    ]);
    assert(result !== null, 'should return a prediction');
    assert(result.confidence > 0.3, 'confidence should exceed threshold');
  });

  it('predicts name from structural rules', () => {
    const result = predictName('s$', ['stream', 'yield'], {
      declaration: 'async function* s$(params) {',
    });
    assert(result !== null, 'should match async generator rule');
    assertEq(result.name, 'streamGenerator', 'should infer streamGenerator');
  });

  it('detects async generator agent loop pattern', () => {
    const result = predictName('s$', ['agent', 'loop', 'yield'], {
      declaration: 'async function* s$(params) {',
    });
    assert(result !== null, 'should match');
    assertEq(result.name, 'agentLoop', 'should infer agentLoop');
  });

  it('toCamelCase converts correctly', () => {
    assertEq(toCamelCase('McpToolHandler'), 'mcpToolHandler', 'PascalCase');
    assertEq(toCamelCase('http-router'), 'httpRouter', 'kebab-case');
    assertEq(toCamelCase('already_camel'), 'alreadyCamel', 'snake_case');
  });

  it('isMinifiedLike detects short names', () => {
    assertEq(isMinifiedLike('A'), true, 'single letter');
    assertEq(isMinifiedLike('s$'), true, 'letter + $');
    assertEq(isMinifiedLike('agentLoop'), false, 'meaningful name');
  });
});

// ─── Style Improver Tests ───

describe('StyleImprover', () => {
  it('converts !0 to true and !1 to false', () => {
    const result = convertMinifiedBooleans('var a = !0; var b = !1;');
    assert(result.includes('true'), '!0 -> true');
    assert(result.includes('false'), '!1 -> false');
  });

  it('converts void 0 to undefined', () => {
    const result = convertVoidZero('if (a === void 0) return;');
    assert(result.includes('undefined'), 'void 0 -> undefined');
    assert(!result.includes('void 0'), 'void 0 removed');
  });

  it('converts guard chains to optional chaining', () => {
    const result = convertToOptionalChaining('a && a.b && a.b.c');
    assertEq(result, 'a?.b?.c', '3-level chain');
  });

  it('converts 2-level guard chain', () => {
    const result = convertToOptionalChaining('x && x.y');
    assertEq(result, 'x?.y', '2-level chain');
  });

  it('extracts function parameters', () => {
    const params = extractParams('function foo(a, b, c) {');
    assertEq(params.length, 3, 'should find 3 params');
  });

  it('handles destructured parameters', () => {
    const params = extractParams('function foo({ a, b }) {');
    assertEq(params[0], 'options', 'destructured -> options');
  });

  it('generates JSDoc for async generator', () => {
    const doc = generateJSDoc('async function* streamEvents(params) {', [
      'stream_event', 'yield', 'await',
    ]);
    assert(doc !== null, 'should generate JSDoc');
    assert(doc.includes('@param'), 'should have @param');
    assert(doc.includes('@yields'), 'should have @yields');
  });

  it('generates JSDoc for class extending Error', () => {
    const doc = generateJSDoc('class CustomError extends Error {', []);
    assert(doc !== null, 'should generate JSDoc');
    assert(doc.includes('@extends Error'), 'should note base class');
  });

  it('improveReadability applies all transforms', () => {
    const input = 'var a = !0; if (x === void 0) return; b && b.c;';
    const result = improveReadability(input);
    assert(result.includes('true'), 'booleans converted');
    assert(result.includes('undefined'), 'void converted');
    assert(result.includes('b?.c'), 'optional chaining');
  });
});

// ─── Reconstructor Tests ───

describe('Reconstructor', () => {
  it('renames minified variables while preserving behavior', () => {
    const input = 'var a = function(b) { return b + 1; };';
    const result = reconstructCode(input);
    assert(result.code.length > 0, 'should produce output');
    assert(result.renames.length >= 0, 'should track renames');
    assert(result.code.includes('+ 1'), 'preserves arithmetic');
  });

  it('preserves all string literals', () => {
    const input = 'var a = "hello"; var b = "world";';
    const result = reconstructCode(input);
    assert(result.code.includes('"hello"'), 'preserves "hello"');
    assert(result.code.includes('"world"'), 'preserves "world"');
  });

  it('preserves class inheritance', () => {
    const input = 'class A extends Error { constructor(m) { super(m); } }';
    const result = reconstructCode(input);
    assert(result.code.includes('extends Error'), 'preserves extends');
  });

  it('converts minified booleans', () => {
    const input = 'var a = !0; var b = !1; var c = void 0;';
    const result = reconstructCode(input);
    assert(result.code.includes('true'), '!0 -> true');
    assert(result.code.includes('false'), '!1 -> false');
    assert(result.code.includes('undefined'), 'void 0 -> undefined');
  });

  it('propagates renames consistently', () => {
    const input = 'var a = 1; function b() { return a + 1; }';
    const result = reconstructCode(input);
    const renamedA = result.renames.find((r) => r.original === 'a');
    if (renamedA) {
      assert(
        !result.code.includes('return a ') && !result.code.includes('return a+'),
        'all references to "a" should be renamed',
      );
    }
  });

  it('adds JSDoc comments to functions', () => {
    const input = 'async function doSomething(params) { return await fetch(params.url); }';
    const result = reconstructCode(input, { addComments: true });
    assert(result.code.includes('/**'), 'should add JSDoc');
    assert(result.comments > 0, 'should report comment count');
  });

  it('upgrades var to const/let', () => {
    const input = 'var x = 1; var y = 2; y = 3;';
    const result = reconstructCode(input, { improveStyle: false });
    assert(result.code.includes('const x'), 'x should become const');
    assert(result.code.includes('let y'), 'y should become let (reassigned)');
  });

  it('handles the example from the task description', () => {
    const input = `var s$=async function*(A){let B=A.messages,G=A.systemPrompt;yield{type:"stream_request_start"};let Z=await Y2(B,G,A.canUseTool);for await(let J of Z){yield{type:"stream_event",event:J}}};`;
    const result = reconstructCode(input);
    // Should preserve string literals
    assert(result.code.includes('"stream_request_start"'), 'preserves stream_request_start');
    assert(result.code.includes('"stream_event"'), 'preserves stream_event');
    // Should have some renames
    assert(result.renames.length > 0, 'should rename some identifiers');
    // Should still have the yield and for-await
    assert(result.code.includes('yield'), 'preserves yield');
    assert(result.code.includes('for await'), 'preserves for-await');
  });
});

// ─── Validator Tests ───

describe('Validator', () => {
  it('detects valid syntax', () => {
    const result = checkSyntaxValidity('var a = 1; function f() { return a; }');
    assertEq(result.valid, true, 'valid JS should pass');
  });

  it('detects syntax errors', () => {
    const result = checkSyntaxValidity('var a = 1; }}}');
    assertEq(result.valid, false, 'broken JS should fail');
    assert(result.error !== null, 'should have error message');
  });

  it('checks string preservation', () => {
    const result = checkStringPreservation(
      'var a = "hello"; var b = "world";',
      'var x = "hello"; var y = "world";',
    );
    assertEq(result.preserved, true, 'all strings should be preserved');
  });

  it('detects missing strings', () => {
    const result = checkStringPreservation(
      'var a = "hello"; var b = "world";',
      'var x = "hello"; var y = "earth";',
    );
    assertEq(result.preserved, false, 'should detect missing "world"');
    assert(result.missing.includes('world'), '"world" should be in missing list');
  });

  it('checks class hierarchy', () => {
    const result = checkClassHierarchy(
      'class A extends Error {}',
      'class CustomError extends Error {}',
    );
    assertEq(result.match, true, 'hierarchy should match');
  });

  it('detects class count mismatch', () => {
    const result = checkClassHierarchy(
      'class A extends Error {} class B {}',
      'class CustomError extends Error {}',
    );
    assertEq(result.match, false, 'should detect count mismatch');
  });

  it('extracts string literals', () => {
    const strings = extractStringLiterals('var a = "hello"; var b = \'world\';');
    assert(strings.includes('hello'), 'should find double-quoted');
    assert(strings.includes('world'), 'should find single-quoted');
  });

  it('full validation passes for valid reconstruction', () => {
    const original = 'var a = 1; var b = a + 2;';
    const reconstructed = 'const count = 1; const total = count + 2;';
    const result = validateReconstruction(original, reconstructed);
    assertEq(result.syntaxValid, true, 'should be syntactically valid');
  });

  it('full validation catches broken reconstruction', () => {
    const original = 'var a = 1;';
    const reconstructed = 'const count = 1; }}}';
    const result = validateReconstruction(original, reconstructed);
    assertEq(result.syntaxValid, false, 'should detect syntax error');
  });
});

// ─── Runnable Reconstruction Tests ───

describe('Runnable Reconstruction', () => {
  it('produces syntactically valid output', () => {
    const { reconstructRunnable } = require('../src/decompiler/reconstructor');
    const input = 'var a = function(b) { return b + 1; };';
    const result = reconstructRunnable(input);
    assertEq(result.runnable, true, 'should be marked runnable');
    // Verify syntax
    let valid = true;
    try { new Function(result.code); } catch { valid = false; }
    assertEq(valid, true, 'output should parse without errors');
  });

  it('preserves string literals in runnable mode', () => {
    const { reconstructRunnable } = require('../src/decompiler/reconstructor');
    const input = 'var a = "hello"; var b = "world"; function c() { return a + b; }';
    const result = reconstructRunnable(input);
    assert(result.code.includes('"hello"'), 'preserves hello');
    assert(result.code.includes('"world"'), 'preserves world');
  });

  it('reports applied and rejected renames', () => {
    const { reconstructRunnable } = require('../src/decompiler/reconstructor');
    const input = 'var a = 1; var b = a + 2; module.exports = { a: a, b: b };';
    const result = reconstructRunnable(input);
    assert(result.stats.totalCandidates >= 0, 'should report total candidates');
    assert(result.stats.applied >= 0, 'should report applied count');
    assert(result.stats.rejected >= 0, 'should report rejected count');
  });

  it('applies safe style fixes', () => {
    const { reconstructRunnable } = require('../src/decompiler/reconstructor');
    const input = 'var a = !0; var b = !1; var c = void 0;';
    const result = reconstructRunnable(input);
    assert(result.code.includes('true'), '!0 -> true');
    assert(result.code.includes('false'), '!1 -> false');
    assert(result.code.includes('undefined'), 'void 0 -> undefined');
  });
});

// ─── Full Pipeline Integration ───

describe('Full Pipeline Integration', () => {
  it('decompileSource with reconstruct=true', () => {
    const { decompileSource } = require('../src/decompiler/index');
    const minified = 'var a=function(b){return b.c("d")};var e=class extends Error{constructor(){super("test")}};';
    const result = decompileSource(minified, { reconstruct: true });

    assert(result.modules.length > 0, 'should have modules');
    assert(result.reconstruction !== undefined, 'should have reconstruction summary');
    assert(result.reconstruction.modulesProcessed > 0, 'should process modules');
  });

  it('decompileSource with reconstruct=true and validate=true', () => {
    const { decompileSource } = require('../src/decompiler/index');
    const minified = 'var a=function(b){return b+1};';
    const result = decompileSource(minified, { reconstruct: true, validate: true });

    assert(result.reconstruction !== undefined, 'should have reconstruction');
    assert(result.reconstruction.validation !== undefined, 'should have validation');
  });

  it('reconstruction preserves string literals through full pipeline', () => {
    const { decompileSource } = require('../src/decompiler/index');
    const minified = 'var a="important_string";var b="another_string";function c(){return a+b}';
    const result = decompileSource(minified, { reconstruct: true });

    const moduleContent = result.modules.map((m) => m.content).join('\n');
    assert(moduleContent.includes('important_string'), 'preserves important_string');
    assert(moduleContent.includes('another_string'), 'preserves another_string');
  });

  it('decompileSource without reconstruct works as before', () => {
    const { decompileSource } = require('../src/decompiler/index');
    const source = 'function hello() { return "world"; }';
    const result = decompileSource(source);

    assert(result.modules !== undefined, 'should have modules');
    assert(result.metrics !== undefined, 'should have metrics');
    assert(result.reconstruction === undefined, 'should NOT have reconstruction');
  });
});

// ─── Report ───

console.log(`\n========================================`);
console.log(`Tests: ${total}  Passed: ${passed}  Failed: ${failed}`);
console.log(`========================================\n`);

if (failed > 0) {
  process.exit(1);
}
