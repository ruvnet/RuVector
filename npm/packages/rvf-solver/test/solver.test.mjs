#!/usr/bin/env node

import assert from 'assert';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const packageDir = path.dirname(__dirname);

/**
 * Test suite for RvfSolver TypeScript SDK
 *
 * Validates:
 * 1. Import/export of all types and classes
 * 2. Type structure and defaults
 * 3. WASM integration (if binary available)
 */

console.log('Starting RvfSolver TypeScript SDK validation tests...\n');

// Test 1: Import/export validation
console.log('Test 1: Import/export validation');
try {
  // Try both CommonJS (.js) and ESM (.mjs) outputs
  let distIndex = path.join(packageDir, 'dist', 'index.mjs');
  if (!fs.existsSync(distIndex)) {
    distIndex = path.join(packageDir, 'dist', 'index.js');
  }

  if (!fs.existsSync(distIndex)) {
    throw new Error('dist/index.mjs or dist/index.js not found - run: npm run build');
  }

  const module = await import(`file://${distIndex}`);

  // Handle both default and named exports (CommonJS)
  const RvfSolver = module.RvfSolver || module.default?.RvfSolver;

  assert(typeof RvfSolver === 'function', 'RvfSolver class should be exported');
  console.log('  ✓ RvfSolver class exported');

  // Verify all type exports (TypeScript types don't exist at runtime, but we verify the structure)
  const exportedNames = Object.keys(module).sort();
  console.log(`  ✓ Module exports: ${exportedNames.join(', ')}`);

  // Check minimum expected exports
  assert(exportedNames.includes('RvfSolver') || module.default?.RvfSolver, 'Should export RvfSolver');
  console.log('  ✓ All required exports present (RvfSolver)');

} catch (error) {
  console.error(`  ✗ FAILED: ${error.message}`);
  process.exit(1);
}

// Test 2: Type structure validation (via TypeScript definitions)
console.log('\nTest 2: Type structure validation');
try {
  const dtsFile = path.join(packageDir, 'dist', 'types.d.ts');

  if (!fs.existsSync(dtsFile)) {
    throw new Error(`dist/types.d.ts not found - run: npm run build`);
  }

  const dtsContent = fs.readFileSync(dtsFile, 'utf8');

  // Verify TrainOptions interface
  assert(dtsContent.includes('interface TrainOptions'), 'TrainOptions interface should be defined');
  assert(dtsContent.includes('count: number'), 'TrainOptions should have count property');
  assert(dtsContent.includes('minDifficulty'), 'TrainOptions should have minDifficulty property');
  assert(dtsContent.includes('maxDifficulty'), 'TrainOptions should have maxDifficulty property');
  assert(dtsContent.includes('seed'), 'TrainOptions should have seed property');
  console.log('  ✓ TrainOptions has expected shape (count, minDifficulty, maxDifficulty, seed)');

  // Verify TrainResult interface
  assert(dtsContent.includes('interface TrainResult'), 'TrainResult interface should be defined');
  assert(dtsContent.includes('trained: number'), 'TrainResult should have trained property');
  assert(dtsContent.includes('correct: number'), 'TrainResult should have correct property');
  assert(dtsContent.includes('accuracy: number'), 'TrainResult should have accuracy property');
  assert(dtsContent.includes('patternsLearned: number'), 'TrainResult should have patternsLearned property');
  console.log('  ✓ TrainResult has expected fields (trained, correct, accuracy, patternsLearned)');

  // Verify AcceptanceOptions interface
  assert(dtsContent.includes('interface AcceptanceOptions'), 'AcceptanceOptions interface should be defined');
  assert(dtsContent.includes('holdoutSize'), 'AcceptanceOptions should have holdoutSize property');
  assert(dtsContent.includes('trainingPerCycle'), 'AcceptanceOptions should have trainingPerCycle property');
  assert(dtsContent.includes('cycles'), 'AcceptanceOptions should have cycles property');
  assert(dtsContent.includes('stepBudget'), 'AcceptanceOptions should have stepBudget property');
  console.log('  ✓ AcceptanceOptions has expected defaults (holdoutSize, trainingPerCycle, cycles, stepBudget)');

  // Verify AcceptanceManifest interface
  assert(dtsContent.includes('interface AcceptanceManifest'), 'AcceptanceManifest interface should be defined');
  assert(dtsContent.includes('modeA: AcceptanceModeResult'), 'AcceptanceManifest should have modeA property');
  assert(dtsContent.includes('modeB: AcceptanceModeResult'), 'AcceptanceManifest should have modeB property');
  assert(dtsContent.includes('modeC: AcceptanceModeResult'), 'AcceptanceManifest should have modeC property');
  assert(dtsContent.includes('allPassed: boolean'), 'AcceptanceManifest should have allPassed property');
  console.log('  ✓ AcceptanceManifest has expected fields (modeA, modeB, modeC, allPassed)');

  // Verify AcceptanceModeResult interface
  assert(dtsContent.includes('interface AcceptanceModeResult'), 'AcceptanceModeResult interface should be defined');
  assert(dtsContent.includes('passed: boolean'), 'AcceptanceModeResult should have passed property');
  assert(dtsContent.includes('finalAccuracy: number'), 'AcceptanceModeResult should have finalAccuracy property');
  assert(dtsContent.includes('cycles: CycleMetrics[]'), 'AcceptanceModeResult should have cycles property');
  console.log('  ✓ AcceptanceModeResult has expected structure');

  // Verify PolicyState interface
  assert(dtsContent.includes('interface PolicyState'), 'PolicyState interface should be defined');
  assert(dtsContent.includes('contextStats'), 'PolicyState should have contextStats property');
  assert(dtsContent.includes('earlyCommitPenalties'), 'PolicyState should have earlyCommitPenalties property');
  console.log('  ✓ PolicyState has expected properties (contextStats, earlyCommitPenalties, etc.)');

  // Verify SkipMode type
  assert(dtsContent.includes("type SkipMode = 'none' | 'weekday' | 'hybrid'"), 'SkipMode type should be defined');
  console.log('  ✓ SkipMode type defined (none | weekday | hybrid)');

  // Verify SkipModeStats interface
  assert(dtsContent.includes('interface SkipModeStats'), 'SkipModeStats interface should be defined');
  assert(dtsContent.includes('attempts: number'), 'SkipModeStats should have attempts property');
  assert(dtsContent.includes('successes: number'), 'SkipModeStats should have successes property');
  console.log('  ✓ SkipModeStats has expected structure (attempts, successes, totalSteps, etc.)');

  // Verify CompiledConfig interface
  assert(dtsContent.includes('interface CompiledConfig'), 'CompiledConfig interface should be defined');
  assert(dtsContent.includes('maxSteps: number'), 'CompiledConfig should have maxSteps property');
  assert(dtsContent.includes('avgSteps: number'), 'CompiledConfig should have avgSteps property');
  console.log('  ✓ CompiledConfig has expected fields (maxSteps, avgSteps, observations, etc.)');

  // Verify CycleMetrics interface
  assert(dtsContent.includes('interface CycleMetrics'), 'CycleMetrics interface should be defined');
  assert(dtsContent.includes('cycle: number'), 'CycleMetrics should have cycle property');
  assert(dtsContent.includes('accuracy: number'), 'CycleMetrics should have accuracy property');
  assert(dtsContent.includes('costPerSolve: number'), 'CycleMetrics should have costPerSolve property');
  console.log('  ✓ CycleMetrics has expected structure (cycle, accuracy, costPerSolve)');

} catch (error) {
  console.error(`  ✗ FAILED: ${error.message}`);
  process.exit(1);
}

// Test 3: WASM integration test (conditional)
console.log('\nTest 3: WASM integration test');
try {
  const wasmBinaryPath = path.join(packageDir, 'pkg', 'rvf_solver_bg.wasm');

  if (!fs.existsSync(wasmBinaryPath)) {
    console.warn('  ⊘ SKIPPED: WASM binary not available at pkg/rvf_solver_bg.wasm');
  } else {
    console.log('  Found WASM binary, loading solver...');

    // Try both CommonJS and ESM outputs
    let distIndex = path.join(packageDir, 'dist', 'index.mjs');
    if (!fs.existsSync(distIndex)) {
      distIndex = path.join(packageDir, 'dist', 'index.js');
    }

    const module = await import(`file://${distIndex}`);
    const RvfSolver = module.RvfSolver || module.default?.RvfSolver;

    // Create solver instance
    console.log('  Creating RvfSolver instance...');
    const solver = await RvfSolver.create();
    assert(solver !== null, 'Solver should be created');
    console.log('  ✓ RvfSolver.create() succeeded');

    // Run small training session
    console.log('  Running training with count=10...');
    const trainResult = solver.train({ count: 10 });

    // Verify result structure
    assert(typeof trainResult.trained === 'number', 'trained should be a number');
    assert(typeof trainResult.correct === 'number', 'correct should be a number');
    assert(typeof trainResult.accuracy === 'number', 'accuracy should be a number');
    // patternsLearned may be undefined if not populated by WASM, but if present should be a number
    assert(trainResult.patternsLearned === undefined || typeof trainResult.patternsLearned === 'number', 'patternsLearned should be a number or undefined');

    // Verify expected values
    assert(trainResult.trained === 10, `trained should be 10, got ${trainResult.trained}`);
    assert(trainResult.correct >= 0 && trainResult.correct <= 10, 'correct should be between 0 and 10');
    assert(trainResult.accuracy >= 0 && trainResult.accuracy <= 1, 'accuracy should be between 0 and 1');
    assert(trainResult.patternsLearned === undefined || trainResult.patternsLearned >= 0, 'patternsLearned should be non-negative or undefined');

    console.log(`  ✓ Training result valid: trained=${trainResult.trained}, correct=${trainResult.correct}, accuracy=${(trainResult.accuracy * 100).toFixed(1)}%`);

    // Clean up
    solver.destroy();
    console.log('  ✓ Solver destroyed successfully');

    // Test 4: results match the declared camelCase types
    console.log('\nTest 4: result shapes match the declared types');
    const s4 = await RvfSolver.create();
    const t4 = s4.train({ count: 20, seed: 1 });
    assert(typeof t4.patternsLearned === 'number', `patternsLearned should be a number, got ${t4.patternsLearned}`);
    const m4 = s4.acceptance({ holdoutSize: 5, trainingPerCycle: 5, cycles: 2, seed: 1 });
    for (const key of ['modeA', 'modeB', 'modeC']) {
      const mode = m4[key];
      assert(typeof mode.passed === 'boolean', `${key}.passed should be boolean`);
      assert(typeof mode.finalAccuracy === 'number', `${key}.finalAccuracy should be a number, got ${mode.finalAccuracy}`);
      assert(mode.cycles.length === 2, `${key} should have 2 cycles`);
      assert(typeof mode.cycles[0].costPerSolve === 'number', `${key}.cycles[0].costPerSolve should be a number`);
      assert(mode.finalAccuracy === mode.cycles[1].accuracy, `${key}.finalAccuracy should equal the last cycle accuracy`);
    }
    const p4 = s4.policy();
    assert(p4 && typeof p4.contextStats === 'object', 'policy.contextStats should be an object');
    assert(typeof p4.earlyCommitPenalties === 'number', 'policy.earlyCommitPenalties should be a number');
    const buckets = Object.values(p4.contextStats);
    assert(buckets.length > 0, 'policy should have at least one context bucket after training');
    const arm = Object.values(buckets[0])[0];
    assert(typeof arm.alphaSafety === 'number', 'SkipModeStats.alphaSafety should be a number');
    assert(typeof arm.totalSteps === 'number', 'SkipModeStats.totalSteps should be a number');
    console.log('  ✓ train/acceptance/policy return camelCase fields');

    // Test 5: invalid arguments are rejected without trapping the WASM module
    console.log('\nTest 5: argument validation');
    const bad = [
      ['train count 0', () => s4.train({ count: 0 })],
      ['train count -1', () => s4.train({ count: -1 })],
      ['train count 1.5', () => s4.train({ count: 1.5 })],
      ['train count NaN', () => s4.train({ count: NaN })],
      ['train count 2^32', () => s4.train({ count: 2 ** 32 })],
      ['train minDifficulty 0', () => s4.train({ count: 1, minDifficulty: 0 })],
      ['train maxDifficulty 11', () => s4.train({ count: 1, maxDifficulty: 11 })],
      ['train min > max', () => s4.train({ count: 1, minDifficulty: 5, maxDifficulty: 2 })],
      ['acceptance cycles 0', () => s4.acceptance({ cycles: 0 })],
      ['acceptance holdoutSize 0', () => s4.acceptance({ holdoutSize: 0, cycles: 1 })],
      ['acceptance trainingPerCycle -1', () => s4.acceptance({ trainingPerCycle: -1, cycles: 1 })],
      ['acceptance stepBudget 0', () => s4.acceptance({ stepBudget: 0, cycles: 1 })],
    ];
    for (const [label, fn] of bad) {
      assert.throws(fn, RangeError, `${label} should throw RangeError`);
    }
    const after = s4.train({ count: 3, seed: 2 });
    assert(after.trained === 3 && Number.isFinite(after.accuracy), 'solver should still work after rejected calls');
    s4.destroy();
    console.log(`  ✓ ${bad.length} invalid argument sets rejected with RangeError`);

    // Test 6: the raw WASM ABI rejects bad arguments instead of trapping
    console.log('\nTest 6: WASM ABI argument checks');
    const initWasm = (await import(`file://${path.join(packageDir, 'pkg', 'rvf_solver.js')}`)).default;
    const wasm = await initWasm();
    const h = wasm.rvf_solver_create();
    assert(h > 0, 'raw create should return a handle');
    assert.strictEqual(wasm.rvf_solver_train(h, -1, 1, 10, 1, 0), -2, 'raw train count=-1 should return -2');
    assert.strictEqual(wasm.rvf_solver_train(h, 0, 1, 10, 1, 0), -2, 'raw train count=0 should return -2');
    assert.strictEqual(wasm.rvf_solver_acceptance(h, 5, 5, 0, 100, 1, 0), -2, 'raw acceptance cycles=0 should return -2');
    assert.strictEqual(wasm.rvf_solver_acceptance(h, 0, 5, 1, 100, 1, 0), -2, 'raw acceptance holdout=0 should return -2');
    assert.strictEqual(wasm.rvf_solver_acceptance(h, 5, -1, 1, 100, 1, 0), -2, 'raw acceptance training=-1 should return -2');
    assert.strictEqual(wasm.rvf_solver_train(-2147483648, 1, 1, 10, 1, 0), -1, 'raw train with i32::MIN handle should return -1');
    wasm.rvf_solver_destroy(h);
    console.log('  ✓ raw exports return -2 for invalid arguments');

  }
} catch (error) {
  console.error(`  ✗ FAILED: ${error.message}`);
  process.exit(1);
}

console.log('\n✅ All tests passed!');
