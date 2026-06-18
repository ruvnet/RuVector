#!/usr/bin/env node

/**
 * Startup-budget CI guard (ADR-256 step 4).
 *
 * The ruvector CLI lazy-loads every heavy dependency to keep cold start fast.
 * ADR-256 borrows harness features (the `harness` router surface, MCP policy)
 * into the CLI — this guard fails CI if any of them regress startup.
 *
 * Strategy (robust across machines / loaded CI boxes):
 *   1. ABSOLUTE ceiling — `--help` cold start must stay under a generous cap.
 *   2. RELATIVE delta   — `harness status --json` must not add more than a small
 *      delta over the `--help` baseline (the real regression signal: eager-loading
 *      a heavy module into the harness path would blow this out).
 *
 * Most of the floor (~100ms) is the Node process spawn itself, which is why the
 * relative delta is the meaningful check. Budgets are overridable via env.
 */
'use strict';

const { execSync } = require('child_process');
const path = require('path');
const assert = require('assert');

const CLI = path.join(__dirname, '..', 'bin', 'cli.js');
const CWD = path.join(__dirname, '..');

const ABS_BUDGET_MS = Number(process.env.RUVECTOR_STARTUP_BUDGET_MS || 2000);
const DELTA_BUDGET_MS = Number(process.env.RUVECTOR_STARTUP_DELTA_MS || 120);
const SAMPLES = Number(process.env.RUVECTOR_STARTUP_SAMPLES || 5);

function timeMs(args) {
  const start = process.hrtime.bigint();
  try {
    execSync(`node ${CLI} ${args}`, { stdio: ['pipe', 'pipe', 'pipe'], timeout: 20000, cwd: CWD });
  } catch {
    // non-zero exit still yields a valid timing
  }
  return Number(process.hrtime.bigint() - start) / 1e6;
}

function median(args) {
  const t = [];
  for (let i = 0; i < SAMPLES; i++) t.push(timeMs(args));
  t.sort((a, b) => a - b);
  return t[Math.floor(t.length / 2)];
}

let passed = 0;
let failed = 0;
const failures = [];
function test(name, fn) {
  try { fn(); passed++; console.log(`  PASS  ${name}`); }
  catch (err) { failed++; failures.push({ name, error: err.message || String(err) }); console.log(`  FAIL  ${name}`); console.log(`        ${err.message || err}`); }
}

console.log('\n--- Startup-budget guard (ADR-256) ---\n');
console.log(`  samples=${SAMPLES}  abs_budget=${ABS_BUDGET_MS}ms  delta_budget=${DELTA_BUDGET_MS}ms\n`);

// Warm up (filesystem cache, AV scan of the file, etc.)
timeMs('--version');

const helpMs = median('--help');
const harnessMs = median('harness status --json');
const delta = harnessMs - helpMs;

console.log(`  --help (cold):            ${helpMs.toFixed(0)}ms`);
console.log(`  harness status --json:    ${harnessMs.toFixed(0)}ms  (Δ ${delta >= 0 ? '+' : ''}${delta.toFixed(0)}ms vs --help)\n`);

test(`--help cold start under ${ABS_BUDGET_MS}ms absolute budget`, () => {
  assert(helpMs < ABS_BUDGET_MS,
    `--help startup ${helpMs.toFixed(0)}ms exceeds ${ABS_BUDGET_MS}ms (set RUVECTOR_STARTUP_BUDGET_MS to override)`);
});

test(`harness status adds < ${DELTA_BUDGET_MS}ms over --help baseline (no lazy-load regression)`, () => {
  assert(delta < DELTA_BUDGET_MS,
    `harness status added ${delta.toFixed(0)}ms over --help — a heavy module may have leaked into the startup path`);
});

console.log('\n' + '='.repeat(60));
console.log(`\nResults: ${passed} passed, ${failed} failed\n`);
if (failures.length > 0) {
  console.log('Failures:');
  for (const f of failures) console.log(`  - ${f.name}: ${f.error}`);
  console.log('');
}
if (failed > 0) { console.log('STARTUP BUDGET EXCEEDED\n'); process.exit(1); }
else { console.log('STARTUP WITHIN BUDGET\n'); }
