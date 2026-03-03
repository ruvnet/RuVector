#!/usr/bin/env node
/**
 * RuVector CLI Startup & Command Benchmark Suite
 *
 * Measures:
 * - CLI startup time (cold and warm)
 * - Per-command execution time
 * - Module loading overhead
 * - Lazy loading effectiveness
 *
 * Usage:
 *   node test/benchmark-cli.js              # Run full benchmark
 *   node test/benchmark-cli.js --quick      # Quick mode (fewer iterations)
 *   node test/benchmark-cli.js --modules    # Module loading profile only
 *   node test/benchmark-cli.js --json       # Output as JSON
 */
'use strict';

const { execSync } = require('child_process');
const path = require('path');

const CLI_PATH = path.join(__dirname, '..', 'bin', 'cli.js');
const ITERATIONS = process.argv.includes('--quick') ? 3 : 5;
const JSON_OUTPUT = process.argv.includes('--json');
const MODULES_ONLY = process.argv.includes('--modules');

// ============================================================================
// Utilities
// ============================================================================

function runCommand(cmd, opts = {}) {
  const start = process.hrtime.bigint();
  try {
    execSync(`node ${CLI_PATH} ${cmd}`, {
      encoding: 'utf8',
      timeout: opts.timeout || 15000,
      cwd: path.join(__dirname, '..'),
      stdio: ['pipe', 'pipe', 'pipe'],
    });
  } catch (e) {
    // Command may fail (e.g., missing deps), timing still valid
  }
  const end = process.hrtime.bigint();
  return Number(end - start) / 1e6;
}

function benchmarkCommand(cmd, iterations) {
  const times = [];
  for (let i = 0; i < iterations; i++) {
    times.push(runCommand(cmd));
  }
  times.sort((a, b) => a - b);
  const avg = times.reduce((a, b) => a + b) / times.length;
  const min = times[0];
  const max = times[times.length - 1];
  const median = times[Math.floor(times.length / 2)];
  const p95 = times[Math.floor(times.length * 0.95)] || max;
  return { avg, min, max, median, p95, samples: times };
}

function measureModuleLoad(name, mod) {
  const start = process.hrtime.bigint();
  let status = 'ok';
  try {
    require(mod);
  } catch (e) {
    status = 'not-found';
  }
  const end = process.hrtime.bigint();
  return { name, module: mod, time: Number(end - start) / 1e6, status };
}

function formatMs(ms) {
  return ms.toFixed(0) + 'ms';
}

function formatPct(before, after) {
  if (!before || before === 0) return '';
  const pct = ((before - after) / before * 100);
  if (pct > 0) return ` (${pct.toFixed(0)}% faster)`;
  if (pct < 0) return ` (${Math.abs(pct).toFixed(0)}% slower)`;
  return '';
}

// ============================================================================
// Module Loading Profile
// ============================================================================

function profileModules() {
  const modules = [
    ['commander', 'commander'],
    ['chalk', 'chalk'],
    ['ora', 'ora'],
    ['fs', 'fs'],
    ['path', 'path'],
    ['@ruvector/core', '@ruvector/core'],
    ['@ruvector/gnn', '@ruvector/gnn'],
    ['@ruvector/attention', '@ruvector/attention'],
    ['@ruvector/sona', '@ruvector/sona'],
    ['@ruvector/rvf', '@ruvector/rvf'],
    ['ruvector dist/index', '../dist/index.js'],
  ];

  const results = [];
  for (const [name, mod] of modules) {
    // Clear relevant require cache entries to measure fresh load
    const cacheKeys = Object.keys(require.cache);
    const toDelete = cacheKeys.filter(k => {
      const bn = path.basename(k, '.js');
      return k.includes(name.replace('@ruvector/', '')) && !k.includes('benchmark-cli');
    });
    toDelete.forEach(k => delete require.cache[k]);

    results.push(measureModuleLoad(name, mod));
  }
  return results;
}

// ============================================================================
// Main Benchmark
// ============================================================================

function runBenchmarks() {
  const results = {};

  // 1. Module profiling
  if (!JSON_OUTPUT) {
    console.log('\n' + '='.repeat(70));
    console.log('  RUVECTOR CLI BENCHMARK SUITE');
    console.log('='.repeat(70));
    console.log(`  Iterations: ${ITERATIONS}`);
    console.log(`  Node: ${process.version}`);
    console.log(`  Platform: ${process.platform} ${process.arch}`);
    console.log('='.repeat(70));
  }

  // Module loading profile
  if (!JSON_OUTPUT) {
    console.log('\n  MODULE LOADING PROFILE');
    console.log('  ' + '-'.repeat(66));
    console.log('  ' + 'Module'.padEnd(30) + 'Time'.padStart(10) + '  Status');
    console.log('  ' + '-'.repeat(66));
  }

  const moduleResults = profileModules();
  let totalModuleTime = 0;

  for (const r of moduleResults) {
    totalModuleTime += r.time;
    if (!JSON_OUTPUT) {
      const statusStr = r.status === 'ok' ? 'loaded' : 'not found';
      console.log('  ' + r.name.padEnd(30) + formatMs(r.time).padStart(10) + '  ' + statusStr);
    }
  }

  if (!JSON_OUTPUT) {
    console.log('  ' + '-'.repeat(66));
    console.log('  ' + 'TOTAL'.padEnd(30) + formatMs(totalModuleTime).padStart(10));
  }

  results.modules = moduleResults;
  results.totalModuleTime = totalModuleTime;

  if (MODULES_ONLY) {
    if (JSON_OUTPUT) {
      console.log(JSON.stringify(results, null, 2));
    }
    return results;
  }

  // 2. Cold start
  if (!JSON_OUTPUT) {
    console.log('\n  COLD START');
    console.log('  ' + '-'.repeat(66));
  }

  const coldStart = runCommand('--version');
  results.coldStart = coldStart;

  if (!JSON_OUTPUT) {
    console.log('  ' + '--version (cold)'.padEnd(30) + formatMs(coldStart).padStart(10));
  }

  // 3. Warm start - commands benchmark
  if (!JSON_OUTPUT) {
    console.log('\n  COMMAND BENCHMARKS (' + ITERATIONS + ' iterations each)');
    console.log('  ' + '-'.repeat(66));
    console.log('  ' + 'Command'.padEnd(25) + 'Avg'.padStart(8) + 'Min'.padStart(8) + 'Max'.padStart(8) + 'Med'.padStart(8) + 'P95'.padStart(8));
    console.log('  ' + '-'.repeat(66));
  }

  // Warm up
  runCommand('--version');

  const commands = [
    { cmd: '--version', label: '--version', category: 'startup' },
    { cmd: '--help', label: '--help', category: 'startup' },
    { cmd: 'info', label: 'info', category: 'info' },
    { cmd: 'gnn info', label: 'gnn info', category: 'gnn' },
    { cmd: 'attention info', label: 'attention info', category: 'attention' },
    { cmd: 'install --list', label: 'install --list', category: 'info' },
    { cmd: 'doctor', label: 'doctor', category: 'diagnostic', timeout: 30000 },
  ];

  results.commands = {};

  for (const { cmd, label, category, timeout } of commands) {
    const bench = benchmarkCommand(cmd, ITERATIONS);
    results.commands[label] = { ...bench, category };

    if (!JSON_OUTPUT) {
      console.log(
        '  ' +
        label.padEnd(25) +
        formatMs(bench.avg).padStart(8) +
        formatMs(bench.min).padStart(8) +
        formatMs(bench.max).padStart(8) +
        formatMs(bench.median).padStart(8) +
        formatMs(bench.p95).padStart(8)
      );
    }
  }

  // 4. Lazy loading effectiveness
  if (!JSON_OUTPUT) {
    console.log('\n  LAZY LOADING ANALYSIS');
    console.log('  ' + '-'.repeat(66));
  }

  const versionTime = results.commands['--version'].avg;
  const infoTime = results.commands['info'].avg;
  const gnnInfoTime = results.commands['gnn info'].avg;
  const attentionInfoTime = results.commands['attention info'].avg;

  const lazyLoadOverhead = {
    gnn: gnnInfoTime - versionTime,
    attention: attentionInfoTime - versionTime,
    info: infoTime - versionTime,
  };

  results.lazyLoadOverhead = lazyLoadOverhead;

  if (!JSON_OUTPUT) {
    console.log('  Base startup (--version):     ' + formatMs(versionTime));
    console.log('  GNN lazy load overhead:       ' + formatMs(lazyLoadOverhead.gnn));
    console.log('  Attention lazy load overhead:  ' + formatMs(lazyLoadOverhead.attention));
    console.log('  Info command overhead:         ' + formatMs(lazyLoadOverhead.info));
  }

  // 5. Summary
  if (!JSON_OUTPUT) {
    console.log('\n' + '='.repeat(70));
    console.log('  SUMMARY');
    console.log('='.repeat(70));

    const startupCommands = Object.entries(results.commands)
      .filter(([, v]) => v.category === 'startup');
    const avgStartup = startupCommands.reduce((s, [, v]) => s + v.avg, 0) / startupCommands.length;

    console.log('  Cold start:           ' + formatMs(results.coldStart));
    console.log('  Avg startup (warm):   ' + formatMs(avgStartup));
    console.log('  Module load total:    ' + formatMs(results.totalModuleTime));

    // Performance budget check
    const BUDGET_MS = 100;
    const withinBudget = versionTime < BUDGET_MS;
    console.log('');
    if (withinBudget) {
      console.log('  PASS: Startup ' + formatMs(versionTime) + ' is within ' + BUDGET_MS + 'ms budget');
    } else {
      console.log('  WARN: Startup ' + formatMs(versionTime) + ' exceeds ' + BUDGET_MS + 'ms budget');
    }

    console.log('='.repeat(70) + '\n');
  }

  if (JSON_OUTPUT) {
    console.log(JSON.stringify(results, null, 2));
  }

  return results;
}

// ============================================================================
// Run
// ============================================================================

runBenchmarks();
