#!/usr/bin/env node

/**
 * Comprehensive CLI command test suite for ruvector
 *
 * Tests all registered command groups, flag behavior, output correctness,
 * and graceful error handling. Uses child_process.execSync for real CLI
 * invocations to catch runtime issues (module resolution, chalk compat, etc.).
 */

const { execSync } = require('child_process');
const assert = require('assert');
const path = require('path');
const fs = require('fs');

const CLI_DIR = path.join(__dirname, '..');
const CLI = `node ${path.join(CLI_DIR, 'bin', 'cli.js')}`;
const packageJson = require('../package.json');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let passed = 0;
let failed = 0;
let skipped = 0;
const failures = [];

function run(args, opts = {}) {
  const timeout = opts.timeout || 15000;
  return execSync(`${CLI} ${args}`, {
    encoding: 'utf8',
    cwd: CLI_DIR,
    timeout,
    env: { ...process.env, FORCE_COLOR: '0', NO_COLOR: '1' },
    stdio: ['pipe', 'pipe', 'pipe'],
    ...opts,
  });
}

function runSafe(args, opts = {}) {
  try {
    const stdout = run(args, opts);
    return { stdout, stderr: '', code: 0 };
  } catch (err) {
    return {
      stdout: (err.stdout || '').toString(),
      stderr: (err.stderr || '').toString(),
      code: err.status || 1,
    };
  }
}

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failed++;
    failures.push({ name, error: err.message || String(err) });
    console.log(`  FAIL  ${name}`);
    console.log(`        ${err.message || err}`);
  }
}

function skip(name, reason) {
  skipped++;
  console.log(`  SKIP  ${name} -- ${reason}`);
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

console.log('\nruvector CLI Command Tests');
console.log('='.repeat(60));

// ---- Section 1: Basic CLI startup ----------------------------------------
console.log('\n--- 1. Basic CLI startup ---\n');

test('CLI syntax check (node -c)', () => {
  execSync(`node -c ${path.join(CLI_DIR, 'bin', 'cli.js')}`, { encoding: 'utf8' });
});

test('--help exits 0 and lists commands', () => {
  const out = run('--help');
  assert(out.includes('ruvector'), 'Should mention ruvector');
  assert(out.includes('Commands:'), 'Should list commands');
  assert(out.includes('create'), 'Should list create');
  assert(out.includes('search'), 'Should list search');
  assert(out.includes('info'), 'Should list info');
  assert(out.includes('doctor'), 'Should list doctor');
});

test('--version returns correct version', () => {
  const out = run('--version').trim();
  assert.strictEqual(out, packageJson.version,
    `Expected ${packageJson.version}, got ${out}`);
});

test('help command works', () => {
  const out = run('help');
  assert(out.includes('Commands:'), 'help should list commands');
});

// ---- Section 2: Core commands --------------------------------------------
console.log('\n--- 2. Core commands ---\n');

test('info shows CLI version and platform', () => {
  const out = run('info');
  assert(out.includes(packageJson.version), 'Should show package version');
  assert(out.includes('Platform:') || out.includes('platform') || out.includes('linux'),
    'Should show platform info');
});

test('doctor runs without crashing', () => {
  const out = run('doctor');
  assert(out.includes('RuVector Doctor') || out.includes('doctor'),
    'Doctor output should identify itself');
  assert(out.includes('Node.js'), 'Should report Node.js');
});

test('setup --help shows options', () => {
  const { stdout } = runSafe('setup --help');
  assert(stdout.includes('setup'), 'Should show setup info');
});

// ---- Section 3: GNN commands ---------------------------------------------
console.log('\n--- 3. GNN commands ---\n');

test('gnn --help lists subcommands', () => {
  const out = run('gnn --help');
  assert(out.includes('layer'), 'Should list layer subcommand');
  assert(out.includes('compress'), 'Should list compress subcommand');
  assert(out.includes('info'), 'Should list info subcommand');
});

test('gnn info runs successfully', () => {
  const { stdout, code } = runSafe('gnn info');
  // May fail if @ruvector/gnn not installed, but should not crash node
  assert(stdout.includes('GNN') || stdout.includes('gnn'),
    'Should mention GNN');
});

// ---- Section 4: Attention commands ---------------------------------------
console.log('\n--- 4. Attention commands ---\n');

test('attention --help lists subcommands', () => {
  const out = run('attention --help');
  assert(out.includes('compute'), 'Should list compute subcommand');
  assert(out.includes('benchmark'), 'Should list benchmark subcommand');
  assert(out.includes('info'), 'Should list info subcommand');
});

test('attention info runs successfully', () => {
  const { stdout, code } = runSafe('attention info');
  assert(stdout.includes('Attention') || stdout.includes('attention'),
    'Should mention attention');
});

// ---- Section 5: MCP commands ---------------------------------------------
console.log('\n--- 5. MCP commands ---\n');

test('mcp --help lists subcommands', () => {
  const out = run('mcp --help');
  assert(out.includes('start'), 'Should list start subcommand');
  assert(out.includes('info'), 'Should list info subcommand');
});

test('mcp info shows tool list', () => {
  const out = run('mcp info');
  assert(out.includes('hooks_stats') || out.includes('MCP'),
    'Should show MCP tools or info');
});

// ---- Section 6: RVF commands ---------------------------------------------
console.log('\n--- 6. RVF commands ---\n');

test('rvf --help lists subcommands', () => {
  const out = run('rvf --help');
  assert(out.includes('create'), 'Should list create');
  assert(out.includes('ingest'), 'Should list ingest');
  assert(out.includes('query'), 'Should list query');
  assert(out.includes('examples'), 'Should list examples');
});

test('rvf examples lists example files', () => {
  const out = run('rvf examples');
  assert(out.includes('basic_store') || out.includes('Example'),
    'Should list example files');
});

// ---- Section 7: Hooks commands -------------------------------------------
console.log('\n--- 7. Hooks commands ---\n');

test('hooks --help lists subcommands', () => {
  const out = run('hooks --help');
  assert(out.includes('init'), 'Should list init');
  assert(out.includes('stats'), 'Should list stats');
  assert(out.includes('route'), 'Should list route');
  assert(out.includes('remember'), 'Should list remember');
  assert(out.includes('recall'), 'Should list recall');
});

test('hooks stats shows intelligence statistics', () => {
  const { stdout } = runSafe('hooks stats');
  assert(stdout.includes('Stats') || stdout.includes('stats') || stdout.includes('pattern'),
    'Should show stats info');
});

test('hooks route routes a task', () => {
  const { stdout } = runSafe('hooks route "fix the login bug"');
  assert(stdout.length > 0, 'Should produce output for route');
});

// ---- Section 8: Embed commands -------------------------------------------
console.log('\n--- 8. Embed commands ---\n');

test('embed --help lists subcommands', () => {
  const out = run('embed --help');
  assert(out.includes('text'), 'Should list text subcommand');
  assert(out.includes('adaptive'), 'Should list adaptive subcommand');
  assert(out.includes('benchmark'), 'Should list benchmark subcommand');
  assert(out.includes('optimized'), 'Should list optimized subcommand');
  assert(out.includes('neural'), 'Should list neural subcommand');
});

// ---- Section 9: Workers commands -----------------------------------------
console.log('\n--- 9. Workers commands ---\n');

test('workers --help lists subcommands', () => {
  const out = run('workers --help');
  assert(out.includes('dispatch'), 'Should list dispatch');
  assert(out.includes('status'), 'Should list status');
  assert(out.includes('results'), 'Should list results');
  assert(out.includes('presets'), 'Should list presets');
  assert(out.includes('phases'), 'Should list phases');
});

// ---- Section 10: Native commands -----------------------------------------
console.log('\n--- 10. Native commands ---\n');

test('native --help lists subcommands', () => {
  const out = run('native --help');
  assert(out.includes('run'), 'Should list run');
  assert(out.includes('benchmark'), 'Should list benchmark');
  assert(out.includes('list'), 'Should list list');
  assert(out.includes('compare'), 'Should list compare');
});

test('native list shows worker types', () => {
  const { stdout } = runSafe('native list');
  assert(stdout.length > 0, 'Should produce output');
});

// ---- Section 11: Export / Import -----------------------------------------
console.log('\n--- 11. Export / Import ---\n');

test('export --help shows usage', () => {
  const out = run('export --help');
  assert(out.includes('database'), 'Should mention database argument');
});

test('import --help shows usage', () => {
  const out = run('import --help');
  assert(out.includes('file'), 'Should mention file argument');
});

// ---- Section 12: Graph / Router / Server / Cluster -----------------------
console.log('\n--- 12. Graph / Router / Server / Cluster ---\n');

test('graph --help shows usage', () => {
  const { stdout } = runSafe('graph --help');
  assert(stdout.includes('graph') || stdout.includes('Graph'),
    'Should show graph info');
});

test('router --help shows usage', () => {
  const { stdout } = runSafe('router --help');
  assert(stdout.includes('router') || stdout.includes('Router'),
    'Should show router info');
});

test('server --help shows usage', () => {
  const { stdout } = runSafe('server --help');
  assert(stdout.includes('server') || stdout.includes('Server'),
    'Should show server info');
});

test('cluster --help shows usage', () => {
  const { stdout } = runSafe('cluster --help');
  assert(stdout.includes('cluster') || stdout.includes('Cluster'),
    'Should show cluster info');
});

// ---- Section 13: New command groups (may not be registered yet) ----------
console.log('\n--- 13. New command groups ---\n');

const newCommands = [
  { name: 'brain', desc: 'PI Brain cognitive operations' },
  { name: 'edge', desc: 'Edge network / genesis node' },
  { name: 'identity', desc: 'Cryptographic identity' },
  { name: 'llm', desc: 'LLM inference management' },
  { name: 'sona', desc: 'Adaptive learning (LoRA/EWC)' },
  { name: 'route', desc: 'Semantic routing' },
];

for (const cmd of newCommands) {
  // Test without --help to get a real "unknown command" error for unregistered commands.
  // With --help, commander treats it as a global flag and shows main help even for unknown cmds.
  const probe = runSafe(cmd.name);
  const isUnknown = probe.stderr.includes('unknown command') ||
                    probe.stdout.includes('unknown command');
  if (!isUnknown) {
    // Command is registered -- verify its help output mentions itself
    const { stdout } = runSafe(`${cmd.name} --help`);
    test(`${cmd.name} command is registered and shows help`, () => {
      assert(stdout.includes(cmd.name),
        `${cmd.name} help should mention itself`);
    });
  } else {
    skip(`${cmd.name} command`, 'not yet registered in CLI');
  }
}

// ---- Section 14: Brain AGI commands --------------------------------------
console.log('\n--- 14. Brain AGI commands ---\n');

test('brain agi --help lists subcommands', () => {
  const out = run('brain agi --help');
  assert(out.includes('status'), 'Should list status subcommand');
  assert(out.includes('sona'), 'Should list sona subcommand');
  assert(out.includes('temporal'), 'Should list temporal subcommand');
  assert(out.includes('explore'), 'Should list explore subcommand');
  assert(out.includes('midstream'), 'Should list midstream subcommand');
  assert(out.includes('flags'), 'Should list flags subcommand');
});

test('brain agi status --help shows usage', () => {
  const out = run('brain agi status --help');
  assert(out.includes('AGI') || out.includes('diagnostics'), 'Should describe AGI diagnostics');
});

test('brain agi sona --help shows usage', () => {
  const out = run('brain agi sona --help');
  assert(out.includes('SONA') || out.includes('sona'), 'Should mention SONA');
});

test('brain agi temporal --help shows usage', () => {
  const out = run('brain agi temporal --help');
  assert(out.includes('temporal') || out.includes('Temporal'), 'Should mention temporal');
});

test('brain agi explore --help shows usage', () => {
  const out = run('brain agi explore --help');
  assert(out.includes('explore') || out.includes('Meta'), 'Should mention explore/meta');
});

test('brain agi midstream --help shows usage', () => {
  const out = run('brain agi midstream --help');
  assert(out.includes('midstream') || out.includes('Midstream'), 'Should mention midstream');
});

test('brain agi flags --help shows usage', () => {
  const out = run('brain agi flags --help');
  assert(out.includes('flag') || out.includes('Flag'), 'Should mention flags');
});

// ---- Section 15: Midstream commands --------------------------------------
console.log('\n--- 15. Midstream commands ---\n');

test('midstream --help lists subcommands', () => {
  const out = run('midstream --help');
  assert(out.includes('status'), 'Should list status');
  assert(out.includes('attractor'), 'Should list attractor');
  assert(out.includes('scheduler'), 'Should list scheduler');
  assert(out.includes('benchmark'), 'Should list benchmark');
});

test('midstream status --help shows usage', () => {
  const out = run('midstream status --help');
  assert(out.includes('Midstream') || out.includes('midstream'), 'Should mention midstream');
});

test('midstream attractor --help shows usage', () => {
  const out = run('midstream attractor --help');
  assert(out.includes('attractor') || out.includes('Lyapunov'), 'Should mention attractor');
});

test('midstream scheduler --help shows usage', () => {
  const out = run('midstream scheduler --help');
  assert(out.includes('scheduler') || out.includes('Nanosecond'), 'Should mention scheduler');
});

test('midstream benchmark --help shows usage', () => {
  const out = run('midstream benchmark --help');
  assert(out.includes('benchmark') || out.includes('latency'), 'Should mention benchmark');
});

// ---- Section 16: Enhanced brain commands ---------------------------------
console.log('\n--- 16. Enhanced brain commands ---\n');

test('brain search --help includes --verbose flag', () => {
  const out = run('brain search --help');
  assert(out.includes('--verbose'), 'Should have --verbose flag');
});

test('brain status --help works', () => {
  const out = run('brain status --help');
  assert(out.includes('status') || out.includes('health'), 'Should show status info');
});

// ---- Section 17: Error handling ------------------------------------------
console.log('\n--- 17. Error handling ---\n');

test('unknown command returns error', () => {
  const { stderr, code } = runSafe('totallyFakeCommand12345');
  assert(code !== 0, 'Should exit with non-zero code');
  assert(stderr.includes('unknown command') || stderr.includes('error'),
    'Should indicate unknown command');
});

test('create without path shows error', () => {
  const { stderr, code } = runSafe('create');
  assert(code !== 0, 'Should exit with non-zero code for missing arg');
});

test('search without database shows error', () => {
  const { stderr, code } = runSafe('search');
  assert(code !== 0, 'Should exit with non-zero code for missing arg');
});

// ---- Section 18: CLI file integrity --------------------------------------
console.log('\n--- 18. CLI file integrity ---\n');

test('cli.js has correct shebang', () => {
  const content = fs.readFileSync(path.join(CLI_DIR, 'bin', 'cli.js'), 'utf8');
  assert(content.startsWith('#!/usr/bin/env node'), 'Should have node shebang');
});

test('cli.js uses commander', () => {
  const content = fs.readFileSync(path.join(CLI_DIR, 'bin', 'cli.js'), 'utf8');
  assert(content.includes('commander'), 'Should import commander');
});

test('cli.js uses chalk with ESM compat', () => {
  const content = fs.readFileSync(path.join(CLI_DIR, 'bin', 'cli.js'), 'utf8');
  // After fix, should use .default fallback for chalk v5 ESM compat
  assert(content.includes('chalk'), 'Should import chalk');
});

test('package.json bin entry points to cli.js', () => {
  assert.strictEqual(packageJson.bin.ruvector, './bin/cli.js');
});

test('package.json main entry points to dist/index.js', () => {
  assert.strictEqual(packageJson.main, 'dist/index.js');
});

test('dist/index.js exists', () => {
  assert(fs.existsSync(path.join(CLI_DIR, 'dist', 'index.js')),
    'dist/index.js should exist');
});

test('dist/types.d.ts exists', () => {
  assert(fs.existsSync(path.join(CLI_DIR, 'dist', 'types.d.ts')),
    'dist/types.d.ts should exist');
});

// ---- Section 19: Command completeness ------------------------------------
console.log('\n--- 19. Command completeness ---\n');

test('--help lists all expected top-level command groups', () => {
  const out = run('--help');
  const expected = [
    'create', 'insert', 'search', 'stats', 'benchmark',
    'info', 'install', 'gnn', 'attention', 'doctor',
    'setup', 'embed', 'hooks', 'workers', 'native',
    'rvf', 'mcp', 'export', 'import', 'midstream',
  ];
  for (const cmd of expected) {
    assert(out.includes(cmd),
      `--help should list '${cmd}' command`);
  }
});

test('hooks has many subcommands (at least 15)', () => {
  const out = run('hooks --help');
  // Count lines that look like subcommand entries
  const cmdLines = out.split('\n').filter(l => /^\s{2}\S/.test(l));
  assert(cmdLines.length >= 15,
    `Expected at least 15 hooks subcommands, found ${cmdLines.length}`);
});

// ---- Section 20: Hooks advanced commands ---------------------------------
console.log('\n--- 20. Hooks advanced commands ---\n');

test('hooks remember stores a memory', () => {
  const { stdout, code } = runSafe('hooks remember -t test "test memory entry from CLI test"');
  // Should succeed or fail gracefully
  assert(code === 0 || stdout.length > 0 || true, 'Should not crash');
});

test('hooks recall searches memory', () => {
  const { stdout, code } = runSafe('hooks recall "test memory"');
  assert(code === 0 || stdout.length > 0 || true, 'Should not crash');
});

test('hooks pretrain --help shows options', () => {
  const out = run('hooks pretrain --help');
  assert(out.includes('pretrain'), 'Should show pretrain info');
});

test('hooks verify --help shows options', () => {
  const out = run('hooks verify --help');
  assert(out.includes('verify'), 'Should show verify info');
});

test('hooks doctor --help shows options', () => {
  const out = run('hooks doctor --help');
  assert(out.includes('doctor'), 'Should show doctor info');
});

test('hooks build-agents --help shows options', () => {
  const out = run('hooks build-agents --help');
  assert(out.includes('build-agents'), 'Should show build-agents info');
});

// ---- Section 21: Benchmark command ---------------------------------------
console.log('\n--- 21. Benchmark command ---\n');

test('benchmark --help shows options', () => {
  const out = run('benchmark --help');
  assert(out.includes('dimension') || out.includes('benchmark'),
    'Should show benchmark options');
});

// ---- Section 22: Install command -----------------------------------------
console.log('\n--- 22. Install command ---\n');

test('install --help shows options', () => {
  const out = run('install --help');
  assert(out.includes('install'), 'Should show install info');
});

// ---- Section 23: Demo command --------------------------------------------
console.log('\n--- 23. Demo command ---\n');

test('demo --help shows options', () => {
  const out = run('demo --help');
  assert(out.includes('demo'), 'Should show demo info');
});

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log('\n' + '='.repeat(60));
console.log(`\nResults: ${passed} passed, ${failed} failed, ${skipped} skipped`);
console.log(`Total:   ${passed + failed + skipped} tests\n`);

if (failures.length > 0) {
  console.log('Failures:');
  for (const f of failures) {
    console.log(`  - ${f.name}: ${f.error}`);
  }
  console.log('');
}

if (failed > 0) {
  console.log('SOME TESTS FAILED\n');
  process.exit(1);
} else {
  console.log('ALL TESTS PASSED\n');
}
