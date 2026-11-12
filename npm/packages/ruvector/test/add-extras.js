#!/usr/bin/env node
/**
 * `ruvector add <extra>` — opt-in installer for optional @ruvector packages.
 *
 * Unit tests drive bin/extras.js with a mock spawn (no network, no installs);
 * CLI tests run bin/cli.js with --dry-run / --list in scratch directories.
 * Security contract: only allowlisted extras are accepted, and the installer
 * is spawned with an argv array and no shell.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync, spawnSync } = require('child_process');

const extras = require('../bin/extras');
const CLI = path.join(__dirname, '..', 'bin', 'cli.js');

let passed = 0;
let failed = 0;
function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL  ${name}\n        ${e.message}`);
  }
}

function tmpDir(files = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-add-'));
  for (const [name, content] of Object.entries(files)) {
    fs.writeFileSync(path.join(dir, name), content);
  }
  return dir;
}

function mockSpawn(status = 0) {
  const calls = [];
  const fn = (cmd, args, opts) => {
    calls.push({ cmd, args, opts });
    return { status, error: undefined };
  };
  fn.calls = calls;
  return fn;
}

const notInstalled = () => false;

console.log('\nruvector add — extras allowlist');

test('allowlist maps extra names to exact package names', () => {
  const expected = {
    rabitq: '@ruvector/rabitq-wasm',
    diskann: '@ruvector/diskann',
    router: '@ruvector/router',
    graph: '@ruvector/graph-node',
    rvlite: 'rvlite',
    rvf: '@ruvector/rvf',
    'tiny-dancer': '@ruvector/tiny-dancer',
    sona: '@ruvector/sona',
  };
  for (const [key, pkg] of Object.entries(expected)) {
    const extra = extras.resolveExtra(key);
    assert.ok(extra, `extra ${key} should exist`);
    assert.strictEqual(extra.pkg, pkg);
    assert.ok(extra.spec.startsWith(`${pkg}@`), `${key} spec must pin ${pkg}@<range>`);
    assert.ok(extra.why && extra.why.length > 10, `${key} needs a reason`);
    assert.ok(extra.usage && extra.usage.length > 0, `${key} needs a usage hint`);
  }
});

test('every spec is a plain package@range (no shell metacharacters)', () => {
  for (const extra of extras.listExtras()) {
    assert.match(extra.spec, /^(@ruvector\/)?[a-z0-9-]+@[\^~>=<0-9.\s]+$/, extra.spec);
  }
});

test('package names in the allowlist are accepted as aliases', () => {
  assert.strictEqual(extras.resolveExtra('@ruvector/rabitq-wasm').key, 'rabitq');
  assert.strictEqual(extras.resolveExtra('RaBitQ').key, 'rabitq');
});

test('unknown or hostile names are rejected', () => {
  for (const bad of ['lodash', 'rabitq;touch pwned', '$(id)', '../rabitq', '__proto__',
    'constructor', 'toString', '', '@ruvector/rabitq-wasm@latest', 'rabitq --registry=http://evil']) {
    assert.strictEqual(extras.resolveExtra(bad), null, `should reject ${JSON.stringify(bad)}`);
  }
});

console.log('\nruvector add — package manager detection');

test('detects pnpm / yarn / bun / npm from lockfiles', () => {
  assert.strictEqual(extras.detectPackageManager(tmpDir({ 'pnpm-lock.yaml': '' })), 'pnpm');
  assert.strictEqual(extras.detectPackageManager(tmpDir({ 'yarn.lock': '' })), 'yarn');
  assert.strictEqual(extras.detectPackageManager(tmpDir({ 'bun.lockb': '' })), 'bun');
  assert.strictEqual(extras.detectPackageManager(tmpDir({ 'bun.lock': '' })), 'bun');
  assert.strictEqual(extras.detectPackageManager(tmpDir({ 'package-lock.json': '{}' })), 'npm');
  assert.strictEqual(extras.detectPackageManager(tmpDir()), 'npm');
});

test('falls back to package.json packageManager when there is no lockfile', () => {
  const dir = tmpDir({ 'package.json': JSON.stringify({ packageManager: 'pnpm@9.1.0' }) });
  assert.strictEqual(extras.detectPackageManager(dir), 'pnpm');
  const odd = tmpDir({ 'package.json': JSON.stringify({ packageManager: 'evil; rm -rf /' }) });
  assert.strictEqual(extras.detectPackageManager(odd), 'npm');
});

test('builds an argv array per package manager', () => {
  const spec = ['@ruvector/rabitq-wasm@^0.1.1'];
  assert.deepStrictEqual(extras.buildInstallCommand('npm', spec, 'linux'), { cmd: 'npm', args: ['install', ...spec] });
  assert.deepStrictEqual(extras.buildInstallCommand('pnpm', spec, 'linux'), { cmd: 'pnpm', args: ['add', ...spec] });
  assert.deepStrictEqual(extras.buildInstallCommand('yarn', spec, 'linux'), { cmd: 'yarn', args: ['add', ...spec] });
  assert.deepStrictEqual(extras.buildInstallCommand('bun', spec, 'linux'), { cmd: 'bun', args: ['add', ...spec] });
  assert.strictEqual(extras.buildInstallCommand('npm', spec, 'win32').cmd, 'npm.cmd');
  assert.throws(() => extras.buildInstallCommand('sh', spec, 'linux'), /package manager/i);
});

console.log('\nruvector add — runAdd (mock spawn)');

test('dry run plans the install but never spawns', () => {
  const spawn = mockSpawn();
  const cwd = tmpDir({ 'yarn.lock': '' });
  const res = extras.runAdd(['rabitq'], { cwd, dryRun: true, spawn, isInstalled: notInstalled });
  assert.strictEqual(spawn.calls.length, 0);
  assert.strictEqual(res.ok, true);
  assert.strictEqual(res.dryRun, true);
  assert.deepStrictEqual(res.command, { cmd: 'yarn', args: ['add', '@ruvector/rabitq-wasm@^0.1.1'] });
});

test('install spawns once with an argv array, no shell, in the project cwd', () => {
  const spawn = mockSpawn(0);
  const cwd = tmpDir({ 'pnpm-lock.yaml': '' });
  const res = extras.runAdd(['rabitq', 'diskann'], { cwd, spawn, isInstalled: notInstalled });
  assert.strictEqual(res.ok, true);
  assert.strictEqual(spawn.calls.length, 1);
  const { cmd, args, opts } = spawn.calls[0];
  assert.strictEqual(cmd, 'pnpm');
  assert.ok(Array.isArray(args));
  assert.deepStrictEqual(args, ['add', '@ruvector/rabitq-wasm@^0.1.1', '@ruvector/diskann@^0.1.2']);
  assert.ok(!opts.shell, 'must not use a shell');
  assert.strictEqual(opts.cwd, cwd);
});

test('unknown extra fails before anything is spawned', () => {
  const spawn = mockSpawn();
  const res = extras.runAdd(['rabitq', 'left-pad'], { cwd: tmpDir(), spawn, isInstalled: notInstalled });
  assert.strictEqual(res.ok, false);
  assert.deepStrictEqual(res.unknown, ['left-pad']);
  assert.strictEqual(spawn.calls.length, 0);
});

test('already-installed and bundled extras are skipped', () => {
  const spawn = mockSpawn();
  const res = extras.runAdd(['sona', 'diskann'], {
    cwd: tmpDir(), spawn, isInstalled: (pkg) => pkg === '@ruvector/diskann',
  });
  assert.strictEqual(res.ok, true);
  assert.strictEqual(spawn.calls.length, 0);
  assert.deepStrictEqual(res.skipped.map((s) => s.key).sort(), ['diskann', 'sona']);
});

test('a failing installer is reported as not ok', () => {
  const spawn = mockSpawn(1);
  const res = extras.runAdd(['router'], { cwd: tmpDir(), spawn, isInstalled: notInstalled });
  assert.strictEqual(res.ok, false);
  assert.strictEqual(res.status, 1);
});

test('extraHint names the add command and the package', () => {
  const hint = extras.extraHint('rabitq');
  assert.match(hint, /npx ruvector add rabitq/);
  assert.match(hint, /@ruvector\/rabitq-wasm/);
});

console.log('\nruvector add — CLI');

function cli(args, cwd) {
  return spawnSync(process.execPath, [CLI, ...args], { cwd, encoding: 'utf8', env: { ...process.env, FORCE_COLOR: '0' } });
}

test('add --list prints every extra with its package', () => {
  const r = cli(['add', '--list'], tmpDir());
  assert.strictEqual(r.status, 0, r.stderr);
  for (const extra of extras.listExtras()) {
    assert.ok(r.stdout.includes(extra.key), `missing ${extra.key}`);
    assert.ok(r.stdout.includes(extra.pkg), `missing ${extra.pkg}`);
  }
});

test('add --list --json is machine readable', () => {
  const r = cli(['add', '--list', '--json'], tmpDir());
  assert.strictEqual(r.status, 0, r.stderr);
  const parsed = JSON.parse(r.stdout);
  assert.ok(parsed.extras.some((e) => e.key === 'rabitq' && typeof e.installed === 'boolean'));
});

test('add <extra> --dry-run prints the exact command and installs nothing', () => {
  const dir = tmpDir({ 'pnpm-lock.yaml': '', 'package.json': '{"name":"x","version":"1.0.0"}' });
  // Use an extra that does not resolve here, with the CLI's own check. Which
  // extras are pre-installed depends on dev/optional deps and on workspace
  // hoisting (rvlite became a devDependency for the rvlite MCP tests), so a
  // hard-coded name would hit the "already installed" path instead.
  const extra = extras.listExtras().find((e) => !extras.isPackageInstalled(e.pkg, dir));
  assert.ok(extra, 'every extra is pre-installed; no dry-run install path to test');
  const r = cli(['add', extra.key, '--dry-run'], dir);
  assert.strictEqual(r.status, 0, r.stderr);
  assert.ok(r.stdout.includes(`pnpm add ${extra.spec}`), r.stdout);
  assert.ok(r.stdout.includes(extra.why), 'should explain why');
  assert.ok(!fs.existsSync(path.join(dir, 'node_modules')), 'dry run must not install');
});

test('add with a hostile name exits non-zero and executes nothing', () => {
  const dir = tmpDir();
  const r = cli(['add', 'rabitq;touch pwned', '--dry-run'], dir);
  assert.notStrictEqual(r.status, 0);
  assert.match(r.stderr + r.stdout, /Unknown extra/);
  assert.ok(!fs.existsSync(path.join(dir, 'pwned')));
});

test('add with no extra and no --list exits non-zero with usage', () => {
  const r = cli(['add'], tmpDir());
  assert.notStrictEqual(r.status, 0);
  assert.match(r.stderr + r.stdout, /ruvector add --list/);
});

test('--help lists the add command', () => {
  const out = execFileSync(process.execPath, [CLI, '--help'], { encoding: 'utf8' });
  assert.match(out, /add \[options\] \[extras\.\.\.\]/);
});

console.log(`\nResults: ${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
