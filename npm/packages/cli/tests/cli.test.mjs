// Regression tests for @ruvector/cli. Run with `npm test` (builds first).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const PKG_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CLI = path.join(PKG_DIR, 'dist', 'cli.js');
const PKG = JSON.parse(fs.readFileSync(path.join(PKG_DIR, 'package.json'), 'utf-8'));

function sandbox() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-cli-test-'));
  const home = path.join(root, 'home');
  const cwd = path.join(root, 'proj');
  fs.mkdirSync(home);
  fs.mkdirSync(cwd);
  const run = (...args) => execFileSync(process.execPath, [CLI, ...args], {
    cwd,
    env: { ...process.env, HOME: home, USERPROFILE: home },
    encoding: 'utf-8'
  });
  return { root, home, cwd, run };
}

test('--version reports the package version', () => {
  const { run } = sandbox();
  assert.equal(run('--version').trim(), PKG.version);
});

test('remember returns a distinct id for each memory', () => {
  const { run } = sandbox();
  const a = JSON.parse(run('hooks', 'remember', '-t', 't', 'one')).id;
  const b = JSON.parse(run('hooks', 'remember', '-t', 't', 'two')).id;
  assert.notEqual(a, b);
});

test('learn returns a distinct id for each trajectory', () => {
  const { run } = sandbox();
  const a = JSON.parse(run('hooks', 'learn', 's', 'a1')).id;
  const b = JSON.parse(run('hooks', 'learn', 's', 'a2')).id;
  assert.notEqual(a, b);
});

for (const cmd of ['install', 'init']) {
  test(`hooks ${cmd} keeps existing hooks and settings`, () => {
    const { cwd, run } = sandbox();
    const claudeDir = path.join(cwd, '.claude');
    fs.mkdirSync(claudeDir);
    const settingsPath = path.join(claudeDir, 'settings.json');
    const guard = { matcher: 'Bash', hooks: [{ type: 'command', command: 'my-guard.sh' }] };
    fs.writeFileSync(settingsPath, JSON.stringify({ model: 'x', hooks: { PreToolUse: [guard] } }));

    run('hooks', cmd, ...(cmd === 'init' ? ['--force'] : []));
    run('hooks', cmd, ...(cmd === 'init' ? ['--force'] : []));

    const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));
    assert.equal(settings.model, 'x');
    assert.deepEqual(settings.hooks.PreToolUse[0], guard, 'user guard hook was removed');
    // Our entries are added once, not duplicated by a second install.
    const commands = settings.hooks.PreToolUse.flatMap(m => m.hooks.map(h => h.command));
    assert.equal(commands.filter(c => c.includes('hooks pre-command')).length, 1);
    // Every hook uses the Claude Code object schema, never a bare string.
    for (const matchers of Object.values(settings.hooks)) {
      for (const m of matchers) {
        assert.ok(Array.isArray(m.hooks), 'matcher group must have a hooks array');
        for (const h of m.hooks) {
          assert.equal(typeof h, 'object');
          assert.equal(h.type, 'command');
          assert.equal(typeof h.command, 'string');
        }
      }
    }
  });
}

test('npm pack ships the built dist (bin target exists)', () => {
  const dest = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-cli-pack-'));
  fs.rmSync(path.join(PKG_DIR, 'dist'), { recursive: true, force: true });
  const out = execFileSync('npm', ['pack', '--json', '--pack-destination', dest], {
    cwd: PKG_DIR,
    encoding: 'utf-8',
    stdio: ['ignore', 'pipe', 'ignore']
  });
  const jsonStart = out.indexOf('[');
  const files = JSON.parse(out.slice(jsonStart))[0].files.map(f => f.path);
  assert.ok(files.includes(PKG.bin.ruvector), `tarball is missing ${PKG.bin.ruvector}`);
});
