/**
 * Claude Code hook integration.
 *
 * #1222: `hooks init` must merge with hooks other tools already wrote, feed its
 *        commands from the JSON Claude Code sends on stdin, write timeouts in
 *        seconds, and leave other MCP servers and npm packages alone.
 * #1223: hooks run in parallel must not lose each other's store writes,
 *        compressedPatterns must survive a save, and the session pair must run
 *        once per session (SessionEnd), not after every reply (Stop).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PKG = path.join(__dirname, '..');
const CLI = path.join(PKG, 'bin', 'cli.js');
const ENV = { ...process.env, NO_COLOR: '1', FORCE_COLOR: '0' };
const isWindows = process.platform === 'win32';

function project(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rv-hooks-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  fs.mkdirSync(path.join(dir, '.claude'));
  return dir;
}

const INIT_FLAGS = ['hooks', 'init', '--no-claude-md', '--no-gitignore', '--no-statusline'];
const init = (cwd, extra = []) =>
  execFileSync(process.execPath, [CLI, ...INIT_FLAGS, ...extra], { cwd, env: ENV, encoding: 'utf8' });
const readSettings = (dir) => JSON.parse(fs.readFileSync(path.join(dir, '.claude', 'settings.json'), 'utf8'));
const allCommands = (settings) => Object.values(settings.hooks || {})
  .flatMap((groups) => groups.flatMap((g) => (g.hooks || []).map((h) => h.command)));

const FOREIGN = {
  enabledMcpjsonServers: ['ruflo'],
  hooks: {
    PreToolUse: [
      { matcher: 'Edit|Write', hooks: [{ type: 'command', command: 'node .claude/helpers/guard-master.js', timeout: 5 }] },
      { matcher: 'Bash', hooks: [{ type: 'command', command: 'npx ruflo hooks pre-command', timeout: 10 }] },
    ],
    SessionStart: [{ hooks: [{ type: 'command', command: 'npx ruflo hooks session-restore', timeout: 15 }] }],
    Stop: [{ hooks: [{ type: 'command', command: 'npx ruflo hooks session-end', timeout: 10 }] }],
    SubagentStop: [{ hooks: [{ type: 'command', command: 'npx ruflo hooks subagent-stop' }] }],
  },
};

test('hooks init keeps foreign hooks and is idempotent', (t) => {
  const dir = project(t);
  fs.writeFileSync(path.join(dir, '.claude', 'settings.json'), JSON.stringify(FOREIGN, null, 2));
  init(dir);
  const once = readSettings(dir);
  // Every foreign hook survives verbatim, in its original event and matcher.
  for (const [event, groups] of Object.entries(FOREIGN.hooks)) {
    for (const g of groups) {
      const kept = once.hooks[event].find((x) => x.matcher === g.matcher &&
        JSON.stringify(x.hooks) === JSON.stringify(g.hooks));
      assert.ok(kept, `foreign ${event} hook ${g.hooks[0].command} was dropped`);
    }
  }
  assert.ok(allCommands(once).some((c) => /ruvector\S* hooks pre-edit/.test(c)), 'ruvector hooks were not added');
  init(dir);
  const twice = readSettings(dir);
  assert.deepEqual(twice.hooks, once.hooks, 'a second init must not duplicate or reorder hooks');
});

test('hooks init --force still replaces the settings file', (t) => {
  const dir = project(t);
  fs.writeFileSync(path.join(dir, '.claude', 'settings.json'), JSON.stringify(FOREIGN, null, 2));
  init(dir, ['--force']);
  assert.ok(!allCommands(readSettings(dir)).some((c) => c.includes('ruflo')));
});

test('hooks init writes Claude Code compatible commands and timeouts', (t) => {
  const dir = project(t);
  fs.writeFileSync(path.join(dir, '.claude', 'settings.json'), JSON.stringify(FOREIGN, null, 2));
  init(dir);
  const s = readSettings(dir);
  const ours = Object.values(s.hooks).flatMap((gs) => gs.flatMap((g) => g.hooks))
    .filter((h) => /ruvector/.test(h.command));
  assert.ok(ours.length >= 8);
  for (const h of ours) {
    // Claude Code passes hook input as JSON on stdin; these variables are never set.
    assert.doesNotMatch(h.command, /\$TOOL_INPUT|\$CLAUDE_USER_PROMPT/, h.command);
    assert.doesNotMatch(h.command, /agentic-flow/, h.command);
    // `timeout` is in seconds; it must outlast the 10 s store-lock wait.
    assert.ok(Number.isInteger(h.timeout) && h.timeout > 10 && h.timeout <= 120, `timeout ${h.timeout}: ${h.command}`);
  }
  // Session pair runs at SessionEnd, not after every reply.
  const stopCmds = (s.hooks.Stop || []).flatMap((g) => g.hooks.map((h) => h.command));
  assert.ok(!stopCmds.some((c) => /ruvector/.test(c)), `ruvector hooks in Stop: ${stopCmds}`);
  const endCmds = s.hooks.SessionEnd.flatMap((g) => g.hooks.map((h) => h.command));
  assert.ok(endCmds.some((c) => /hooks session-end/.test(c)));
  assert.ok(endCmds.some((c) => /hooks trajectory-end/.test(c)));
  // Does not enable another product's MCP server.
  assert.deepEqual(s.enabledMcpjsonServers, ['ruflo']);
  assert.equal(s.mcpServers, undefined);
});

test('hooks init upgrades an old ruvector config in place', (t) => {
  const dir = project(t);
  const old = {
    hooks: {
      Stop: [{ hooks: [
        { type: 'command', timeout: 2000, command: 'npx ruvector@latest hooks trajectory-end --success --quality 0.8 2>/dev/null || true' },
        { type: 'command', timeout: 2000, command: 'npx ruvector@latest hooks session-end 2>/dev/null || true' },
      ] }, ...FOREIGN.hooks.Stop],
      UserPromptSubmit: [{ hooks: [
        { type: 'command', timeout: 2000, command: 'npx agentic-flow@3.0.0-alpha.2 workers dispatch-prompt "$CLAUDE_USER_PROMPT" 2>/dev/null || true' },
      ] }],
    },
  };
  fs.writeFileSync(path.join(dir, '.claude', 'settings.json'), JSON.stringify(old, null, 2));
  init(dir);
  const s = readSettings(dir);
  assert.deepEqual(s.hooks.Stop, FOREIGN.hooks.Stop, 'old ruvector Stop hooks removed, foreign Stop hook kept');
  assert.ok(!allCommands(s).some((c) => /agentic-flow|\$CLAUDE_USER_PROMPT|\$TOOL_INPUT/.test(c)));
});

// Run a generated hook command the way Claude Code does: through a shell, with
// the event JSON on stdin and CLAUDE_PROJECT_DIR set. The npx prefix is swapped
// for this checkout's CLI so the test exercises the code under test.
function runHook(dir, command, payload, cwd = dir) {
  const local = command.replace(/npx (-y )?ruvector@\S+/, `"${process.execPath}" "${CLI}"`);
  assert.notEqual(local, command, `unexpected hook command shape: ${command}`);
  const r = spawnSync('bash', ['-c', local], {
    cwd, input: JSON.stringify(payload), encoding: 'utf8', env: { ...ENV, CLAUDE_PROJECT_DIR: dir },
  });
  assert.equal(r.status, 0, r.stderr);
  return r;
}

test('generated hooks read tool input from stdin', { skip: isWindows && 'needs bash' }, (t) => {
  const dir = project(t);
  init(dir);
  const s = readSettings(dir);
  const find = (event, matcher, re) => s.hooks[event].find((g) => g.matcher === matcher)
    .hooks.find((h) => re.test(h.command)).command;
  const sub = path.join(dir, 'src', 'deep');
  fs.mkdirSync(sub, { recursive: true });

  runHook(dir, find('PreToolUse', 'Read', /remember/), { hook_event_name: 'PreToolUse', tool_name: 'Read', tool_input: { file_path: '/x/read-me.ts' } }, sub);
  runHook(dir, find('PreToolUse', 'Bash', /pre-command/), { tool_name: 'Bash', tool_input: { command: 'cargo test -p demo' } });
  const pre = runHook(dir, find('PreToolUse', 'Edit|Write|MultiEdit', /pre-edit/), { tool_name: 'Edit', tool_input: { file_path: '/x/y/edited.rs' } });
  assert.match(pre.stdout, /edited\.rs/);
  runHook(dir, find('PostToolUse', 'Edit|Write|MultiEdit', /post-edit/), { tool_name: 'Edit', tool_input: { file_path: '/x/y/edited.rs' } });

  // Written to the project store even though the Read hook ran from a subdirectory.
  const store = JSON.parse(fs.readFileSync(path.join(dir, '.ruvector', 'intelligence.json'), 'utf8'));
  const contents = store.memories.map((m) => m.content);
  assert.ok(contents.includes('Reading: /x/read-me.ts'), JSON.stringify(contents));
  assert.ok(Object.keys(store.patterns).some((k) => k.startsWith('edit_rs_in_')), JSON.stringify(Object.keys(store.patterns)));
  assert.ok(!fs.existsSync(path.join(sub, '.ruvector')), 'hook wrote a store in the subdirectory');
});

const runAsync = (cwd, args) => new Promise((resolve) => {
  const p = spawn(process.execPath, [CLI, ...args], { cwd, env: ENV, stdio: ['ignore', 'ignore', 'pipe'] });
  let err = '';
  p.stderr.on('data', (d) => { err += d; });
  p.on('close', (code) => resolve({ code, err }));
});

test('parallel hook processes do not lose each other\'s writes', async (t) => {
  const dir = project(t);
  const N = 10;
  const results = await Promise.all(Array.from({ length: N }, (_, i) =>
    runAsync(dir, ['hooks', 'remember', `parallel-mem-${i}`, '-t', 'test', '--silent'])));
  results.forEach((r) => assert.equal(r.code, 0, r.err));
  const store = JSON.parse(fs.readFileSync(path.join(dir, '.ruvector', 'intelligence.json'), 'utf8'));
  const kept = store.memories.filter((m) => /^parallel-mem-\d+$/.test(m.content)).map((m) => m.content).sort();
  assert.equal(kept.length, N, `lost writes: kept ${kept}`);

  // The SessionStart pair runs in parallel; trajectory-end must still find the trajectory.
  await Promise.all([runAsync(dir, ['hooks', 'session-start']), runAsync(dir, ['hooks', 'trajectory-begin', '-c', 'ctx', '-a', 'coder'])]);
  const after = JSON.parse(fs.readFileSync(path.join(dir, '.ruvector', 'intelligence.json'), 'utf8'));
  assert.equal(Object.keys(after.activeTrajectories).length, 1);
  assert.equal(after.stats.session_count, 1);
});

test('compressedPatterns survive later saves and compress stores the Q-table', (t) => {
  const dir = project(t);
  execFileSync(process.execPath, [CLI, 'hooks', 'post-edit', 'src/a.ts'], { cwd: dir, env: ENV });
  execFileSync(process.execPath, [CLI, 'hooks', 'post-command', 'npm', 'test'], { cwd: dir, env: ENV });
  const out = JSON.parse(execFileSync(process.execPath, [CLI, 'hooks', 'compress'], { cwd: dir, env: ENV, encoding: 'utf8' }).trim().split('\n').pop());
  assert.equal(out.success, true, JSON.stringify(out));
  const storePath = path.join(dir, '.ruvector', 'intelligence.json');
  const compressed = JSON.parse(fs.readFileSync(storePath, 'utf8')).compressedPatterns;
  assert.ok(compressed && Object.keys(compressed.tensors).length > 0, `compress wrote ${JSON.stringify(compressed)}`);

  // A later hook (load → save through the Intelligence class) keeps them.
  execFileSync(process.execPath, [CLI, 'hooks', 'remember', 'after-compress', '-t', 'test', '--silent'], { cwd: dir, env: ENV });
  assert.deepEqual(JSON.parse(fs.readFileSync(storePath, 'utf8')).compressedPatterns, compressed);

  // session-end auto-compress now has something to work on.
  execFileSync(process.execPath, [CLI, 'hooks', 'session-end'], { cwd: dir, env: { ...ENV, RUVECTOR_AUTO_COMPRESS: 'true' } });
  const end = JSON.parse(fs.readFileSync(storePath, 'utf8')).compressedPatterns;
  assert.deepEqual(Object.keys(end.tensors).sort(), Object.keys(compressed.tensors).sort());
});
