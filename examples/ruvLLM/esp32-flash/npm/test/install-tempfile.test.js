'use strict';
// Regression: `ruvllm-esp32 install` must not download espup to a fixed,
// shared temp path and then execute it. Another local user could pre-create
// that path and have their own binary executed.
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const childProcess = require('child_process');

test('install downloads espup into a private temp dir, not a shared fixed path', async (t) => {
  if (process.platform === 'win32') return t.skip('POSIX install path');

  const calls = [];
  const record = (kind) => (cmd, argv) => {
    calls.push({ kind, cmd, argv: Array.isArray(argv) ? argv : [] });
    // Simulate curl writing the download so the follow-up chmod succeeds.
    if (kind === 'execFileSync' && cmd === 'curl') fs.writeFileSync(argv[argv.indexOf('-o') + 1], '');
    return '';
  };
  const orig = { execSync: childProcess.execSync, execFileSync: childProcess.execFileSync };
  childProcess.execSync = record('execSync');
  childProcess.execFileSync = record('execFileSync');
  const origLog = console.log;
  console.log = () => {};
  try {
    delete require.cache[require.resolve('../bin/cli.js')];
    const cli = require('../bin/cli.js');
    assert.strictEqual(typeof cli.installToolchain, 'function', 'cli.js must export installToolchain');
    const ok = await cli.installToolchain();
    assert.strictEqual(ok, true);
  } finally {
    Object.assign(childProcess, orig);
    console.log = origLog;
  }

  const shared = [path.join(os.tmpdir(), 'espup'), '/tmp/espup'];
  for (const c of calls) {
    const text = [c.cmd, ...c.argv].join(' ');
    for (const p of shared) {
      assert.ok(!new RegExp(`${p.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&')}(\\s|$)`).test(text),
        `command uses shared temp path ${p}: ${text}`);
    }
    if (text.includes('espup')) {
      assert.notStrictEqual(c.kind, 'execSync', `espup download/run must not go through a shell: ${text}`);
    }
  }

  const run = calls.find(c => c.argv[0] === 'install' && path.basename(c.cmd) === 'espup');
  assert.ok(run, 'espup install was executed');
  const dir = path.dirname(run.cmd);
  assert.ok(path.basename(dir).startsWith('ruvllm-espup-'), `espup lives in a fresh mkdtemp dir: ${dir}`);
  fs.rmSync(dir, { recursive: true, force: true });
});
