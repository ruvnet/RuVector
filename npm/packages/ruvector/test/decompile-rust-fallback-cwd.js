#!/usr/bin/env node
/**
 * The decompiler's Rust fallback (cargo / target/release/examples/run_on_cli
 * under the monorepo root) is only used from a source checkout. When the
 * package is installed under node_modules, a run_on_cli in the installing
 * project's target/ directory must not be executed.
 */

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const PKG_ROOT = path.join(__dirname, '..');

if (process.platform === 'win32') {
  console.log('decompile-rust-fallback-cwd: skipped on Windows');
  process.exit(0);
}

const proj = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-decomp-cwd-'));
try {
  const pkg = path.join(proj, 'node_modules', 'ruvector');
  fs.cpSync(path.join(PKG_ROOT, 'src', 'decompiler'), path.join(pkg, 'src', 'decompiler'), { recursive: true });
  fs.cpSync(path.join(PKG_ROOT, 'wasm'), path.join(pkg, 'wasm'), { recursive: true });
  // chalk / js-beautify resolve from the real package's node_modules.
  fs.symlinkSync(path.join(PKG_ROOT, 'node_modules'), path.join(pkg, 'node_modules'));

  const marker = path.join(proj, 'PLANTED_BINARY_RAN');
  const binDir = path.join(proj, 'target', 'release', 'examples');
  fs.mkdirSync(binDir, { recursive: true });
  fs.writeFileSync(path.join(binDir, 'run_on_cli'), `#!/bin/sh\necho ran > "${marker}"\n`, { mode: 0o755 });

  const input = path.join(proj, 'data.js');
  fs.writeFileSync(input, '/*' + 'x'.repeat(120000) + '*/\n');

  const script = `
    const d = require(${JSON.stringify(path.join(pkg, 'src', 'decompiler', 'index.js'))});
    try { d.decompileFile(${JSON.stringify(input)}, { witness: false }); } catch (e) {}
  `;
  // PATH without cargo so only the planted binary could run.
  const r = spawnSync(process.execPath, ['-e', script], {
    cwd: proj,
    env: { ...process.env, PATH: '/usr/bin:/bin' },
    timeout: 120000,
    encoding: 'utf8',
  });
  assert.strictEqual(r.status, 0, `decompile child failed: ${r.stderr}`);
  assert.ok(!fs.existsSync(marker), 'Rust fallback ran from an installed package');
  console.log('decompile-rust-fallback-cwd: OK');
} catch (e) {
  console.error('decompile-rust-fallback-cwd: FAIL');
  console.error(e && e.stack || e);
  process.exitCode = 1;
} finally {
  fs.rmSync(proj, { recursive: true, force: true });
}
