'use strict';

// Regression test: the published tarball must install without a compiler.
// binding.gyp is not shipped (see "files"), so npm must not inject an
// implicit `node-gyp rebuild` install script at publish time.

const test = require('node:test');
const assert = require('node:assert');
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const pkgDir = path.resolve(__dirname, '..');
const pkg = require('../package.json');

test('package.json disables implicit node-gyp install', () => {
  assert.strictEqual(pkg.gypfile, false);
  assert.strictEqual(pkg.scripts.install, undefined);
});

test('packed tarball installs and loads with the JS fallback', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'snn-pack-'));
  try {
    const out = execFileSync('npm', ['pack', '--json', '--pack-destination', tmp], {
      cwd: pkgDir,
      encoding: 'utf8',
    });
    const tgz = path.join(tmp, JSON.parse(out)[0].filename);
    const app = path.join(tmp, 'app');
    fs.mkdirSync(app);
    fs.writeFileSync(path.join(app, 'package.json'), '{"name":"t","private":true}');
    execFileSync('npm', ['install', '--no-workspaces', '--no-audit', '--no-fund', tgz], {
      cwd: app,
      stdio: 'pipe',
    });
    const loaded = execFileSync(
      process.execPath,
      ['-e', "const m=require('@ruvector/spiking-neural');console.log(m.version)"],
      { cwd: app, encoding: 'utf8' }
    ).trim();
    assert.strictEqual(loaded, pkg.version);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
