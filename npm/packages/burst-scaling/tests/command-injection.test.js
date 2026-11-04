'use strict';
// Regression: claude-flow notify hooks must not go through a shell.
// Run with: npm test
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const DIST = path.join(__dirname, '..', 'dist');

function withFakeNpx(fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'burst-scaling-test-'));
  const log = path.join(dir, 'argv.log');
  const npx = path.join(dir, 'npx');
  // Fake npx records its argv (one per line) instead of fetching claude-flow.
  fs.writeFileSync(npx, `#!/bin/sh\nfor a in "$@"; do printf '%s\\n' "$a" >> "${log}"; done\n`, { mode: 0o755 });
  const oldPath = process.env.PATH;
  process.env.PATH = `${dir}${path.delimiter}${oldPath}`;
  return Promise.resolve(fn(dir, log)).finally(() => {
    process.env.PATH = oldPath;
    fs.rmSync(dir, { recursive: true, force: true });
  });
}

test('ReactiveScaler default notify hook does not evaluate shell syntax in region', { skip: process.platform === 'win32' }, () =>
  withFakeNpx(async (dir, log) => {
    const { ReactiveScaler } = require(path.join(DIST, 'reactive-scaler.js'));
    const marker = path.join(dir, 'pwned');
    const region = `us-central1$(touch ${marker})\`touch ${marker}\``;
    const scaler = new ReactiveScaler([region]);
    const action = await scaler.processMetrics({
      region,
      timestamp: new Date(),
      cpuUtilization: 0.99,
      memoryUtilization: 0.95,
      activeConnections: 45_000_000,
      requestRate: 150_000,
      errorRate: 0.005,
      p99Latency: 45,
      currentInstances: 50,
    });
    assert.notStrictEqual(action.action, 'none', 'test needs a scaling action to trigger the hook');
    assert.strictEqual(fs.existsSync(marker), false, 'shell command substitution was executed');
    const argv = fs.readFileSync(log, 'utf8').split('\n');
    assert.deepStrictEqual(argv.slice(0, 4), ['claude-flow@alpha', 'hooks', 'notify', '--message']);
    assert.ok(argv[4].includes(region), 'message must be passed through verbatim');
  }));

test('no module invokes child_process.exec (shell)', () => {
  for (const f of ['index.js', 'burst-predictor.js', 'reactive-scaler.js', 'capacity-manager.js']) {
    const src = fs.readFileSync(path.join(DIST, f), 'utf8');
    assert.doesNotMatch(src, /child_process_1\.exec\b(?!File)/, `${f} uses exec`);
    assert.doesNotMatch(src, /\bexecAsync\s*\(/, `${f} uses execAsync`);
  }
});
