#!/usr/bin/env node
/**
 * @ruvector/attention must be >= 2.2.3 (H1 fix): on 0.1.x a malformed input
 * panicked inside the NAPI binding and aborted the whole Node process
 * ("fatal runtime error: failed to initiate panic"). The call now throws a
 * catchable error. Run in a child so a regression shows up as a test failure
 * instead of killing the runner.
 */

const assert = require('assert');
const path = require('path');
const { spawnSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const { isAttentionAvailable } = require(path.join(ROOT, 'dist', 'core', 'attention-fallbacks.js'));
if (!isAttentionAvailable()) {
  console.log('  SKIP  @ruvector/attention not loadable on this platform');
  process.exit(0);
}

const child = spawnSync(process.execPath, ['-e', `
  const { MultiHeadAttention } = require(${JSON.stringify(path.join(ROOT, 'dist', 'core', 'attention-fallbacks.js'))});
  const mha = new MultiHeadAttention(16, 4);
  try {
    mha.computeRaw(new Float32Array(16), [new Float32Array(8)], [new Float32Array(16)]);
    console.log('RESULT returned');
  } catch (e) {
    console.log('RESULT threw ' + e.message);
  }
`], { encoding: 'utf8', timeout: 30_000 });

assert.strictEqual(child.status, 0, `process must survive a malformed input (status ${child.status}, signal ${child.signal}):\n${child.stderr}`);
assert.match(child.stdout, /RESULT threw .*length/, child.stdout + child.stderr);
const installed = require('@ruvector/attention/package.json').version;
console.log(`  PASS  @ruvector/attention ${installed}: malformed input throws instead of aborting Node`);
