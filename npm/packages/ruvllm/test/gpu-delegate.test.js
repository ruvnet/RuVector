/**
 * `ruvllm gpu ...`, `ruvllm-gpu-loop` and `ruvllm-adapters` delegate to
 * @ruvector/ruvllm-gpu-loop. Runs the real delegation when the package is
 * installed; otherwise checks the missing-package error.
 */
const { test, describe } = require('node:test');
const assert = require('node:assert');
const { spawnSync } = require('node:child_process');
const path = require('node:path');

const bin = (name) => path.join(__dirname, '..', 'bin', name);
const run = (file, args) => spawnSync(process.execPath, [file, ...args], { encoding: 'utf8', timeout: 30000 });
let installed = true;
try { require.resolve('@ruvector/ruvllm-gpu-loop/package.json'); } catch { installed = false; }

describe('GPU-loop CLI delegation', () => {
  const cases = [
    ['ruvllm gpu --help', bin('cli.js'), ['gpu', '--help'], /run --config/],
    ['ruvllm gpu loop', bin('cli.js'), ['gpu', 'loop'], /tick\|status\|run/],
    ['ruvllm gpu adapters', bin('cli.js'), ['gpu', 'adapters'], /export\|training-plan\|select/],
    ['ruvllm-gpu-loop', bin('ruvllm-gpu-loop.js'), [], /tick\|status\|run/],
    ['ruvllm-adapters', bin('ruvllm-adapters.js'), [], /export\|training-plan\|select/],
  ];
  for (const [label, file, args, expected] of cases) {
    test(label, () => {
      const r = run(file, args);
      if (!installed) {
        assert.strictEqual(r.status, 1);
        assert.match(r.stderr, /npm install @ruvector\/ruvllm-gpu-loop/);
        return;
      }
      assert.match(r.stdout + r.stderr, expected);
    });
  }

  test('ruvllm gpu does not load the runtime first', () => {
    const r = spawnSync(process.execPath, [bin('cli.js'), 'gpu', '--help'], {
      encoding: 'utf8', env: { ...process.env, NODE_OPTIONS: '--require=' + path.join(__dirname, 'fixtures', 'forbid-dist.js') },
    });
    assert.doesNotMatch(r.stderr, /dist\/cjs was loaded/);
  });
});
