/**
 * The ESM build must load the native binding exactly like the CJS build.
 * Before 4.1.0, dist/esm/native.js called require()/__dirname (undefined in
 * ESM), so `import('@ruvector/ruvllm')` silently ran the JS fallback.
 */
const { test } = require('node:test');
const assert = require('node:assert');
const { spawnSync } = require('node:child_process');

test('ESM and CJS entry points agree on native loading', () => {
  const script = `
    const { createRequire } = await import('node:module');
    const cjs = createRequire(import.meta.url)('@ruvector/ruvllm');
    const esm = await import('@ruvector/ruvllm');
    console.log(JSON.stringify({ cjs: new cjs.RuvLLM().isNativeLoaded(), esm: new esm.RuvLLM().isNativeLoaded(), version: esm.version() }));`;
  const r = spawnSync(process.execPath, ['--input-type=module', '-e', script], { cwd: __dirname, encoding: 'utf8' });
  assert.strictEqual(r.status, 0, r.stderr);
  const out = JSON.parse(r.stdout.trim().split('\n').at(-1));
  assert.strictEqual(out.esm, out.cjs, `esm native=${out.esm} but cjs native=${out.cjs}`);
});

test('dist/esm/native.js carries the CommonJS-globals preamble', () => {
  const r = spawnSync(process.execPath, [require.resolve('../scripts/fix-esm-extensions.mjs'), '--verify'], { encoding: 'utf8' });
  assert.strictEqual(r.status, 0, r.stderr);
});
