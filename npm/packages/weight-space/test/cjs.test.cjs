// CommonJS consumers: require() must load the CJS build, and ruv-fann (ESM-only) must still load.
const { test } = require('node:test');
const assert = require('node:assert/strict');

test('require() works and reaches the ESM-only ruv-fann runtime', async () => {
  const ws = require('../dist/cjs/index.js');
  assert.equal(typeof ws.WeightRegistry, 'function');
  assert.equal(ws.VERSION, require('../package.json').version);
  const rt = await ws.loadFann();
  const model = ws.createMlp([2, 2, 1], { seed: 1 });
  const prepared = rt.prepare(model);
  assert.equal(prepared.predict(new Float64Array([0.1, 0.2])).length, 1);
  prepared.dispose();
  const reg = await ws.WeightRegistry.open({ dimensions: 2, store: 'flat' });
  const rec = await reg.save(model, [1, 0]);
  assert.equal((await reg.nearest([1, 0], 1))[0].id, rec.id);
});
