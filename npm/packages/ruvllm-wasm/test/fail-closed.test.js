// loadModel/generate/chat must fail closed instead of returning placeholder output.
const test = require('node:test');
const assert = require('node:assert/strict');

test('inference methods throw RuvLLMWasmNotImplementedError', async () => {
  const mod = await import('../dist/index.js');
  assert.equal(mod.INFERENCE_AVAILABLE, false);
  const llm = await mod.RuvLLMWasm.create();
  for (const [name, call] of [
    ['loadModel', () => llm.loadModel('https://example.com/model.gguf')],
    ['generate', () => llm.generate('hello')],
    ['chat', () => llm.chat([{ role: 'user', content: 'hello' }])],
  ]) {
    await assert.rejects(call, (e) =>
      e instanceof mod.RuvLLMWasmNotImplementedError &&
      e.code === 'RUVLLM_WASM_NOT_IMPLEMENTED' &&
      e.message.includes(name));
  }
  assert.equal(llm.getStatus(), 'error');
});

test('the publish guard refuses while inference is unavailable', () => {
  const { spawnSync } = require('node:child_process');
  const path = require('node:path');
  const env = { ...process.env };
  delete env.RUVLLM_WASM_ALLOW_FACADE_PUBLISH;
  const r = spawnSync(process.execPath, [path.join(__dirname, '..', 'scripts', 'guard-publish.mjs')], { env, encoding: 'utf8' });
  assert.equal(r.status, 1);
  assert.match(r.stderr, /refusing to publish/);
});
