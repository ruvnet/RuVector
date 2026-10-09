/**
 * Text comes only from a loaded language model. Without one, generation
 * throws RUVLLM_NO_LANGUAGE_MODEL instead of returning text that is not model
 * output; a labelled placeholder is available only by explicit opt-in.
 *
 * Runs against whatever native binary loads (none, a binary without model
 * support, or one with it — set RUVLLM_NATIVE_PATH to pick one); the
 * expectations below hold for all three.
 */
const { test } = require('node:test');
const assert = require('node:assert');
const { spawnSync } = require('node:child_process');
const path = require('node:path');

const pkg = path.join(__dirname, '..');
const run = (body, env = {}) =>
  spawnSync(process.execPath, ['-e', `const { RuvLLM, PLACEHOLDER_TEXT } = require('./dist/cjs/index.js');\n${body}`], {
    cwd: pkg,
    encoding: 'utf8',
    env: { ...process.env, RUVLLM_ALLOW_MOCK: '', ...env },
  });
const count = (s, code) => (s.match(new RegExp(code, 'g')) || []).length;
// Prints the error code of each call, or "ok".
const codesOf = (calls) =>
  calls
    .map((c) => `try { ${c}; console.log('ok'); } catch (e) { console.log(e.code || e.message.split(':')[0]); }`)
    .join('\n');

test('generation without a loaded model throws RUVLLM_NO_LANGUAGE_MODEL', () => {
  const r = run(
    'const l = new RuvLLM();\n' +
      codesOf([
        "l.generate('hi', { maxTokens: 4 })",
        "l.query('hi')",
        "l.batchQuery({ queries: ['a', 'b'] })",
        "l.generateDetailed('hi')",
        "l.chat([{ role: 'user', content: 'hi' }])",
      ]),
  );
  assert.strictEqual(r.status, 0, r.stderr);
  assert.deepStrictEqual(r.stdout.trim().split('\n'), Array(5).fill('RUVLLM_NO_LANGUAGE_MODEL'));
});

test('the error says how to get a model for the binary that is installed', () => {
  const r = run(
    "const l = new RuvLLM(); try { l.generate('hi'); } catch (e) { console.log(JSON.stringify({ m: e.message, native: l.isNativeLoaded(), loads: l.supportsModelLoading() })); }",
  );
  const { m, native, loads } = JSON.parse(r.stdout);
  if (!native) assert.match(m, /no native RuvLLM binary/);
  else if (!loads) assert.match(m, /cannot load models/);
  else assert.match(m, /call loadModel\(/);
});

test('allowPlaceholder returns a labelled placeholder and warns once', () => {
  const r = run(
    'const l = new RuvLLM({ allowPlaceholder: true });\n' +
      "const g = l.generate('hi'); const q = l.query('hi'); l.generate('again');\n" +
      'console.log(JSON.stringify({ g, q: q.text, same: g === PLACEHOLDER_TEXT, model: typeof q.model }));',
  );
  assert.strictEqual(r.status, 0, r.stderr);
  const out = JSON.parse(r.stdout);
  assert.ok(out.same);
  assert.strictEqual(out.q, out.g);
  assert.match(out.g, /not model output/);
  assert.strictEqual(out.model, 'string', 'routing fields are still filled in');
  assert.strictEqual(count(r.stderr, 'RUVLLM_PLACEHOLDER_TEXT'), 1);
});

test('RUVLLM_ALLOW_MOCK=1 opts in like allowPlaceholder; other values do not', () => {
  const on = run("console.log(new RuvLLM().generate('hi') === PLACEHOLDER_TEXT)", { RUVLLM_ALLOW_MOCK: '1' });
  assert.strictEqual(on.stdout.trim(), 'true', on.stderr);
  const off = run(codesOf(["new RuvLLM().generate('hi')"]), { RUVLLM_ALLOW_MOCK: 'true' });
  assert.strictEqual(off.stdout.trim(), 'RUVLLM_NO_LANGUAGE_MODEL');
});

test('the placeholder opt-in does not apply to generateDetailed() or chat()', () => {
  const r = run(
    'const l = new RuvLLM({ allowPlaceholder: true });\n' +
      codesOf(["l.generateDetailed('hi')", "l.chat([{ role: 'user', content: 'hi' }])"]),
  );
  assert.deepStrictEqual(r.stdout.trim().split('\n'), ['RUVLLM_NO_LANGUAGE_MODEL', 'RUVLLM_NO_LANGUAGE_MODEL']);
});

test('routing, memory and embeddings work without a model, silently', () => {
  const r = run(
    "const l = new RuvLLM(); l.embed('hi'); l.route('hi'); l.similarity('a', 'b'); console.log(l.isModelLoaded(), l.modelInfo());",
  );
  assert.strictEqual(r.status, 0, r.stderr);
  assert.strictEqual(r.stdout.trim(), 'false null');
  assert.strictEqual(count(r.stderr, 'RUVLLM_'), 0);
});

test('modelPath loads in the constructor, and a load failure throws instead of being ignored', () => {
  const r = run(
    "const probe = new RuvLLM();\n" +
      "console.log(probe.isNativeLoaded(), typeof probe.native?.loadModel === 'function', probe.supportsModelLoading());\n" +
      codesOf(["new RuvLLM({ modelPath: './does-not-exist.gguf' })"]),
  );
  const [state, code] = r.stdout.trim().split('\n');
  const expected = {
    'false false false': 'RUVLLM_NATIVE_UNAVAILABLE',
    'true false false': 'RUVLLM_NATIVE_TOO_OLD',
    'true true false': 'RUVLLM_NO_INFERENCE_BACKEND',
    'true true true': 'RUVLLM_MODEL_NOT_FOUND',
  }[state];
  assert.strictEqual(code, expected, r.stdout + r.stderr);
});

test('loadModel rejects an empty path', () => {
  const r = run("try { new RuvLLM().loadModel(''); } catch (e) { console.log(e.name, e.message); }");
  assert.match(r.stdout, /^TypeError .*path must be a non-empty string/);
});

test('backend is reported as unsupported, once; strict throws', () => {
  const r = run("new RuvLLM({ backend: 'x' }); new RuvLLM({ backend: 'x' });");
  assert.strictEqual(r.status, 0, r.stderr);
  assert.strictEqual(count(r.stderr, 'RUVLLM_UNSUPPORTED_OPTION'), 1);
  const s = run(codesOf(["new RuvLLM({ backend: 'x', strict: true })"]));
  assert.strictEqual(s.stdout.trim(), 'RUVLLM_UNSUPPORTED_OPTION');
});
