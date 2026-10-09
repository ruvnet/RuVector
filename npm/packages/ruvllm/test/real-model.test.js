/**
 * Real inference: load a GGUF and generate from Node. Runs only when
 * RUVLLM_TEST_GGUF points at a chat/instruct GGUF (written against
 * Qwen2.5-0.5B-Instruct Q4_K_M; Qwen3 adds a <think> block, so prompts may
 * need `/no_think`), and fails if the native binary cannot load it:
 *
 *   npm run build:native      # or set RUVLLM_NATIVE_PATH to a built binary
 *   RUVLLM_TEST_GGUF=./qwen2.5-0.5b-instruct-q4_k_m.gguf npm test
 */
const { describe, test, before } = require('node:test');
const assert = require('node:assert');

const { RuvLLM } = require('../dist/cjs/index.js');

const MODEL = process.env.RUVLLM_TEST_GGUF;
const greedy = (maxTokens) => ({ maxTokens, temperature: 0 });
const ask = (content) => [{ role: 'user', content }];

describe('real model', { skip: !MODEL && 'set RUVLLM_TEST_GGUF to a GGUF file to run' }, () => {
  let llm;
  let info;
  before(() => {
    llm = new RuvLLM();
    assert.ok(llm.supportsModelLoading(), 'RUVLLM_TEST_GGUF is set but the native binary cannot load models');
    info = llm.loadModel(MODEL, { maxContext: 2048 });
  });

  test('loadModel reports the model', () => {
    assert.strictEqual(llm.isModelLoaded(), true);
    assert.deepStrictEqual(llm.modelInfo(), info);
    assert.ok(info.numLayers > 0 && info.vocabSize > 0 && info.hiddenSize > 0);
    assert.strictEqual(info.maxContextLength, 2048);
    assert.ok(info.chatTemplate, 'a chat template is known for the model');
  });

  test('chat answers from the model, with real token counts', () => {
    const out = llm.chat(ask('What is the capital of France? Answer in one word.'), greedy(16));
    assert.match(out.text, /Paris/);
    assert.strictEqual(out.finishReason, 'stop');
    assert.ok(out.promptTokens > 5, `promptTokens ${out.promptTokens}`);
    assert.ok(out.completionTokens >= 1 && out.completionTokens <= 16, `completionTokens ${out.completionTokens}`);
  });

  test('greedy generation is repeatable (no state leaks between requests)', () => {
    const a = llm.chat(ask('Name three colours.'), greedy(24));
    const b = llm.chat(ask('Name three colours.'), greedy(24));
    assert.deepStrictEqual(a, b);
  });

  test('maxTokens is honoured and reported as finishReason "length"', () => {
    const out = llm.chat(ask('Count from 1 to 50, one number per line.'), greedy(3));
    assert.strictEqual(out.finishReason, 'length');
    assert.strictEqual(out.completionTokens, 3);
  });

  test('a stop sequence ends generation and is not returned', () => {
    const out = llm.chat(ask('What is the capital of France? Answer in one sentence.'), {
      ...greedy(32),
      stopSequences: ['Paris'],
    });
    assert.strictEqual(out.finishReason, 'stop');
    assert.doesNotMatch(out.text, /Paris/);
  });

  test('generate() continues a raw prompt and matches generateDetailed()', () => {
    const detailed = llm.generateDetailed('The capital of France is', greedy(8));
    assert.match(detailed.text, /Paris/);
    assert.strictEqual(llm.generate('The capital of France is', greedy(8)), detailed.text);
  });

  test('query() routes and answers with the model', () => {
    const r = llm.query('What is 2+2? Answer with a number only.', greedy(8));
    assert.match(r.text, /4/);
    assert.strictEqual(r.finishReason, 'stop');
    assert.ok(r.promptTokens > 0 && r.completionTokens > 0);
    assert.strictEqual(typeof r.confidence, 'number');
    assert.ok(r.requestId);
  });

  test('invalid input is rejected', () => {
    assert.throws(() => llm.generate('x', { temperature: -1 }), /temperature/);
    assert.throws(() => llm.generate('x', { topP: 0 }), /topP/);
    assert.throws(() => llm.generate('x', { maxTokens: 0 }), /maxTokens/);
    assert.throws(() => llm.chat([{ role: 'robot', content: 'x' }]), /unknown chat role 'robot'/);
    assert.throws(() => llm.chat([]), /at least one message/);
  });

  test('unloadModel frees the model; generation then fails loudly', () => {
    llm.unloadModel();
    assert.strictEqual(llm.isModelLoaded(), false);
    assert.throws(() => llm.generate('hi'), { code: 'RUVLLM_NO_LANGUAGE_MODEL' });
  });
});
