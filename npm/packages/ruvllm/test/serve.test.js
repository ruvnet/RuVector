/**
 * `ruvllm serve` HTTP contract (the one @claude-flow/providers' RuVector
 * provider calls), driven with a stub engine, plus the strict CLI exit paths.
 */
const { test, describe, before, after } = require('node:test');
const assert = require('node:assert');
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const { createServer } = require('../bin/serve.js');

function stubEngine() {
  const calls = [];
  return {
    calls,
    isModelLoaded: () => true,
    isNativeLoaded: () => true,
    query(text, config) {
      calls.push(['query', text, config]);
      return { text: `answer:${text}`, confidence: 0.75, model: 'M700', contextSize: 1024, latencyMs: 3,
        requestId: 'req-1', promptTokens: 5, completionTokens: 7, finishReason: 'stop' };
    },
    chat(messages, config) {
      calls.push(['chat', messages, config]);
      if (messages.some((m) => m.content === 'fail')) {
        const error = new Error('no model'); error.code = 'RUVLLM_NO_LANGUAGE_MODEL'; throw error;
      }
      return { text: 'hi there', promptTokens: 9, completionTokens: 2, finishReason: 'length' };
    },
  };
}

describe('ruvllm serve HTTP API', () => {
  let server; let base; const engine = stubEngine();
  before(async () => {
    server = createServer(engine, { modelId: 'qwen-test', version: '9.9.9', defaults: { maxTokens: 32 } });
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    base = `http://127.0.0.1:${server.address().port}`;
  });
  after(() => new Promise((r) => server.close(r)));

  const post = (p, body) => fetch(base + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: typeof body === 'string' ? body : JSON.stringify(body) });

  test('GET /health reports the loaded model', async () => {
    const res = await fetch(`${base}/health`);
    assert.strictEqual(res.status, 200);
    const body = await res.json();
    assert.deepStrictEqual(body, { status: 'ok', model: 'qwen-test', modelLoaded: true, native: true, sona: true, hnsw: true, version: '9.9.9' });
  });

  test('GET /v1/models lists the model', async () => {
    const body = await (await fetch(`${base}/v1/models`)).json();
    assert.deepStrictEqual(body.data.map((m) => m.id), ['qwen-test']);
  });

  test('POST /query returns the provider response shape', async () => {
    const res = await post('/query', { query: 'what is 2+2', session_id: 's1', max_tokens: 16, temperature: 0 });
    assert.strictEqual(res.status, 200);
    const body = await res.json();
    assert.strictEqual(body.content, 'answer:what is 2+2');
    assert.strictEqual(body.done, true);
    assert.strictEqual(body.session_id, 's1');
    assert.deepStrictEqual(body.usage, { prompt_tokens: 5, completion_tokens: 7, total_tokens: 12 });
    assert.strictEqual(body.router_metrics.route, 'M700');
    const [, , config] = engine.calls.at(-1);
    assert.deepStrictEqual(config, { maxTokens: 16, temperature: 0 });
  });

  test('POST /v1/chat/completions (non-stream)', async () => {
    const res = await post('/v1/chat/completions', { messages: [{ role: 'system', content: 'be brief' }, { role: 'user', content: [{ type: 'text', text: 'hello' }] }] });
    const body = await res.json();
    assert.strictEqual(body.object, 'chat.completion');
    assert.strictEqual(body.choices[0].message.content, 'hi there');
    assert.strictEqual(body.choices[0].finish_reason, 'length');
    const [, messages, config] = engine.calls.at(-1);
    assert.deepStrictEqual(messages, [{ role: 'system', content: 'be brief' }, { role: 'user', content: 'hello' }]);
    assert.deepStrictEqual(config, { maxTokens: 32 });
  });

  test('POST /v1/chat/completions (stream) sends SSE chunks and [DONE]', async () => {
    const res = await post('/v1/chat/completions', { stream: true, messages: [{ role: 'user', content: 'hello' }] });
    assert.match(res.headers.get('content-type'), /text\/event-stream/);
    const events = (await res.text()).trim().split('\n\n').map((e) => e.replace(/^data: /, ''));
    assert.strictEqual(events.at(-1), '[DONE]');
    const chunks = events.slice(0, -1).map((e) => JSON.parse(e));
    assert.strictEqual(chunks[0].choices[0].delta.content, 'hi there');
    assert.strictEqual(chunks.at(-1).choices[0].finish_reason, 'length');
  });

  test('bad input is rejected with 400', async () => {
    assert.strictEqual((await post('/query', '{nope')).status, 400);
    assert.strictEqual((await post('/query', { query: '' })).status, 400);
    assert.strictEqual((await post('/query', { query: 'x', temperature: 'hot' })).status, 400);
    assert.strictEqual((await post('/v1/chat/completions', { messages: [{ role: 'tool', content: 'x' }] })).status, 400);
  });

  test('oversized bodies are rejected with 413', async () => {
    const small = createServer(stubEngine(), { maxBodyBytes: 64 });
    await new Promise((r) => small.listen(0, '127.0.0.1', r));
    try {
      const res = await fetch(`http://127.0.0.1:${small.address().port}/query`, { method: 'POST', body: JSON.stringify({ query: 'x'.repeat(200) }) }).catch((e) => e);
      if (res instanceof Error) return; // connection reset after the 413 is also acceptable
      assert.strictEqual(res.status, 413);
    } finally {
      await new Promise((r) => small.close(r));
    }
  });

  test('engine RUVLLM_NO_LANGUAGE_MODEL maps to 503', async () => {
    const res = await post('/v1/chat/completions', { messages: [{ role: 'user', content: 'fail' }] });
    assert.strictEqual(res.status, 503);
    assert.strictEqual((await res.json()).error.code, 'RUVLLM_NO_LANGUAGE_MODEL');
  });

  test('unknown routes are 404', async () => {
    assert.strictEqual((await fetch(`${base}/nope`)).status, 404);
  });
});

describe('ruvllm serve CLI is strict', () => {
  const cli = path.join(__dirname, '..', 'bin', 'cli.js');

  test('exits 1 without a model', () => {
    const r = spawnSync(process.execPath, [cli, 'serve'], { encoding: 'utf8' });
    assert.strictEqual(r.status, 1);
    assert.match(r.stderr, /needs a model/);
  });

  test('exits 1 when the model cannot be loaded, and never listens', () => {
    const r = spawnSync(process.execPath, [cli, 'serve', '/nonexistent/model.gguf', '--port', '0'], { encoding: 'utf8', timeout: 30000 });
    assert.strictEqual(r.status, 1);
    assert.match(r.stderr, /server was not started/);
    assert.doesNotMatch(r.stderr, /listening/);
  });
});
