'use strict';
/**
 * `ruvllm serve` - local HTTP server over a loaded GGUF model.
 *
 * Endpoints (the contract used by @claude-flow/providers' RuVector provider):
 *   GET  /health               -> { status, model, sona, hnsw, native, version }
 *   GET  /v1/models            -> OpenAI-style model list
 *   POST /query                -> { query, session_id? } -> routed answer
 *   POST /v1/chat/completions  -> OpenAI-compatible chat (stream or not)
 *
 * Strict by design: the server only starts once a model is loaded. A healthy
 * /health tells clients that /query will be answered by a real model, so a
 * server that cannot generate must not be running (clients then fall back,
 * e.g. to Ollama). Generation is synchronous in the native binding, so
 * requests are served one at a time; `stream: true` returns the finished
 * completion as a single SSE chunk followed by [DONE] (no token streaming).
 */
const http = require('node:http');
const { randomUUID } = require('node:crypto');

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 3000;
const MAX_BODY_BYTES = 1024 * 1024;

function httpError(status, code, message) {
  const error = new Error(message);
  error.status = status;
  error.code = code;
  return error;
}

function readJson(req, limit) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(httpError(413, 'request_too_large', `request body exceeds ${limit} bytes`));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      try {
        const text = Buffer.concat(chunks).toString('utf8');
        const body = text.length ? JSON.parse(text) : {};
        if (!body || typeof body !== 'object' || Array.isArray(body)) {
          throw new Error('body must be a JSON object');
        }
        resolve(body);
      } catch (error) {
        reject(httpError(400, 'invalid_json', `invalid JSON body: ${error.message}`));
      }
    });
    req.on('error', reject);
  });
}

const LOOPBACK_ADDRESSES = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);
const LOOPBACK_HOSTNAMES = new Set(['localhost', '127.0.0.1', '[::1]']);

/**
 * DNS-rebinding guard: when a request arrives on a loopback address, its Host
 * header must name loopback too. A server bound to another address on purpose
 * (--host 0.0.0.0) still answers LAN clients by whatever name they use.
 */
function hostAllowed(req) {
  if (!LOOPBACK_ADDRESSES.has(req.socket.localAddress)) return true;
  const host = String(req.headers.host || '').toLowerCase();
  const name = host.startsWith('[') ? host.slice(0, host.indexOf(']') + 1) : host.split(':')[0];
  return LOOPBACK_HOSTNAMES.has(name);
}

/**
 * Browsers send cross-origin POSTs without a CORS preflight only for "simple"
 * content types, so requiring application/json keeps web pages from driving
 * generation on a local server.
 */
function requireJsonContentType(req) {
  const type = String(req.headers['content-type'] || '').split(';')[0].trim().toLowerCase();
  if (type !== 'application/json') {
    throw httpError(415, 'unsupported_media_type', 'content-type must be application/json');
  }
}

function send(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function finiteNumber(value, name, min, max) {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    throw httpError(400, 'invalid_request', `${name} must be a number in [${min}, ${max}]`);
  }
  return value;
}

/** Map OpenAI-style request fields onto the engine's GenerationConfig. */
function generationConfig(body, defaults) {
  const config = { ...defaults };
  const maxTokens = finiteNumber(body.max_tokens ?? body.max_completion_tokens, 'max_tokens', 1, 1 << 20);
  const temperature = finiteNumber(body.temperature, 'temperature', 0, 10);
  const topP = finiteNumber(body.top_p, 'top_p', 0, 1);
  const topK = finiteNumber(body.top_k, 'top_k', 0, 1 << 20);
  const seed = finiteNumber(body.seed, 'seed', 0, Number.MAX_SAFE_INTEGER);
  if (maxTokens !== undefined) config.maxTokens = Math.floor(maxTokens);
  if (temperature !== undefined) config.temperature = temperature;
  if (topP !== undefined) config.topP = topP;
  if (topK !== undefined) config.topK = Math.floor(topK);
  if (seed !== undefined) config.seed = Math.floor(seed);
  if (body.stop !== undefined) {
    const stop = typeof body.stop === 'string' ? [body.stop] : body.stop;
    if (!Array.isArray(stop) || stop.some((s) => typeof s !== 'string')) {
      throw httpError(400, 'invalid_request', 'stop must be a string or an array of strings');
    }
    config.stopSequences = stop;
  }
  return config;
}

function chatMessages(body) {
  const { messages } = body;
  if (!Array.isArray(messages) || messages.length === 0) {
    throw httpError(400, 'invalid_request', 'messages must be a non-empty array');
  }
  return messages.map((m, i) => {
    if (!m || !['system', 'user', 'assistant'].includes(m.role)) {
      throw httpError(400, 'invalid_request', `messages[${i}].role must be system, user or assistant`);
    }
    let content = m.content;
    if (Array.isArray(content)) {
      content = content.map((part) => (part && part.type === 'text' && typeof part.text === 'string' ? part.text : '')).join('');
    }
    if (typeof content !== 'string') {
      throw httpError(400, 'invalid_request', `messages[${i}].content must be a string`);
    }
    return { role: m.role, content };
  });
}

function usage(promptTokens, completionTokens) {
  const p = promptTokens || 0;
  const c = completionTokens || 0;
  return { prompt_tokens: p, completion_tokens: c, total_tokens: p + c };
}

/**
 * Build the HTTP server around an engine that already has a model loaded.
 * `engine` needs: query(text, config), chat(messages, config), route(text),
 * isModelLoaded(), isNativeLoaded(). Exported for tests.
 */
function createServer(engine, options = {}) {
  const modelId = options.modelId || 'ruvllm';
  const version = options.version || 'unknown';
  const defaults = options.defaults || {};
  const limit = options.maxBodyBytes || MAX_BODY_BYTES;
  const learning = options.learningEnabled !== false;

  async function handle(req, res) {
    const url = new URL(req.url, 'http://localhost');
    const route = `${req.method} ${url.pathname}`;

    if (!hostAllowed(req)) throw httpError(403, 'forbidden_host', 'invalid Host header');
    if (req.method === 'POST') requireJsonContentType(req);

    if (route === 'GET /health') {
      return send(res, 200, {
        status: 'ok',
        model: modelId,
        modelLoaded: engine.isModelLoaded(),
        native: engine.isNativeLoaded(),
        sona: learning,
        hnsw: engine.isNativeLoaded(),
        version,
      });
    }

    if (route === 'GET /v1/models') {
      return send(res, 200, { object: 'list', data: [{ id: modelId, object: 'model', owned_by: 'ruvllm' }] });
    }

    if (route === 'POST /query') {
      const body = await readJson(req, limit);
      if (typeof body.query !== 'string' || body.query.length === 0) {
        throw httpError(400, 'invalid_request', 'query must be a non-empty string');
      }
      const config = generationConfig(body, defaults);
      const response = engine.query(body.query, config);
      return send(res, 200, {
        id: response.requestId || `ruvllm-${randomUUID()}`,
        model: modelId,
        content: response.text,
        done: response.finishReason !== 'length',
        finish_reason: response.finishReason || 'stop',
        session_id: typeof body.session_id === 'string' ? body.session_id : undefined,
        usage: usage(response.promptTokens, response.completionTokens),
        router_metrics: { route: response.model, confidence: response.confidence, context_size: response.contextSize },
        latency_ms: response.latencyMs,
      });
    }

    if (route === 'POST /v1/chat/completions') {
      const body = await readJson(req, limit);
      const messages = chatMessages(body);
      const config = generationConfig(body, defaults);
      const result = engine.chat(messages, config);
      const id = `chatcmpl-${randomUUID()}`;
      const created = Math.floor(Date.now() / 1000);
      const finish = result.finishReason === 'length' ? 'length' : 'stop';
      if (body.stream === true) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        const chunk = (delta, finishReason, extra) => `data: ${JSON.stringify({
          id, object: 'chat.completion.chunk', created, model: modelId,
          choices: [{ index: 0, delta, finish_reason: finishReason }], ...extra,
        })}\n\n`;
        res.write(chunk({ role: 'assistant', content: result.text }, null));
        res.write(chunk({}, finish, { usage: usage(result.promptTokens, result.completionTokens) }));
        res.end('data: [DONE]\n\n');
        return undefined;
      }
      return send(res, 200, {
        id, object: 'chat.completion', created, model: modelId,
        choices: [{ index: 0, message: { role: 'assistant', content: result.text }, finish_reason: finish }],
        usage: usage(result.promptTokens, result.completionTokens),
      });
    }

    throw httpError(404, 'not_found', `no route for ${route}`);
  }

  return http.createServer((req, res) => {
    handle(req, res).catch((error) => {
      if (res.headersSent) {
        res.destroy(error);
        return;
      }
      const status = error.status
        || (error.code === 'RUVLLM_NO_LANGUAGE_MODEL' ? 503 : 500);
      send(res, status, { error: { code: error.code || 'server_error', message: error.message } });
    });
  });
}

/**
 * CLI entry: `ruvllm serve <model.gguf> [--host H] [--port P] [--max-context N]`.
 * `RuvLLM` is passed in so this file does not load dist/ by itself.
 */
function runServe({ RuvLLM, version }, positional, flags) {
  const modelPath = (typeof flags.model === 'string' && flags.model) || positional[0];
  if (!modelPath) {
    console.error('Error: ruvllm serve needs a model: ruvllm serve <path/to/model.gguf> (or --model <path>)');
    console.error('Download one with: ruvllm models download claude-code');
    process.exit(1);
  }
  const host = typeof flags.host === 'string' ? flags.host : DEFAULT_HOST;
  const port = flags.port === undefined ? DEFAULT_PORT : Number(flags.port);
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    console.error('Error: --port must be an integer in [0, 65535]');
    process.exit(1);
  }

  const llm = new RuvLLM({ learningEnabled: !flags['no-learning'] });
  const loadOptions = {};
  if (flags['max-context']) loadOptions.maxContext = parseInt(flags['max-context'], 10);
  let info;
  try {
    info = llm.loadModel(modelPath, loadOptions);
  } catch (error) {
    console.error(`Error: ${error.message}`);
    if (error.code) console.error(`(${error.code}; the server was not started)`);
    process.exit(1);
  }

  const defaults = {};
  if (flags['max-tokens']) defaults.maxTokens = parseInt(flags['max-tokens'], 10);
  if (flags.temperature) defaults.temperature = parseFloat(flags.temperature);
  const server = createServer(llm, {
    modelId: info.name, version, defaults, learningEnabled: !flags['no-learning'],
  });
  server.listen(port, host, () => {
    const address = server.address();
    console.error(`Loaded ${info.name} (${info.architecture}, ${info.quantization ?? 'quantization unknown'})`);
    console.error(`ruvllm serve listening on http://${address.address}:${address.port} (GET /health, POST /query, POST /v1/chat/completions)`);
  });
  const stop = () => server.close(() => process.exit(0));
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
  return server;
}

module.exports = { createServer, runServe, DEFAULT_HOST, DEFAULT_PORT };
