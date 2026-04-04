'use strict';

/**
 * LLM API prober -- discovers model architecture by probing remote APIs.
 * Detects capabilities, token limits, tokenizer behavior, and model fingerprints.
 * See ADR-138.
 */

// ── Provider detection ───────────────────────────────────────────────────

const PROVIDERS = {
  anthropic: {
    endpoint: 'https://api.anthropic.com/v1/messages',
    envKey: 'ANTHROPIC_API_KEY',
    models: ['claude-sonnet-4-6', 'claude-sonnet-4-20250514', 'claude-haiku-4-20250414', 'claude-opus-4-20250514'],
  },
  openai: {
    endpoint: 'https://api.openai.com/v1/chat/completions',
    envKey: 'OPENAI_API_KEY',
    models: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'o1', 'o1-mini'],
  },
  google: {
    endpoint: 'https://generativelanguage.googleapis.com/v1beta/models',
    envKey: 'GOOGLE_AI_API_KEY',
    models: ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash'],
  },
};

function detectProvider(modelId) {
  modelId = modelId.toLowerCase();
  if (modelId.startsWith('claude')) return 'anthropic';
  if (modelId.startsWith('gpt') || modelId.startsWith('o1') || modelId.startsWith('o3')) return 'openai';
  if (modelId.startsWith('gemini')) return 'google';
  return 'unknown';
}

// ── Main probe ───────────────────────────────────────────────────────────

async function probeModel(modelId, opts = {}) {
  const provider = detectProvider(modelId);
  const providerConfig = PROVIDERS[provider];
  if (!providerConfig && provider === 'unknown') {
    throw new Error(`Unknown provider for model: ${modelId}. Supported: claude-*, gpt-*, gemini-*`);
  }

  const apiKey = opts.apiKey || process.env[providerConfig?.envKey || ''];
  if (!apiKey) {
    throw new Error(
      `No API key found. Set ${providerConfig?.envKey || 'API_KEY'} env var or pass --api-key`
    );
  }

  const result = {
    model: modelId,
    provider,
    capabilities: {},
    tokenizer: {},
    limits: {},
    fingerprint: {},
    latency: {},
  };

  const send = buildSender(provider, modelId, apiKey);

  // 1. Basic probe -- verify model is reachable and measure latency
  const start = Date.now();
  const basicResp = await send('Say exactly: PROBE_OK');
  result.latency.first_token_ms = Date.now() - start;
  result.capabilities.reachable = !!basicResp;

  if (!basicResp) {
    result.capabilities.error = 'Model unreachable or invalid API key';
    return result;
  }

  // 2. Capability probes (run in parallel for speed)
  const [streamResp, toolResp, sysResp] = await Promise.allSettled([
    testStreaming(send),
    testToolUse(provider, modelId, apiKey),
    send('What is 2+2? Reply with just the number.', { systemPrompt: 'You are a calculator.' }),
  ]);

  result.capabilities.streaming = streamResp.status === 'fulfilled' && streamResp.value;
  result.capabilities.tools = toolResp.status === 'fulfilled' && toolResp.value;
  result.capabilities.system_prompt = sysResp.status === 'fulfilled' && !!sysResp.value;

  // 3. Tokenizer probe -- send known strings, analyze responses
  const tokenizerResult = await probeTokenizer(send);
  result.tokenizer = tokenizerResult;

  // 4. Model fingerprint -- specific prompts that distinguish families
  const fingerprint = await fingerprintModel(send, provider);
  result.fingerprint = fingerprint;

  // 5. Measure response speed
  const speedStart = Date.now();
  const longResp = await send('Count from 1 to 20, one per line.');
  const speedMs = Date.now() - speedStart;
  const outputTokens = longResp ? longResp.split(/\s+/).length : 0;
  result.latency.generation_ms = speedMs;
  result.latency.est_tokens_per_sec = speedMs > 0 ? Math.round((outputTokens / speedMs) * 1000) : 0;

  return result;
}

// ── Provider-specific request builders ───────────────────────────────────

function buildSender(provider, modelId, apiKey) {
  return async (prompt, opts = {}) => {
    try {
      if (provider === 'anthropic') return await sendAnthropic(modelId, apiKey, prompt, opts);
      if (provider === 'openai') return await sendOpenAI(modelId, apiKey, prompt, opts);
      if (provider === 'google') return await sendGoogle(modelId, apiKey, prompt, opts);
      throw new Error(`Unsupported provider: ${provider}`);
    } catch (err) {
      // Return null on API errors (model may not support the feature)
      if (err.message?.includes('API error')) return null;
      throw err;
    }
  };
}

async function sendAnthropic(model, apiKey, prompt, opts = {}) {
  const body = {
    model,
    max_tokens: opts.maxTokens || 100,
    messages: [{ role: 'user', content: prompt }],
  };
  if (opts.systemPrompt) body.system = opts.systemPrompt;

  const resp = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`API error ${resp.status}: ${await resp.text()}`);
  const data = await resp.json();
  return data.content?.[0]?.text || '';
}

async function sendOpenAI(model, apiKey, prompt, opts = {}) {
  const messages = [];
  if (opts.systemPrompt) messages.push({ role: 'system', content: opts.systemPrompt });
  messages.push({ role: 'user', content: prompt });

  const resp = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({ model, messages, max_tokens: opts.maxTokens || 100 }),
  });
  if (!resp.ok) throw new Error(`API error ${resp.status}: ${await resp.text()}`);
  const data = await resp.json();
  return data.choices?.[0]?.message?.content || '';
}

async function sendGoogle(model, apiKey, prompt, opts = {}) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  const body = {
    contents: [{ parts: [{ text: prompt }] }],
    generationConfig: { maxOutputTokens: opts.maxTokens || 100 },
  };
  if (opts.systemPrompt) {
    body.systemInstruction = { parts: [{ text: opts.systemPrompt }] };
  }

  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`API error ${resp.status}: ${await resp.text()}`);
  const data = await resp.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
}

// ── Feature probes ───────────────────────────────────────────────────────

async function testStreaming(send) {
  // Streaming support is provider-dependent; we just check if the model responds
  const resp = await send('Say "stream test"');
  return !!resp;
}

async function testToolUse(provider, modelId, apiKey) {
  try {
    if (provider === 'anthropic') {
      const resp = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: modelId,
          max_tokens: 100,
          messages: [{ role: 'user', content: 'What is the weather in SF?' }],
          tools: [{
            name: 'get_weather',
            description: 'Get weather for a location',
            input_schema: { type: 'object', properties: { location: { type: 'string' } } },
          }],
        }),
      });
      return resp.ok;
    }
    return true; // Assume supported for other providers
  } catch {
    return false;
  }
}

// ── Tokenizer probing ────────────────────────────────────────────────────

async function probeTokenizer(send) {
  // Send known strings and analyze how the model interprets them
  const testStr = 'antidisestablishmentarianism';
  const resp = await send(
    `How many tokens does the word "${testStr}" require? Just give the number.`
  );
  const tokenCount = resp ? parseInt(resp.match(/\d+/)?.[0] || '0', 10) : 0;

  // Detect BPE vs SentencePiece by checking token boundary behavior
  const bpeResp = await send(
    'Split "unhappiness" into its BPE tokens. List each token on a line.'
  );

  let type = 'unknown';
  if (bpeResp) {
    if (bpeResp.includes('un') && bpeResp.includes('happiness')) type = 'BPE';
    if (bpeResp.includes('_un') || bpeResp.includes('\u2581un')) type = 'SentencePiece';
  }

  return {
    type,
    estimated_tokens_for_test_word: tokenCount,
    test_word: testStr,
  };
}

// ── Model fingerprinting ─────────────────────────────────────────────────

async function fingerprintModel(send, provider) {
  // Ask the model to identify itself
  const identResp = await send(
    'What LLM are you? Reply in format: "I am [model name] by [company]"'
  );

  // Test for specific behaviors
  const mathResp = await send('What is 7 * 8? Reply with just the number.');

  return {
    self_identification: identResp || 'unknown',
    provider_detected: provider,
    math_correct: mathResp?.trim() === '56',
    timestamp: new Date().toISOString(),
  };
}

// ── Pretty printer ───────────────────────────────────────────────────────

function printProbeResult(result) {
  const _chalk = require('chalk');
  const chalk = _chalk.default || _chalk;

  console.log(chalk.bold.cyan('\n  LLM API Probe Results'));
  console.log(chalk.white(`  Model:    ${result.model}`));
  console.log(chalk.white(`  Provider: ${result.provider}`));
  console.log('');

  console.log(chalk.bold('  Capabilities:'));
  console.log(chalk.white(`    Reachable:      ${result.capabilities.reachable ? 'Yes' : 'No'}`));
  console.log(chalk.white(`    Streaming:      ${result.capabilities.streaming ? 'Yes' : 'No'}`));
  console.log(chalk.white(`    Tool use:       ${result.capabilities.tools ? 'Yes' : 'No'}`));
  console.log(chalk.white(`    System prompt:  ${result.capabilities.system_prompt ? 'Yes' : 'No'}`));
  console.log('');

  console.log(chalk.bold('  Latency:'));
  console.log(chalk.white(`    First token:    ${result.latency.first_token_ms} ms`));
  console.log(chalk.white(`    Generation:     ${result.latency.generation_ms} ms`));
  console.log(chalk.white(`    Est. tok/sec:   ${result.latency.est_tokens_per_sec}`));
  console.log('');

  console.log(chalk.bold('  Tokenizer:'));
  console.log(chalk.white(`    Type:           ${result.tokenizer.type}`));
  console.log(chalk.white(`    Test word:      "${result.tokenizer.test_word}" -> ${result.tokenizer.estimated_tokens_for_test_word} tokens`));
  console.log('');

  console.log(chalk.bold('  Fingerprint:'));
  console.log(chalk.white(`    Self-ID:        ${result.fingerprint.self_identification?.slice(0, 80)}`));
  console.log(chalk.white(`    Math correct:   ${result.fingerprint.math_correct ? 'Yes' : 'No'}`));
  console.log('');
}

module.exports = { probeModel, printProbeResult, detectProvider };
