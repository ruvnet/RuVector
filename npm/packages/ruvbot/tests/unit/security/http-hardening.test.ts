/**
 * HTTP API hardening tests
 *
 * Covers request body limits, API authentication, session listing redaction,
 * session lifecycle bounds, and per-session memory recall scoping.
 */

import { describe, it, expect, vi } from 'vitest';
import { PassThrough } from 'node:stream';
import type { IncomingMessage, ServerResponse } from 'node:http';
import {
  readJsonBody,
  isRequestAuthorized,
  isPublicPath,
  summarizeSession,
  PayloadTooLargeError,
} from '../../../src/api/httpGuard.js';
import { RuvBot } from '../../../src/RuvBot.js';
import { ChatEnhancer } from '../../../src/core/ChatEnhancer.js';

function fakeReq(opts: {
  method?: string;
  url?: string;
  headers?: Record<string, string>;
  body?: string | Buffer;
}): IncomingMessage {
  const stream = new PassThrough() as unknown as IncomingMessage & PassThrough;
  stream.method = opts.method ?? 'GET';
  stream.url = opts.url ?? '/';
  stream.headers = { host: 'localhost', ...(opts.headers ?? {}) };
  setImmediate(() => {
    if (opts.body !== undefined) stream.write(opts.body);
    stream.end();
  });
  return stream;
}

function fakeRes(): ServerResponse & { statusCode: number; body: string; done: Promise<void> } {
  let resolveDone!: () => void;
  const done = new Promise<void>((r) => { resolveDone = r; });
  const res = {
    statusCode: 200,
    body: '',
    headersSent: false,
    headers: {} as Record<string, unknown>,
    done,
    setHeader(k: string, v: unknown) { this.headers[k.toLowerCase()] = v; },
    writeHead(code: number) { this.statusCode = code; this.headersSent = true; return this; },
    end(chunk?: string) { if (chunk) this.body += chunk; resolveDone(); return this; },
  };
  return res as unknown as ServerResponse & { statusCode: number; body: string; done: Promise<void> };
}

function quietBot(overrides: Record<string, unknown> = {}): RuvBot {
  return new RuvBot({
    config: {
      api: { enabled: false, port: 3000, host: '127.0.0.1' },
      logging: { level: 'fatal', pretty: false },
      ...overrides,
    } as never,
  });
}

async function callApi(bot: RuvBot, req: IncomingMessage) {
  const res = fakeRes();
  await (bot as unknown as {
    handleApiRequest: (q: IncomingMessage, s: ServerResponse) => Promise<void>;
  }).handleApiRequest(req, res);
  await res.done;
  return res;
}

describe('readJsonBody', () => {
  it('parses a JSON body under the limit', async () => {
    const body = await readJsonBody(fakeReq({ method: 'POST', body: '{"a":1}' }), 1024);
    expect(body).toEqual({ a: 1 });
  });

  it('rejects bodies over the limit with PayloadTooLargeError', async () => {
    const big = 'x'.repeat(2048);
    await expect(
      readJsonBody(fakeReq({ method: 'POST', body: `{"m":"${big}"}` }), 1024)
    ).rejects.toBeInstanceOf(PayloadTooLargeError);
  });
});

describe('isRequestAuthorized', () => {
  const secret = 's3cret-token';

  it('allows everything when auth is disabled', () => {
    expect(isRequestAuthorized({}, { enabled: false, type: 'bearer' })).toBe(true);
  });

  it('fails closed when enabled without a secret', () => {
    expect(isRequestAuthorized({ authorization: 'Bearer ' }, { enabled: true, type: 'bearer' })).toBe(false);
  });

  it('checks bearer tokens', () => {
    const auth = { enabled: true, type: 'bearer' as const, secret };
    expect(isRequestAuthorized({ authorization: `Bearer ${secret}` }, auth)).toBe(true);
    expect(isRequestAuthorized({ authorization: 'Bearer wrong' }, auth)).toBe(false);
    expect(isRequestAuthorized({}, auth)).toBe(false);
  });

  it('checks api keys', () => {
    const auth = { enabled: true, type: 'apikey' as const, secret };
    expect(isRequestAuthorized({ 'x-api-key': secret }, auth)).toBe(true);
    expect(isRequestAuthorized({ 'x-api-key': `${secret}x` }, auth)).toBe(false);
  });

  it('checks basic credentials', () => {
    const auth = { enabled: true, type: 'basic' as const, secret: 'user:pass' };
    const good = Buffer.from('user:pass').toString('base64');
    expect(isRequestAuthorized({ authorization: `Basic ${good}` }, auth)).toBe(true);
    expect(isRequestAuthorized({ authorization: 'Basic dXNlcjpub3Bl' }, auth)).toBe(false);
  });

  it('treats only landing and health paths as public', () => {
    expect(isPublicPath('/health')).toBe(true);
    expect(isPublicPath('/ready')).toBe(true);
    expect(isPublicPath('/api/sessions')).toBe(false);
  });
});

describe('summarizeSession', () => {
  it('omits message contents and metadata', () => {
    const summary = summarizeSession({
      id: 's1', agentId: 'a1', userId: 'u1', platform: 'api',
      messages: [{ id: 'm', sessionId: 's1', role: 'user', content: 'my secret', createdAt: new Date() }],
      context: { topics: [], entities: [] },
      metadata: { email: 'x@y.z' },
      createdAt: new Date(), updatedAt: new Date(),
    });
    expect(JSON.stringify(summary)).not.toContain('my secret');
    expect(JSON.stringify(summary)).not.toContain('x@y.z');
    expect(summary.messageCount).toBe(1);
  });
});

describe('RuvBot API server', () => {
  it('enforces configured api.auth on /api routes', async () => {
    const bot = quietBot({
      api: { enabled: false, port: 3000, host: '127.0.0.1', auth: { enabled: true, type: 'bearer', secret: 'tok' } },
    });
    const denied = await callApi(bot, fakeReq({ url: '/api/sessions' }));
    expect(denied.statusCode).toBe(401);

    const allowed = await callApi(bot, fakeReq({ url: '/api/sessions', headers: { authorization: 'Bearer tok' } }));
    expect(allowed.statusCode).toBe(200);

    const health = await callApi(bot, fakeReq({ url: '/health' }));
    expect(health.statusCode).toBe(200);
  });

  it('does not expose message contents via GET /api/sessions', async () => {
    const bot = quietBot();
    (bot as unknown as { llmProvider: unknown }).llmProvider = {
      complete: vi.fn().mockResolvedValue({ content: 'ok', usage: { inputTokens: 1, outputTokens: 1 }, finishReason: 'stop' }),
    };
    await bot.spawnAgent({ id: 'a', name: 'a', model: 'm' } as never);
    const s = await bot.createSession('a');
    await bot.chat(s.id, 'private-message-content');

    const res = await callApi(bot, fakeReq({ url: '/api/sessions' }));
    expect(res.statusCode).toBe(200);
    expect(res.body).not.toContain('private-message-content');
  });

  it('rejects oversized request bodies with 413', async () => {
    const bot = quietBot();
    const res = await callApi(bot, fakeReq({
      method: 'POST', url: '/api/chat', body: `{"message":"${'x'.repeat(2 * 1024 * 1024)}"}`,
    }));
    expect(res.statusCode).toBe(413);
  });
});

describe('RuvBot sessions', () => {
  function botWithLLM(session?: Record<string, unknown>) {
    const bot = quietBot(session ? { session } : {});
    const complete = vi.fn().mockResolvedValue({
      content: 'reply', usage: { inputTokens: 1, outputTokens: 1 }, finishReason: 'stop',
    });
    (bot as unknown as { llmProvider: unknown }).llmProvider = { complete };
    return { bot, complete };
  }

  it('sends the current user turn to the LLM exactly once', async () => {
    const { bot, complete } = botWithLLM();
    await bot.spawnAgent({ id: 'a', name: 'a', model: 'm' } as never);
    const s = await bot.createSession('a');
    await bot.chat(s.id, 'hello-once');

    const messages = complete.mock.calls[0][0] as Array<{ role: string; content: string }>;
    expect(messages.filter((m) => m.content === 'hello-once')).toHaveLength(1);
  });

  it('refuses to chat in an expired session and drops it', async () => {
    const { bot } = botWithLLM();
    await bot.spawnAgent({ id: 'a', name: 'a', model: 'm' } as never);
    const s = await bot.createSession('a');
    s.expiresAt = new Date(Date.now() - 1000);

    await expect(bot.chat(s.id, 'hi')).rejects.toThrow(/expired/i);
    expect(bot.getSession(s.id)).toBeUndefined();
  });

  it('caps stored history at session.maxMessages', async () => {
    const { bot } = botWithLLM({ maxMessages: 10 });
    await bot.spawnAgent({ id: 'a', name: 'a', model: 'm' } as never);
    const s = await bot.createSession('a');
    for (let i = 0; i < 8; i++) await bot.chat(s.id, `msg ${i}`);

    expect(bot.getSession(s.id)!.messages.length).toBeLessThanOrEqual(10);
  });
});

describe('ChatEnhancer memory scoping', () => {
  it('does not recall memories stored by a different session', async () => {
    const enhancer = new ChatEnhancer({ enableProactiveAssistance: false });
    await enhancer.processMessage('remember that the vault code is 4242', {
      sessionId: 'session-A', userId: 'alice', tenantId: 'default', conversationHistory: [],
    });
    const other = await enhancer.processMessage('what is the vault code', {
      sessionId: 'session-B', userId: 'mallory', tenantId: 'default', conversationHistory: [],
    });
    expect(JSON.stringify(other.memoriesRecalled)).not.toContain('4242');

    const same = await enhancer.processMessage('what is the vault code', {
      sessionId: 'session-A', userId: 'alice', tenantId: 'default', conversationHistory: [],
    });
    expect(JSON.stringify(same.memoriesRecalled)).toContain('4242');
  });
});

describe('ConfigManager.fromEnv API auth', () => {
  it('enables bearer auth when RUVBOT_API_KEY is set', async () => {
    const { ConfigManager } = await import('../../../src/core/BotConfig.js');
    const prev = process.env.RUVBOT_API_KEY;
    process.env.RUVBOT_API_KEY = 'env-key';
    try {
      const auth = ConfigManager.fromEnv().getConfig().api.auth;
      expect(auth).toMatchObject({ enabled: true, type: 'bearer', secret: 'env-key' });
    } finally {
      if (prev === undefined) delete process.env.RUVBOT_API_KEY;
      else process.env.RUVBOT_API_KEY = prev;
    }
  });
});

describe('chat UI markdown renderer', () => {
  async function loadRenderer(): Promise<(text: string) => string> {
    const { readFileSync } = await import('node:fs');
    const { join } = await import('node:path');
    const html = readFileSync(join(__dirname, '../../../src/api/public/index.html'), 'utf-8');
    const start = html.indexOf('function escapeHtml(');
    const fmt = html.indexOf('function formatMarkdown(');
    const end = html.indexOf('// Check API health and status');
    const src = start >= 0 && start < fmt ? html.slice(start, end) : html.slice(fmt, end);
    return new Function(`${src}; return formatMarkdown;`)() as (text: string) => string;
  }

  it('escapes raw HTML in model output', async () => {
    const render = await loadRenderer();
    expect(render('hello <img src=x onerror=alert(1)>')).not.toContain('<img');
  });

  it('does not render javascript: links', async () => {
    const render = await loadRenderer();
    expect(render('[click](javascript:alert(1))')).not.toMatch(/href="javascript:/i);
  });

  it('still renders basic markdown', async () => {
    const render = await loadRenderer();
    const out = render('**bold** and [site](https://example.com)\n> quote');
    expect(out).toContain('<strong>bold</strong>');
    expect(out).toContain('href="https://example.com"');
    expect(out).toContain('<blockquote>quote</blockquote>');
  });
});
