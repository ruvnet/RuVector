/**
 * Tests for the bundled web chat UI's fetch wrapper (src/api/public/index.html,
 * <script id="ruvbot-api-client">). The script is extracted from the shipped
 * HTML and evaluated with mocked fetch / sessionStorage / prompt, so these
 * tests exercise the exact code the browser runs.
 */

import { describe, it, expect, vi } from 'vitest';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import vm from 'vm';

const html = readFileSync(resolve(__dirname, '../../../src/api/public/index.html'), 'utf8');

function loadFactory() {
  const m = /<script id="ruvbot-api-client">([\s\S]*?)<\/script>/.exec(html);
  if (!m) throw new Error('ruvbot-api-client script block not found in index.html');
  // Evaluate only the factory; the trailing wiring touches `window`.
  const src = m[1].slice(0, m[1].indexOf('const ruvbotApi'));
  const ctx = vm.createContext({ Headers, Promise });
  vm.runInContext(src + '\nthis.createRuvBotApiClient = createRuvBotApiClient;', ctx);
  return ctx.createRuvBotApiClient as (deps: unknown) => {
    apiFetch: (url: string, init?: RequestInit) => Promise<Response>;
    getKey: () => string | null;
  };
}

class MemoryStorage {
  map = new Map<string, string>();
  getItem(k: string) { return this.map.has(k) ? (this.map.get(k) as string) : null; }
  setItem(k: string, v: string) { this.map.set(k, String(v)); }
  removeItem(k: string) { this.map.delete(k); }
}

/** A fake server: 401 unless the request carries the expected bearer key. */
function fakeServer(requiredKey: string | null) {
  const calls: { url: string; auth: string | null }[] = [];
  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    const auth = new Headers(init?.headers).get('authorization');
    calls.push({ url, auth });
    const ok = requiredKey === null || auth === `Bearer ${requiredKey}`;
    return new Response(JSON.stringify(ok ? { ok: true } : { error: 'Unauthorized' }), {
      status: ok ? 200 : 401,
      headers: { 'content-type': 'application/json' },
    });
  });
  return { fetch, calls };
}

const createClient = loadFactory();

describe('web UI api client', () => {
  it('attaches Authorization: Bearer from sessionStorage to /api/* requests', async () => {
    const server = fakeServer('k1');
    const storage = new MemoryStorage();
    storage.setItem('ruvbot.apiKey', 'k1');
    const prompt = vi.fn();
    const client = createClient({ fetch: server.fetch, storage, prompt });

    const res = await client.apiFetch('/api/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });

    expect(res.status).toBe(200);
    expect(server.calls).toEqual([{ url: '/api/sessions', auth: 'Bearer k1' }]);
    const sent = new Headers(server.fetch.mock.calls[0][1]?.headers);
    expect(sent.get('content-type')).toBe('application/json');
    expect(prompt).not.toHaveBeenCalled();
  });

  it('on 401 prompts once, stores the key, retries, and reuses it afterwards', async () => {
    const server = fakeServer('secret-key');
    const storage = new MemoryStorage();
    const prompt = vi.fn(() => '  secret-key  ');
    const client = createClient({ fetch: server.fetch, storage, prompt });

    const first = await client.apiFetch('/api/status');
    expect(first.status).toBe(200);
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(storage.getItem('ruvbot.apiKey')).toBe('secret-key');
    expect(server.calls.map((c) => c.auth)).toEqual([null, 'Bearer secret-key']);

    const second = await client.apiFetch('/api/sessions/abc/chat', { method: 'POST', body: '{}' });
    expect(second.status).toBe(200);
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(server.calls[2]).toEqual({ url: '/api/sessions/abc/chat', auth: 'Bearer secret-key' });
  });

  it('never puts the key in the URL', async () => {
    const server = fakeServer('secret-key');
    const client = createClient({ fetch: server.fetch, storage: new MemoryStorage(), prompt: () => 'secret-key' });
    await client.apiFetch('/api/status');
    for (const c of server.calls) expect(c.url).not.toContain('secret-key');
  });

  it('shares one prompt between concurrent 401s', async () => {
    const server = fakeServer('k');
    const prompt = vi.fn(() => 'k');
    const client = createClient({ fetch: server.fetch, storage: new MemoryStorage(), prompt });
    const results = await Promise.all([client.apiFetch('/api/status'), client.apiFetch('/api/sessions')]);
    expect(results.map((r) => r.status)).toEqual([200, 200]);
    expect(prompt).toHaveBeenCalledTimes(1);
  });

  it('returns the 401 when the user cancels the prompt', async () => {
    const server = fakeServer('k');
    const storage = new MemoryStorage();
    const client = createClient({ fetch: server.fetch, storage, prompt: () => null });
    const res = await client.apiFetch('/api/status');
    expect(res.status).toBe(401);
    expect(storage.getItem('ruvbot.apiKey')).toBeNull();
    expect(server.calls).toHaveLength(1);
  });

  it('drops a rejected stored key and asks again', async () => {
    const server = fakeServer('new');
    const storage = new MemoryStorage();
    storage.setItem('ruvbot.apiKey', 'old');
    const prompt = vi.fn(() => 'new');
    const client = createClient({ fetch: server.fetch, storage, prompt });
    const res = await client.apiFetch('/api/status');
    expect(res.status).toBe(200);
    expect(prompt).toHaveBeenCalledTimes(1);
    expect(storage.getItem('ruvbot.apiKey')).toBe('new');
  });

  it('is a no-op when the server has no key configured', async () => {
    const server = fakeServer(null);
    const prompt = vi.fn();
    const client = createClient({ fetch: server.fetch, storage: new MemoryStorage(), prompt });
    expect((await client.apiFetch('/api/status')).status).toBe(200);
    expect((await client.apiFetch('/health')).status).toBe(200);
    expect(prompt).not.toHaveBeenCalled();
    expect(server.calls.every((c) => c.auth === null)).toBe(true);
  });

  it('does not attach the key to non-/api requests', async () => {
    const server = fakeServer(null);
    const storage = new MemoryStorage();
    storage.setItem('ruvbot.apiKey', 'k');
    const client = createClient({ fetch: server.fetch, storage, prompt: vi.fn() });
    await client.apiFetch('/health');
    await client.apiFetch('https://example.com/api/x');
    expect(server.calls.map((c) => c.auth)).toEqual([null, null]);
  });

  it('routes every /api/* call in the page through apiFetch (no bare fetch to /api)', () => {
    const page = html.replace(/<script id="ruvbot-api-client">[\s\S]*?<\/script>/, '');
    expect(page).not.toMatch(/\bfetch\(\s*['"`]\/api/);
    expect(page).not.toMatch(/EventSource/);
    expect(page).toMatch(/apiFetch\('\/api\/sessions'/);
  });
});
