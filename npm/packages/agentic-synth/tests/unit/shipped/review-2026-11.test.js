/**
 * Regression tests for the shipped TypeScript sources (built to dist/).
 * The older unit suites exercise legacy src/*.js files that dist/ does not
 * contain, so these import the published entry point directly.
 * No real LLM calls: fetch and the Gemini client are stubbed.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import {
  AgenticSynth,
  EventGenerator,
  ModelRouter,
  StructuredGenerator,
  TimeSeriesGenerator
} from '../../../dist/index.js';

const pkgRoot = join(dirname(fileURLToPath(import.meta.url)), '../../..');
const FAKE_GEMINI = 'AIzaFAKE-gemini-key-000000';
const FAKE_OPENROUTER = 'sk-or-FAKE-openrouter-key-0000';

function okResponse(content) {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => ({ choices: [{ message: { content } }] })
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe('model routing', () => {
  it('uses the model the caller asked for even when it is not in the built-in table', () => {
    const router = new ModelRouter({ defaultProvider: 'openrouter', providerKeys: {} });
    const route = router.selectModel({
      provider: 'openrouter',
      preferredModel: 'openai/gpt-4o-mini',
      capabilities: ['text', 'json']
    });
    expect(route.provider).toBe('openrouter');
    expect(route.model).toBe('openai/gpt-4o-mini');
  });
});

describe('parsing model output', () => {
  const structured = new StructuredGenerator({ provider: 'gemini', apiKey: 'x' });

  it('ignores bracketed prose after a fenced JSON array', () => {
    const out = structured.parseResult('```json\n[{"n":1}]\n```\nNote: values in [0,10]', {});
    expect(out).toEqual([{ n: 1 }]);
  });

  it('handles brackets inside JSON strings', () => {
    const out = structured.parseResult('Here: [{"s":"a]b[c"}] done [x]', {});
    expect(out).toEqual([{ s: 'a]b[c' }]);
  });

  it('skips a bracketed citation before the record array', () => {
    const out = structured.parseResult('See [1] above: [{"n":1},{"n":2}]', {});
    expect(out).toEqual([{ n: 1 }, { n: 2 }]);
  });

  it('accepts JSON Schema "integer" fields holding integers and rejects floats', () => {
    const schema = { n: { type: 'integer', required: true } };
    expect(structured.parseResult('[{"n":3}]', { schema })).toEqual([{ n: 3 }]);
    expect(() => structured.parseResult('[{"n":3.5}]', { schema })).toThrow(/integer/);
  });

  it('time-series output keeps the normalised ISO timestamp', () => {
    const ts = new TimeSeriesGenerator({ provider: 'gemini', apiKey: 'x' });
    const out = ts.parseResult('[{"timestamp":1700000000000,"value":1}]', {});
    expect(out[0].timestamp).toBe(new Date(1700000000000).toISOString());
  });
});

describe('local generation', () => {
  it('poisson distribution always yields the requested number of events', async () => {
    const gen = new EventGenerator({ provider: 'gemini', apiKey: 'x' });
    for (let i = 0; i < 50; i++) {
      const events = await gen.generateLocal({ count: 20, distribution: 'poisson' });
      expect(events).toHaveLength(20);
    }
  });
});

describe('option validation', () => {
  it.each([Number.NaN, Number.POSITIVE_INFINITY, 2.5, 0])('rejects count=%s', async (count) => {
    const gen = new StructuredGenerator({ provider: 'gemini', apiKey: 'x' });
    await expect(gen.generate({ count, schema: { a: { type: 'string' } } })).rejects.toThrow(/Count/);
  });

  it('generateBatch with concurrency 0 terminates', async () => {
    const gen = new StructuredGenerator({ provider: 'gemini', apiKey: 'x' });
    vi.spyOn(gen, 'generate').mockResolvedValue({ data: [], metadata: {} });
    const results = await gen.generateBatch([{}, {}, {}], 0);
    expect(results).toHaveLength(3);
  });
});

describe('OpenRouter calls', () => {
  it('sends the configured apiKey, with a timeout signal', async () => {
    vi.stubEnv('OPENROUTER_API_KEY', '');
    const fetchMock = vi.fn().mockResolvedValue(okResponse('[{"a":"x"}]'));
    vi.stubGlobal('fetch', fetchMock);
    const synth = new AgenticSynth({
      provider: 'openrouter',
      apiKey: FAKE_OPENROUTER,
      model: 'anthropic/claude-3.5-sonnet',
      enableFallback: false
    });
    const res = await synth.generateStructured({ count: 1, schema: { a: { type: 'string' } } });
    expect(res.data).toEqual([{ a: 'x' }]);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/^https:\/\/openrouter\.ai\//);
    expect(init.headers.Authorization).toBe(`Bearer ${FAKE_OPENROUTER}`);
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it('never sends the Gemini key to OpenRouter', async () => {
    vi.stubEnv('GEMINI_API_KEY', FAKE_GEMINI);
    vi.stubEnv('OPENROUTER_API_KEY', FAKE_OPENROUTER);
    const fetchMock = vi.fn().mockResolvedValue(okResponse('[{"a":"x"}]'));
    vi.stubGlobal('fetch', fetchMock);

    const fresh = new AgenticSynth({ provider: 'openrouter', enableFallback: false });
    await fresh.generateStructured({ count: 1, schema: { a: { type: 'string' } } });

    const switched = new AgenticSynth({ provider: 'gemini' });
    switched.configure({ provider: 'openrouter', enableFallback: false });
    await switched.generateStructured({ count: 1, schema: { a: { type: 'string' } } });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    for (const [, init] of fetchMock.mock.calls) {
      expect(init.headers.Authorization).toBe(`Bearer ${FAKE_OPENROUTER}`);
    }
  });
});

describe('streaming', () => {
  it('yields the parsed items once the stream ends', async () => {
    const gen = new StructuredGenerator({ provider: 'gemini', apiKey: 'x', streaming: true });
    const chunks = ['[{"a":', '"x"},{"a":"y"}]'];
    gen.gemini = {
      getGenerativeModel: () => ({
        generateContentStream: async () => ({
          stream: (async function* () {
            for (const c of chunks) yield { text: () => c };
          })()
        })
      })
    };
    const items = [];
    for await (const item of gen.generateStream({ count: 2, schema: { a: { type: 'string' } } })) {
      items.push(item);
    }
    expect(items).toEqual([{ a: 'x' }, { a: 'y' }]);
  });
});

describe('CLI secret handling', () => {
  const run = (args, extraEnv = {}) =>
    execFileSync('node', [join(pkgRoot, 'bin/cli.js'), ...args], {
      cwd: pkgRoot,
      env: { ...process.env, OPENROUTER_API_KEY: '', GEMINI_API_KEY: FAKE_GEMINI, ...extraEnv },
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe']
    });

  it('`config` does not print the API key', () => {
    const out = run(['config']);
    expect(out).toContain('apiKey');
    expect(out).not.toContain(FAKE_GEMINI);
  });

  it('`doctor --verbose` does not print the key prefix', () => {
    let out;
    try {
      out = run(['doctor', '--verbose']);
    } catch (e) {
      out = `${e.stdout}`;
    }
    expect(out).not.toContain(FAKE_GEMINI.substring(0, 10));
  });
});
