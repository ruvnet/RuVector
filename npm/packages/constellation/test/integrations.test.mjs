import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { routeWithMetaHarness, createMcpConfiguration } from '../src/integrations.mjs';

const input = () => ({ rows: [{ embedding: [1, 0], scores: { small: 0.9, large: 1 } }], prices: { small: 1, large: 10 }, embedding: [1, 0], qualityBar: 0.8 });
const loader = async () => ({ Router: { fromExamples(rows, prices, options) {
  assert.equal(rows[0].scores.small, 0.9);
  assert.equal(options.qualityBar, 0.8);
  return { route(embedding) { assert.deepEqual(embedding, [1, 0]); return { id: 'small', predictedQuality: 0.9, costPerMTok: prices.small, metBar: true }; } };
} } });

test('uses real Router API contract without executing a model', async () => {
  assert.deepEqual(await routeWithMetaHarness(input(), { loader }), { id: 'small', predictedQuality: 0.9, costPerMTok: 1, metBar: true });
});
test('missing optional module produces actionable failure; other loader errors propagate', async () => {
  await assert.rejects(routeWithMetaHarness(input(), { loader: async () => { throw Object.assign(new Error('missing'), { code: 'ERR_MODULE_NOT_FOUND' }); } }), { code: 'OPTIONAL_INTEGRATION_UNAVAILABLE' });
  const failure = new Error('broken module');
  await assert.rejects(routeWithMetaHarness(input(), { loader: async () => { throw failure; } }), error => error === failure);
});
test('rejects malformed measured data before loading optional code', async () => {
  const mutations = [
    x => { x.rows = []; }, x => { x.rows[0].scores.small = NaN; }, x => { x.rows[0].scores.small = 1.1; },
    x => { x.rows[0].embedding = [1]; }, x => { x.embedding = [Infinity, 0]; },
    x => { delete x.prices.small; }, x => { x.prices.small = -1; }, x => { x.prices.large = Infinity; },
    x => { x.qualityBar = NaN; }, x => { x.k = 0; },
  ];
  for (const mutate of mutations) {
    const value = input(); mutate(value);
    await assert.rejects(routeWithMetaHarness(value, { loader: () => { assert.fail('must validate before loading'); } }), TypeError);
  }
});
test('never returns a below-bar or invalid recommendation', async () => {
  for (const result of [
    { id: 'small', predictedQuality: 0.7, costPerMTok: 1, metBar: true },
    { id: 'small', predictedQuality: 0.9, costPerMTok: 1, metBar: false },
    { id: 'unknown', predictedQuality: 1, costPerMTok: 0, metBar: true },
    { id: 'small', predictedQuality: NaN, costPerMTok: 1, metBar: true },
  ]) await assert.rejects(routeWithMetaHarness(input(), { loader: async () => ({ Router: { fromExamples: () => ({ route: () => result }) } }) }));
});
test('creates inert config from installed executables and preserves unrelated configuration', () => {
  const directory = mkdtempSync(join(tmpdir(), 'constellation-integrations-'));
  try {
    const executable = join(directory, 'installed-tool');
    writeFileSync(executable, '#!/bin/sh\nexit 99\n', { mode: 0o700 });
    const existing = { preference: 'keep', mcpServers: { other: { command: '/other' } } };
    const config = createMcpConfiguration({ existing, executables: { constellation: executable, ruvector: executable, ruflo: executable }, reportPath: '/reports/inventory.json' });
    assert.deepEqual(config.mcpServers['ruvector-constellation'], { command: executable, args: ['mcp', '/reports/inventory.json'] });
    assert.deepEqual(config.mcpServers.ruflo.args, ['mcp', 'start']);
    assert.equal(config.preference, 'keep');
    assert.deepEqual(Object.keys(existing.mcpServers), ['other']);
    assert.throws(() => createMcpConfiguration({ existing: config, executables: { ruflo: executable } }), /already exists/);
    assert.throws(() => createMcpConfiguration({ executables: { ruflo: 'npx' } }), /absolute/);
    assert.throws(() => createMcpConfiguration({ executables: { ruflo: directory } }), /file/);
    assert.throws(() => createMcpConfiguration({ executables: { constellation: executable }, reportPath: 'relative.json' }), /absolute/);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('rejects sparse query and training vectors', async () => {
  const sparse = [1, , 2];
  const valid = { rows: [{ embedding: [1, 0, 2], scores: { good: 0.9 } }], prices: {good: 1}, embedding: [1, 0, 2] };
  await assert.rejects(routeWithMetaHarness({...valid, embedding: sparse}), /finite vector/);
  await assert.rejects(routeWithMetaHarness({...valid, rows: [{embedding:sparse, scores:{good:0.9}}]}), /finite dimensions/);
});
