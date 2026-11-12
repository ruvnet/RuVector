#!/usr/bin/env node

const assert = require('assert');
const meta = require('../dist');

async function main() {
  const capabilities = await meta.getMetaHarnessCapabilities();
  assert.strictEqual(capabilities.length, 9);
  assert.deepStrictEqual(
    capabilities.filter((capability) => !capability.available),
    [],
    'all pinned MetaHarness dependencies must load from the published SDK surface',
  );

  const route = await meta.routeWithMetaHarness({
    rows: [{ embedding: [1, 0], scores: { cheap: 0.91, frontier: 0.99 } }],
    prices: { cheap: 1, frontier: 20 },
    queryEmbedding: [1, 0],
    qualityBar: 0.9,
  });
  assert.strictEqual(route.id, 'cheap');
  assert.strictEqual(route.metBar, true);

  const decision = await meta.evaluateMetaHarnessPromotion({
    baseline: { primary: 0.8, noopRate: 0.2, costPerWin: 2, regressed: false },
    candidate: { primary: 0.81, noopRate: 0.1, costPerWin: 1.9, regressed: false },
    anchor: { baseline: 0.7, candidate: 0.7 },
  });
  assert.deepStrictEqual(decision, { promote: true, reasons: [] });

  await assert.rejects(
    () => meta.runMetaHarnessDarwin({}),
    /requires \{ execute: true \}/,
    'Darwin must never execute through an accidental SDK call',
  );

  const scan = await meta.scanMetaHarnessRewardHacks({
    instance_id: 'fixture-1',
    model: 'fixture',
    tier: 'cheap',
    resolved: false,
    messages: [],
    model_patch: '',
  });
  assert.deepStrictEqual(scan, { clean: true, findings: [] });

  console.log('MetaHarness SDK dependency integration checks passed');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
