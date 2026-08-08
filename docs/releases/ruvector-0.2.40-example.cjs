'use strict';

const assert = require('node:assert/strict');
const {
  getMetaHarnessCapabilities,
  routeWithMetaHarness,
  evaluateMetaHarnessPromotion,
} = require('ruvector');

async function main() {
  const capabilities = await getMetaHarnessCapabilities();
  assert.equal(capabilities.length, 9);
  assert.equal(capabilities.every((capability) => capability.available), true);

  const route = await routeWithMetaHarness({
    rows: [
      {
        embedding: [1, 0],
        scores: { economical: 0.91, frontier: 0.99 },
      },
    ],
    prices: { economical: 1, frontier: 20 },
    queryEmbedding: [1, 0],
    qualityBar: 0.9,
  });
  assert.equal(route.id, 'economical');

  const decision = await evaluateMetaHarnessPromotion({
    baseline: {
      primary: 0.8,
      noopRate: 0.2,
      costPerWin: 2,
      regressed: false,
    },
    candidate: {
      primary: 0.81,
      noopRate: 0.1,
      costPerWin: 1.9,
      regressed: false,
    },
    anchor: {
      baseline: 0.7,
      candidate: 0.7,
    },
  });
  assert.equal(decision.promote, true);

  console.log({
    availableCapabilities: capabilities.length,
    selectedRoute: route,
    promotionDecision: decision,
  });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
