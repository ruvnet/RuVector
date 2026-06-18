/**
 * Regression tests for #553: processInstantLearning was a no-op stub, so
 * learn-from-feedback never updated micro-LoRA weights and deltaNorm stayed
 * 0.000000 forever in downstream consumers.
 */

const { test, describe } = require('node:test');
const assert = require('node:assert');

const { SonaCoordinator } = require('../dist/cjs/sona.js');

function signal(quality, requestId = 'req-1', type = 'explicit') {
  return { requestId, quality, type, timestamp: new Date() };
}

describe('SONA instant learning (#553)', () => {
  test('a single feedback signal produces a non-zero micro-LoRA delta', () => {
    const sona = new SonaCoordinator();
    assert.strictEqual(sona.microLoraDeltaNorm(), 0, 'B is zero-init: delta must start at exactly 0');

    sona.recordSignal(signal(0.9));

    const delta = sona.microLoraDeltaNorm();
    assert.ok(delta > 0, `deltaNorm must be non-zero after one signal, got ${delta}`);
    assert.strictEqual(sona.stats().microLora.updates, 1);
    assert.strictEqual(sona.stats().microLora.deltaNorm, delta);
  });

  test('applyMicroLora output changes after feedback', () => {
    const sona = new SonaCoordinator();
    const input = Array.from({ length: 64 }, (_, i) => Math.sin(i + 1));

    const before = sona.applyMicroLora(input);
    assert.deepStrictEqual(before, input.slice(0, 64), 'untrained adapter is the identity (residual)');

    for (let i = 0; i < 5; i++) sona.recordSignal(signal(0.95, `req-${i}`));

    const after = sona.applyMicroLora(input);
    const moved = after.some((v, i) => Math.abs(v - before[i]) > 1e-12);
    assert.ok(moved, 'forward output must change after feedback updates');
  });

  test('negative feedback adapts in the opposite direction of positive', () => {
    const pos = new SonaCoordinator();
    const neg = new SonaCoordinator();
    // Same request -> same embedding; opposite reward signs.
    pos.recordSignal(signal(1.0, 'same-request'));
    neg.recordSignal(signal(0.0, 'same-request'));

    assert.ok(pos.microLoraDeltaNorm() > 0);
    assert.ok(neg.microLoraDeltaNorm() > 0, 'low quality must also adapt (unlearn), not be gated off');
  });

  test('neutral feedback (quality exactly 0.5) is a no-op', () => {
    const sona = new SonaCoordinator();
    sona.recordSignal(signal(0.5));
    assert.strictEqual(sona.microLoraDeltaNorm(), 0);
    assert.strictEqual(sona.stats().microLora.updates, 0);
  });

  test('instantLoopEnabled=false disables adaptation', () => {
    const sona = new SonaCoordinator({ instantLoopEnabled: false });
    sona.recordSignal(signal(0.9));
    assert.strictEqual(sona.microLoraDeltaNorm(), 0);
  });

  test('repeated feedback accumulates (deltaNorm grows)', () => {
    const sona = new SonaCoordinator();
    sona.recordSignal(signal(0.9, 'r1'));
    const d1 = sona.microLoraDeltaNorm();
    for (let i = 0; i < 9; i++) sona.recordSignal(signal(0.9, `r${i + 2}`));
    const d10 = sona.microLoraDeltaNorm();
    assert.ok(d10 > d1, `delta must grow with feedback: ${d1} -> ${d10}`);
  });
});
