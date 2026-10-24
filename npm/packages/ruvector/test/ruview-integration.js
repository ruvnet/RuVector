'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const { createRuViewIntegration } = require('../dist');

test('loads the real RuView SDK explicitly and preserves its registry', async () => {
  const ruview = await createRuViewIntegration();
  const tools = ruview.tools();
  assert.ok(tools.length > 0);
  assert.ok(tools.some(tool => tool.name === 'ruview_train_gate'));
  assert.ok(tools.some(tool => tool.name === 'ruview_esp32_capture'));
  assert.ok(Object.isFrozen(ruview));
});

test('real report gate rejects missing baseline and preserves synthetic labels', async () => {
  const ruview = await createRuViewIntegration();
  const missing = await ruview.training.gate({ model_score: 0.8 });
  assert.equal(missing.ok, false);
  assert.equal(missing.verdict, 'FAIL');
  const synthetic = await ruview.training.gate({
    model_score: 0.8, baseline_score: 0.5, split: 'grouped-subject',
    train_subjects: ['a'], test_subjects: ['b'], n_test: 200, data: 'synthetic',
  });
  assert.equal(synthetic.evidence, 'SYNTHETIC');
  assert.match(synthetic.statement, /SYNTHETIC/);
});

test('forwards upstream strict errors without replacing negative results', async () => {
  const relaxed = await createRuViewIntegration({ strict: false });
  const result = await relaxed.call('ruview_this_tool_does_not_exist');
  assert.equal(result.ok, false);
  const strict = await createRuViewIntegration({ strict: true });
  await assert.rejects(strict.call('ruview_this_tool_does_not_exist'), error =>
    error.name === 'RuViewError' && error.result.ok === false);
  await assert.rejects(createRuViewIntegration({ strict: 'yes' }), /must be a boolean/);
});
