// Invalid sizes and vectors must surface as JavaScript errors (or be ignored),
// never abort the process.
//
// Local build:        node --test test/
// Published package:  SONA_BINDING=@ruvector/sona node --test test/
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');

const { SonaEngine } = process.env.SONA_BINDING
  ? require(require.resolve(process.env.SONA_BINDING, { paths: [process.cwd()] }))
  : require(path.join(__dirname, '..', 'index.js'));

const vec = (n, v = 0.1) => new Array(n).fill(v);

test('applyMicroLora rejects a vector of the wrong length', () => {
  assert.throws(() => new SonaEngine(16).applyMicroLora(vec(5)));
  assert.throws(() => new SonaEngine(16).applyMicroLora(vec(64)));
  assert.strictEqual(new SonaEngine(16).applyMicroLora(vec(16)).length, 16);
});

test('applyBaseLora rejects a vector of the wrong length or survives a bad layer', () => {
  assert.throws(() => new SonaEngine(16).applyBaseLora(0, vec(5)));
  assert.strictEqual(new SonaEngine(16).applyBaseLora(4e9, vec(16)).length, 16);
});

test('constructor and withConfig reject out-of-range sizes', () => {
  assert.throws(() => new SonaEngine(0));
  assert.throws(() => new SonaEngine(1 << 20));
  assert.throws(() => SonaEngine.withConfig({ hiddenDim: 0 }));
  assert.throws(() => SonaEngine.withConfig({ hiddenDim: 16, microLoraRank: 3 }));
  assert.throws(() => SonaEngine.withConfig({ hiddenDim: 16, embeddingDim: 0 }));
  assert.throws(() => SonaEngine.withConfig({ hiddenDim: 16, patternClusters: 0 }));
  assert.throws(() => SonaEngine.withConfig({ hiddenDim: 65536, baseLoraRank: 1024 }));
});

test('trajectory and pattern entry points tolerate mismatched vectors', async () => {
  const e = new SonaEngine(16);
  for (let i = 0; i < 120; i++) {
    const t = e.beginTrajectory(vec(i % 2 ? 3 : 40, 0.2));
    e.addTrajectoryStep(t, vec(7), vec(99), 0.5);
    e.setTrajectoryRoute(t, 'r');
    e.addTrajectoryContext(t, 'c');
    e.endTrajectory(t, 0.9);
  }
  e.addTrajectoryStep(123456, vec(3), vec(3), 0.1); // unknown id is ignored
  e.endTrajectory(123456, 0.5);
  e.flush();
  e.tick();
  e.forceLearn();
  await e.forceLearnAsync();
  e.findPatterns(vec(3), 5);
  e.findPatterns(vec(16), 4e9);
  assert.strictEqual(e.applyMicroLora(vec(16)).length, 16);
  assert.ok(typeof e.getStats() === 'string');
});
