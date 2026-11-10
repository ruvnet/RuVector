#!/usr/bin/env node
/**
 * 'neural-q' learning algorithm (optional peer ruv-fann).
 *
 * 1. Real import: with ruv-fann installed, neural-q trains an MLP and routes states it never
 *    saw better than tabular q-learning; the model survives export()/import().
 * 2. Fallback: with ruv-fann unresolvable (module resolve hook in a child process), neural-q
 *    behaves exactly like tabular q-learning and reports backend 'tabular'.
 * 3. CommonJS: require() of the package entry still works and lists the new algorithm.
 */
'use strict';
const assert = require('assert');
const path = require('path');
const { execFileSync } = require('child_process');

const DIST = path.join(__dirname, '..', 'dist');
const AGENTS = ['tester', 'coder', 'rust-developer', 'python-developer', 'documenter'];
const EXT = { '.ts': 'coder', '.js': 'coder', '.mjs': 'coder', '.rs': 'rust-developer', '.py': 'python-developer', '.pyi': 'python-developer', '.md': 'documenter', '.rst': 'documenter' };
const TASKS = ['fix', 'edit', 'refactor', 'test', 'docs'];
const truth = (task, ext) => (task === 'test' ? 'tester' : task === 'docs' ? 'documenter' : EXT[ext]);

// Train on every (task, ext) except this held-out set.
const HELD_OUT = ['refactor:.py', 'fix:.mjs', 'edit:.rst', 'test:.rs', 'docs:.ts', 'refactor:.rs', 'fix:.pyi', 'edit:.js'];

function stream() {
  const out = [];
  for (let pass = 0; pass < 3; pass++) {
    for (const task of TASKS) {
      for (const ext of Object.keys(EXT)) {
        const state = `${task}:${ext}`;
        if (HELD_OUT.includes(state)) continue;
        for (const action of AGENTS) {
          out.push({ state, action, reward: action === truth(task, ext) ? 1 : -0.5, nextState: state, done: true });
        }
      }
    }
  }
  return out;
}

function accuracy(engine, states) {
  let ok = 0;
  for (const state of states) {
    const [task, ext] = state.split(':');
    if (engine.getBestAction('agent-routing', state, AGENTS).action === truth(task, ext)) ok++;
  }
  return ok / states.length;
}

async function realImport() {
  const LearningEngine = require(path.join(DIST, 'core', 'learning-engine')).default;
  const neural = new LearningEngine();
  neural.configure('agent-routing', { algorithm: 'neural-q' });
  assert.strictEqual(await neural.enableNeural(), true, 'ruv-fann must load (devDependency)');
  const tabular = new LearningEngine();
  tabular.configure('agent-routing', { algorithm: 'q-learning' });
  for (const exp of stream()) {
    neural.update('agent-routing', exp);
    tabular.update('agent-routing', exp);
  }
  const loss = neural.trainNeural('agent-routing', 3);
  assert.ok(Number.isFinite(loss), `finite training loss, got ${loss}`);

  const best = neural.getBestAction('agent-routing', 'refactor:.py', AGENTS);
  assert.strictEqual(best.backend, 'ruv-fann');
  const neuralAcc = accuracy(neural, HELD_OUT);
  const tabularAcc = accuracy(tabular, HELD_OUT);
  const info = neural.getNeuralInfo('agent-routing');
  console.log(`  real import: backend=${info.backend} pairs=${info.pairs} loss=${loss.toFixed(4)} unseen acc neural-q=${neuralAcc.toFixed(3)} q-learning=${tabularAcc.toFixed(3)}`);
  assert.ok(neuralAcc >= 0.75, `neural-q should route most unseen states, got ${neuralAcc}`);
  assert.ok(neuralAcc > tabularAcc, `neural-q (${neuralAcc}) must beat tabular (${tabularAcc}) on unseen states`);

  // Persistence: a fresh engine restored from export() predicts the same.
  const restored = new LearningEngine();
  restored.import(JSON.parse(JSON.stringify(neural.export())));
  assert.strictEqual(await restored.enableNeural(), true);
  for (const state of HELD_OUT) {
    assert.strictEqual(
      restored.getBestAction('agent-routing', state, AGENTS).action,
      neural.getBestAction('agent-routing', state, AGENTS).action,
      `restored model disagrees on ${state}`,
    );
  }
  console.log('  ok: model round-trips through export()/import()');
}

function fallback() {
  const hook = 'data:text/javascript,' + encodeURIComponent(
    'export async function resolve(s, c, next) {' +
    ' if (s === "ruv-fann") { const e = new Error("Cannot find package \'ruv-fann\'"); e.code = "ERR_MODULE_NOT_FOUND"; throw e; }' +
    ' return next(s, c); }');
  const child = `
    const { register } = require('node:module');
    register(${JSON.stringify(hook)});
    const assert = require('assert');
    const LearningEngine = require(${JSON.stringify(path.join(DIST, 'core', 'learning-engine'))}).default;
    (async () => {
      const neural = new LearningEngine();
      neural.configure('agent-routing', { algorithm: 'neural-q' });
      assert.strictEqual(await neural.enableNeural(), false, 'ruv-fann must be unresolvable here');
      const tabular = new LearningEngine();
      tabular.configure('agent-routing', { algorithm: 'q-learning' });
      const exps = ${JSON.stringify(stream().slice(0, 200))};
      for (const e of exps) {
        assert.strictEqual(neural.update('agent-routing', e), tabular.update('agent-routing', e));
      }
      const actions = ${JSON.stringify(AGENTS)};
      for (const s of ${JSON.stringify(HELD_OUT.concat(['fix:.ts', 'edit:.rs']))}) {
        const a = neural.getBestAction('agent-routing', s, actions);
        const b = tabular.getBestAction('agent-routing', s, actions);
        assert.strictEqual(a.backend, 'tabular');
        assert.strictEqual(a.action, b.action);
        assert.strictEqual(a.confidence, b.confidence);
      }
      assert.strictEqual(neural.trainNeural('agent-routing'), null);
      const pairs = neural.getNeuralInfo('agent-routing').pairs;
      assert.ok(pairs > 0, 'buffer is still recorded for a later install');
      console.log('  fallback: no ruv-fann -> identical to q-learning, ' + pairs + ' pairs buffered');
    })().catch(e => { console.error(e); process.exit(1); });
  `;
  process.stdout.write(execFileSync(process.execPath, ['-e', child], { encoding: 'utf8' }));
}

function commonjs() {
  const ruvector = require(path.join(__dirname, '..'));
  assert.strictEqual(typeof ruvector.LearningEngine, 'function', 'LearningEngine exported from the CJS entry');
  const names = ruvector.LearningEngine.getAlgorithms().map(a => a.algorithm);
  assert.ok(names.includes('neural-q') && names.includes('q-learning') && names.length === 10, names.join(','));
  console.log('  ok: require() entry works, algorithms=' + names.length);
}

(async () => {
  console.log('neural-q');
  commonjs();
  fallback();
  await realImport();
  console.log('neural-q: all tests passed');
})().catch(e => { console.error(e); process.exit(1); });
