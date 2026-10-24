// Acceptance test for micro-LoRA learning and EWC++ through the WASM API.
//
// Build the web target first, then run from the crate directory:
//   wasm-pack build --target web --release --features wasm
//   node tests-js/lora-ewc.mjs
//
// Set SONA_WASM_PKG to point at another build directory.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgDir = process.env.SONA_WASM_PKG ?? path.resolve(here, '..', 'pkg');

const wasm = await import(pathToFileURL(path.join(pkgDir, 'ruvector_sona.js')).href);
wasm.initSync({ module: fs.readFileSync(path.join(pkgDir, 'ruvector_sona_bg.wasm')) });
const { WasmSonaEngine } = wasm;

const DIM = 64;

// ---- Vectors --------------------------------------------------------------

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (a) => Math.sqrt(dot(a, a));
const normalize = (a) => {
  const n = norm(a);
  return Float32Array.from(a, (v) => v / n);
};
const unit = (seed) => {
  const r = rng(seed);
  return normalize(Array.from({ length: DIM }, () => r() - 0.5));
};
const near = (centre, seed, weight) => {
  const noise = unit(seed);
  return normalize(Array.from(centre, (c, i) => c + weight * noise[i]));
};
const delta = (engine, x) => {
  const out = engine.applyLora(x);
  return Array.from(out, (v, i) => v - x[i]);
};

// ---- Engine helpers -------------------------------------------------------

// Configs and stats cross the boundary as JSON strings.
function makeEngine(overrides) {
  const config = JSON.parse(new WasmSonaEngine(DIM).getConfig());
  return WasmSonaEngine.withConfig(JSON.stringify({ ...config, ...overrides }));
}

function feedback(engine, x, target, score) {
  const id = engine.startTrajectory(x);
  engine.recordStepWithActivations(id, 1, target, score);
  engine.endTrajectory(id, score);
}

function alignment(engine, inputs, target) {
  let total = 0;
  for (const x of inputs) {
    const out = engine.applyLora(x);
    total += dot(out, target) / norm(out);
  }
  return total / inputs.length;
}

const stats = (engine) => JSON.parse(engine.getStats());

const results = [];
function check(name, fn) {
  fn();
  results.push(name);
  console.log(`ok - ${name}`);
}

// ---- Checks ---------------------------------------------------------------

check('one feedback moves applyLora(x) toward the step activation', () => {
  let toward = 0;
  for (let k = 0; k < 20; k++) {
    const engine = makeEngine({ micro_lora_lr: 0.02 });
    const x = unit(100 + k);
    const target = unit(900 + k);
    feedback(engine, x, target, 1.0);
    engine.runInstantCycle();
    if (dot(delta(engine, x), target) > 0) toward++;
  }
  assert.equal(toward, 20);
});

check('applyLora stays bounded under repeated feedback at a high learning rate', () => {
  const engine = makeEngine({ micro_lora_lr: 0.5 });
  const x = unit(7);
  const target = unit(8);
  for (let i = 0; i < 2000; i++) {
    feedback(engine, x, target, 1.0);
    engine.runInstantCycle();
  }
  for (let k = 0; k < 10; k++) {
    const probe = k === 0 ? x : unit(50 + k);
    const gain = norm(delta(engine, probe));
    assert.ok(Number.isFinite(gain) && gain <= 1.0001, `gain ${gain}`);
  }
});

check('learnFromFeedback still changes applyLora output', () => {
  const engine = makeEngine({});
  const x = unit(11);
  const before = engine.applyLora(x);
  engine.startTrajectory(x);
  for (let i = 0; i < 5; i++) engine.learnFromFeedback(true, 50, 0.9);
  const after = engine.applyLora(x);
  const change = before.reduce((s, v, i) => s + Math.abs(v - after[i]), 0);
  assert.ok(change > 0, 'no change after feedback');
});

// Task A, then task B on overlapping inputs with a different target.
function taskAThenB(ewcLambda) {
  const engine = makeEngine({ micro_lora_lr: 0.5, ewc_lambda: ewcLambda });
  const centreA = unit(31);
  const centreB = near(centreA, 32, 1.2); // cosine about 0.6 with task A
  const targetA = unit(33);
  const targetB = unit(34);
  const train = (centre, target, seed) => {
    for (let k = 0; k < 300; k++) {
      feedback(engine, near(centre, seed + (k % 100), 0.4), target, 1.0);
      if (k % 5 === 4) engine.runInstantCycle();
    }
  };
  const heldOutA = Array.from({ length: 30 }, (_, k) => near(centreA, 5000 + k, 0.4));
  const heldOutB = Array.from({ length: 30 }, (_, k) => near(centreB, 6000 + k, 0.4));
  const start = alignment(engine, heldOutA, targetA);
  train(centreA, targetA, 1000);
  const afterA = alignment(engine, heldOutA, targetA);
  engine.startNewTask();
  train(centreB, targetB, 2000);
  return {
    start,
    afterA,
    afterB: alignment(engine, heldOutA, targetA),
    taskB: alignment(engine, heldOutB, targetB),
    stats: stats(engine),
  };
}

const plain = taskAThenB(0);
const ewc = taskAThenB(2000);

check('feedback with step activations teaches task A', () => {
  assert.ok(plain.afterA > plain.start + 0.4, JSON.stringify(plain));
});

check('startNewTask registers an EWC++ task in both loops', () => {
  assert.equal(ewc.stats.instant_ewc_tasks, 1);
  assert.equal(ewc.stats.ewc_tasks, 1);
  assert.equal(plain.stats.instant_ewc_tasks, 0);
});

check('EWC++ reduces forgetting of task A while task B trains', () => {
  const forgotPlain = plain.afterA - plain.afterB;
  const forgotEwc = ewc.afterA - ewc.afterB;
  assert.ok(forgotPlain > 0.4, `forgot ${forgotPlain} without EWC++`);
  assert.ok(forgotEwc < forgotPlain * 0.5, `forgot ${forgotEwc} with EWC++ vs ${forgotPlain}`);
});

const f = (v) => v.toFixed(3);
console.log(
  `task A alignment: start ${f(plain.start)}, after A ${f(plain.afterA)}; ` +
    `after B without EWC++ ${f(plain.afterB)} (task B ${f(plain.taskB)}), ` +
    `with EWC++ ${f(ewc.afterB)} (task B ${f(ewc.taskB)})`,
);
console.log(`${results.length}/${results.length} checks passed`);
