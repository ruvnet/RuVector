#!/usr/bin/env node
// Does warm-starting from registry.nearest() reach a target loss faster than a random init?
//
// SYNTHETIC task family (labelled as such): y = c * sin(a * x0 + b) + d * x1 on x in [-1, 1]^2,
// with a in [1, 3], b in [-1, 1], c in [0.5, 1], d in [-0.5, 0.5]. The task embedding is the
// centred parameter vector. A library of past tasks is trained and saved; each new task is then
// trained from (1) a fresh random init, (2) the nearest() compatible snapshot, and (3) a random
// library snapshot (control: "any pretrained model" vs "the nearest one").
//
// Usage: node eval/warmstart.mjs [--out results.json]
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { WeightRegistry, createMlp, loadFann, meanSquaredError } from '../dist/esm/index.js';

const outPath = process.argv.includes('--out') ? process.argv[process.argv.indexOf('--out') + 1] : null;
const SEEDS = [1, 2, 3];
const LIBRARY = 40;
const NEW_TASKS = 20;
const ROWS = 128;
const TARGET_MSE = 0.01;
const MAX_EPOCHS = 600;
const CHUNK = 5;
const LR = 0.1;
const SIZES = [2, 16, 1];

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
function sampleTask(rand) {
  return { a: 1 + 2 * rand(), b: 2 * rand() - 1, c: 0.5 + 0.5 * rand(), d: rand() - 0.5 };
}
const embed = (t) => [t.a - 2, t.b, 4 * (t.c - 0.75), 2 * t.d];
function dataset(task, rand, rows = ROWS) {
  const x = new Float64Array(rows * 2), y = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    const x0 = 2 * rand() - 1, x1 = 2 * rand() - 1;
    x[2 * r] = x0; x[2 * r + 1] = x1;
    y[r] = task.c * Math.sin(task.a * x0 + task.b) + task.d * x1;
  }
  return { x, y };
}

/** Train in CHUNK-epoch calls until train MSE <= target; returns epochs used (or null) and the curve. */
function trainToTarget(rt, model, data, test) {
  const curve = { 0: meanSquaredError(rt, model, test.x, test.y, ROWS) };
  let m = model;
  let reached = meanSquaredError(rt, m, data.x, data.y, ROWS) <= TARGET_MSE ? 0 : null;
  for (let e = CHUNK; e <= MAX_EPOCHS && reached === null; e += CHUNK) {
    m = rt.train(m, { inputs: data.x, targets: data.y, rows: ROWS, epochs: CHUNK, learningRate: LR, optimizer: 'sgd' }).model;
    if ([25, 50, 100].includes(e)) curve[e] = meanSquaredError(rt, m, test.x, test.y, ROWS);
    if (meanSquaredError(rt, m, data.x, data.y, ROWS) <= TARGET_MSE) reached = e;
  }
  for (const e of [25, 50, 100]) if (curve[e] === undefined) curve[e] = meanSquaredError(rt, m, test.x, test.y, ROWS);
  return { epochs: reached, curve };
}

const median = (xs) => { const s = xs.slice().sort((a, b) => a - b); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

const rt = await loadFann();
const arms = ['random-init', 'nearest', 'random-snapshot'];
const all = Object.fromEntries(arms.map((a) => [a, []]));
const perSeed = [];

for (const seed of SEEDS) {
  const rand = rng(seed * 104729);
  const reg = await WeightRegistry.open({ dimensions: 4, store: 'flat' });
  const ids = [];
  for (let i = 0; i < LIBRARY; i++) {
    const task = sampleTask(rand);
    const data = dataset(task, rand);
    const init = createMlp(SIZES, { seed: seed * 1000 + i });
    const fit = rt.train(init, { inputs: data.x, targets: data.y, rows: ROWS, epochs: 300, learningRate: LR, optimizer: 'sgd' });
    ids.push((await reg.save(fit.model, embed(task), { task: `lib-${i}`, outcomes: [1 - Math.min(1, fit.finalLoss)] })).id);
  }
  const seedRows = Object.fromEntries(arms.map((a) => [a, []]));
  for (let i = 0; i < NEW_TASKS; i++) {
    const task = sampleTask(rand);
    const data = dataset(task, rand);
    const test = dataset(task, rand);
    const fresh = createMlp(SIZES, { seed: seed * 5000 + i });
    const [hit] = await reg.nearest(embed(task), 1, { compatibleWith: fresh });
    const other = reg.get(ids[Math.floor(rand() * ids.length)]);
    const starts = { 'random-init': fresh, nearest: hit.record.model, 'random-snapshot': other.model };
    for (const arm of arms) {
      const r = trainToTarget(rt, starts[arm], data, test);
      seedRows[arm].push(r);
      all[arm].push(r);
    }
  }
  perSeed.push({ seed, ...Object.fromEntries(arms.map((a) => [a, {
    reached: seedRows[a].filter((r) => r.epochs !== null).length,
    medianEpochs: median(seedRows[a].map((r) => r.epochs ?? MAX_EPOCHS)),
    testMse0: mean(seedRows[a].map((r) => r.curve[0])),
    testMse50: mean(seedRows[a].map((r) => r.curve[50])),
  }])) });
  console.log(JSON.stringify(perSeed.at(-1)));
  await reg.close();
}

const summary = Object.fromEntries(arms.map((a) => [a, {
  tasks: all[a].length,
  reachedTarget: all[a].filter((r) => r.epochs !== null).length,
  medianEpochsToTarget: median(all[a].map((r) => r.epochs ?? MAX_EPOCHS)),
  meanTestMse: { 0: mean(all[a].map((r) => r.curve[0])), 25: mean(all[a].map((r) => r.curve[25])), 50: mean(all[a].map((r) => r.curve[50])), 100: mean(all[a].map((r) => r.curve[100])) },
}]));
// Paired: fraction of tasks where nearest needed fewer epochs than random init.
const wins = all.nearest.filter((r, i) => (r.epochs ?? MAX_EPOCHS) < (all['random-init'][i].epochs ?? MAX_EPOCHS)).length;
const pairWins = (a, b) => {
  const ea = (r) => r.epochs ?? MAX_EPOCHS;
  const w = all[a].filter((r, i) => ea(r) < ea(all[b][i])).length;
  const l = all[a].filter((r, i) => ea(r) > ea(all[b][i])).length;
  return { wins: w, losses: l, ties: all[a].length - w - l };
};
const report = {
  task: 'synthetic: y = c*sin(a*x0+b) + d*x1', model: SIZES, library: LIBRARY, newTasksPerSeed: NEW_TASKS, seeds: SEEDS,
  targetTrainMse: TARGET_MSE, maxEpochs: MAX_EPOCHS, learningRate: LR, optimizer: 'sgd',
  summary, nearestFasterThanRandomInit: `${wins}/${all.nearest.length}`,
  paired: { 'nearest-vs-random-init': pairWins('nearest', 'random-init'), 'nearest-vs-random-snapshot': pairWins('nearest', 'random-snapshot') }, perSeed,
};
console.log(JSON.stringify(summary, null, 2), '\npaired:', JSON.stringify(report.paired));
if (outPath) writeFileSync(resolve(outPath), JSON.stringify(report, null, 2) + '\n');
