#!/usr/bin/env node
/**
 * Held-out routing benchmark: tabular 'q-learning' vs 'neural-q' (ruv-fann MLP).
 *
 * Ground truth routes a state to one of 9 agents from its task type and file extension.
 * States are split into train/test with no overlap, so every test state is unseen. Both
 * engines receive the identical exploration stream (uniform random agent per visit,
 * reward +1 for the right agent, -0.5 otherwise, done=true), then route the test states
 * greedily.
 *
 *   A. route keys in the hooks' own format, e.g. "refactor:.tsx" (routeStateKey)
 *   B. free-text states, e.g. "fix src/api/users_handler.rs"
 *
 * Usage: npm run build && node scripts/neural-q-bench.js [--seeds 5] [--json]
 * Requires the optional peer `ruv-fann` (npm i ruv-fann@0.1.0).
 */
'use strict';
const { performance } = require('perf_hooks');
const LearningEngine = require('../dist/core/learning-engine').default;

const FAMILIES = {
  coder: ['.ts', '.mts', '.cts', '.js', '.mjs', '.cjs'],
  'frontend-developer': ['.tsx', '.jsx', '.vue', '.svelte', '.css', '.scss', '.html'],
  'rust-developer': ['.rs'],
  'python-developer': ['.py', '.pyi', '.ipynb'],
  'go-developer': ['.go'],
  'database-specialist': ['.sql', '.prisma'],
  'devops-engineer': ['.yml', '.yaml', '.toml', '.sh', '.tf', '.dockerfile'],
  documenter: ['.md', '.mdx', '.rst', '.txt'],
};
const AGENTS = ['tester', ...Object.keys(FAMILIES)];
const TASKS = ['fix', 'test', 'refactor', 'docs', 'edit'];
const EXT_AGENT = new Map(Object.entries(FAMILIES).flatMap(([agent, exts]) => exts.map(e => [e, agent])));
const DIRS = ['src', 'src/api', 'src/core', 'lib', 'app/components', 'pkg/server', 'scripts', 'db', 'infra', 'docs'];
const WORDS = ['user', 'auth', 'cache', 'router', 'index', 'handler', 'model', 'config', 'session', 'billing', 'search', 'queue', 'parser', 'client', 'schema', 'worker'];

function truth(task, ext) {
  if (task === 'test') return 'tester';
  if (task === 'docs') return 'documenter';
  return EXT_AGENT.get(ext);
}

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(arr, rand) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function datasetA() {
  const out = [];
  for (const task of TASKS) for (const ext of EXT_AGENT.keys()) out.push({ state: `${task}:${ext}`, agent: truth(task, ext) });
  return out;
}

function datasetB(rand, n = 400) {
  const seen = new Set();
  const out = [];
  const exts = [...EXT_AGENT.keys()];
  while (out.length < n) {
    const task = TASKS[Math.floor(rand() * TASKS.length)];
    const ext = exts[Math.floor(rand() * exts.length)];
    const dir = DIRS[Math.floor(rand() * DIRS.length)];
    const name = `${WORDS[Math.floor(rand() * WORDS.length)]}_${WORDS[Math.floor(rand() * WORDS.length)]}`;
    const state = `${task} ${dir}/${name}${ext}`;
    if (seen.has(state)) continue;
    seen.add(state);
    out.push({ state, agent: truth(task, ext) });
  }
  return out;
}

function evaluate(engine, rows) {
  let correct = 0;
  const times = [];
  for (const { state, agent } of rows) {
    const t0 = performance.now();
    const best = engine.getBestAction('agent-routing', state, AGENTS);
    times.push(performance.now() - t0);
    if (best.action === agent) correct++;
  }
  times.sort((a, b) => a - b);
  return { accuracy: correct / rows.length, p50Us: times[Math.floor(times.length / 2)] * 1000, meanUs: (times.reduce((a, b) => a + b, 0) / times.length) * 1000 };
}

async function runOnce(rows, seed, passes) {
  const rand = rng(seed);
  shuffle(rows, rand);
  const cut = Math.floor(rows.length * 0.7);
  const train = rows.slice(0, cut);
  const test = rows.slice(cut);
  const stream = [];
  for (let p = 0; p < passes; p++) {
    for (const { state, agent } of shuffle(train.slice(), rand)) {
      const action = AGENTS[Math.floor(rand() * AGENTS.length)];
      stream.push({ state, action, reward: action === agent ? 1 : -0.5, nextState: state, done: true });
    }
  }

  const result = {};
  for (const algorithm of ['q-learning', 'neural-q']) {
    const engine = new LearningEngine();
    engine.configure('agent-routing', { algorithm });
    if (algorithm === 'neural-q' && !(await engine.enableNeural())) throw new Error('ruv-fann is not installed');
    const t0 = performance.now();
    for (const exp of stream) engine.update('agent-routing', exp);
    const trainMs = performance.now() - t0;
    engine.getBestAction('agent-routing', test[0].state, AGENTS); // warm the prepared model
    result[algorithm] = { trainMs, seen: evaluate(engine, train).accuracy, unseen: evaluate(engine, test), updates: stream.length };
  }
  return { trainStates: train.length, testStates: test.length, ...result };
}

function summarize(runs, algorithm, pick) {
  const v = runs.map(r => pick(r[algorithm]));
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length);
  return { mean, sd };
}

async function main() {
  const seedsArg = process.argv.indexOf('--seeds');
  const seeds = seedsArg >= 0 ? Number(process.argv[seedsArg + 1]) : 5;
  const report = {};
  for (const [name, make, passes] of [['A:route-keys', () => datasetA(), 30], ['B:free-text', s => datasetB(rng(s ^ 0x9e3779b9)), 10]]) {
    const runs = [];
    for (let s = 1; s <= seeds; s++) runs.push(await runOnce(make(s), s, passes));
    report[name] = { seeds, trainStates: runs[0].trainStates, testStates: runs[0].testStates, updatesPerRun: runs[0]['q-learning'].updates };
    for (const alg of ['q-learning', 'neural-q']) {
      report[name][alg] = {
        unseenAccuracy: summarize(runs, alg, r => r.unseen.accuracy),
        seenAccuracy: summarize(runs, alg, r => r.seen),
        decisionP50Us: summarize(runs, alg, r => r.unseen.p50Us),
        decisionMeanUs: summarize(runs, alg, r => r.unseen.meanUs),
        trainMs: summarize(runs, alg, r => r.trainMs),
      };
    }
  }
  if (process.argv.includes('--json')) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }
  const f = ({ mean, sd }, d = 3) => `${mean.toFixed(d)} ±${sd.toFixed(d)}`;
  console.log(`agents=${AGENTS.length} (chance ${(1 / AGENTS.length).toFixed(3)})`);
  for (const [name, r] of Object.entries(report)) {
    console.log(`\n${name}: ${r.trainStates} train / ${r.testStates} unseen test states, ${r.updatesPerRun} updates, ${r.seeds} seeds`);
    for (const alg of ['q-learning', 'neural-q']) {
      const m = r[alg];
      console.log(`  ${alg.padEnd(10)} unseen acc ${f(m.unseenAccuracy)}  seen acc ${f(m.seenAccuracy)}  decision p50 ${f(m.decisionP50Us, 1)} us  mean ${f(m.decisionMeanUs, 1)} us  train ${f(m.trainMs, 0)} ms`);
    }
  }
}

main().catch(e => { console.error(e); process.exit(1); });
