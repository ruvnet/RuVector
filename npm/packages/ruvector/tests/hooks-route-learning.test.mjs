/**
 * Regression test for issue #517: `ruvector hooks route` never returned
 * learned routing (always default mapping / confidence 0) because the
 * Q-pattern state keys written by the learning hooks did not match the
 * state keys route() reads, and no path ever wrote agent-name actions.
 *
 * Covers the full loop via real CLI invocations in an isolated temp project:
 *   1. sane fallback when nothing has been learned,
 *   2. learned routing from a seeded .ruvector/intelligence.json,
 *   3. trajectory-begin/trajectory-end writing the agent-routing pattern
 *      that a subsequent `hooks route` picks up (cross-process).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CLI = path.join(__dirname, '..', 'bin', 'cli.js');

function makeProject(intelligence) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-route-'));
  fs.mkdirSync(path.join(dir, '.ruvector'), { recursive: true });
  fs.writeFileSync(
    path.join(dir, '.ruvector', 'intelligence.json'),
    JSON.stringify(intelligence ?? {}, null, 2)
  );
  return dir;
}

function cli(cwd, args) {
  const out = execFileSync(process.execPath, [CLI, ...args], {
    cwd,
    encoding: 'utf8',
    timeout: 30000,
    env: { ...process.env, FORCE_COLOR: '0', NO_COLOR: '1' },
  });
  return JSON.parse(out);
}

test('hooks route falls back to default mapping when nothing learned', (t) => {
  const dir = makeProject({});
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  const res = cli(dir, ['hooks', 'route', 'fix a failing test', '--file', 'src/index.ts']);
  assert.equal(res.recommended, 'typescript-developer');
  assert.equal(res.confidence, 0);
  assert.match(res.reasoning, /default for ts files/);
});

test('hooks route returns learned agent from persisted Q-patterns', (t) => {
  const dir = makeProject({
    patterns: {
      'fix:.ts|tester': { state: 'fix:.ts', action: 'tester', q_value: 0.85, visits: 12, last_update: 0 },
    },
    stats: { total_patterns: 1 },
  });
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  const res = cli(dir, ['hooks', 'route', 'fix a failing test', '--file', 'src/index.ts']);
  assert.equal(res.recommended, 'tester');
  assert.ok(res.confidence > 0.5, `confidence should reflect learned q-value, got ${res.confidence}`);
  assert.match(res.reasoning, /learned/);
  assert.doesNotMatch(res.reasoning, /default/);
});

test('trajectory-end writes the routing pattern route() reads (cross-process loop)', (t) => {
  const dir = makeProject({});
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  const begin = cli(dir, [
    'hooks', 'trajectory-begin',
    '-c', 'fix a failing test',
    '-a', 'tester',
    '-f', 'src/index.ts',
  ]);
  assert.equal(begin.success, true);

  const end = cli(dir, ['hooks', 'trajectory-end', '--success']);
  assert.equal(end.success, true);
  assert.equal(end.learned_route, 'fix:.ts|tester');

  // The learned outcome must now influence routing.
  const res = cli(dir, ['hooks', 'route', 'fix a failing test', '--file', 'src/index.ts']);
  assert.equal(res.recommended, 'tester');
  assert.ok(res.confidence > 0.5, `expected non-zero learned confidence, got ${res.confidence}`);
  assert.match(res.reasoning, /learned/);

  // A different task type is untouched and still falls back sanely.
  const other = cli(dir, ['hooks', 'route', 'refactor the parser', '--file', 'src/index.ts']);
  assert.equal(other.confidence, 0);
  assert.match(other.reasoning, /default/);

  // The persisted pattern lives in the namespace engine.route() imports
  // (state `taskType:ext`, action = agent name).
  const saved = JSON.parse(fs.readFileSync(path.join(dir, '.ruvector', 'intelligence.json'), 'utf8'));
  assert.ok(saved.patterns['fix:.ts|tester'], 'pattern persisted under canonical state key');
  assert.ok(saved.patterns['fix:.ts|tester'].q_value > 0.5);
});

// ---------------------------------------------------------------------------
// 0.3.4 self-learning review regressions
// ---------------------------------------------------------------------------
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';

function cliRaw(cwd, args) {
  try {
    return { code: 0, out: execFileSync(process.execPath, [CLI, ...args], { cwd, encoding: 'utf8', timeout: 30000, env: { ...process.env, FORCE_COLOR: '0', NO_COLOR: '1' } }) };
  } catch (e) {
    return { code: e.status, out: String(e.stdout || '') };
  }
}

test('hooks learn rewards reach hooks route', (t) => {
  const dir = makeProject({});
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  for (let i = 0; i < 6; i++) cli(dir, ['hooks', 'learn', '-s', 'docs:.md', '-a', 'tech-writer', '-r', '1']);
  const res = cli(dir, ['hooks', 'route', 'document the API', '--file', 'README.md']);
  assert.equal(res.recommended, 'tech-writer');
  assert.ok(res.confidence > 0.5);
});

test('an agent with a failure record does not beat the default', (t) => {
  const dir = makeProject({});
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  for (let i = 0; i < 5; i++) {
    cli(dir, ['hooks', 'trajectory-begin', '-c', 'refactor module', '-a', 'bad-agent', '-f', 'src/m.rs']);
    cli(dir, ['hooks', 'trajectory-end', '--quality', '0']);
  }
  const res = cli(dir, ['hooks', 'route', 'refactor module', '--file', 'src/m.rs']);
  assert.notEqual(res.recommended, 'bad-agent');
});

test('non-finite rewards and qualities are rejected, not stored as null', (t) => {
  const dir = makeProject({});
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const r = cliRaw(dir, ['hooks', 'learn', '-s', 'x', '-a', 'x', '-r', 'abc']);
  assert.notEqual(r.code, 0);
  assert.match(r.out, /finite number/);
  const cfg = cliRaw(dir, ['hooks', 'learning-config', '-t', 'agent-routing', '-a', 'qlearning']);
  assert.match(cfg.out, /Unknown learning algorithm/);
  const stats = cliRaw(dir, ['hooks', 'learning-stats']);
  assert.equal(stats.code, 0);
});

test('concurrent hook processes do not lose learning updates', async (t) => {
  const dir = makeProject({});
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const N = 12;
  await Promise.all(Array.from({ length: N }, (_, i) => new Promise((resolve, reject) => {
    const p = spawn(process.execPath, [CLI, 'hooks', 'learn', '-s', `s${i}:.x`, '-a', 'a', '-r', '1'], { cwd: dir, stdio: 'ignore' });
    p.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`exit ${code}`))));
  })));
  const data = JSON.parse(fs.readFileSync(path.join(dir, '.ruvector', 'intelligence.json'), 'utf8'));
  const kept = Object.keys(data.patterns).filter((k) => /^s\d+:\.x\|a$/.test(k)).length;
  assert.equal(kept, N);
  assert.ok(!fs.existsSync(path.join(dir, '.ruvector', 'intelligence.json.lock')), 'lock file left behind');
});

test('LearningEngine: double-q uses both tables, DT does not reward failure, config validated', () => {
  const require = createRequire(import.meta.url);
  const LearningEngine = require('../dist/core/learning-engine.js').default;

  const dq = new LearningEngine();
  dq.configure('agent-routing', { algorithm: 'double-q' });
  dq.update('agent-routing', { state: 's', action: 'good', reward: 1, nextState: 's', done: true, timestamp: 0 });
  for (let i = 0; i < 50; i++) {
    assert.equal(dq.getBestAction('agent-routing', 's', ['bad', 'good']).action, 'good');
  }

  const dt = new LearningEngine();
  dt.configure('agent-routing', { algorithm: 'decision-transformer' });
  for (let i = 0; i < 5; i++) {
    dt.update('agent-routing', { state: 's', action: 'fail', reward: -1, nextState: 's', done: true, timestamp: 0 });
  }
  assert.equal(dt.getBestAction('agent-routing', 's', ['fail', 'unseen']).action, 'unseen');

  assert.throws(() => dq.configure('agent-routing', { algorithm: 'qlearning' }), /Unknown learning algorithm/);
  assert.throws(() => dq.configure('agent-routing', { learningRate: 5 }), /\[0, 1\]/);
  assert.throws(() => dq.update('agent-routing', { state: 's', action: 'a', reward: NaN, nextState: 's', done: true, timestamp: 0 }), /finite/);
});
