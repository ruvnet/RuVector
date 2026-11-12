#!/usr/bin/env node

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const ROOT = path.join(__dirname, '..');
const CLI = path.join(ROOT, 'bin', 'cli.js');

function run(args) {
  const result = spawnSync(process.execPath, [CLI, ...args], {
    cwd: ROOT,
    encoding: 'utf8',
    env: { ...process.env, FORCE_COLOR: '0', NO_COLOR: '1' },
  });
  assert.strictEqual(result.status, 0, result.stderr);
  return result.stdout;
}

const status = JSON.parse(run(['harness', 'doctor', '--json']));
assert.deepStrictEqual(
  Object.entries(status.primitives)
    .filter(([name]) => name.startsWith('metaharness.'))
    .filter(([, capability]) => !capability.available),
  [],
);
assert.strictEqual(status.primitives['metaharness.darwin'].directDependency, true);

const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-metaharness-cli-'));
const examples = path.join(temporary, 'examples.json');
const prices = path.join(temporary, 'prices.json');
const query = path.join(temporary, 'query.json');
const evidence = path.join(temporary, 'evidence.json');
fs.writeFileSync(examples, JSON.stringify([
  { embedding: [1, 0], scores: { cheap: 0.91, frontier: 0.99 } },
]));
fs.writeFileSync(prices, JSON.stringify({ cheap: 1, frontier: 20 }));
fs.writeFileSync(query, JSON.stringify([1, 0]));
fs.writeFileSync(evidence, JSON.stringify({
  baseline: { primary: 0.8, noopRate: 0.2, costPerWin: 2, regressed: false },
  candidate: { primary: 0.81, noopRate: 0.1, costPerWin: 1.9, regressed: false },
  anchor: { baseline: 0.7, candidate: 0.7 },
}));

const route = JSON.parse(run([
  'harness', 'route',
  '--examples', examples,
  '--prices', prices,
  '--query', query,
]));
assert.strictEqual(route.id, 'cheap');

const gate = JSON.parse(run(['harness', 'flywheel', 'gate', evidence]));
assert.strictEqual(gate.promote, true);

const denied = spawnSync(process.execPath, [CLI, 'harness', 'darwin', evidence], {
  cwd: ROOT,
  encoding: 'utf8',
});
assert.notStrictEqual(denied.status, 0);
assert.match(denied.stderr, /without --execute/);

console.log('MetaHarness CLI integration checks passed');
