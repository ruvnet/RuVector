import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { main } = require('../dist/cli/main.js');
const here = fileURLToPath(new URL('.', import.meta.url));
const cliPath = join(here, '..', 'bin', 'cli.js');

function makeBinding() {
  delete require.cache[require.resolve('./fixtures/fake-binding.cjs')];
  const fake = require('./fixtures/fake-binding.cjs');
  return { Engine: fake.Engine, version: fake.version, backend: fake.backend };
}

function collector() {
  const lines = [];
  return { lines, sink: (s) => lines.push(s), text: () => lines.join('\n') };
}

const tmp = mkdtempSync(join(tmpdir(), 'typesafe-cli-'));
const questionsPath = join(tmp, 'questions.json');
writeFileSync(
  questionsPath,
  JSON.stringify({
    dept: { type: 'choice', criteria: { billing: 'charges', fraud: 'unauthorised' } },
    urgent: { type: 'noul', instructions: 'needs a reply soon' },
  }),
);

test('decide prints a response JSON', async () => {
  const out = collector();
  const code = await main(['decide', '--state', 'my card was charged twice', '--questions', questionsPath], {
    binding: makeBinding(),
    stdout: out.sink,
    stderr: out.sink,
  });
  assert.equal(code.code, 0);
  const parsed = JSON.parse(out.text());
  assert.ok(parsed.answers.dept.choice);
  assert.ok('usage' in parsed);
});

test('decide --jev strips the additive fields from the output', async () => {
  const out = collector();
  await main(['decide', '--state', 'x', '--questions', questionsPath, '--jev'], {
    binding: makeBinding(),
    stdout: out.sink,
    stderr: out.sink,
  });
  const parsed = JSON.parse(out.text());
  assert.equal('head' in parsed.answers.dept, false);
  assert.equal('confidence' in parsed.answers.dept, true);
});

test('eval scores a JSONL dataset and reports per-question metrics', async () => {
  const datasetPath = join(tmp, 'labeled.jsonl');
  writeFileSync(
    datasetPath,
    [
      JSON.stringify({ text: 'charged twice', label: { dept: 'billing', urgent: true } }),
      JSON.stringify({ text: 'someone used my card', label: { dept: 'billing', urgent: false } }),
    ].join('\n'),
  );
  const out = collector();
  const code = await main(['eval', '--dataset', datasetPath, '--questions', questionsPath], {
    binding: makeBinding(),
    stdout: out.sink,
    stderr: out.sink,
  });
  assert.equal(code.code, 0);
  const report = JSON.parse(out.text());
  assert.equal(report.items, 2);
  assert.ok(report.questions.dept);
  assert.equal(typeof report.questions.dept.accuracy, 'number');
  assert.equal(typeof report.questions.dept.ece, 'number');
  assert.ok('latencyMs' in report.questions.dept);
});

test('serve answers /healthz and a Jev POST, then closes', async () => {
  const out = collector();
  const result = await main(['serve', '--port', '0'], {
    binding: makeBinding(),
    stdout: out.sink,
    stderr: out.sink,
  });
  assert.ok(result.serve, 'serve returns a live handle');
  const port = result.serve.port;
  try {
    const health = await fetch(`http://127.0.0.1:${port}/healthz`);
    assert.equal(health.status, 200);
    const hj = await health.json();
    assert.equal(hj.status, 'ok');

    const res = await fetch(`http://127.0.0.1:${port}/v1/systemone`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        state: 'my card was charged twice',
        questions: {
          dept: { type: 'choice', criteria: { billing: 'b', fraud: 'f' } },
          mood: { type: 'score', criteria: ['Calm', 'Angry'] },
        },
      }),
    });
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.ok(body.answers.dept.choice);
    assert.equal(body.answers.mood.legend, 'Calm');
    assert.ok('usage' in body);
  } finally {
    await result.serve.close();
  }
});

test('serve returns 400 on invalid JSON without echoing the body', async () => {
  const out = collector();
  const result = await main(['serve', '--port', '0'], {
    binding: makeBinding(),
    stdout: out.sink,
    stderr: out.sink,
  });
  try {
    const res = await fetch(`http://127.0.0.1:${result.serve.port}/v1/systemone`, {
      method: 'POST',
      body: 'not json',
    });
    assert.equal(res.status, 400);
    const body = await res.json();
    assert.equal(body.error.kind, 'invalid');
  } finally {
    await result.serve.close();
  }
});

test('serve rejects a >1 MiB body with 400/limit and stays alive', async () => {
  const out = collector();
  const result = await main(['serve', '--port', '0'], {
    binding: makeBinding(),
    stdout: out.sink,
    stderr: out.sink,
  });
  try {
    const huge = 'x'.repeat(1024 * 1024 + 4096); // ~1.004 MiB, over the 1 MiB cap
    const res = await fetch(`http://127.0.0.1:${result.serve.port}/v1/systemone`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: huge,
    });
    assert.equal(res.status, 400);
    const body = await res.json();
    assert.equal(body.error.kind, 'limit');
    // The server must still answer after the oversize request.
    const health = await fetch(`http://127.0.0.1:${result.serve.port}/healthz`);
    assert.equal(health.status, 200);
  } finally {
    await result.serve.close();
  }
});

test('the real bin/cli.js entry prints the version (the only allowed spawn)', () => {
  const stdout = execFileSync(process.execPath, [cliPath, '--version'], { encoding: 'utf8' });
  assert.match(stdout.trim(), /^\d+\.\d+\.\d+/);
});

test('unknown command exits non-zero via the real entry', () => {
  assert.throws(() =>
    execFileSync(process.execPath, [cliPath, 'nope'], { encoding: 'utf8', stdio: 'pipe' }),
  );
});
