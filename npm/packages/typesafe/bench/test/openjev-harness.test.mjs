// OpenJev v0 harness additions (ADR-007 plan Step 1): Assertion B leakage,
// sha256-verified --model-dir resolution, --no-test, per-item records,
// optimize.mjs flags, fetch-models OpenJev entries, derived fixture pins.
// No network: a fake in-process binding stands in for the engine and the
// "model files" are small dummy byte strings (hashing only).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, writeFileSync, readFileSync, rmSync, mkdirSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { loadTickets, excludeHeldOutText, BENCH_DIR, FIXTURE_DIR } from '../lib/fixture.mjs';
import { sha256Norm } from '../lib/norm.mjs';
import { parseTrainHashes, checkLeakage, assertNoLeakage } from '../lib/leakage.mjs';
import { resolveOnnxModel } from '../lib/model-dir.mjs';
import { main as runMain } from '../run.mjs';
import { main as optimizeMain } from '../optimize.mjs';
import { main as vsJevMain } from '../vs-jev.mjs';
import { computeHashes, FROZEN_FILES, DERIVED_FILES } from '../../scripts/hash-fixtures.mjs';
import { openjevManifestEntry, planPath, parseArgs as fetchArgs, assertHfRevision } from '../../scripts/fetch-models.mjs';

const sha = (s) => createHash('sha256').update(s).digest('hex');
const tmp = (p) => mkdtempSync(join(tmpdir(), p));
const tickets = loadTickets();
const poolHashes = () => excludeHeldOutText(tickets.bySplit).trainItems.map((it) => sha256Norm(it.text)).sort();

/** Stage DIR/<name>/{model.onnx,tokenizer.json[,train-text-hashes.txt]} + DIR/manifest.json. */
function stageModel({ name = 'openjev-small-v0', hashes = poolHashes(), manifest = true, pins = {} } = {}) {
  const dir = tmp('ts-model-');
  mkdirSync(join(dir, name));
  writeFileSync(join(dir, name, 'model.onnx'), 'fake onnx bytes');
  writeFileSync(join(dir, name, 'tokenizer.json'), '{"fake":true}');
  if (hashes) writeFileSync(join(dir, name, 'train-text-hashes.txt'), hashes.join('\n') + '\n');
  if (manifest) {
    const entry = {
      name, file: `${name}/model.onnx`, sha256: sha('fake onnx bytes'), dims: 384, license: 'MIT',
      source_url: 'local:test', added: '2026-09-26', review_by: '2027-03-26', pooling: 'cls',
      tokenizer_file: `${name}/tokenizer.json`, tokenizer_sha256: sha('{"fake":true}'), max_tokens: 256, ...pins,
    };
    writeFileSync(join(dir, 'manifest.json'), JSON.stringify({ models: [entry] }));
  }
  return dir;
}

/** Fake binding: records the constructor options; answers every question. */
function fakeBinding({ optimizeReport } = {}) {
  const seen = { options: [], optimizeSpecs: [] };
  class Engine {
    constructor(json) { seen.options.push(JSON.parse(json)); }
    trainJson() { return JSON.stringify({ ok: true }); }
    statsJson() { return JSON.stringify({ backend: 'fake' }); }
    decideJson(reqJson) {
      const req = JSON.parse(reqJson);
      const depts = Object.keys(req.questions.department.criteria);
      const choice = /charge|refund|invoice/i.test(req.state) ? 'billing' : depts[0];
      return JSON.stringify({ answers: {
        department: { choice, probabilities: Object.fromEntries(depts.map((d) => [d, d === choice ? 0.9 : 0.1 / 7])), confidence: 0.9, abstain: 0.05 },
        urgent: { noul: /urgent|asap/i.test(req.state) ? 0.8 : 0.2 },
        frustration: { score: 0 },
      } });
    }
    optimizeJson(specJson) {
      seen.optimizeSpecs.push(JSON.parse(specJson));
      return JSON.stringify(optimizeReport);
    }
  }
  return { binding: { Engine, version: () => 'fake', backend: 'fake' }, seen };
}

// ---------------------------------------------------------------------------
// leakage (Assertion B)
// ---------------------------------------------------------------------------

test('train-text-hashes format: strict, sorted, lowercase 64-hex, LF only', () => {
  const a = sha('a');
  const b = sha('b');
  const [lo, hi] = [a, b].sort();
  assert.deepEqual(parseTrainHashes(`${lo}\n${hi}\n`), [lo, hi]);
  assert.deepEqual(parseTrainHashes(`${lo}\n${lo}`), [lo, lo], 'duplicates allowed (one line per training row)');
  assert.throws(() => parseTrainHashes(`${hi}\n${lo}\n`), /sorted/);
  assert.throws(() => parseTrainHashes(`${lo.toUpperCase()}\n`), /64-hex/);
  assert.throws(() => parseTrainHashes(`${lo}\r\n`), /CR/);
  assert.throws(() => parseTrainHashes(`${lo}\n\n${hi}\n`), /64-hex/);
  assert.throws(() => parseTrainHashes(''), /empty/);
  assert.throws(() => parseTrainHashes('# comment\n'), /64-hex/);
});

test('leakage refuses a planted collision, and matches through norm()', () => {
  const victim = tickets.bySplit.test[7];
  // Same text after norm (case + punctuation) — must still collide.
  const planted = [...poolHashes(), sha256Norm(`  ${victim.text.toUpperCase()}!!! `)].sort();
  const report = checkLeakage(planted, { test: tickets.bySplit.test, transfer: tickets.bySplit.transfer });
  assert.equal(report.ok, false);
  assert.equal(report.splits.test.intersection, 1);
  assert.deepEqual(report.splits.test.colliding_ids, [victim.id]);
  assert.throws(() => assertNoLeakage(report), (e) => e.code === 'LEAKAGE' && e.message.includes(victim.id));
  const clean = checkLeakage(poolHashes(), { test: tickets.bySplit.test, transfer: tickets.bySplit.transfer, calibration: tickets.bySplit.calibration });
  assert.equal(clean.ok, true);
  assert.equal(clean.train_rows, 136);
});

// ---------------------------------------------------------------------------
// --model-dir resolution (sha256 discipline)
// ---------------------------------------------------------------------------

test('model-dir: staged manifest resolves one verified entry, inlined for the binding', () => {
  const dir = stageModel();
  try {
    const r = resolveOnnxModel({ modelDir: dir, model: 'openjev-small-v0' });
    assert.equal(r.record.sha256, sha('fake onnx bytes'));
    assert.equal(r.record.id, `openjev-small-v0@${sha('fake onnx bytes').slice(0, 12)}`);
    assert.equal(r.record.train_hashes.via, 'model-dir');
    assert.equal(JSON.parse(r.spec.manifest).name, 'openjev-small-v0', 'single entry passed inline');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('model-dir: fails closed on mismatch, missing pin, unknown name, traversal, openjev without hashes', () => {
  const cases = [
    [{ pins: { sha256: '0'.repeat(64) } }, 'MODEL_HASH_MISMATCH'],
    [{ pins: { tokenizer_sha256: '' } }, 'MODEL_UNPINNED'],
    [{ pins: { file: '../escape.onnx' } }, 'MODEL_INVALID'],
    [{ pins: { train_hashes_file: 'openjev-small-v0/train-text-hashes.txt', train_hashes_sha256: '1'.repeat(64) } }, 'MODEL_HASH_MISMATCH'],
    [{ hashes: null }, 'MODEL_INVALID'],
  ];
  for (const [opts, code] of cases) {
    const dir = stageModel(opts);
    try {
      assert.throws(() => resolveOnnxModel({ modelDir: dir, model: 'openjev-small-v0' }), (e) => e.code === code, JSON.stringify(opts));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }
  const dir = stageModel();
  try {
    assert.throws(() => resolveOnnxModel({ modelDir: dir, model: 'bge-small-en-v1.5' }), (e) => e.code === 'MODEL_INVALID', 'no silent first-entry fallback');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('model-dir: a bare unpublished dir gets a synthesized, hash-recorded entry', () => {
  const dir = stageModel({ manifest: false });
  try {
    const r = resolveOnnxModel({ modelDir: join(dir, 'openjev-small-v0') });
    assert.equal(r.record.manifest.synthesized, true);
    assert.equal(r.record.name, 'openjev-small-v0');
    assert.equal(r.record.pooling, 'cls');
    assert.equal(r.record.dims, 384);
    assert.equal(r.record.tokenizer_sha256, sha('{"fake":true}'));
    assert.ok(r.trainHashesPath.endsWith('train-text-hashes.txt'));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// run.mjs: leakage gate, --no-test, per-item records, --emit-records
// ---------------------------------------------------------------------------

test('run.mjs refuses to score a model whose train hashes contain a test text', async () => {
  const dir = stageModel({ hashes: [...poolHashes(), sha256Norm(tickets.bySplit.test[0].text)].sort() });
  try {
    const { binding, seen } = fakeBinding();
    await assert.rejects(
      runMain(['--suite', 'tickets', '--arm', 'local', '--embedder', 'onnx', '--model-dir', dir, '--model', 'openjev-small-v0', '--out', join(dir, 'r.json')], { binding }),
      (e) => e.code === 'LEAKAGE',
    );
    assert.equal(seen.options.length, 0, 'no engine was built');
    assert.ok(!existsSync(join(dir, 'r.json')));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('run.mjs --no-test: validation + transfer only, no test block, no jev replay; --gate rejected', async () => {
  const dir = stageModel();
  try {
    const { binding } = fakeBinding();
    const { code, receipt } = await runMain(
      ['--suite', 'tickets', '--arm', 'both', '--embedder', 'onnx', '--model-dir', dir, '--model', 'openjev-small-v0', '--no-test', '--out', join(dir, 'r.json')],
      { binding },
    );
    assert.equal(code, 0);
    assert.equal(receipt.no_test, true);
    assert.equal(receipt.metrics.jev, undefined);
    assert.deepEqual(receipt.arms, ['local'], 'the skipped jev arm is not claimed');
    assert.ok(!JSON.stringify(receipt).includes(dir), 'no absolute paths in the receipt');
    assert.equal(receipt.leakage.file, 'train-text-hashes.txt');
    assert.equal(receipt.embedder_model.train_hashes.via, 'model-dir');
    assert.match(receipt.embedder_model.train_hashes.sha256, /^[0-9a-f]{64}$/);
    assert.deepEqual(Object.keys(receipt.metrics.local).sort(), ['transfer', 'validation']);
    assert.deepEqual(Object.keys(receipt.item_records.local).sort(), ['transfer', 'validation']);
    assert.ok(!JSON.stringify(receipt.metrics).includes('"test"'));
    assert.equal(receipt.leakage.intersection, 0);
    assert.equal(receipt.embedder_model.name, 'openjev-small-v0');
    await assert.rejects(runMain(['--suite', 'tickets', '--no-test', '--gate'], { binding }), /--no-test cannot be combined with --gate/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('run.mjs receipts carry per-item records; --emit-records feeds vs-jev', async () => {
  const dir = stageModel();
  try {
    const { binding, seen } = fakeBinding();
    const recPath = join(dir, 'records.json');
    const { receipt } = await runMain(
      ['--suite', 'tickets', '--arm', 'both', '--embedder', 'onnx', '--model-dir', dir, '--emit-records', recPath, '--out', join(dir, 'r.json')],
      { binding },
    );
    assert.equal(JSON.parse(seen.options[0].embedder.manifest).sha256, sha('fake onnx bytes'));
    const recs = receipt.item_records.local.test;
    assert.equal(recs.length, 150);
    const r0 = recs[0];
    assert.deepEqual(Object.keys(r0.correct).sort(), ['department', 'frustration', 'urgent']);
    assert.equal(typeof r0.confidence, 'number');
    assert.equal(r0.correct.department, r0.predicted.department === r0.truth.department);
    assert.ok(!JSON.stringify(receipt).includes(tickets.bySplit.test[0].text), 'no item text in the receipt');
    const doc = JSON.parse(readFileSync(recPath, 'utf8'));
    assert.equal(doc.schema, 'ruvector-typesafe-bench/item-records@1');
    assert.deepEqual(doc.records, recs);
    const viaRecords = vsJevMain(['--records', recPath]).out;
    const viaReceipt = vsJevMain(['--receipt', join(dir, 'r.json')]).out;
    assert.deepEqual(viaRecords.results, viaReceipt.results);
    assert.equal(viaReceipt.inputs.embedder_model.name, 'openjev-small-v0');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// optimize.mjs: --model-dir, --no-test, leakage
// ---------------------------------------------------------------------------

const metrics = (acc) => ({ baseline_accuracy: acc, champion_accuracy: acc, n: 0, ece: 0.1, brier: null });
const OPT_REPORT = {
  promotions: 0, budget_consumed: 1, champion_options: {}, arms: [{ promoted: false, statistic: { max_wealth: 1, threshold: 20 } }],
  receipts: { receipts: [{ hash: 'h0' }] },
  baseline_val: metrics(0.8), baseline_transfer: metrics(0.8), baseline_test: metrics(0),
  champion_val: { ...metrics(0.8), n: 150 }, champion_transfer: metrics(0.8), champion_test: metrics(0),
};

test('optimize.mjs --no-test --model-dir: test rows withheld, test fields null, entry recorded', () => {
  const dir = stageModel();
  try {
    const { binding, seen } = fakeBinding({ optimizeReport: OPT_REPORT });
    const { campaign } = optimizeMain(
      ['--models', 'openjev-small-v0', '--model-dir', dir, '--no-test', '--budget', '4', '--out', join(dir, 'c.json')],
      { binding, resultsDir: dir },
    );
    const rows = seen.optimizeSpecs[0].rows;
    assert.equal(rows.length, 350);
    assert.ok(rows.every((r) => r.split !== 'test'));
    assert.equal(campaign.no_test, true);
    assert.equal(campaign.jev_test, null);
    assert.equal(campaign.arms[0].champion.test, null);
    assert.equal(campaign.arms[0].baseline.test_ece, null);
    assert.equal(campaign.arms[0].embedder_model.sha256, sha('fake onnx bytes'));
    assert.equal(campaign.arms[0].leakage.intersection, 0);
    assert.equal(JSON.parse(seen.options[0].embedder.manifest).name, 'openjev-small-v0');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('optimize.mjs records an unfetched model as a per-arm error and continues', () => {
  const dir = stageModel();
  try {
    rmSync(join(dir, 'openjev-small-v0', 'model.onnx'));
    const { binding } = fakeBinding({ optimizeReport: OPT_REPORT });
    const { campaign } = optimizeMain(
      ['--models', 'openjev-small-v0', '--model-dir', dir, '--no-test', '--out', join(dir, 'c.json')],
      { binding, resultsDir: dir },
    );
    assert.match(campaign.arms[0].error, /missing/);
    assert.equal(campaign.champion, null);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('optimize.mjs refuses a leaking model before its campaign runs', () => {
  const dir = stageModel({ hashes: [...poolHashes(), sha256Norm(tickets.bySplit.calibration[0].text)].sort() });
  try {
    const { binding, seen } = fakeBinding({ optimizeReport: OPT_REPORT });
    assert.throws(
      () => optimizeMain(['--models', 'openjev-small-v0', '--model-dir', dir, '--no-test', '--out', join(dir, 'c.json')], { binding, resultsDir: dir }),
      (e) => e.code === 'LEAKAGE',
    );
    assert.equal(seen.optimizeSpecs.length, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// fetch-models: OpenJev download map + manifest entry shape
// ---------------------------------------------------------------------------

test('fetch-models: OpenJev entry shape, immutable revision, derived download map', () => {
  const rev = 'a'.repeat(40);
  const e = openjevManifestEntry({ revision: rev, added: '2026-09-26' });
  assert.equal(e.name, 'openjev-small-v0');
  assert.equal(e.sha256, '', 'bootstrap state: empty pins');
  assert.equal(e.pooling, 'cls');
  assert.equal(e.dims, 384);
  assert.equal(e.max_tokens, 256);
  assert.equal(e.license, 'MIT');
  assert.equal(e.review_by, '2027-03-26');
  assert.equal(e.source_url, `https://huggingface.co/ruvnet/openjev-small-v0/resolve/${rev}/onnx/model.onnx`);
  assert.equal(planPath('openjev-small-v0/model.onnx', e), `ruvnet/openjev-small-v0/resolve/${rev}/onnx/model.onnx`);
  assert.equal(planPath('openjev-small-v0/train-text-hashes.txt', e), `ruvnet/openjev-small-v0/resolve/${rev}/train-text-hashes.txt`);
  assert.equal(planPath('bge-small-en-v1.5/model.onnx'), 'Xenova/bge-small-en-v1.5/resolve/main/onnx/model.onnx');
  assert.equal(planPath('openjev-small-v0/model.onnx'), null, 'no revision → no download source');
  for (const bad of ['main', 'A'.repeat(40), 'a'.repeat(39), undefined]) {
    assert.throws(() => assertHfRevision(bad), /40-hex/);
  }
  assert.equal(fetchArgs(['--add-openjev', rev]).modelName, 'openjev-small-v0');
  assert.throws(() => fetchArgs(['--add-openjev', rev, '--check']), /cannot be combined/);
  // The committed manifest carries no unpublished entry (release preflight needs real pins).
  const committed = JSON.parse(readFileSync(join(BENCH_DIR, '..', 'models', 'manifest.json'), 'utf8')).models;
  assert.ok(committed.every((m) => /^[0-9a-f]{64}$/.test(m.sha256) && /^[0-9a-f]{64}$/.test(m.tokenizer_sha256)));
});

// ---------------------------------------------------------------------------
// fixture pins: derived files never leak into the release fixture pin
// ---------------------------------------------------------------------------

test('HASHES.json: `files` is unchanged (release pin); the novel slice is pinned under `derived`', () => {
  const manifest = JSON.parse(readFileSync(join(FIXTURE_DIR, 'HASHES.json'), 'utf8'));
  assert.deepEqual(Object.keys(manifest.files).sort(), [...FROZEN_FILES].sort());
  const releaseBaseline = JSON.parse(readFileSync(join(BENCH_DIR, 'results', 'tickets-onnx-bge-2026-09-21.json'), 'utf8'));
  assert.deepEqual(manifest.files, releaseBaseline.fixtures.hashes);
  assert.deepEqual(manifest.derived, computeHashes(BENCH_DIR, DERIVED_FILES));
});
