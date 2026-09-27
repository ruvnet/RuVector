// ADR-007 M0 "honesty fixes" for the benchmark harness (node --test). No
// network, no child processes: every binding here is in-process.
//
//   - gates fail closed (engine down / nothing measured / tie-check unusable)
//   - entity counts include the transfer split
//   - unpinned datasets fail closed; YAGO3-10 is pinned
//   - the tie-check reaches an all-tied model on a toJson/fromJson binding
//   - receipts carry provenance and a verifiable body hash, no absolute paths
//   - --help, --train-config whitelist, baseline-receipt validation, throughput label

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { evaluateGates, loadGates } from '../bench/lib/gates.mjs';
import { graphCounts, loadStandardTriples } from '../bench/datasets/lib.mjs';
import { PINS as YAGO_PINS } from '../bench/datasets/yago310.mjs';
import { cacheFile as FB_CACHE_FILE } from '../bench/datasets/fb15k237.mjs';
import { zeroTablesText, zeroTableModel } from '../bench/lib/arms.mjs';
import { verifyReceipt, sanitizeBindingSource, canonicalJson } from '../bench/lib/receipt.mjs';
import { main, parseArgs, parseTrainConfig, baselineMismatch } from '../bench/run.mjs';

const require = createRequire(import.meta.url);
const FAKE = require('./fixtures/fake-binding.cjs');
const PKG_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const GIT = { sha: 'a'.repeat(40), engineSrc: 'b'.repeat(64) }; // injected provenance (no fs walk)
const out = () => join(mkdtempSync(join(tmpdir(), 'kge-m0-')), 'receipt.json');
const sha = (s) => createHash('sha256').update(s).digest('hex');

// ---------------------------------------------------------------------------
// A binding that mimics the real napi Model's contract: a zero-epoch model is
// NOT tied (random init), and `fromJson` verifies sha256 over the body text in
// the Rust serde format ({"dims":D,"entities":[f32..],"relations":[f32..]} as
// the last field). A zeroed-table model scores every triple 0 (all tied).
// ---------------------------------------------------------------------------
function rustLikeBinding({ rejectFromJson = false } = {}) {
  class Model extends FAKE.Model {
    constructor(cfgJson) {
      const c = JSON.parse(cfgJson || '{}');
      const zero = c._zeroTables === true;
      super(JSON.stringify({ ...c, epochs: zero ? 0 : Math.max(1, c.epochs ?? 1) }));
      this._cfg = c;
      this._tagged = [];
    }
    addTriplesJson(json) {
      this._tagged.push(...JSON.parse(json));
      return super.addTriplesJson(json);
    }
    toJson() {
      const floats = (n, k) => Array.from({ length: n }, (_, i) => (((i * 7 + k) % 13) / 10 - 0.6).toFixed(3)).join(',');
      const ne = this.entities.size;
      const nr = this.relations.size;
      const body =
        `{"version":1,"config":${JSON.stringify({ scorer: this._cfg.scorer ?? 'hole', dims: 4, seed: 42 })},` +
        `"entities":{"labels":[]},"relations":{"labels":[]},"triples":${JSON.stringify(this._tagged)},` +
        `"tables":{"dims":4,"entities":[${floats(ne * 4, 1)}],"relations":[${floats(nr * 4, 2)}]}}`;
      return `{"sha256":"${sha(body)}","model":${body}}`;
    }
    static fromJson(env) {
      if (rejectFromJson) throw new Error('model hash mismatch: payload is tampered or corrupt');
      const m = /^\{"sha256":"([0-9a-f]{64})","model":(.*)\}$/s.exec(env);
      if (!m || sha(m[2]) !== m[1]) throw new Error('model hash mismatch: payload is tampered or corrupt');
      const model = JSON.parse(m[2]);
      const allZero = [...model.tables.entities, ...model.tables.relations].every((x) => x === 0);
      const copy = new Model(JSON.stringify({ ...model.config, _zeroTables: allZero }));
      copy.addTriplesJson(JSON.stringify(model.triples));
      return copy;
    }
  }
  return { Model, version: () => '0-rustlike', backend: 'fake' };
}

// ---------------------------------------------------------------------------
// gates fail closed
// ---------------------------------------------------------------------------

test('gates: engine unavailable is a FAIL, never PASS', () => {
  const r = evaluateGates({}, { suite: 'synthetic', engineAvailable: false }, loadGates());
  assert.equal(r.pass, false);
  assert.equal(r.anyFail, true);
  assert.equal(r.rows.find((x) => x.name === 'engine_available').status, 'FAIL');
});

test('gates: a run where every gate SKIPs is a FAIL (nothing measured)', () => {
  const r = evaluateGates({}, { suite: 'synthetic', engineAvailable: true }, loadGates());
  assert.equal(r.pass, false);
  assert.equal(r.rows.find((x) => x.name === 'measured_rows').status, 'FAIL');
});

test('gates: tie-check SKIPs only when not requested; unusable → FAIL', () => {
  const g = loadGates();
  const row = (tieCheck) =>
    evaluateGates({}, { suite: 'synthetic', engineAvailable: true, tieCheck }, g).rows.find((x) => x.name === 'tie_break_random');
  assert.equal(row(null).status, 'SKIP');
  assert.equal(row({ available: false, reason: 'x' }).status, 'FAIL');
  // requested, but the model was not all-tied (the pre-M0 real-binding case)
  assert.equal(row({ available: true, topMr: 40, randomMr: 150, bottomMr: 290, nEntities: 300 }).status, 'FAIL');
  assert.equal(row({ available: true, topMr: 1, randomMr: 150.5, bottomMr: 300, nEntities: 300 }).status, 'PASS');
});

test('run.mjs: engine-down synthetic --gate exits 1; --report-only exits 0 but records FAIL', async () => {
  const strict = await main(['--suite', 'synthetic', '--gate', '--out', out()], { binding: {}, git: GIT });
  assert.equal(strict.code, 1, 'engine-down --gate must not exit 0');
  assert.equal(strict.receipt.gates.pass, false);
  const lax = await main(['--suite', 'synthetic', '--gate', '--report-only', '--out', out()], { binding: {}, git: GIT });
  assert.equal(lax.code, 0);
  assert.equal(lax.receipt.gates.pass, false, 'report-only records the honest verdict');
});

test('run.mjs: a skipped suite (tampered sha256-pinned file) under --gate exits 1, not 0', async () => {
  // A tampered cached file: fetchCached reads it (no network) and the pin fails.
  const cacheDir = mkdtempSync(join(tmpdir(), 'kge-m0-cache-'));
  for (const k of ['train', 'valid', 'test']) writeFileSync(join(cacheDir, FB_CACHE_FILE(k)), 'a\tr\tb\n');
  const deps = { binding: FAKE, git: GIT, cacheDir };
  const strict = await main(['--suite', 'fb15k237', '--gate', '--out', out()], deps);
  assert.equal(strict.code, 1, 'a suite that did not run must fail the gate');
  const [row] = strict.receipt.runs;
  assert.match(row.skipped, /sha256 mismatch/);
  assert.equal(row.gates.pass, false);
  assert.equal(row.gates.rows[0].name, 'suite_available');
  assert.equal(row.gates.rows[0].status, 'FAIL');

  const lax = await main(['--suite', 'fb15k237', '--gate', '--report-only', '--out', out()], deps);
  assert.equal(lax.code, 0, '--report-only keeps exit 0');
  assert.equal(lax.receipt.runs[0].gates.pass, false, 'but still records the FAIL');

  // Without --gate a skip is informational only (no gate row, exit 0).
  const plain = await main(['--suite', 'fb15k237', '--out', out()], deps);
  assert.equal(plain.code, 0);
  assert.equal(plain.receipt.runs[0].gates, undefined);
});

// ---------------------------------------------------------------------------
// datasets
// ---------------------------------------------------------------------------

test('graphCounts: entities in the transfer split are counted', () => {
  const t = (s, r, o) => ({ s, r, o });
  const c = graphCounts({ train: [t('a', 'r', 'b')], valid: [t('a', 'r', 'b')], transfer: [t('a', 'q', 'z')], test: [] });
  assert.equal(c.entities, 3, 'z appears only in transfer');
  assert.equal(c.relations, 2);
  assert.equal(c.transfer, 1);
});

test('datasets: an unpinned file fails closed before any fetch', async () => {
  await assert.rejects(
    loadStandardTriples({ name: 'x', sources: { train: 'http://invalid/', valid: 'http://invalid/', test: 'http://invalid/' }, pins: { train: 'a', valid: null, test: 'b' } }),
    /no sha256 pin for valid/,
  );
});

test('YAGO3-10 is sha256-pinned', () => {
  for (const k of ['train', 'valid', 'test']) assert.match(YAGO_PINS[k], /^[0-9a-f]{64}$/);
});

// ---------------------------------------------------------------------------
// tie-check reachability
// ---------------------------------------------------------------------------

test('zeroTablesText rewrites only the table floats, keeps counts, leaves vocab alone', () => {
  const body = '{"entities":{"labels":["x[1]"]},"triples":[[0,0,1]],"tables":{"dims":2,"entities":[0.5,-1.25e-3,2.0,1.0],"relations":[3.5,-0.0]}}';
  const z = zeroTablesText(body);
  assert.equal(z.body, '{"entities":{"labels":["x[1]"]},"triples":[[0,0,1]],"tables":{"dims":2,"entities":[0.0,0.0,0.0,0.0],"relations":[0.0,0.0]}}');
  assert.match(zeroTablesText('{"tables":null}').error, /no built tables/);
});

test('zeroTableModel re-hashes so a verifying fromJson accepts the zeroed model', () => {
  const b = rustLikeBinding();
  const m = new b.Model('{"dims":4}');
  m.addTriplesJson(JSON.stringify([{ s: 'a', r: 'r', o: 'b', split: 'train' }]));
  const z = zeroTableModel(b, m);
  assert.equal(z.available, true, z.error);
  assert.equal(z.model.epochs, 0, 'all-zero tables → constant scorer');
  const bad = zeroTableModel(rustLikeBinding({ rejectFromJson: true }), m);
  assert.equal(bad.available, false);
  assert.match(bad.error, /fromJson rejected/);
});

test('run.mjs --tie-check reaches an all-tied model on a toJson/fromJson binding → PASS', async () => {
  const { code, receipt } = await main(['--suite', 'synthetic', '--tie-check', '--gate', '--out', out()], {
    binding: rustLikeBinding(),
    git: GIT,
  });
  assert.equal(receipt.tie_check.method, 'zero-tables');
  assert.ok(Math.abs(receipt.tie_check.topMr - 1) < 1e-9, `TOP MR ${receipt.tie_check.topMr}`);
  assert.equal(receipt.gates.rows.find((r) => r.name === 'tie_break_random').status, 'PASS');
  assert.equal(code, 0);
});

test('run.mjs --tie-check with a fromJson that rejects the zeroed model → FAIL, not SKIP', async () => {
  const { code, receipt } = await main(['--suite', 'synthetic', '--tie-check', '--gate', '--out', out()], {
    binding: rustLikeBinding({ rejectFromJson: true }),
    git: GIT,
  });
  assert.equal(receipt.tie_check.available, false);
  assert.equal(receipt.gates.rows.find((r) => r.name === 'tie_break_random').status, 'FAIL');
  assert.equal(code, 1);
});

// ---------------------------------------------------------------------------
// receipts: provenance, body hash, no absolute paths, throughput label, control
// ---------------------------------------------------------------------------

test('receipt carries provenance + a verifiable receipt_sha256 and no absolute path', async () => {
  const { receipt } = await main(['--suite', 'synthetic', '--adversarial', '--out', out()], { binding: rustLikeBinding(), git: GIT });
  const p = receipt.provenance;
  assert.equal(p.git_sha, GIT.sha);
  assert.equal(p.engine_src_sha256, GIT.engineSrc);
  assert.match(p.config_sha256, /^[0-9a-f]{64}$/);
  assert.equal(p.config_sha256, sha(canonicalJson(receipt.config)));
  assert.match(p.dataset_sha256, /^[0-9a-f]{64}$/);
  assert.ok(verifyReceipt(receipt), 'body hash verifies');
  assert.ok(verifyReceipt(JSON.parse(JSON.stringify(receipt))), 'and survives a JSON round trip');
  assert.equal(verifyReceipt({ ...receipt, metrics: { test: { mrr: 0.99 } } }), false, 'tampering is detected');
  assert.ok(!JSON.stringify(receipt).includes(PKG_ROOT), 'no absolute package path in the receipt');
  // throughput is labelled as what it is
  assert.equal(receipt.training.triplesPerSec, undefined);
  assert.ok(receipt.training.triple_epochs_per_sec > 0);
  // the adversarial report carries the no-decoy retrain control
  assert.ok(receipt.adversarial.control && typeof receipt.adversarial.control.net_drop === 'number', JSON.stringify(receipt.adversarial.control));
});

test('receipt real provenance: git SHA read without child_process', async () => {
  const { receipt } = await main(['--suite', 'synthetic', '--out', out()], { binding: FAKE });
  const p = receipt.provenance;
  assert.ok(p.git_sha === null || /^[0-9a-f]{40}$/.test(p.git_sha));
  assert.ok(p.engine_src_sha256 === null || /^[0-9a-f]{64}$/.test(p.engine_src_sha256));
  assert.equal(receipt.adversarial, null);
});

test('sanitizeBindingSource drops absolute host paths', () => {
  assert.equal(sanitizeBindingSource(join(PKG_ROOT, 'index.js'), PKG_ROOT), 'index.js');
  assert.equal(sanitizeBindingSource('/elsewhere/x/binding.cjs', PKG_ROOT), '<external>/binding.cjs');
  assert.equal(sanitizeBindingSource(`env:KGE_BENCH_BINDING(${join(PKG_ROOT, 'test/f.cjs')})`, PKG_ROOT), 'env:KGE_BENCH_BINDING(test/f.cjs)');
  assert.equal(sanitizeBindingSource('injected', PKG_ROOT), 'injected');
});

// ---------------------------------------------------------------------------
// CLI surface
// ---------------------------------------------------------------------------

test('--help prints usage and exits 0 without running', async () => {
  const r = await main(['--help']);
  assert.equal(r.code, 0);
  assert.equal(r.receipt, null);
});

test('--train-config: whitelisted knobs pass through and are recorded; unknown keys rejected', async () => {
  assert.deepEqual(parseTrainConfig('{"lr":0.05,"batch_size":128}'), { lr: 0.05, batch_size: 128 });
  assert.throws(() => parseTrainConfig('{"learning_rate":0.05}'), /unsupported key/);
  assert.throws(() => parseTrainConfig('[1]'), /JSON object/);
  assert.throws(() => parseArgs(['--train-config', '{bad']), /not valid JSON/);
  const { receipt } = await main(['--suite', 'synthetic', '--train-config', '{"lr":0.05}', '--out', out()], { binding: FAKE, git: GIT });
  assert.equal(receipt.config.lr, 0.05);
});

test('--baseline-receipt must match suite / splits_hash / scorer / dims', async () => {
  const base = { suite: 'synthetic', scorer: 'hole', config: { dims: 256 }, dataset: { splits_hash: 'h' } };
  assert.equal(baselineMismatch(base, { suite: 'synthetic', splitsHash: 'h', scorer: 'hole', dims: 256 }), null);
  assert.match(baselineMismatch(base, { suite: 'fb15k237', splitsHash: 'h', scorer: 'hole', dims: 256 }), /suite/);
  assert.match(baselineMismatch(base, { suite: 'synthetic', splitsHash: 'h', scorer: 'hole', dims: 64 }), /dims/);
});
