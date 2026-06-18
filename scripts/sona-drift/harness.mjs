#!/usr/bin/env node
/**
 * SONA cross-implementation behavioral-parity harness.
 *
 * Drives each of the three independent SONA learn-from-feedback
 * implementations through an identical, fully deterministic feedback
 * scenario and extracts a BEHAVIORAL FINGERPRINT (numeric vector) per
 * implementation, plus a pass/fail CONTRACT MATRIX:
 *
 *   C1 fresh == 0            a fresh engine has zero adaptation
 *   C2 single-positive > 0   ONE positive feedback adapts (#519/#553 tripwire)
 *   C3 negative-adapts > 0   negative feedback also adapts (unlearning)
 *   C4 neutral == 0          neutral feedback is a no-op — only enforced for
 *                            impls that define neutral as a no-op
 *   C5 inference-changes > 0 forward/apply output actually changes
 *
 * Fingerprint layout (6 components, all deterministic — NOTHING timing-based):
 *   [0] m1 baseline adaptation metric on a fresh engine            (C1)
 *   [1] m2 adaptation metric after 1 positive feedback (q=0.9)     (C2)
 *   [2] m3 adaptation metric after 6 positive feedbacks (growth)
 *   [3] m4 adaptation metric after 1 negative feedback (q=0.1)     (C3)
 *   [4] m5 adaptation metric after 1 neutral feedback (q=0.5)      (C4)
 *   [5] m6 inference-output change after adaptation                (C5)
 *
 * Usage:  node scripts/sona-drift/harness.mjs --json
 *   --json   print the machine-readable JSON report on stdout
 *            (human-readable progress always goes to stderr)
 * Exit code: non-zero if any non-skipped implementation fails any contract.
 */

import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const REPO = path.resolve(__dirname, '..', '..');

// ---------------------------------------------------------------------------
// Shared deterministic scenario constants
// ---------------------------------------------------------------------------

const DIM = 64;
/** Fixed probe vector used to observe inference output. */
const PROBE = Array.from({ length: DIM }, (_, i) => Math.sin(i + 1));
/** Second fixed probe, used for the m6 inference-change metric. */
const PROBE2 = Array.from({ length: DIM }, () => 1 / Math.sqrt(DIM));
/** Fixed feedback embedding fed to trajectory-based engines. */
const FEEDBACK_EMB = Array.from({ length: DIM }, (_, i) => Math.cos(i + 1) * 0.5);

const Q_POS = 0.9;
const Q_NEG = 0.1;
const Q_NEUTRAL = 0.5;
const N_GROWTH = 6;

/** "Adapted at all" threshold for C2/C3/C5. */
const EPS = 1e-9;
/** "Exactly zero" threshold for C1/C4 (allows f64 noise only). */
const ZERO = 1e-12;

export const FINGERPRINT_DIM = 6;

function l2(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - (b ? b[i] : 0);
    s += d * d;
  }
  return Math.sqrt(s);
}

function log(...args) {
  console.error('[harness]', ...args);
}

/** Deterministic PRNG (xorshift32) for seeding the ruvllm micro-LoRA. */
function xorshift32(seed) {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

function runBuild(cwd, script) {
  log(`building (npm run ${script}) in ${cwd} ...`);
  const r = spawnSync('npm', ['run', script], { cwd, shell: true, encoding: 'utf8', timeout: 600000 });
  if (r.status !== 0) {
    throw new Error(`npm run ${script} failed in ${cwd}: ${String(r.stderr).slice(-800)}`);
  }
}

// ---------------------------------------------------------------------------
// Adapter: ruvllm-ts (SonaCoordinator, npm/packages/ruvllm — #553 fix)
// ---------------------------------------------------------------------------

const ruvllmAdapter = {
  name: 'ruvllm-ts',
  neutralIsNoop: true, // processInstantLearning: reward = quality - 0.5 === 0 -> early return
  async run() {
    // Env override exists ONLY so the failure mode can be demonstrated against
    // a scratch copy without touching the tree (see README).
    const distPath = process.env.SONA_DRIFT_RUVLLM_SONA_JS
      || path.join(REPO, 'npm', 'packages', 'ruvllm', 'dist', 'cjs', 'sona.js');
    if (!existsSync(distPath)) {
      runBuild(path.join(REPO, 'npm', 'packages', 'ruvllm'), 'build:cjs');
    }
    const { SonaCoordinator } = require(distPath);

    // SonaCoordinator initializes its micro-LoRA A matrix with Math.random();
    // re-seed it deterministically (A := xorshift-init, B := 0) so fingerprints
    // are reproducible. B == 0 keeps the LoRA delta exactly zero at baseline,
    // matching the production zero-init contract.
    const fresh = () => {
      const c = new SonaCoordinator();
      const lora = c.microLora; // TS-private, plain property at runtime
      const w = lora.getWeights();
      const rnd = xorshift32(0x51f1ed5);
      const scale = Math.sqrt(2 / w.loraA.length);
      lora.setWeights({
        ...w,
        loraA: w.loraA.map((row) => row.map(() => (rnd() - 0.5) * scale)),
        loraB: w.loraB.map((row) => row.map(() => 0)),
      });
      return c;
    };
    const signal = (quality) => ({
      requestId: 'sona-drift-probe',
      quality,
      type: quality > 0.5 ? 'positive' : quality < 0.5 ? 'negative' : 'implicit',
      timestamp: new Date(0),
    });
    const metric = (c) => c.microLoraDeltaNorm();

    const m1 = metric(fresh());

    const c2 = fresh();
    c2.recordSignal(signal(Q_POS));
    const m2 = metric(c2);

    const c3 = fresh();
    for (let i = 0; i < N_GROWTH; i++) c3.recordSignal(signal(Q_POS));
    const m3 = metric(c3);

    const c4 = fresh();
    c4.recordSignal(signal(Q_NEG));
    const m4 = metric(c4);

    const c5 = fresh();
    c5.recordSignal(signal(Q_NEUTRAL));
    const m5 = metric(c5);

    // m6: forward/apply output before vs after adaptation (same probe, the
    // "before" coordinator is bit-identical to c3's pre-feedback state).
    const m6 = l2(c3.applyMicroLora(PROBE), fresh().applyMicroLora(PROBE));

    return {
      metrics: { m1, m2, m3, m4, m5, m6 },
      detail: 'SonaCoordinator.recordSignal -> processInstantLearning; metric = microLoraDeltaNorm()',
    };
  },
};

// ---------------------------------------------------------------------------
// Adapter: rust-sona (crates/sona via @ruvector/sona N-API — #519 fix)
// ---------------------------------------------------------------------------

const rustSonaAdapter = {
  name: 'rust-sona',
  // The Rust instant loop weights updates by quality but does NOT subtract a
  // 0.5 baseline: a quality-0.5 trajectory still produces a small update.
  neutralIsNoop: false,
  async run() {
    const pkgDir = path.join(REPO, 'npm', 'packages', 'sona');
    const indexJs = path.join(pkgDir, 'index.js');
    let SonaEngine;
    try {
      ({ SonaEngine } = require(indexJs));
    } catch {
      // The platform .node binary is not committed — build it (napi build
      // against crates/sona, ~40s).
      runBuild(pkgDir, 'build');
      ({ SonaEngine } = require(indexJs));
    }

    const fresh = () => new SonaEngine(DIM); // deterministic init (verified: fresh-vs-fresh L2 == 0)
    const feedback = (engine, quality, n = 1) => {
      for (let i = 0; i < n; i++) {
        const t = engine.beginTrajectory(FEEDBACK_EMB);
        engine.addTrajectoryStep(t, FEEDBACK_EMB, FEEDBACK_EMB, quality);
        engine.endTrajectory(t, quality);
      }
      engine.flush(); // apply instant-loop micro-LoRA updates immediately
    };
    const freshOut = fresh().applyMicroLora(PROBE);
    const freshOut2 = fresh().applyMicroLora(PROBE2);
    const metric = (e) => l2(e.applyMicroLora(PROBE), freshOut);

    const m1 = metric(fresh());

    const e2 = fresh();
    feedback(e2, Q_POS, 1);
    const m2 = metric(e2);

    const e3 = fresh();
    feedback(e3, Q_POS, N_GROWTH);
    const m3 = metric(e3);

    const e4 = fresh();
    feedback(e4, Q_NEG, 1);
    const m4 = metric(e4);

    const e5 = fresh();
    feedback(e5, Q_NEUTRAL, 1);
    const m5 = metric(e5);

    // m6: inference change on a second, independent probe (m2..m5 already use
    // PROBE, so PROBE2 adds information instead of duplicating m3).
    const m6 = l2(e3.applyMicroLora(PROBE2), freshOut2);

    return {
      metrics: { m1, m2, m3, m4, m5, m6 },
      detail: 'SonaEngine begin/addStep/endTrajectory + flush; metric = L2(applyMicroLora(PROBE) - freshEngineOutput)',
    };
  },
};

// ---------------------------------------------------------------------------
// Adapter: ruvector-cli (IntelligenceEngine route-confidence learning — #517)
// ---------------------------------------------------------------------------

const cliAdapter = {
  name: 'ruvector-cli',
  neutralIsNoop: true, // value update: v + lr*(reward - v) with v=0.5, reward=0.5 -> unchanged
  async run() {
    const distPath = path.join(
      REPO, 'npm', 'packages', 'ruvector', 'dist', 'core', 'intelligence-engine.js');
    if (!existsSync(distPath)) {
      return { skipped: 'skipped: #517 fix not on main (npm/packages/ruvector/dist not present and not built)' };
    }
    let mod;
    try {
      mod = require(distPath);
    } catch (e) {
      return { skipped: `skipped: ruvector dist not loadable (${e.message})` };
    }
    const Engine = mod.IntelligenceEngine ?? mod.default;
    const probeEngine = new Engine({});
    if (typeof probeEngine.recordRouteOutcome !== 'function') {
      return { skipped: 'skipped: #517 fix not on main (recordRouteOutcome absent)' };
    }

    const TASK = 'fix the auth bug';
    const FILE = 'src/auth.rs';
    const AGENT = 'security-auditor';
    const fresh = () => new Engine({});
    // Adaptation metric: drift of the learned route-policy value for AGENT
    // away from its 0.5 prior (recordRouteOutcome writes routingPatterns).
    const metric = (e) => {
      for (const stateMap of e.routingPatterns.values()) {
        if (stateMap.has(AGENT)) return Math.abs(stateMap.get(AGENT) - 0.5);
      }
      return 0;
    };
    const feedback = (e, reward, n = 1) => {
      for (let i = 0; i < n; i++) e.recordRouteOutcome(TASK, FILE, AGENT, reward);
    };

    const m1 = metric(fresh());

    const e2 = fresh();
    feedback(e2, Q_POS, 1);
    const m2 = metric(e2);

    const e3 = fresh();
    feedback(e3, Q_POS, N_GROWTH);
    const m3 = metric(e3);

    const e4 = fresh();
    feedback(e4, Q_NEG, 1);
    const m4 = metric(e4);

    const e5 = fresh();
    feedback(e5, Q_NEUTRAL, 1);
    const m5 = metric(e5);

    // m6: route() confidence ("inference output") before vs after adaptation.
    const confBefore = (await fresh().route(TASK, FILE)).confidence;
    const confAfter = (await e3.route(TASK, FILE)).confidence;
    const m6 = Math.abs(confAfter - confBefore);

    return {
      metrics: { m1, m2, m3, m4, m5, m6 },
      detail: 'IntelligenceEngine.recordRouteOutcome -> route(); metric = |learned route value - 0.5|, m6 = |route confidence delta|',
    };
  },
};

// ---------------------------------------------------------------------------
// Contract evaluation + report
// ---------------------------------------------------------------------------

function evalContracts(adapter, m) {
  return {
    'C1-fresh-zero': { pass: m.m1 <= ZERO, value: m.m1 },
    'C2-single-positive-adapts': { pass: m.m2 > EPS, value: m.m2 },
    'C3-negative-adapts': { pass: m.m4 > EPS, value: m.m4 },
    'C4-neutral-noop': adapter.neutralIsNoop
      ? { pass: m.m5 <= ZERO, value: m.m5 }
      : {
          pass: true,
          value: m.m5,
          note: 'n/a: impl defines neutral as a deterministic low-weight update, not a no-op',
        },
    'C5-inference-changes': { pass: m.m6 > EPS, value: m.m6 },
  };
}

const ADAPTERS = [ruvllmAdapter, rustSonaAdapter, cliAdapter];

/**
 * Run the scenario for every implementation. Each adapter is executed TWICE
 * and the fingerprints compared bit-for-bit — any mismatch means the metric
 * is jittery and must not be fingerprinted, so the impl is marked as error.
 */
export async function collectFingerprints() {
  const results = [];
  for (const adapter of ADAPTERS) {
    log(`running scenario for ${adapter.name} ...`);
    let entry;
    try {
      const a = await adapter.run();
      if (a.skipped) {
        entry = { impl: adapter.name, status: 'skipped', reason: a.skipped, contracts: null, fingerprint: null };
      } else {
        const b = await adapter.run(); // determinism check
        const fpA = [a.metrics.m1, a.metrics.m2, a.metrics.m3, a.metrics.m4, a.metrics.m5, a.metrics.m6];
        const fpB = [b.metrics.m1, b.metrics.m2, b.metrics.m3, b.metrics.m4, b.metrics.m5, b.metrics.m6];
        if (fpA.some((v, i) => v !== fpB[i])) {
          entry = {
            impl: adapter.name,
            status: 'error',
            reason: `non-deterministic fingerprint: ${JSON.stringify(fpA)} vs ${JSON.stringify(fpB)}`,
            contracts: null,
            fingerprint: null,
          };
        } else {
          const contracts = evalContracts(adapter, a.metrics);
          const pass = Object.values(contracts).every((c) => c.pass);
          entry = {
            impl: adapter.name,
            status: pass ? 'pass' : 'fail',
            detail: a.detail,
            contracts,
            fingerprint: fpA,
          };
        }
      }
    } catch (e) {
      entry = { impl: adapter.name, status: 'error', reason: e.message, contracts: null, fingerprint: null };
    }
    log(`  ${adapter.name}: ${entry.status}${entry.reason ? ` (${entry.reason})` : ''}`);
    results.push(entry);
  }
  const pass = results.every((r) => r.status === 'pass' || r.status === 'skipped');
  return {
    schema: 1,
    scenario: {
      dim: DIM,
      qualityPositive: Q_POS,
      qualityNegative: Q_NEG,
      qualityNeutral: Q_NEUTRAL,
      growthCount: N_GROWTH,
      fingerprint: ['m1-fresh', 'm2-1pos', 'm3-6pos', 'm4-1neg', 'm5-neutral', 'm6-inference-change'],
    },
    results,
    pass,
  };
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const report = await collectFingerprints();
  if (process.argv.includes('--json')) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    for (const r of report.results) {
      console.log(`${r.impl}: ${r.status}${r.reason ? ` — ${r.reason}` : ''}`);
      if (r.contracts) {
        for (const [name, c] of Object.entries(r.contracts)) {
          console.log(`  ${c.pass ? 'PASS' : 'FAIL'} ${name} (value=${c.value}${c.note ? `, ${c.note}` : ''})`);
        }
      }
    }
    console.log(report.pass ? 'HARNESS PASS' : 'HARNESS FAIL');
  }
  process.exit(report.pass ? 0 : 1);
}
