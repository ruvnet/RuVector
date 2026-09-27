// Gate evaluation (ADR-006 §Release gates). Consumes a run's flat metric bag
// plus a context (suite, which optional runs happened, tie-check result,
// baseline receipt) and returns PASS / FAIL / SKIP rows. Pure: no I/O beyond
// loading the gates doc.
//
// Four gate kinds:
//   numeric      metric `op` threshold, gated only when `applies_when` holds
//   tie_break    the Sun-et-al RANDOM assertion (needs the --tie-check result)
//   external     always SKIP — asserted elsewhere (the crate's HolE≡ComplEx test)
//   skip_pending always SKIP — threshold not set yet (latency, ADR-002 spike)

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
export const GATES_PATH = join(HERE, '..', 'gates.json');
const BENCH_DIR = join(HERE, '..');

export function loadGates(path = GATES_PATH) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

function compare(op, a, b) {
  switch (op) {
    case '>=':
      return a >= b;
    case '<=':
      return a <= b;
    case '>':
      return a > b;
    case '<':
      return a < b;
    default:
      throw new Error(`unknown gate op ${op}`);
  }
}

const fmtNum = (x) => (typeof x === 'number' ? Number(x.toFixed(4)) : x);

/**
 * @param metrics flat bag: { mrr_test, ann_recall_at_10,
 *                adversarial_confidence_drop, transfer_regression_pt }
 * @param ctx     { suite, hasAnn, hasAdversarial, hasBaselineReceipt,
 *                engineAvailable, tieCheck }
 *                tieCheck: { available, topMr, randomMr, bottomMr, nEntities } | null
 * @returns { rows:[{name,status,detail,source}], pass, anyFail, skipped }
 */
export function evaluateGates(metrics, ctx, gatesDoc = loadGates()) {
  const rows = [];
  const engineDown = ctx.engineAvailable === false;

  for (const [name, g] of Object.entries(gatesDoc.gates)) {
    if (g.kind === 'external' || g.kind === 'skip_pending') {
      rows.push({ name, status: 'SKIP', detail: g.reason, source: g.source });
      continue;
    }
    if (g.kind === 'tie_break') {
      rows.push({ ...tieBreakRow(name, g, ctx), source: g.source });
      continue;
    }
    if (g.kind === 'latency') {
      rows.push({ ...latencyRow(name, g, ctx), source: g.source });
      continue;
    }
    // numeric
    const applies = numericApplies(g, ctx);
    if (!applies.ok) {
      rows.push({ name, status: 'SKIP', detail: applies.why, source: g.source });
      continue;
    }
    if (engineDown) {
      rows.push({ name, status: 'FAIL', detail: 'engine unavailable', source: g.source });
      continue;
    }
    const actual = metrics[g.metric];
    if (actual === undefined || actual === null || Number.isNaN(actual)) {
      rows.push({ name, status: 'SKIP', detail: `metric ${g.metric} not measured`, source: g.source });
      continue;
    }
    const ok = compare(g.op, actual, g.threshold);
    rows.push({
      name,
      status: ok ? 'PASS' : 'FAIL',
      detail: `${fmtNum(actual)} ${g.op} ${g.threshold}`,
      source: g.source,
    });
  }

  // Fail closed (ADR-007 M0). Pre-M0, `pass = !anyFail` over measured rows
  // only, so an engine-down synthetic `--gate` run (every row SKIP) printed
  // PASS and exited 0. Now an unavailable engine, or a run where no gate was
  // measured at all, is itself a visible FAIL row. `--report-only` still exits
  // 0 (run.mjs), but the verdict it records is FAIL, never PASS.
  if (engineDown) {
    rows.push({ name: 'engine_available', status: 'FAIL', detail: 'engine unavailable — nothing was measured', source: 'ADR-007 M0' });
  } else if (!rows.some((r) => r.status !== 'SKIP')) {
    rows.push({ name: 'measured_rows', status: 'FAIL', detail: 'no gate was measured (every row SKIP)', source: 'ADR-007 M0' });
  }
  const anyFail = rows.some((r) => r.status === 'FAIL');
  return { rows, pass: !anyFail, anyFail, skipped: rows.filter((r) => r.status === 'SKIP').length };
}

function numericApplies(g, ctx) {
  const w = g.applies_when;
  if (!w) return { ok: true };
  if (w.startsWith('suite:')) {
    const want = w.slice('suite:'.length);
    if (ctx.suite !== want) return { ok: false, why: `suite is ${ctx.suite}, not ${want}` };
    // A --limit slice is a subgraph, not the full graph the cited baseline was
    // measured on, so its MRR is not comparable — the link-prediction gate is
    // informational on a slice, never a pass/fail.
    if (typeof ctx.limit === 'number') {
      return { ok: false, why: `subgraph slice (--limit ${ctx.limit}): informational, not comparable to the full-graph baseline` };
    }
    return { ok: true };
  }
  if (w === 'has_ann') return ctx.hasAnn ? { ok: true } : { ok: false, why: 'no --ann run' };
  if (w === 'has_adversarial') return ctx.hasAdversarial ? { ok: true } : { ok: false, why: 'no --adversarial run' };
  if (w === 'has_baseline_receipt')
    return ctx.hasBaselineReceipt ? { ok: true } : { ok: false, why: 'no --baseline-receipt' };
  return { ok: true };
}

/**
 * The RANDOM tie-break assertion. PASS iff the model is provably all-tied (TOP
 * mean rank ≈ 1) AND the RANDOM mean rank ≈ (|E|+1)/2 within tolerance. SKIP
 * only when no `--tie-check` was requested. Once it was requested, an
 * unavailable check or a model that is not all-tied is a FAIL (ADR-007 M0) —
 * pre-M0 both SKIPped, so the gate never actually ran on the real binding.
 */
function tieBreakRow(name, g, ctx) {
  const tc = ctx.tieCheck;
  if (!tc) {
    return { name, status: 'SKIP', detail: 'no --tie-check run' };
  }
  if (!tc.available) {
    return { name, status: 'FAIL', detail: `tie-check requested but unavailable: ${tc.reason ?? 'unknown'}` };
  }
  const expected = (tc.nEntities + 1) / 2;
  const tol = g.tolerance_frac ?? 0.1;
  const allTied = Math.abs(tc.topMr - 1) <= 0.5; // TOP puts the true first → MR≈1
  if (!allTied) {
    return {
      name,
      status: 'FAIL',
      detail: `tie-check model is not all-tied (TOP MR=${fmtNum(tc.topMr)}≉1); cannot assert tie-break`,
    };
  }
  const within = Math.abs(tc.randomMr - expected) <= tol * expected;
  return {
    name,
    status: within ? 'PASS' : 'FAIL',
    detail: `RANDOM MR=${fmtNum(tc.randomMr)} vs (|E|+1)/2=${fmtNum(expected)} (±${tol * 100}%), TOP MR=${fmtNum(tc.topMr)}`,
  };
}

/**
 * Latency gate. Activates only once the ADR-002 spike file exists (ADR-006
 * §protocol 7). SKIPs without the spike, without a predict-latency measurement
 * (needs --ann), or on a backend whose threshold is not yet set (wasm stays
 * pending: compile-only in the spike).
 */
function latencyRow(name, g, ctx) {
  const spikePath = join(BENCH_DIR, g.spike_file);
  if (!existsSync(spikePath)) {
    return { name, status: 'SKIP', detail: `spike ${g.spike_file} absent — thresholds not set yet (ADR-002)` };
  }
  const p95 = ctx.latencyP95;
  if (typeof p95 !== 'number') {
    return { name, status: 'SKIP', detail: 'no predict-latency measured (run with --ann)' };
  }
  const backend = ctx.backend;
  const threshold = backend === 'native' ? g.native_p95_ms : backend === 'wasm' ? g.wasm_p95_ms : undefined;
  if (threshold == null) {
    const why = backend === 'wasm' ? 'wasm p95 threshold pending (compile-only in the spike)' : `backend ${backend ?? 'unknown'} is not native/wasm`;
    return { name, status: 'SKIP', detail: why };
  }
  const ok = p95 <= threshold;
  return { name, status: ok ? 'PASS' : 'FAIL', detail: `p95=${fmtNum(p95)}ms <= ${threshold}ms (${backend})` };
}

/** Render the gate rows as a compact markdown table. */
export function gatesTable(result) {
  const head = '| Gate | Status | Detail |\n|---|---|---|';
  const body = result.rows.map((r) => `| ${r.name} | ${r.status} | ${r.detail ?? ''} |`).join('\n');
  return `${head}\n${body}`;
}
