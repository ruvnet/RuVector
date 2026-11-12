// Gate evaluation (ADR-006 §Release gates). Consumes a run's computed metric
// bag plus a context (embedder kind, suite capabilities, jev reference numbers,
// optional baseline receipt) and returns PASS / FAIL / SKIP rows. Pure: no I/O.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
export const GATES_PATH = join(HERE, '..', 'gates.json');

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

/**
 * @param metrics  flat bag: { choice_accuracy_test, ece_test, oos_auroc,
 *                 latency_p95_test, transfer_regression_pp,
 *                 jev_choice_accuracy_test, ... }
 * @param ctx      { embedderKind:'hash'|'onnx', embedderTarget:'native'|'wasm',
 *                 hasOos, hasBaselineReceipt, engineAvailable }
 * @param gatesDoc loaded gates.json
 * @returns { rows:[{name,status,detail,threshold,actual}], pass, anyFail }
 */
export function evaluateGates(metrics, ctx, gatesDoc = loadGates()) {
  const rows = [];
  // Engine unavailability short-circuits every measured gate to FAIL (strict)
  // — the caller decides whether --report-only downgrades the exit code.
  const engineDown = ctx.engineAvailable === false;

  for (const [name, g] of Object.entries(gatesDoc.gates)) {
    const applies = gateApplies(g, ctx);
    if (!applies.ok) {
      rows.push({ name, status: 'SKIP', detail: applies.why, threshold: null, actual: null });
      continue;
    }
    if (engineDown && g.metric !== 'oos_auroc') {
      rows.push({ name, status: 'FAIL', detail: 'engine unavailable', threshold: fmtThreshold(g), actual: 'n/a' });
      continue;
    }
    const actual = metrics[g.metric];
    if (actual === undefined || actual === null || Number.isNaN(actual)) {
      // A suite declaring an OOS slice cannot silently pass when its AUROC
      // has only positive rows (or no prediction for either class).
      const required = (name === 'oos_auroc' && ctx.hasOos) ||
        (g.applies_when === 'has_secondary' && ctx.hasSecondary);
      rows.push({ name, status: required ? 'FAIL' : 'SKIP', detail: `metric ${g.metric} not measured`, threshold: fmtThreshold(g), actual: null });
      continue;
    }
    let target;
    if (g.reference) {
      const ref = metrics[g.reference];
      if (ref === undefined || ref === null) {
        const required = g.applies_when === 'has_secondary' && ctx.hasSecondary;
        rows.push({ name, status: required ? 'FAIL' : 'SKIP', detail: `reference ${g.reference} not measured`, threshold: null, actual });
        continue;
      }
      target = ref + (g.margin_pp ?? 0) / 100; // margin_pp is in percentage points
    } else {
      target = g.threshold;
    }
    const ok = compare(g.op, actual, target);
    rows.push({
      name,
      status: ok ? 'PASS' : 'FAIL',
      detail: `${fmtNum(actual)} ${g.op} ${fmtNum(target)}`,
      threshold: fmtNum(target),
      actual: fmtNum(actual),
    });
  }

  const measured = rows.filter((r) => r.status !== 'SKIP');
  const anyFail = measured.some((r) => r.status === 'FAIL');
  return { rows, pass: !anyFail, anyFail, skipped: rows.filter((r) => r.status === 'SKIP').length };
}

function gateApplies(g, ctx) {
  switch (g.applies_when) {
    case undefined:
      return { ok: true };
    case 'has_oos':
      return ctx.hasOos ? { ok: true } : { ok: false, why: 'no OOS slice in this suite' };
    case 'has_secondary':
      return ctx.hasSecondary ? { ok: true } : { ok: false, why: 'no secondary ticket heads in this suite' };
    case 'embedder_native':
      return ctx.embedderTarget === 'native'
        ? { ok: true }
        : { ok: false, why: `embedder target is ${ctx.embedderTarget}, not native` };
    case 'embedder_wasm':
      return ctx.embedderTarget === 'wasm'
        ? { ok: true }
        : { ok: false, why: `embedder target is ${ctx.embedderTarget}, not wasm` };
    case 'has_baseline_receipt':
      return ctx.hasBaselineReceipt ? { ok: true } : { ok: false, why: 'no --baseline-receipt for transfer comparison' };
    default:
      return { ok: true };
  }
}

const fmtNum = (x) => (typeof x === 'number' ? Number(x.toFixed(4)) : x);
function fmtThreshold(g) {
  if (g.reference) return `${g.reference}${g.margin_pp ? ` ${g.margin_pp >= 0 ? '+' : ''}${g.margin_pp}pp` : ''}`;
  return `${g.op} ${g.threshold}`;
}

/** Render the gate rows as a compact markdown table. */
export function gatesTable(result) {
  const head = '| Gate | Status | Detail |\n|---|---|---|';
  const body = result.rows
    .map((r) => `| ${r.name} | ${r.status} | ${r.detail ?? ''} |`)
    .join('\n');
  return `${head}\n${body}`;
}
