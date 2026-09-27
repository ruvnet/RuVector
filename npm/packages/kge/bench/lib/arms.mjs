// Binding adapter (ADR-006 §protocol). EVERY call into the native/WASM Model
// contract goes through this one file, so when the real binding's exact JSON
// shape lands (built concurrently by kge-bindings-ts) the blast radius is here,
// not spread through run.mjs.
//
// Contract (assumed shapes flagged where the binding is not yet frozen):
//   require('../index.js') -> { Model, version(), backend }
//   new Model('{"scorer":"hole","dims":256,"seed":42,"epochs":N}')
//   model.addTriplesJson('[{"s","r","o","split"}]')            -> '{"added":N}' | {error}
//   model.trainJson('{"loss":...,"epochs":N}')                 -> '{"epochs":N,"triplesPerSec":X}' | {error}
//   model.evalJson('{"split":"test","tieBreak":"random"}')     -> {mrr,mr,hits:{1,3,10},perSide} | {error}
//   model.predictJson('{"s","r","k":10,"useIndex":true}')      -> {candidates:[{o,score}]} | {error}
//   model.predictJson('{"s","r","o"}')                         -> {score} | {error}
//   model.buildIndexJson('{"metric":"dot","m":16,"efConstruction":200}') -> {built} | {error}
//   model.optimizeJson('{...}')                                -> {...} | {error}
//   model.statsJson()                                          -> {...}
//
// Any method that is missing, throws, returns non-JSON, or returns an {error}
// envelope on the FIRST call is surfaced as { available:false, error } — the
// harness then reports "engine unavailable" for that arm and never crashes.

import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';

/**
 * Resolve the binding module. Order: injected (tests) -> KGE_BENCH_BINDING env
 * (dev-only override, e.g. the fake) -> ../index.js -> ../dist/index.js.
 */
export function resolveBinding(pkgRoot, injected) {
  if (injected) return { binding: injected, source: 'injected' };
  const require = createRequire(import.meta.url);
  const envPath = process.env.KGE_BENCH_BINDING;
  if (envPath) {
    try {
      const abs = envPath.startsWith('.') ? join(pkgRoot, envPath) : envPath;
      return { binding: require(abs), source: `env:KGE_BENCH_BINDING(${envPath})`, path: safeResolve(require, abs) };
    } catch (e) {
      return { binding: null, error: `require(${envPath}) failed: ${e && e.message}` };
    }
  }
  for (const p of [join(pkgRoot, 'index.js'), join(pkgRoot, 'dist', 'index.js')]) {
    if (existsSync(p)) {
      try {
        return { binding: require(p), source: p, path: p };
      } catch (e) {
        return { binding: null, error: `require(${p}) failed: ${e && e.message}` };
      }
    }
  }
  return { binding: null, error: 'no index.js / dist/index.js — core not built yet' };
}

/** Absolute file a module id resolves to (for provenance hashing), or null. */
function safeResolve(require, id) {
  try {
    return require.resolve(id);
  } catch {
    return null;
  }
}

const parse = (raw) => (typeof raw === 'string' ? JSON.parse(raw) : raw);

/** Construct a Model. Returns { model } or { error }. */
export function constructModel(binding, config) {
  if (!binding || typeof binding.Model !== 'function') return { error: 'binding has no Model constructor' };
  try {
    return { model: new binding.Model(JSON.stringify(config)) };
  } catch (e) {
    return { error: `new Model failed: ${e && e.message}` };
  }
}

/** Call one JSON method; classify a not-implemented/error envelope uniformly. */
function callJson(model, method, argObj) {
  if (!model || typeof model[method] !== 'function') {
    return { available: false, error: `binding has no ${method}` };
  }
  let raw;
  try {
    raw = argObj === undefined ? model[method]() : model[method](JSON.stringify(argObj));
  } catch (e) {
    return { available: false, error: `${method} threw: ${e && e.message ? e.message : e}` };
  }
  let resp;
  try {
    resp = parse(raw);
  } catch {
    return { available: false, error: `${method} returned non-JSON: ${String(raw).slice(0, 120)}` };
  }
  if (resp && resp.error) {
    return { available: false, error: resp.error.message ?? JSON.stringify(resp.error), errorKind: resp.error.kind };
  }
  return { available: true, resp };
}

export function addTriples(model, tagged) {
  const r = callJson(model, 'addTriplesJson', tagged);
  return r.available ? { available: true, added: r.resp.added ?? tagged.length } : r;
}

export function train(model, config = {}) {
  const t0 = performance.now();
  const r = callJson(model, 'trainJson', config);
  if (!r.available) return r;
  const m = r.resp;
  return {
    available: true,
    epochs: m.epochs ?? config.epochs ?? null,
    triplesPerSec: m.triplesPerSec ?? null,
    loss: m.loss ?? null,
    n3Penalty: m.n3Penalty ?? null,
    batches: m.batches ?? null,
    wallMs: performance.now() - t0,
  };
}

/** Normalise a MetricSet ({mrr,mr,hits1/3/10} or {mrr,mr,hits:{...}}) to {mrr,mr,hits}. */
function metricSet(m) {
  if (!m) return null;
  const hits = m.hits ?? { 1: m.hits1, 3: m.hits3, 10: m.hits10 };
  return { mrr: m.mrr, mr: m.mr, hits };
}

/**
 * Filtered MRR/Hits for a split at a tie-break mode. The binding returns
 * `{report:{combined,head,tail}, split, splitSource, filtered}`; older stubs
 * returned a flat `{mrr,mr,hits,perSide}`. Both are accepted.
 */
export function evalSplit(model, split, tieBreak) {
  const r = callJson(model, 'evalJson', { split, tieBreak });
  if (!r.available) return r;
  const m = r.resp;
  const combined = m.report ? metricSet(m.report.combined) : metricSet(m);
  const perSide = m.report
    ? { head: metricSet(m.report.head), tail: metricSet(m.report.tail) }
    : m.perSide ?? null;
  return {
    available: true,
    mrr: combined?.mrr,
    mr: combined?.mr,
    hits: combined?.hits ?? {},
    perSide,
    splitSource: m.splitSource ?? null,
    filtered: m.filtered ?? null,
  };
}

/**
 * Top-k tail candidates for (s,r,?), with or without the ANN index. Timed.
 * The binding returns candidates as { entity, score } (older stubs used { o }).
 */
export function predictTopK(model, { s, r, k = 10, useIndex = true }) {
  const t0 = performance.now();
  const res = callJson(model, 'predictJson', { s, r, k, useIndex });
  const latencyMs = performance.now() - t0;
  if (!res.available) return { ...res, latencyMs };
  const candidates = (res.resp.candidates ?? []).map((c) => ({ o: c.entity ?? c.o, score: c.score }));
  return { available: true, candidates, latencyMs };
}

/**
 * Confidence/score of one specific triple (adversarial drop + hard-negatives).
 * The binding requires exactly one of s/o OPEN in a predict, so there is no
 * all-three scoring call: we open the tail, ask for a deep candidate list
 * (capped at the ADR-005 k≤1000 limit), and read the score of `o`. A triple
 * ranked beyond the cap returns score:null (excluded by callers).
 */
export function scoreTriple(model, { s, r, o }, { k = 1000 } = {}) {
  const res = callJson(model, 'predictJson', { s, r, k, useIndex: false });
  if (!res.available) return res;
  const cand = (res.resp.candidates ?? []).find((c) => (c.entity ?? c.o) === o);
  return { available: true, score: cand ? cand.score : null };
}

export function buildIndex(model, params = { metric: 'dot', m: 16, efConstruction: 200 }) {
  return callJson(model, 'buildIndexJson', params);
}

export function optimize(model, params = {}) {
  return callJson(model, 'optimizeJson', params);
}

export function readStats(model) {
  if (!model || typeof model.statsJson !== 'function') return null;
  try {
    const s = model.statsJson();
    return typeof s === 'string' ? JSON.parse(s) : s;
  } catch {
    return null;
  }
}

export const safeVersion = (binding) => {
  try {
    return typeof binding.version === 'function' ? binding.version() : null;
  } catch {
    return null;
  }
};

// ---------------------------------------------------------------------------
// Tie-check model (ADR-007 M0). A zero-epoch real model still has Xavier-random
// tables, so it is never all-tied and the RANDOM tie-break gate could never run
// on the real binding. Instead we zero the tables of a built model through the
// public toJson/fromJson envelope. `fromJson` verifies sha256 over serde_json's
// RE-serialization of the parsed model, so the body must stay byte-identical
// except for the table floats: we never JSON.parse/stringify it, we rewrite only
// the number tokens of the two table arrays to `0.0` (serde's f32 zero — pinned
// by the Rust test eval::tests::zeroed_tables_serialize_as_literal_zero_point_zero)
// and re-hash. An all-zero HolE/RotatE model scores every triple exactly 0.
// ---------------------------------------------------------------------------

const TABLES_KEY = '"tables":{"dims":';
const ENVELOPE_HEAD = /^\{"sha256":"[0-9a-f]{64}","model":/;

/** Count the comma-separated tokens of an array body without materialising them. */
function tokenCount(inner) {
  if (inner.trim() === '') return 0;
  let n = 1;
  for (let i = 0; i < inner.length; i++) if (inner.charCodeAt(i) === 44) n++;
  return n;
}

/**
 * Rewrite the `entities`/`relations` float arrays of the (last) `"tables"`
 * object in a serialized model body to all `0.0`. `tables` is the last
 * serialized field of the FFI model, and an unescaped `"` cannot occur inside a
 * JSON string, so `lastIndexOf` finds the real key. Returns { body } | { error }.
 */
export function zeroTablesText(body) {
  const at = body.lastIndexOf(TABLES_KEY);
  if (at < 0) return { error: 'model JSON has no built tables' };
  let replaced = 0;
  const tail = body.slice(at).replace(/("(?:entities|relations)":\[)([^\]]*)(\])/g, (_, open, inner, close) => {
    replaced++;
    const n = tokenCount(inner);
    return open + (n ? '0.0,'.repeat(n - 1) + '0.0' : '') + close;
  });
  if (replaced !== 2) return { error: `expected 2 table arrays, found ${replaced}` };
  return { body: body.slice(0, at) + tail };
}

/** True when the binding can round-trip a model through toJson/fromJson. */
export function canZeroTables(binding, model) {
  return typeof binding?.Model?.fromJson === 'function' && typeof model?.toJson === 'function';
}

/**
 * A copy of `model` (tables already built) with every embedding zeroed, via the
 * hash-carrying envelope. Returns { available:true, model } | { available:false, error }.
 */
export function zeroTableModel(binding, model) {
  if (!canZeroTables(binding, model)) return { available: false, error: 'binding has no toJson/fromJson' };
  let env;
  try {
    env = model.toJson();
  } catch (e) {
    return { available: false, error: `toJson threw: ${e && e.message ? e.message : e}` };
  }
  const head = typeof env === 'string' ? ENVELOPE_HEAD.exec(env) : null;
  if (!head || !env.endsWith('}')) return { available: false, error: 'toJson envelope has an unexpected shape' };
  const z = zeroTablesText(env.slice(head[0].length, -1));
  if (z.error) return { available: false, error: z.error };
  const digest = createHash('sha256').update(z.body).digest('hex');
  try {
    return { available: true, model: binding.Model.fromJson(`{"sha256":"${digest}","model":${z.body}}`) };
  } catch (e) {
    return { available: false, error: `fromJson rejected the zeroed model: ${e && e.message ? e.message : e}` };
  }
}
