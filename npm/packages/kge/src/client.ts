/**
 * The public client: `createKge(opts)` returns a `Kge` bound to one model
 * instance; `loadKge(json, opts)` rebuilds one from a saved envelope. Every
 * call is one hop across the binding seam; nothing here touches the network,
 * the filesystem, or a subprocess.
 */

import {
  Binding,
  EngineOptions,
  ModelInstance,
  resolveDefaultBinding,
  toOptionsJson,
} from './binding';
import { isErrorShape, KgeError } from './errors';
import type { Schema } from './schema';
import type {
  AddReport,
  OptimizeReport,
  OptimizeSpec,
  PredictResult,
  SimilarResult,
  Stats,
  Triple,
} from './types';

export interface KgeOptions<R extends string = string> extends EngineOptions {
  /** Inject a binding (tests, or a non-default addon). Defaults to `../index.js`. */
  binding?: Binding;
  /** Optional schema; narrows the `r` argument of the query verbs to its union. */
  schema?: Schema<R>;
}

/** Tail query `(s, r, ?)` — resolve the object. */
export interface TailQuery<R extends string> {
  s: string;
  r: R;
  o?: never;
  k?: number;
  /** Force exhaustive scoring even if an ANN index exists (default true). */
  useIndex?: boolean;
}
/** Head query `(?, r, o)` — resolve the subject. */
export interface HeadQuery<R extends string> {
  o: string;
  r: R;
  s?: never;
  k?: number;
  /** Force exhaustive scoring even if an ANN index exists (default true). */
  useIndex?: boolean;
}
export type PredictQuery<R extends string> = TailQuery<R> | HeadQuery<R>;

export interface ComposeQuery<R extends string> {
  r1: R;
  r2: R;
  s: string;
  k?: number;
}

export interface Kge<R extends string = string> {
  readonly version: string;
  readonly backend: 'native' | 'wasm';
  /** Admit `[{s,r,o}]` facts, interning labels. */
  addTriples(triples: readonly Triple[]): AddReport;
  /** Rank candidates for the open slot of a link-prediction query. */
  predict(query: PredictQuery<R>): PredictResult;
  /** Rank relations by cosine similarity to `r`. */
  similarRelations(query: { r: R; k?: number }): SimilarResult;
  /** Rank tails for the composed relation `r1 ∘ r2` from `s` (RotatE only). */
  compose(query: ComposeQuery<R>): PredictResult;
  /** Build the ANN index over entities; `predict` then uses it. */
  buildIndex(): Record<string, unknown>;
  /** Train the tables in place (mini-batch, ADR-003). */
  train(config?: Record<string, unknown>): Promise<Record<string, unknown>>;
  /** Filtered evaluation over a (derived) split. */
  evaluate(config?: Record<string, unknown>): Record<string, unknown>;
  /**
   * Run one self-optimization campaign (ADR-004): fit and gate the HPO/model
   * arms over the model's splits, then install the champion's trained tables.
   */
  optimize(spec?: OptimizeSpec): OptimizeReport;
  /**
   * Serialize to a hash-carrying envelope (pass to {@link loadKge}). With
   * `{ key }` (at least 16 bytes) the envelope also carries an HMAC-SHA256,
   * so `loadKge(json, { key })` rejects any edit made without the key. The
   * plain sha256 only catches corruption: anyone can recompute it.
   */
  save(opts?: { key?: string }): string;
  /** Model introspection. */
  stats(): Stats;
}

function parse<T>(json: string): T {
  const value: unknown = JSON.parse(json);
  if (isErrorShape(value)) {
    throw new KgeError(value.error.message, value.error.kind);
  }
  return value as T;
}

function wrap<R extends string>(binding: Binding, model: ModelInstance): Kge<R> {
  return {
    version: binding.version(),
    backend: binding.backend,
    addTriples: (triples) =>
      parse<AddReport>(model.addTriplesJson(JSON.stringify(triples))),
    predict: (query) => parse<PredictResult>(model.predictJson(JSON.stringify(query))),
    similarRelations: (query) =>
      parse<SimilarResult>(model.similarRelationsJson(JSON.stringify(query))),
    compose: (query) => parse<PredictResult>(model.composeJson(JSON.stringify(query))),
    buildIndex: () => parse<Record<string, unknown>>(model.buildIndexJson()),
    train: async (config) => {
      const payload = JSON.stringify(config ?? {});
      const raw =
        typeof model.train === 'function'
          ? await model.train(payload)
          : model.trainJson(payload);
      return parse<Record<string, unknown>>(raw);
    },
    evaluate: (config) =>
      parse<Record<string, unknown>>(model.evalJson(JSON.stringify(config ?? {}))),
    optimize: (spec) =>
      parse<OptimizeReport>(model.optimizeJson(JSON.stringify(spec ?? {}))),
    save: (opts) => {
      if (!hasKey(opts)) return model.toJson();
      if (typeof model.toJsonSigned !== 'function') {
        throw new KgeError('this binding cannot sign models; rebuild @ruvector/kge', 'unavailable');
      }
      return callOrInvalid(() => model.toJsonSigned!(opts!.key!));
    },
    stats: () => parse<Stats>(model.statsJson()),
  };
}

function requireBinding(binding?: Binding): Binding {
  const resolved = binding ?? resolveDefaultBinding();
  if (!resolved) {
    throw new KgeError(
      'no @ruvector/kge binding found — build the native addon ' +
        '(scripts/build-native.sh) or the WASM fallback (scripts/build-wasm.sh); ' +
        'pass { binding } to inject one',
      'unavailable',
    );
  }
  return resolved;
}

/** Create a fresh model. */
export function createKge<R extends string = string>(
  opts: KgeOptions<R> = {},
): Kge<R> {
  const binding = requireBinding(opts.binding);
  const model = new binding.Model(toOptionsJson(opts));
  return wrap<R>(binding, model);
}

/**
 * Rebuild a model from a `save()` envelope; throws on a hash mismatch. With
 * `{ key }`, the envelope must also carry an HMAC that verifies under that
 * key (see `save({ key })`); unsigned or edited envelopes throw
 * `KgeError{kind:'invalid'}`.
 */
export function loadKge<R extends string = string>(
  modelJson: string,
  opts: { binding?: Binding; schema?: Schema<R>; key?: string } = {},
): Kge<R> {
  const binding = requireBinding(opts.binding);
  if (hasKey(opts)) {
    const verify = binding.Model.fromJsonVerified;
    if (typeof verify !== 'function') {
      throw new KgeError('this binding cannot verify signed models; rebuild @ruvector/kge', 'unavailable');
    }
    const model = callOrInvalid(() => verify.call(binding.Model, modelJson, opts.key!));
    return wrap<R>(binding, model);
  }
  const model = callOrInvalid(() => binding.Model.fromJson(modelJson));
  return wrap<R>(binding, model);
}

/**
 * True when the caller asked for signing/verification. A `key` property that
 * is present but not a string (e.g. an unset `process.env` variable) throws
 * rather than silently falling back to the unsigned path.
 */
function hasKey(opts?: { key?: unknown }): boolean {
  if (!opts || !Object.prototype.hasOwnProperty.call(opts, 'key')) return false;
  if (typeof opts.key !== 'string') {
    throw new KgeError('key must be a string; got ' + typeof opts.key, 'invalid');
  }
  return true;
}

/** Run a binding call that throws plain errors, rethrowing as KgeError{invalid}. */
function callOrInvalid<T>(fn: () => T): T {
  try {
    return fn();
  } catch (e) {
    if (e instanceof KgeError) throw e;
    throw new KgeError(e instanceof Error ? e.message : String(e), 'invalid');
  }
}
