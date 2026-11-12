/**
 * The native/WASM binding seam. `require('../index.js')` (the loader) resolves
 * to a `Binding`; until it exists, tests inject a fake through
 * `createKge({ binding })`. Resolution of the default binding is lazy, so
 * importing this package never loads a native addon on its own.
 */

import type { ScorerKind } from './types';

/** One model instance: JSON-in / JSON-out, mirroring the Rust `Model` class. */
export interface ModelInstance {
  toJson(): string;
  /** Signed envelope (HMAC-SHA256 under `key`); present in bindings built after 0.1.0. */
  toJsonSigned?(key: string): string;
  addTriplesJson(triplesJson: string): string;
  predictJson(queryJson: string): string;
  similarRelationsJson(queryJson: string): string;
  composeJson(queryJson: string): string;
  buildIndexJson(): string;
  trainJson(configJson: string): string;
  evalJson(configJson: string): string;
  optimizeJson(campaignJson: string): string;
  statsJson(): string;
  /** Native-only async train (used by the client when present). */
  train?(configJson: string): Promise<string>;
}

/** The `Model` constructor, with the static `fromJson` factory. */
export interface ModelCtor {
  new (optionsJson: string): ModelInstance;
  fromJson(modelJson: string): ModelInstance;
  /** Load a signed envelope, requiring its HMAC to verify under `key`. */
  fromJsonVerified?(modelJson: string, key: string): ModelInstance;
}

export interface Binding {
  Model: ModelCtor;
  version(): string;
  backend: 'native' | 'wasm';
}

export interface EngineOptions {
  /** Which scorer to build. Default `"hole"`. */
  scorer?: ScorerKind;
  /** Embedding width; must be even and > 0. Default `256`. */
  dims?: number;
  /** Deterministic init seed. Default `42`. */
  seed?: number;
}

/** The options JSON the binding's `Model` constructor expects. */
export function toOptionsJson(opts: EngineOptions): string {
  return JSON.stringify({
    scorer: opts.scorer ?? 'hole',
    dims: opts.dims ?? 256,
    seed: opts.seed ?? 42,
  });
}

let cached: Binding | null | undefined;

/**
 * Resolve the default binding lazily. Returns `null` (never throws) when the
 * native addon and the WASM fallback are both absent, so a caller can surface a
 * clean "run npm run build" message instead of a require stack.
 */
export function resolveDefaultBinding(): Binding | null {
  if (cached !== undefined) return cached;
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require('../index.js') as Partial<Binding>;
    if (mod && typeof mod.Model === 'function') {
      cached = mod as Binding;
      return cached;
    }
    cached = null;
  } catch {
    cached = null;
  }
  return cached;
}

/** Test hook: clear the memoised default binding. */
export function _resetBindingCache(): void {
  cached = undefined;
}
