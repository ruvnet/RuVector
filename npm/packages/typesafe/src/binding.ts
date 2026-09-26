/**
 * The native/WASM binding seam. `require('../index.js')` (built by the bindings
 * agent) resolves to a `Binding`; until it exists, tests inject a fake through
 * `createTypesafe({ binding })`. Resolution of the default binding is lazy, so
 * importing this package never loads a native addon on its own.
 */

/** One engine instance, constructed from an options JSON string. */
export interface EngineInstance {
  /** Synchronous decision: DecisionResponse JSON or an error-envelope JSON. */
  decideJson(requestJson: string): string;
  /** Admit labeled examples for one question: TrainReport JSON or error JSON. */
  trainJson(trainJson: string): string;
  /** Engine stats (bank sizes, per-question heads, calibration state) as JSON. */
  statsJson(): string;
  /** Optional async decision on the native path (used by the batch pool). */
  decide?(requestJson: string): Promise<string>;
  /** Optional: run an optimize campaign — CampaignReport JSON or error JSON. */
  optimizeJson?(campaignJson: string): string;
  /** Optional: export the full example bank JSON (the user's own examples). */
  exportBankJson?(): string;
  /** Optional: replace the bank from JSON — `{"ok":true}` or error JSON. */
  importBankJson?(bankJson: string): string;
}

export interface Binding {
  Engine: new (optionsJson: string) => EngineInstance;
  version(): string;
  backend: 'native' | 'wasm';
}

import type { EngineTuning } from './optimize';

/** Embedder selection for the engine constructor (ADR-002). */
export type EmbedderConfig =
  | 'hash'
  | { kind: 'onnx'; modelDir: string; manifest: string };

export interface EngineOptions {
  /**
   * Which embedder to build. Defaults to `"hash"`: a deterministic bag-of-words
   * test-double (no weights, runs everywhere) whose answers are always reported
   * `calibrated: false`. Production accuracy needs the `onnx` embedder.
   */
  embedder?: EmbedderConfig;
  /** Embedding width for the hash embedder (ignored by onnx). */
  dims?: number;
  /** Tunable engine knobs (ADR-004). Absent → the core defaults. */
  engine?: EngineTuning;
}

/** The JSON the binding's `Engine` constructor expects. */
export function toOptionsJson(opts: EngineOptions): string {
  const embedder = opts.embedder ?? 'hash';
  const base: Record<string, unknown> =
    embedder === 'hash'
      ? { embedder: 'hash', dims: opts.dims ?? 256 }
      : { embedder };
  if (opts.engine) base.engine = opts.engine;
  return JSON.stringify(base);
}

let cached: Binding | null | undefined;

/**
 * Resolve the default binding lazily. Returns `null` when no artifact was
 * built, so the caller can suggest a build. A present but broken addon must
 * retain its load error (including its code and cause) for diagnosis.
 */
export function resolveDefaultBinding(): Binding | null {
  if (cached !== undefined) return cached;
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require('../index.js') as Partial<Binding>;
    if (mod && typeof mod.Engine === 'function') {
      cached = mod as Binding;
      return cached;
    }
    cached = null;
  } catch (error) {
    // The package loader falls back to WASM when no native artifact exists.
    // Only its missing WASM module (or a missing loader during development) is
    // an expected absence. In particular, ERR_DLOPEN_FAILED from a *present*
    // native binary must not be misreported as "no binding found".
    if (!isMissingArtifact(error)) throw error;
    cached = null;
  }
  return cached;
}

function isMissingArtifact(error: unknown): boolean {
  if (typeof error !== 'object' || error === null || !('code' in error) ||
      error.code !== 'MODULE_NOT_FOUND' || !('message' in error) ||
      typeof error.message !== 'string') {
    return false;
  }
  const firstLine = error.message.split('\n', 1)[0];
  return firstLine === "Cannot find module '../index.js'" ||
    firstLine === "Cannot find module './wasm/ruvector_typesafe_wasm.js'";
}

/** Test hook: clear the memoised default binding. */
export function _resetBindingCache(): void {
  cached = undefined;
}
