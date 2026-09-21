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
}

export interface Binding {
  Engine: new (optionsJson: string) => EngineInstance;
  version(): string;
  backend: 'native' | 'wasm';
}

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
}

/** The JSON the binding's `Engine` constructor expects. */
export function toOptionsJson(opts: EngineOptions): string {
  const embedder = opts.embedder ?? 'hash';
  if (embedder === 'hash') {
    return JSON.stringify({ embedder: 'hash', dims: opts.dims ?? 256 });
  }
  return JSON.stringify({ embedder });
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
    if (mod && typeof mod.Engine === 'function') {
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
