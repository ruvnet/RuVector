// Hand-written type contract for @ruvector/typesafe's binding surface.
// Both the native (napi-rs) and wasm (wasm-bindgen) backends implement it; the
// only differences are noted per-member. The JSON strings on the wire follow
// `ruvector_typesafe_core::types` (DecisionRequest / DecisionResponse / etc.).

/** Backend that was loaded at require() time. */
export type Backend = 'native' | 'wasm';

/** The one error shape every request-level failure returns (never thrown). */
export interface TypesafeErrorJson {
  error: {
    kind: 'limit' | 'invalid' | 'embedder';
    message: string;
  };
}

/**
 * A compiled typed-decision engine over one embedder.
 *
 * `decideJson` / `trainJson` / `statsJson` return a JSON **string**: either the
 * success payload (a `DecisionResponse` / `TrainReport` / stats object) or a
 * {@link TypesafeErrorJson}. They never throw for a request-level failure.
 * The constructor and `fromBytes` DO throw for programmer errors (malformed
 * options JSON, an unsupported embedder).
 */
export declare class Engine {
  /**
   * @param optionsJson e.g. `{"embedder":"hash","dims":256}`, or
   *   `{"embedder":{"kind":"onnx","modelDir":"...","manifest":"..."}}`
   *   (onnx is native-only and not yet built — currently throws).
   */
  constructor(optionsJson: string);

  /** Answer every question in the request. Returns response-or-error JSON. */
  decideJson(requestJson: string): string;

  /** Admit labeled examples for one question. Returns report-or-error JSON. */
  trainJson(trainJson: string): string;

  /** `{"embedderId","dims","questionsCompiled","examples"}` as JSON. */
  statsJson(): string;

  /**
   * Native backend only: the async form of `decideJson`, run off the JS thread
   * as a napi `AsyncTask`. Absent on the wasm backend.
   */
  decide?(requestJson: string): Promise<string>;

  /**
   * WASM backend only: build an onnx-backed engine from in-memory bytes.
   * Not yet implemented — currently throws the embedder error JSON.
   */
  static fromBytes?(
    optionsJson: string,
    modelBytes: Uint8Array,
    tokenizerBytes: Uint8Array,
  ): Engine;
}

/** Crate (== Cargo workspace) version, e.g. `"2.3.0"`. */
export declare function version(): string;

/** Which backend this process loaded. */
export declare const backend: Backend;
