// Hand-written type contract for @ruvector/kge's binding surface.
// Both the native (napi-rs) and wasm (wasm-bindgen) backends implement it; the
// only difference is the native-only async `train`. JSON strings on the wire
// follow the shapes documented per method.

/** Backend that was loaded at require() time. */
export type Backend = 'native' | 'wasm';

/** The closed set of request-error kinds (never thrown for a request). */
export type KgeErrorKind =
  | 'limit'
  | 'invalid'
  | 'unavailable'
  | 'unsupported'
  | 'scorer';

/** The one error shape every request-level failure returns. */
export interface KgeErrorJson {
  error: {
    kind: KgeErrorKind;
    message: string;
  };
}

/**
 * A knowledge-graph embedding model over one scorer (HolE or RotatE).
 *
 * Every `*Json` method returns a JSON **string**: either the success payload or
 * a {@link KgeErrorJson}. They never throw for a request-level failure. The
 * constructor and `fromJson` DO throw for programmer errors (malformed options
 * JSON, an odd/zero `dims`, a tampered model envelope).
 */
export declare class Model {
  /** @param optionsJson e.g. `{"scorer":"hole","dims":256,"seed":42}`. */
  constructor(optionsJson: string);

  /** Rebuild from a `toJson` envelope; throws on a hash mismatch. */
  static fromJson(modelJson: string): Model;

  /** Serialize to a hash-carrying `{"sha256","model"}` envelope. */
  toJson(): string;

  /** Admit `[{"s","r","o"}]`. Returns a summary or error JSON. */
  addTriplesJson(triplesJson: string): string;

  /** Link prediction: `{"s","r","k"}` or `{"r","o","k"}`. */
  predictJson(queryJson: string): string;

  /** Relation similarity (cosine): `{"r","k"}`. */
  similarRelationsJson(queryJson: string): string;

  /** 2-hop composition (RotatE only): `{"r1","r2","s","k"}`. */
  composeJson(queryJson: string): string;

  /** Build the ANN index over entities. */
  buildIndexJson(): string;

  /** Train synchronously. Returns a report or error JSON. */
  trainJson(configJson: string): string;

  /** Filtered evaluation: `{"split":"test", ...}`. */
  evalJson(configJson: string): string;

  /** Self-optimization campaign. */
  optimizeJson(campaignJson: string): string;

  /** `{scorer,dims,seed,entities,relations,triples,indexed,splits}` as JSON. */
  statsJson(): string;

  /**
   * Native backend only: the async form of `trainJson`, run off the JS thread
   * as a napi `AsyncTask`. Absent on the wasm backend.
   */
  train?(configJson: string): Promise<string>;
}

/** Crate (== Cargo workspace) version, e.g. `"2.3.0"`. */
export declare function version(): string;

/** Which backend this process loaded. */
export declare const backend: Backend;
