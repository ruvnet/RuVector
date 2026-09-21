/**
 * Wire types for @ruvector/kge. Pure types: no runtime code, no I/O. The JSON
 * on the wire is produced by the Rust binding (`crates/ruvector-kge-*`); these
 * mirror it exactly.
 */

/** The two shipped scorers (ADR-002). */
export type ScorerKind = 'hole' | 'rotate';

/** One (subject, relation, object) fact, by label. */
export interface Triple {
  s: string;
  r: string;
  o: string;
}

/** Summary returned by `addTriples`. */
export interface AddReport {
  added: number;
  entities: number;
  relations: number;
  triples: number;
}

/** A ranked entity for an open slot. */
export interface Candidate {
  entity: string;
  score: number;
}

/** Result of `predict` / `compose`: ranked candidates for the open slot. */
export interface PredictResult {
  candidates: Candidate[];
  /** True when scored exhaustively (not via ANN). */
  exact: boolean;
  /** True when the ANN index served the candidates. */
  ann: boolean;
}

/** A relation scored by cosine similarity. */
export interface RelationScore {
  relation: string;
  score: number;
}

/** Result of `similarRelations`. */
export interface SimilarResult {
  relations: RelationScore[];
}

/** Model introspection (`stats`). */
export interface Stats {
  scorer: ScorerKind;
  dims: number;
  seed: number;
  entities: number;
  relations: number;
  triples: number;
  indexed: boolean;
}

/** The closed set of request-error kinds. */
export type KgeErrorKind =
  | 'limit'
  | 'invalid'
  | 'unavailable'
  | 'unsupported'
  | 'scorer';

/** The error envelope the binding returns instead of a result. */
export interface KgeErrorShape {
  error: {
    kind: KgeErrorKind;
    message: string;
  };
}
