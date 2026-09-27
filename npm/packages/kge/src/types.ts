/**
 * Wire types for @ruvector/kge. Pure types: no runtime code, no I/O. The JSON
 * on the wire is produced by the Rust binding (`crates/ruvector-kge-*`); these
 * mirror it exactly.
 */

/**
 * The shipped scorers: HolE (default) and RotatE (ADR-002), and ComplEx — the
 * product scorer of the ComplEx-N3-R recipe (ADR-007 §3; `dims = 2·rank`).
 */
export type ScorerKind = 'hole' | 'rotate' | 'complex';

/** A split tag for a triple (ADR-006 frozen splits). */
export type SplitTag = 'train' | 'valid' | 'test' | 'transfer';

/** One (subject, relation, object) fact, by label, optionally split-tagged. */
export interface Triple {
  s: string;
  r: string;
  o: string;
  /** Pin this triple to a split; `train`/`eval` then honour it verbatim. */
  split?: SplitTag;
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

/** Triple counts per split label. */
export interface SplitCounts {
  train: number;
  valid: number;
  transfer: number;
  test: number;
  unlabelled: number;
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
  /** True for a reciprocal-relations model (`2·R` relation rows, ADR-007 §3). */
  reciprocal: boolean;
  /** Triple counts per split label. */
  splits: SplitCounts;
}

/** One training configuration (the HPO knobs), as an `optimize` report carries it. */
export interface Knobs {
  dims: number;
  lr: number;
  optimizer: 'adam' | 'adagrad';
  loss: 'cross-entropy' | 'bce' | 'margin';
  neg_count: number;
  temperature: number;
  n3_lambda: number;
  epochs: number;
  scorer: ScorerKind;
}

/** Optional HPO grid override for `optimize` (the axes are cartesian-producted). */
export interface OptimizeGrid {
  dims: number[];
  lrs: number[];
  losses: ('cross-entropy' | 'bce' | 'margin')[];
  n3_lambdas: number[];
}

/** Options for a self-optimization campaign (ADR-004); every field is optional. */
export interface OptimizeSpec {
  /** Per-campaign gated-evaluation budget (capped at 64). Default 16. */
  budget?: number;
  /** Anytime paired-test type-I bound, in `(0, 1)`. Default 0.05. */
  alpha?: number;
  /** Seed for the split/train/eval RNGs; defaults to the model's seed. */
  seed?: number;
  /** `[train, valid, transfer, test]` ratios, used only when triples are untagged. */
  splitRatios?: [number, number, number, number];
  /** Transfer-split non-regression tolerance. Default 0.05. */
  transferTolerance?: number;
  /** Cost weight in the bandit reward `MRR − λ·cost`. Default 0.1. */
  lambdaCost?: number;
  /** Override the HPO grid; the default is the model's dims at two learning rates. */
  grid?: OptimizeGrid;
}

/** A baseline-vs-champion MRR pair on one split. */
export interface MrrPair {
  baselineMrr: number;
  championMrr: number;
}

/** One proposal's gate decision (mirrors the Rust receipt's tagged enum). */
export type GateDecision =
  | { decision: 'promote' }
  | { decision: 'reject'; reason: string }
  | { decision: 'paused'; reason: string };

/** One proposal row in an `optimize` report. Ids are content hashes carried as
 * strings (a full u64 would lose precision as a JS number). */
export interface OptimizeProposal {
  id: string;
  parent: string | null;
  arm: 'hpo' | 'model-arm' | 'continual';
  decision: GateDecision;
}

/** Result of a self-optimization campaign (`optimize`). */
export interface OptimizeReport {
  /** The promoted (or, if nothing promoted, the retained baseline) knobs. */
  champion: Knobs;
  /** Content-hash id as a string; matches a receipt's `knobs_hash` exactly. */
  championId: string;
  /** True when a proposal beat the baseline and was promoted. */
  promoted: boolean;
  /** True when the champion's trained tables were written into the model. */
  installed: boolean;
  /** True when the daily budget was exhausted mid-campaign. */
  paused: boolean;
  budgetConsumed: number;
  /** `per-triple` when frozen split tags were used, else `split4`. */
  splitSource: 'per-triple' | 'split4';
  proposalCount: number;
  proposals: OptimizeProposal[];
  /** Filtered validation MRR, baseline vs champion (the reward signal). */
  val: MrrPair;
  /** Filtered transfer MRR, baseline vs champion (the non-regression check). */
  transfer: MrrPair;
  /** Filtered test MRR, scored exactly twice (baseline, champion). */
  test: MrrPair;
  scorer: ScorerKind;
  dims: number;
  receiptsCount: number;
  /** The hash-chained receipt log, one JSON object per line (JSONL). */
  receipts: string;
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

/** 1-N kernel for the batched 1-vs-all path (ComplEx only). */
export type OneNKernel = 'naive' | 'gemm';

/**
 * `train` config (the Rust `TrainConfig`). Every field is optional; omitted
 * fields keep the trainer defaults, so older configs are unchanged. `dims` is
 * always the model's, and `reciprocal` is fixed by the constructor option
 * (omit it, or pass the model's value; a contradicting value is `invalid`).
 * The ComplEx-N3-R recipe (ADR-007 §3) on a `{scorer:"complex",
 * reciprocal:true}` model: `{loss:{kind:'one_vs_all'}, n3_form:'moduli',
 * loss_reduction:'mean', init:{kind:'normal',scale:1e-3},
 * optimizer:{kind:'adagrad'}, optim_state:'dense', lr:0.1, rp_weight:0.05,
 * one_n_kernel:'gemm'}`.
 */
export interface TrainConfig {
  epochs?: number;
  batch_size?: number;
  lr?: number;
  seed?: number;
  optimizer?:
    | { kind: 'adagrad'; epsilon?: number }
    | { kind: 'adam'; beta1?: number; beta2?: number; epsilon?: number };
  loss?:
    | { kind: 'one_vs_all' }
    | { kind: 'self_adversarial'; neg_count?: number; temperature?: number; margin?: number };
  n3_lambda?: number;
  /** Must equal the model's constructor option when given. */
  reciprocal?: boolean;
  init?: { kind: 'keep' } | { kind: 'xavier' } | { kind: 'normal'; scale?: number };
  n3_form?: 'elementwise' | 'moduli';
  loss_reduction?: 'sum' | 'mean';
  /** Relation-prediction auxiliary weight (Chen 2021); 0 = off. */
  rp_weight?: number;
  optim_state?: 'sparse' | 'dense';
  one_n_kernel?: OneNKernel;
}
