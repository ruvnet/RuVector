/**
 * Campaign wire types — a 1:1 mirror of
 * `crates/ruvector-typesafe-core/src/engine/optimize.rs` and `options.rs`.
 * Pure types: no runtime code, no I/O. The optimize campaign (ADR-004) proposes
 * a grid over `EngineTuning` for one embedder and promotes only through the
 * gate; every field is optional on the wire and defaults in the core.
 */

import type { Head, QuestionWire } from './types';

/** The five frozen splits (ADR-006), lowercase to match the Rust serde. */
export type Split =
  | 'train'
  | 'calibration'
  | 'validation'
  | 'transfer'
  | 'test';

/** Trust tier of a label (ADR-004): A verifier, B judge, C user text. */
export type TrustTier = 'A' | 'B' | 'C';

/** Tunable engine knobs (mirrors `EngineOptions`, camelCase). All optional. */
export interface EngineTuning {
  probeL2?: number;
  probeIterations?: number;
  probeLearningRate?: number;
  probeClassBalanced?: boolean;
  notForLambda?: number;
  abstainTau?: number;
  abstainScale?: number;
  /** Inverse-temperature prior applied before calibration (1.0 = no-op). */
  logitScale?: number;
  calibrationFraction?: number;
  minCalibration?: number;
  head?: 'auto' | 'prototype' | 'probe';
}

/** One labeled campaign row with an explicit frozen split. */
export interface CampaignRow {
  text: string;
  label: string;
  split: Split;
  tier?: TrustTier;
}

/** A campaign over one question for a fixed embedder (mirrors `CampaignSpec`). */
export interface CampaignSpec {
  question: string;
  question_def: QuestionWire;
  rows: CampaignRow[];
  base_options?: EngineTuning;
  /** Explicit proposals; empty → the core's deterministic default grid. */
  proposals?: EngineTuning[];
  alpha?: number;
  lambda?: number;
  transfer_tolerance?: number;
  /** Accuracy non-inferiority margin for a calibration-only promotion (gate 2b). */
  accuracy_tolerance?: number;
  budget_per_day?: number;
  day_key?: string;
  created_seq_base?: number;
  created?: string | null;
}

/** A pair of accuracies on one split (mirrors `receipt::Metrics`). */
export interface Metrics {
  baseline_accuracy: number;
  champion_accuracy: number;
  n: number;
  ece?: number | null;
  brier?: number | null;
}

/** The anytime-valid paired-test statistic (mirrors `receipt::TestStatistic`). */
export interface TestStatistic {
  alpha: number;
  lambda: number;
  wealth: number;
  max_wealth: number;
  threshold: number;
  n_champion_wins: number;
  n_baseline_wins: number;
  rejected: boolean;
  n_discordant_at_rejection?: number | null;
}

/** The gate's verdict (mirrors `loop_gate::GateDecision`). */
export type GateDecision =
  | { decision: 'promote' }
  | { decision: 'reject'; reason: string }
  | { decision: 'paused'; reason: string };

/** Which criterion carried a promotion (mirrors `loop_gate::PromotionCriterion`). */
export type PromotionCriterion = 'accuracy' | 'calibration';

/** One arm's outcome inside a campaign (mirrors `optimize::ArmResult`). */
export interface ArmResult {
  proposal_id: number;
  options: EngineTuning;
  decision: GateDecision;
  val: Metrics;
  transfer: Metrics;
  statistic: TestStatistic;
  /** The calibration (paired NLL) test statistic (gate 2b), when run. */
  calibration_statistic?: TestStatistic | null;
  /** Which criterion promoted this arm, if any. */
  promoted_by?: PromotionCriterion | null;
  promoted: boolean;
}

/** One append-only receipt (mirrors `receipt::Receipt`, hashes not text). */
export interface Receipt {
  seq: number;
  proposal: { id: number; parent: number | null; kind: string; description_hash: number };
  kind: string;
  val: Metrics;
  transfer: Metrics;
  test?: Metrics | null;
  statistic: TestStatistic;
  calibration_statistic?: TestStatistic | null;
  promoted_by?: PromotionCriterion | null;
  decision: GateDecision;
  model_id: string;
  head: Head;
  temperature: number;
  budget_consumed: number;
  prev_hash: string;
  hash: string;
  [k: string]: unknown;
}

/** A hash-chained log of receipts (mirrors `receipt::ReceiptLog`). */
export interface ReceiptLog {
  receipts: Receipt[];
}

/** The campaign result (mirrors `optimize::CampaignReport`). */
export interface CampaignReport {
  question: string;
  embedder_id: string;
  baseline_options: EngineTuning;
  champion_options: EngineTuning;
  baseline_val: Metrics;
  champion_val: Metrics;
  baseline_transfer: Metrics;
  champion_transfer: Metrics;
  baseline_test: Metrics;
  champion_test: Metrics;
  arms: ArmResult[];
  promotions: number;
  budget_consumed: number;
  test_scorings: number;
  receipts: ReceiptLog;
}
