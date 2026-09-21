/**
 * Wire types — a 1:1 mirror of `crates/ruvector-typesafe-core/src/types.rs`.
 *
 * Field names follow the Jev (typesafe.ai) contract: `state`, `questions`,
 * `choice`, `probabilities`, `confidence`, `score`, `legend`, `noul`. The five
 * fields `abstain`, `calibrated`, `head`, `model`, `temperature` are additive
 * (ADR-003 "what every answer carries"); `jevShapeOnly` strips exactly those.
 *
 * This module is pure types: no runtime code, no imports, no I/O.
 */

/** One option of a `choice` question: a bare description or the structured form. */
export type Criterion =
  | string
  | {
      what: string;
      not_for?: string;
      examples?: string[];
    };

/** A `choice` question: pick one of up to 255 options (`criteria` is keyed). */
export interface ChoiceQuestionWire {
  type: 'choice';
  instructions?: string;
  criteria: Record<string, Criterion>;
}

/** A `score` question: an ordinal legend, e.g. ["Calm", "Irritated", "Angry"]. */
export interface ScoreQuestionWire {
  type: 'score';
  instructions?: string;
  legend: string[];
}

/** A `noul` question: a 0–1 predicate ("the sender needs a response soon"). */
export interface NoulQuestionWire {
  type: 'noul';
  instructions: string;
}

export type QuestionWire = ChoiceQuestionWire | ScoreQuestionWire | NoulQuestionWire;

export interface DecisionRequest {
  state: string;
  questions: Record<string, QuestionWire>;
}

/** Which head produced an answer (ADR-003) — kebab-case, matching the Rust serde. */
export type Head =
  | 'nearest-prototype'
  | 'linear-probe'
  | 'logistic'
  | 'similarity-uncalibrated';

/** Fields every answer carries in addition to the Jev-shaped payload (ADR-003). */
export interface AnswerMeta {
  confidence: number;
  abstain: number;
  calibrated: boolean;
  head: Head;
  /** Embedder id: model name @ manifest hash (ADR-002 §1). */
  model: string;
  temperature: number;
}

/** The five additive keys stripped by `jevShapeOnly` (confidence is Jev's own). */
export const ADDITIVE_KEYS = [
  'abstain',
  'calibrated',
  'head',
  'model',
  'temperature',
] as const;
export type AdditiveKey = (typeof ADDITIVE_KEYS)[number];

export type ChoiceAnswer<K extends string = string> = {
  /** The winning option key. */
  choice: K;
  /** Sums to 1 over the options (Jev contract); abstain mass is in the meta. */
  probabilities: Record<K, number>;
} & AnswerMeta;

export type ScoreAnswer<L extends string = string> = {
  /** Expected bucket index under the calibrated distribution. */
  score: number;
  /** The label of the chosen bucket. */
  legend: L;
  /** One entry per legend bucket, in legend order. */
  probabilities: number[];
} & AnswerMeta;

export type NoulAnswer = {
  /** Calibrated 0–1 value for the predicate. */
  noul: number;
} & AnswerMeta;

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface Usage {
  embed_calls: number;
  texts_embedded: number;
  state_bytes: number;
}

export interface DecisionResponse {
  answers: Record<string, Answer>;
  usage: Usage;
}

/** Labeled example for `train` (mirrors `engine.rs::LabeledExample`). */
export interface LabeledExample {
  text: string;
  /** Option key, legend bucket, or "yes" | "no" for a `noul` predicate. */
  label: string;
}

/** Result of `train` (mirrors `engine.rs::TrainReport`). */
export interface TrainReport {
  question: string;
  accepted: number;
  rejected: number;
  head: Head;
  calibrated: boolean;
}

/** The error envelope the binding returns instead of a `DecisionResponse`. */
export interface TypesafeErrorShape {
  error: {
    kind: 'limit' | 'invalid' | 'embedder';
    message: string;
  };
}

export type ErrorKind = TypesafeErrorShape['error']['kind'];
