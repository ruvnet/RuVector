/**
 * @ruvector/typesafe — local typed decisions (choice / score / noul) over
 * sentence embeddings, with Jev's (typesafe.ai) System One wire contract.
 *
 * ```ts
 * import { createTypesafe, choice, score, noul } from '@ruvector/typesafe';
 *
 * const ts = createTypesafe();
 * const r = await ts.decide('my card was charged twice', {
 *   dept: choice({
 *     billing: 'charges and refunds',
 *     fraud: { what: 'unauthorised use', not_for: 'duplicate charges' },
 *   }),
 *   mood: score(['Calm', 'Irritated', 'Angry'] as const),
 *   urgent: noul('the sender needs a response soon'),
 * });
 * r.dept.choice;         // "billing" | "fraud"
 * r.mood.legend;         // "Calm" | "Irritated" | "Angry"
 * r.urgent.noul;         // number
 * ```
 */

export { createTypesafe } from './client';
export type {
  DecideManyOptions,
  DecisionResult,
  JevQuestion,
  SystemOneBody,
  SystemOneOptions,
  Typesafe,
  TypesafeOptions,
} from './client';

export { choice, noul, score } from './questions';
export type {
  AnswerOf,
  AnswersOf,
  AnyQuestion,
  ChoiceQuestion,
  NoulQuestion,
  QuestionOptions,
  ScoreQuestion,
} from './questions';

export { TypesafeError } from './errors';

export type {
  Binding,
  EmbedderConfig,
  EngineInstance,
  EngineOptions,
} from './binding';

export type {
  ArmResult,
  CampaignReport,
  CampaignRow,
  CampaignSpec,
  EngineTuning,
  GateDecision,
  Metrics as CampaignMetrics,
  PromotionCriterion,
  Receipt,
  ReceiptLog,
  Split,
  TestStatistic,
  TrustTier,
} from './optimize';

export {
  accuracy,
  brier,
  ece,
  evaluate,
  macroF1,
  percentile,
} from './metrics';
export type { EvalItem, EvalMetrics, EvalOptions } from './metrics';

export { ADDITIVE_KEYS } from './types';
export type {
  Answer,
  AnswerMeta,
  ChoiceAnswer,
  Criterion,
  DecisionRequest,
  DecisionResponse,
  ErrorKind,
  Head,
  LabeledExample,
  NoulAnswer,
  QuestionWire,
  ScoreAnswer,
  TrainReport,
  TypesafeErrorShape,
  Usage,
} from './types';
