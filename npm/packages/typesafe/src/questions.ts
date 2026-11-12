/**
 * Type-safe question builders. A built question is a plain wire object plus a
 * phantom brand carrying its answer type, so `decide` can map each question id
 * to a precisely typed answer (`r.dept.choice` is the union of the criteria
 * keys, `r.mood.legend` is the union of the legend buckets).
 *
 * The brand is an optional property keyed by a unique symbol: it exists only in
 * the type system and is never present at runtime.
 */

import type {
  ChoiceAnswer,
  Criterion,
  NoulAnswer,
  ScoreAnswer,
} from './types';

declare const ANSWER_BRAND: unique symbol;

/** Carries the answer type `A` for a built question, at the type level only. */
export interface Branded<A> {
  readonly [ANSWER_BRAND]?: A;
}

export type ChoiceQuestion<K extends string> = {
  type: 'choice';
  instructions?: string;
  criteria: Record<K, Criterion>;
} & Branded<ChoiceAnswer<K>>;

export type ScoreQuestion<L extends string> = {
  type: 'score';
  instructions?: string;
  legend: readonly L[];
} & Branded<ScoreAnswer<L>>;

export type NoulQuestion = {
  type: 'noul';
  instructions: string;
} & Branded<NoulAnswer>;

export type AnyQuestion =
  | ChoiceQuestion<string>
  | ScoreQuestion<string>
  | NoulQuestion;

/**
 * The answer type for a built question. `Branded<A>` is satisfied by every
 * object (the brand is optional), so an unbranded question infers `A = unknown`;
 * the `unknown extends A` guard maps that back to the wide `Answer`-ish shapes.
 */
export type AnswerOf<T> = T extends Branded<infer A>
  ? unknown extends A
    ? ChoiceAnswer | ScoreAnswer | NoulAnswer
    : A
  : ChoiceAnswer | ScoreAnswer | NoulAnswer;

/** Map a record of built questions to a record of their answer types. */
export type AnswersOf<Q extends Record<string, AnyQuestion>> = {
  [P in keyof Q]: AnswerOf<Q[P]>;
};

export interface QuestionOptions {
  /** Free-text guidance passed through to the engine and recorded in receipts. */
  instructions?: string;
}

/**
 * A `choice` question over a keyed map of options. The option keys are captured
 * as a string-literal union, so the returned answer's `choice` and the keys of
 * `probabilities` are exactly those keys.
 */
export function choice<K extends string>(
  criteria: Record<K, Criterion>,
  opts: QuestionOptions = {},
): ChoiceQuestion<K> {
  const q: { type: 'choice'; criteria: Record<K, Criterion>; instructions?: string } = {
    type: 'choice',
    criteria,
  };
  if (opts.instructions !== undefined) q.instructions = opts.instructions;
  return q as ChoiceQuestion<K>;
}

/**
 * A `score` question over an ordinal legend. TypeScript ≥ 5.0 infers the literal
 * bucket union from a `const` type parameter, so `score(['Calm','Angry'])` and
 * `score(['Calm','Angry'] as const)` both narrow.
 */
export function score<const L extends readonly string[]>(
  legend: L,
  opts: QuestionOptions = {},
): ScoreQuestion<L[number]> {
  const q: { type: 'score'; legend: readonly string[]; instructions?: string } = {
    type: 'score',
    legend,
  };
  if (opts.instructions !== undefined) q.instructions = opts.instructions;
  return q as ScoreQuestion<L[number]>;
}

/** A `noul` question: a 0–1 predicate. `instructions` is required. */
export function noul(instructions: string): NoulQuestion {
  return { type: 'noul', instructions } as NoulQuestion;
}
