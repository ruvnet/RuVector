/**
 * The public client: `createTypesafe(opts)` returns a `Typesafe` bound to one
 * engine instance. Every decision is one call across the binding seam; nothing
 * here touches the network, the filesystem, or a subprocess.
 */

import {
  Binding,
  EngineInstance,
  EngineOptions,
  resolveDefaultBinding,
  toOptionsJson,
} from './binding';
import { isErrorShape, TypesafeError } from './errors';
import type { CampaignReport, CampaignSpec } from './optimize';
import type {
  AnswersOf,
  AnyQuestion,
} from './questions';
import {
  ADDITIVE_KEYS,
  Answer,
  ChoiceQuestionWire,
  Criterion,
  DecisionRequest,
  DecisionResponse,
  LabeledExample,
  QuestionWire,
  ScoreQuestionWire,
  TrainReport,
  Usage,
} from './types';

export interface TypesafeOptions extends EngineOptions {
  /** Inject a binding (tests, or a non-default addon). Defaults to `../index.js`. */
  binding?: Binding;
}

/**
 * The result of `decide`: answers keyed by question id, exposed both under
 * `.answers` and spread at the top level for convenience, plus `.usage`.
 * Reserved top-level keys are `answers` and `usage`; a question with either id
 * is still reachable through `.answers`.
 */
export type DecisionResult<Q extends Record<string, AnyQuestion>> = {
  answers: AnswersOf<Q>;
  usage: Usage;
} & AnswersOf<Q>;

/** A Jev `POST /v1/systemone` question (score buckets may arrive as `criteria`). */
export type JevQuestion =
  | { type: 'choice'; instructions?: string; criteria: Record<string, Criterion> }
  | {
      type: 'score';
      instructions?: string;
      criteria?: readonly string[];
      legend?: readonly string[];
    }
  | { type: 'noul'; instructions: string };

/** A Jev `POST /v1/systemone` request body. `model` is accepted and ignored. */
export interface SystemOneBody {
  state: string;
  model?: string;
  questions: Record<string, JevQuestion>;
}

export interface SystemOneOptions {
  /** Strip the five additive answer fields so the shape is exactly Jev's. */
  jevShapeOnly?: boolean;
}

export interface DecideManyOptions {
  /** Max in-flight decisions on the async native path (default 4). Ignored sync. */
  concurrency?: number;
}

export interface Typesafe {
  readonly version: string;
  readonly backend: 'native' | 'wasm';
  decide<Q extends Record<string, AnyQuestion>>(
    state: string,
    questions: Q,
  ): Promise<DecisionResult<Q>>;
  decideMany<Q extends Record<string, AnyQuestion>>(
    states: readonly string[],
    questions: Q,
    opts?: DecideManyOptions,
  ): Promise<Array<DecisionResult<Q>>>;
  systemOne(body: SystemOneBody, opts?: SystemOneOptions): Promise<DecisionResponse>;
  train(questionId: string, examples: readonly LabeledExample[]): Promise<TrainReport>;
  /**
   * Run an optimize campaign (ADR-004) over `EngineTuning` for this engine's
   * embedder: a gated grid search that promotes only through the paired
   * anytime-valid test. Throws if the binding does not expose `optimizeJson`.
   */
  optimize(spec: CampaignSpec): Promise<CampaignReport>;
  /** Export the full example bank JSON (the user's own examples, with text). */
  exportBank(): string;
  /** Replace the bank from a previously exported JSON (re-embeds every text). */
  importBank(bankJson: string): void;
  stats(): Record<string, unknown>;
}

function toWireQuestion(q: AnyQuestion): QuestionWire {
  if (q.type === 'choice') {
    const w: ChoiceQuestionWire = { type: 'choice', criteria: q.criteria };
    if (q.instructions !== undefined) w.instructions = q.instructions;
    return w;
  }
  if (q.type === 'score') {
    const w: ScoreQuestionWire = { type: 'score', legend: [...q.legend] };
    if (q.instructions !== undefined) w.instructions = q.instructions;
    return w;
  }
  return { type: 'noul', instructions: q.instructions };
}

function jevToWireQuestion(id: string, q: JevQuestion): QuestionWire {
  if (q.type === 'choice') {
    const w: ChoiceQuestionWire = { type: 'choice', criteria: q.criteria };
    if (q.instructions !== undefined) w.instructions = q.instructions;
    return w;
  }
  if (q.type === 'score') {
    const buckets = q.legend ?? q.criteria;
    if (!buckets) {
      throw new TypesafeError(`question ${id}: score needs legend or criteria`, 'invalid');
    }
    const w: ScoreQuestionWire = { type: 'score', legend: [...buckets] };
    if (q.instructions !== undefined) w.instructions = q.instructions;
    return w;
  }
  return { type: 'noul', instructions: q.instructions };
}

function toWireQuestions(
  questions: Record<string, AnyQuestion>,
): Record<string, QuestionWire> {
  const out: Record<string, QuestionWire> = {};
  for (const id of Object.keys(questions)) {
    out[id] = toWireQuestion(questions[id]);
  }
  return out;
}

function parseResponse(json: string): DecisionResponse {
  const parsed: unknown = JSON.parse(json);
  if (isErrorShape(parsed)) {
    throw new TypesafeError(parsed.error.message, parsed.error.kind);
  }
  return parsed as DecisionResponse;
}

function assembleResult<Q extends Record<string, AnyQuestion>>(
  resp: DecisionResponse,
): DecisionResult<Q> {
  const answers = resp.answers as AnswersOf<Q>;
  return Object.assign(
    Object.create(null) as Record<string, unknown>,
    answers,
    { answers, usage: resp.usage },
  ) as DecisionResult<Q>;
}

function stripAdditive(answer: Answer): Answer {
  const out = { ...(answer as unknown as Record<string, unknown>) };
  for (const key of ADDITIVE_KEYS) delete out[key];
  return out as unknown as Answer;
}

async function decideRaw(engine: EngineInstance, requestJson: string): Promise<string> {
  if (typeof engine.decide === 'function') {
    return engine.decide(requestJson);
  }
  return engine.decideJson(requestJson);
}

/** Run `tasks` with at most `limit` in flight, preserving input order. */
async function pool<T, R>(
  items: readonly T[],
  limit: number,
  run: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  const width = Math.max(1, Math.min(limit, items.length || 1));
  const workers = Array.from({ length: width }, async () => {
    for (;;) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await run(items[i], i);
    }
  });
  await Promise.all(workers);
  return results;
}

export function createTypesafe(opts: TypesafeOptions = {}): Typesafe {
  const binding = opts.binding ?? resolveDefaultBinding();
  if (!binding) {
    throw new TypesafeError(
      'no @ruvector/typesafe binding found — build the native addon (npm run build:napi) ' +
        'or ship the WASM fallback; pass { binding } to inject one',
      'embedder',
    );
  }
  const engine = new binding.Engine(toOptionsJson(opts));

  async function decide<Q extends Record<string, AnyQuestion>>(
    state: string,
    questions: Q,
  ): Promise<DecisionResult<Q>> {
    const req: DecisionRequest = { state, questions: toWireQuestions(questions) };
    const resp = parseResponse(await decideRaw(engine, JSON.stringify(req)));
    return assembleResult<Q>(resp);
  }

  async function decideMany<Q extends Record<string, AnyQuestion>>(
    states: readonly string[],
    questions: Q,
    manyOpts: DecideManyOptions = {},
  ): Promise<Array<DecisionResult<Q>>> {
    const wire = toWireQuestions(questions);
    const one = async (state: string): Promise<DecisionResult<Q>> => {
      const req: DecisionRequest = { state, questions: wire };
      const resp = parseResponse(await decideRaw(engine, JSON.stringify(req)));
      return assembleResult<Q>(resp);
    };
    if (typeof engine.decide === 'function') {
      return pool(states, manyOpts.concurrency ?? 4, one);
    }
    const out: Array<DecisionResult<Q>> = [];
    for (const state of states) out.push(await one(state));
    return out;
  }

  async function systemOne(
    body: SystemOneBody,
    systemOpts: SystemOneOptions = {},
  ): Promise<DecisionResponse> {
    const questions: Record<string, QuestionWire> = {};
    for (const id of Object.keys(body.questions)) {
      questions[id] = jevToWireQuestion(id, body.questions[id]);
    }
    const req: DecisionRequest = { state: body.state, questions };
    const resp = parseResponse(await decideRaw(engine, JSON.stringify(req)));
    if (!systemOpts.jevShapeOnly) return resp;
    const answers: Record<string, Answer> = {};
    for (const id of Object.keys(resp.answers)) {
      answers[id] = stripAdditive(resp.answers[id]);
    }
    return { answers, usage: resp.usage };
  }

  async function train(
    questionId: string,
    examples: readonly LabeledExample[],
  ): Promise<TrainReport> {
    const payload = JSON.stringify({ question: questionId, examples });
    const parsed: unknown = JSON.parse(engine.trainJson(payload));
    if (isErrorShape(parsed)) {
      throw new TypesafeError(parsed.error.message, parsed.error.kind);
    }
    return parsed as TrainReport;
  }

  async function optimize(spec: CampaignSpec): Promise<CampaignReport> {
    if (typeof engine.optimizeJson !== 'function') {
      throw new TypesafeError(
        'this binding does not expose optimizeJson — rebuild the native addon',
        'embedder',
      );
    }
    const parsed: unknown = JSON.parse(engine.optimizeJson(JSON.stringify(spec)));
    if (isErrorShape(parsed)) {
      throw new TypesafeError(parsed.error.message, parsed.error.kind);
    }
    return parsed as CampaignReport;
  }

  function exportBank(): string {
    if (typeof engine.exportBankJson !== 'function') {
      throw new TypesafeError('this binding does not expose exportBankJson', 'embedder');
    }
    const json = engine.exportBankJson();
    const parsed: unknown = JSON.parse(json);
    if (isErrorShape(parsed)) {
      throw new TypesafeError(parsed.error.message, parsed.error.kind);
    }
    return json;
  }

  function importBank(bankJson: string): void {
    if (typeof engine.importBankJson !== 'function') {
      throw new TypesafeError('this binding does not expose importBankJson', 'embedder');
    }
    const parsed: unknown = JSON.parse(engine.importBankJson(bankJson));
    if (isErrorShape(parsed)) {
      throw new TypesafeError(parsed.error.message, parsed.error.kind);
    }
  }

  function stats(): Record<string, unknown> {
    return JSON.parse(engine.statsJson()) as Record<string, unknown>;
  }

  return {
    version: binding.version(),
    backend: binding.backend,
    decide,
    decideMany,
    systemOne,
    train,
    optimize,
    exportBank,
    importBank,
    stats,
  };
}
