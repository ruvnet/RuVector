/**
 * `typesafe eval` — score a labeled dataset. Accepts either the repo's
 * decisions JSON (`{split:{train,val,test:[...]}, gen0_questions}`) or a JSONL
 * of `{text|state, label|labels:{qid:value}}`. Metrics are computed in JS
 * (metrics.ts): accuracy, macro-F1, ECE, Brier, mean confidence/abstain, p50/p95
 * latency per question, plus overall latency.
 */

import * as fs from 'node:fs';
import type { ChoiceAnswer, NoulAnswer, ScoreAnswer } from '../types';
import type { JevQuestion, SystemOneBody, Typesafe } from '../client';
import { evaluate, EvalItem, EvalMetrics } from '../metrics';
import { CliContext, flagString, ParsedArgs } from './args';
import { openTypesafe } from './engine';

interface LabeledItem {
  state: string;
  truth: Record<string, unknown>;
}

interface Loaded {
  items: LabeledItem[];
  questions?: Record<string, JevQuestion>;
}

function loadDataset(path: string, split: string): Loaded {
  const text = fs.readFileSync(path, 'utf8');
  let whole: unknown;
  try {
    whole = JSON.parse(text);
  } catch {
    whole = undefined;
  }
  if (whole && typeof whole === 'object' && !Array.isArray(whole) && 'split' in whole) {
    const file = whole as {
      split: Record<string, Array<Record<string, unknown>>>;
      gen0_questions?: Record<string, JevQuestion>;
    };
    const rows = file.split[split] ?? [];
    return {
      items: rows.map((r) => ({
        state: String(r.text ?? r.state ?? ''),
        truth: (r.label ?? r.labels ?? {}) as Record<string, unknown>,
      })),
      questions: file.gen0_questions,
    };
  }
  const rows: Array<Record<string, unknown>> = Array.isArray(whole)
    ? (whole as Array<Record<string, unknown>>)
    : text
        .split('\n')
        .map((l) => l.trim())
        .filter((l) => l.length > 0)
        .map((l) => JSON.parse(l) as Record<string, unknown>);
  return {
    items: rows.map((r) => ({
      state: String(r.text ?? r.state ?? ''),
      truth: (r.label ?? r.labels ?? {}) as Record<string, unknown>,
    })),
  };
}

function scoreLegend(q: JevQuestion): string[] {
  if (q.type === 'score') return [...(q.legend ?? q.criteria ?? [])];
  return [];
}

function isYes(v: unknown): boolean {
  if (typeof v === 'boolean') return v;
  if (typeof v === 'number') return v >= 0.5;
  if (typeof v === 'string') return ['yes', 'true', '1', 'y'].includes(v.toLowerCase());
  return false;
}

/** Predicted and truth labels in a shared string space, per question type. */
function labels(
  q: JevQuestion,
  answer: ChoiceAnswer | ScoreAnswer | NoulAnswer,
  truthRaw: unknown,
): { predicted: string; truth: string } {
  if (q.type === 'choice') {
    return { predicted: (answer as ChoiceAnswer).choice, truth: String(truthRaw) };
  }
  if (q.type === 'score') {
    const legend = scoreLegend(q);
    // A label not in the legend stays as its raw string so it counts as a miss,
    // rather than being coerced to bucket 0 (which could fabricate correctness).
    const idx = legend.indexOf(String(truthRaw));
    const truth =
      typeof truthRaw === 'number' ? String(truthRaw) : idx >= 0 ? String(idx) : String(truthRaw);
    return { predicted: String((answer as ScoreAnswer).score), truth };
  }
  const noulYes = (answer as NoulAnswer).noul >= 0.5;
  return { predicted: noulYes ? 'yes' : 'no', truth: isYes(truthRaw) ? 'yes' : 'no' };
}

export async function runEval(
  flags: ParsedArgs['flags'],
  ctx: CliContext,
): Promise<number> {
  const datasetPath = flagString(flags, 'dataset');
  if (!datasetPath) {
    ctx.err('eval: --dataset <path> is required');
    return 2;
  }
  const split = flagString(flags, 'split') ?? 'test';
  const loaded = loadDataset(datasetPath, split);

  const questionsPath = flagString(flags, 'questions');
  const questions = questionsPath
    ? (JSON.parse(fs.readFileSync(questionsPath, 'utf8')) as Record<string, JevQuestion>)
    : loaded.questions;
  if (!questions) {
    ctx.err('eval: --questions <path> is required (dataset has no gen0_questions)');
    return 2;
  }

  const ts: Typesafe = openTypesafe(flags, ctx);
  const perQuestion: Record<string, EvalItem[]> = {};
  for (const id of Object.keys(questions)) perQuestion[id] = [];

  for (const item of loaded.items) {
    const body: SystemOneBody = { state: item.state, questions };
    // performance.now() has sub-millisecond resolution; Date.now() reads 0 for a
    // single-digit-ms decision, which would make the ADR-006 latency meaningless.
    const started = performance.now();
    const response = await ts.systemOne(body);
    const latencyMs = performance.now() - started;
    for (const id of Object.keys(questions)) {
      const answer = response.answers[id] as ChoiceAnswer | ScoreAnswer | NoulAnswer | undefined;
      if (!answer || !(id in item.truth)) continue;
      const { predicted, truth } = labels(questions[id], answer, item.truth[id]);
      perQuestion[id].push({
        predicted,
        truth,
        confidence: answer.confidence,
        abstain: answer.abstain,
        latencyMs,
      });
    }
  }

  const report: Record<string, EvalMetrics> = {};
  for (const id of Object.keys(perQuestion)) {
    if (perQuestion[id].length > 0) report[id] = evaluate(perQuestion[id]);
  }
  ctx.out(
    JSON.stringify(
      { dataset: datasetPath, split, items: loaded.items.length, questions: report },
      null,
      2,
    ),
  );
  return 0;
}
