/**
 * `typesafe optimize` — run an ADR-004 optimize campaign over one question and
 * write the hash-chained receipts. Rows come from `--dataset` (JSONL of
 * `{text,label,split?}`) and/or `--bank` (an exported bank JSON); a row with no
 * `split` is assigned deterministically by a content hash so a plain labeled
 * file just works. No network, no subprocess — only `node:fs`.
 */

import * as fs from 'node:fs';
import type { CampaignRow, CampaignSpec, Split } from '../optimize';
import type { QuestionWire } from '../types';
import { CliContext, flagString, ParsedArgs, readJsonFile, readJsonl } from './args';
import { openTypesafe } from './engine';

interface DatasetRow {
  text: string;
  label: string;
  split?: Split;
  tier?: 'A' | 'B' | 'C';
}

/** Deterministic FNV-1a bucket in [0,99] over a string (dependency-free). */
function bucket(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h % 100;
}

/** Deterministic 5-way split for a row with no explicit split (60/10/15/10/5). */
function assignSplit(text: string, label: string): Split {
  const b = bucket(label + ' ' + text);
  if (b < 60) return 'train';
  if (b < 70) return 'calibration';
  if (b < 85) return 'validation';
  if (b < 95) return 'transfer';
  return 'test';
}

/** Pick the question definition + id from a `--questions` file. */
function resolveQuestion(
  raw: unknown,
  wanted: string | undefined,
): { id: string; def: QuestionWire } {
  const obj = raw as Record<string, unknown>;
  if (obj && typeof obj.type === 'string') {
    return { id: wanted ?? 'q', def: raw as QuestionWire };
  }
  const keys = Object.keys(obj ?? {});
  const id = wanted ?? (keys.length === 1 ? keys[0] : undefined);
  if (!id || !(id in obj)) {
    throw new Error(
      'optimize: --questions is a map; pass --question <id> (one of ' +
        (keys.join(', ') || 'none') +
        ')',
    );
  }
  return { id, def: obj[id] as QuestionWire };
}

/** Convert exported-bank entries into campaign rows for `question`. */
function bankRows(bankJson: unknown, question: string): CampaignRow[] {
  const entries = (bankJson as { entries?: unknown[] })?.entries ?? [];
  const out: CampaignRow[] = [];
  for (const raw of entries) {
    const e = raw as {
      text?: string;
      question?: string;
      label?: string;
      split?: Split;
      tier?: 'A' | 'B' | 'C';
    };
    if (e.question !== undefined && e.question !== question) continue;
    if (typeof e.text !== 'string' || typeof e.label !== 'string' || !e.split) continue;
    out.push({ text: e.text, label: e.label, split: e.split, tier: e.tier ?? 'A' });
  }
  return out;
}

export async function runOptimize(
  flags: ParsedArgs['flags'],
  ctx: CliContext,
): Promise<number> {
  const questionsPath = flagString(flags, 'questions');
  if (!questionsPath) {
    ctx.err('optimize: --questions <path> is required');
    return 2;
  }
  const datasetPath = flagString(flags, 'dataset');
  const bankPath = flagString(flags, 'bank');
  if (!datasetPath && !bankPath) {
    ctx.err('optimize: one of --dataset <jsonl> or --bank <json> is required');
    return 2;
  }

  const { id, def } = resolveQuestion(readJsonFile(questionsPath), flagString(flags, 'question'));

  const rows: CampaignRow[] = [];
  if (datasetPath) {
    for (const r of readJsonl<DatasetRow>(datasetPath)) {
      rows.push({
        text: r.text,
        label: r.label,
        split: r.split ?? assignSplit(r.text, r.label),
        tier: r.tier ?? 'A',
      });
    }
  }
  if (bankPath) {
    rows.push(...bankRows(readJsonFile(bankPath), id));
  }
  if (!rows.length) {
    ctx.err('optimize: no rows loaded from --dataset / --bank');
    return 2;
  }

  const budget = flagString(flags, 'budget');
  const spec: CampaignSpec = {
    question: id,
    question_def: def,
    rows,
    budget_per_day: budget ? Number(budget) : 64,
    day_key: 'cli',
  };

  const ts = openTypesafe(flags, ctx);
  const report = await ts.optimize(spec);

  const receiptsPath = flagString(flags, 'receipts');
  if (receiptsPath) {
    const jsonl = report.receipts.receipts.map((r) => JSON.stringify(r)).join('\n');
    fs.writeFileSync(receiptsPath, jsonl + (jsonl ? '\n' : ''));
  }
  const outPath = flagString(flags, 'out');
  if (outPath) {
    fs.writeFileSync(outPath, JSON.stringify(report, null, 2) + '\n');
  }

  const pct = (x: number): string => (x * 100).toFixed(1) + '%';
  ctx.out(
    JSON.stringify(
      {
        question: report.question,
        embedder: report.embedder_id,
        promotions: report.promotions,
        arms: report.arms.length,
        budget_consumed: report.budget_consumed,
        test_scorings: report.test_scorings,
        baseline: {
          val: pct(report.baseline_val.baseline_accuracy),
          transfer: pct(report.baseline_transfer.baseline_accuracy),
          test: pct(report.baseline_test.baseline_accuracy),
          test_ece: report.baseline_test.ece ?? null,
        },
        champion: {
          val: pct(report.champion_val.champion_accuracy),
          transfer: pct(report.champion_transfer.champion_accuracy),
          test: pct(report.champion_test.champion_accuracy),
          test_ece: report.champion_test.ece ?? null,
          options: report.champion_options,
        },
        receipts: receiptsPath ?? null,
      },
      null,
      2,
    ),
  );
  return 0;
}
