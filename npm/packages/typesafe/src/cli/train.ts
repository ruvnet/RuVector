/** `typesafe train` — admit labeled examples for one question. */

import type { LabeledExample } from '../types';
import { CliContext, flagString, ParsedArgs, readJsonl } from './args';
import { openTypesafe } from './engine';

export async function runTrain(
  flags: ParsedArgs['flags'],
  ctx: CliContext,
): Promise<number> {
  const question = flagString(flags, 'question');
  const examplesPath = flagString(flags, 'examples');
  if (!question) {
    ctx.err('train: --question <id> is required');
    return 2;
  }
  if (!examplesPath) {
    ctx.err('train: --examples <path> is required');
    return 2;
  }
  const examples = readJsonl<LabeledExample>(examplesPath);
  const ts = openTypesafe(flags, ctx);
  const report = await ts.train(question, examples);
  const bank = flagString(flags, 'bank');
  const persistence = bank
    ? `--bank set; persistence depends on the binding exposing it via statsJson (else in-memory only for this run)`
    : 'in-memory only for this run';
  ctx.out(JSON.stringify({ ...report, persistence }, null, 2));
  return 0;
}
