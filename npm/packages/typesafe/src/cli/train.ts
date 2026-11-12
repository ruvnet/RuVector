/** `typesafe train` — admit labeled examples for one question. With `--bank
 * <path>`, an existing bank is loaded first and the merged bank is written back
 * (append-only, deduplicated), so training persists across runs. */

import * as fs from 'node:fs';
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

  const bank = flagString(flags, 'bank');
  let persistence = 'in-memory only for this run';
  if (bank && fs.existsSync(bank)) {
    ts.importBank(fs.readFileSync(bank, 'utf8'));
  }
  const report = await ts.train(question, examples);
  if (bank) {
    fs.writeFileSync(bank, ts.exportBank());
    persistence = `persisted to ${bank}`;
  }
  ctx.out(JSON.stringify({ ...report, persistence }, null, 2));
  return 0;
}
