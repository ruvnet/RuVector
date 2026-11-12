/** `typesafe decide` — one state + a question batch → the response JSON. */

import type { JevQuestion, SystemOneBody } from '../client';
import {
  CliContext,
  flagBool,
  flagString,
  ParsedArgs,
  readJsonFile,
  readState,
} from './args';
import { openTypesafe } from './engine';

export async function runDecide(
  flags: ParsedArgs['flags'],
  ctx: CliContext,
): Promise<number> {
  const questionsPath = flagString(flags, 'questions');
  if (!questionsPath) {
    ctx.err('decide: --questions <path> is required');
    return 2;
  }
  const state = readState(flags);
  const questions = readJsonFile<Record<string, JevQuestion>>(questionsPath);
  const ts = openTypesafe(flags, ctx);
  const body: SystemOneBody = { state, questions };
  const response = await ts.systemOne(body, { jevShapeOnly: flagBool(flags, 'jev') });
  ctx.out(JSON.stringify(response, null, 2));
  return 0;
}
