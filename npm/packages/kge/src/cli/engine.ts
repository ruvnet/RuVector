/** Open a Kge from CLI flags: load `--model` if given, else create fresh. */

import { readFileSync } from 'node:fs';
import { createKge, Kge, loadKge } from '../client';
import type { EngineOptions } from '../binding';
import { CliContext, flagNumber, flagString, ParsedArgs } from './args';

export function engineOptionsFromFlags(flags: ParsedArgs['flags']): EngineOptions {
  const opts: EngineOptions = {};
  const scorer = flagString(flags, 'scorer');
  if (scorer !== undefined) {
    if (scorer !== 'hole' && scorer !== 'rotate') {
      throw new Error(`unknown --scorer "${scorer}" (expected hole or rotate)`);
    }
    opts.scorer = scorer;
  }
  const dims = flagNumber(flags, 'dims');
  if (dims !== undefined) opts.dims = dims;
  const seed = flagNumber(flags, 'seed');
  if (seed !== undefined) opts.seed = seed;
  return opts;
}

/** `--model <path>` loads a saved envelope; otherwise a fresh model is built. */
export function openKge(flags: ParsedArgs['flags'], ctx: CliContext): Kge {
  const modelPath = flagString(flags, 'model');
  if (modelPath !== undefined) {
    return loadKge(readFileSync(modelPath, 'utf8'), { binding: ctx.binding });
  }
  return createKge({ ...engineOptionsFromFlags(flags), binding: ctx.binding });
}
