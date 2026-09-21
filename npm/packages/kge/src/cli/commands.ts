/**
 * The `kge` subcommand handlers. Each returns an exit code and prints result
 * JSON (or an error line) through the `CliContext`. No `child_process`, no
 * network — only `node:fs` via the helpers in `args.ts`.
 */

import type { Triple } from '../types';
import {
  CliContext,
  flagNumber,
  ParsedArgs,
  readJsonFile,
  readJsonl,
  requireFlag,
  writeFile,
} from './args';
import { openKge } from './engine';

/** `import --triples t.jsonl --out model.json [--scorer --dims --seed]` */
export function runImport(flags: ParsedArgs['flags'], ctx: CliContext): number {
  const triplesPath = requireFlag(flags, 'triples');
  const outPath = requireFlag(flags, 'out');
  const triples = readJsonl<Triple>(triplesPath);
  const kge = openKge(flags, ctx);
  const report = kge.addTriples(triples);
  writeFile(outPath, kge.save());
  ctx.out(JSON.stringify({ ...report, out: outPath }));
  return 0;
}

/** `train --model model.json [--config c.json] [--out other.json]` */
export async function runTrain(flags: ParsedArgs['flags'], ctx: CliContext): Promise<number> {
  const configPath = flags.config;
  const config =
    typeof configPath === 'string'
      ? readJsonFile<Record<string, unknown>>(configPath)
      : {};
  const kge = openKge(flags, ctx);
  const report = await kge.train(config);
  // Persist the trained model: to --out if given, else back to --model.
  const outPath =
    (typeof flags.out === 'string' && flags.out) ||
    (typeof flags.model === 'string' && flags.model) ||
    undefined;
  if (outPath) writeFile(outPath, kge.save());
  ctx.out(JSON.stringify({ ...report, ...(outPath && { saved: outPath }) }));
  return 0;
}

/** `eval --model model.json [--split test]` */
export function runEval(flags: ParsedArgs['flags'], ctx: CliContext): number {
  const kge = openKge(flags, ctx);
  const config: Record<string, unknown> = {};
  const split = flags.split;
  if (typeof split === 'string') config.split = split;
  ctx.out(JSON.stringify(kge.evaluate(config)));
  return 0;
}

/** `predict --model model.json (--s X | --o Z) --r Y [-k 10]` */
export function runPredict(flags: ParsedArgs['flags'], ctx: CliContext): number {
  const kge = openKge(flags, ctx);
  const r = requireFlag(flags, 'r');
  const k = flagNumber(flags, 'k');
  const s = typeof flags.s === 'string' ? flags.s : undefined;
  const o = typeof flags.o === 'string' ? flags.o : undefined;
  if ((s === undefined) === (o === undefined)) {
    throw new Error('predict needs exactly one of --s or --o');
  }
  const query =
    s !== undefined ? { s, r, ...(k !== undefined && { k }) } : { o: o as string, r, ...(k !== undefined && { k }) };
  ctx.out(JSON.stringify(kge.predict(query)));
  return 0;
}

/** `compose --model model.json --r1 A --r2 B --s X [-k 10]` */
export function runCompose(flags: ParsedArgs['flags'], ctx: CliContext): number {
  const kge = openKge(flags, ctx);
  const query = {
    r1: requireFlag(flags, 'r1'),
    r2: requireFlag(flags, 'r2'),
    s: requireFlag(flags, 's'),
    ...(flagNumber(flags, 'k') !== undefined && { k: flagNumber(flags, 'k') }),
  };
  ctx.out(JSON.stringify(kge.compose(query)));
  return 0;
}

/** `similar --model model.json --r X [-k 10]` */
export function runSimilar(flags: ParsedArgs['flags'], ctx: CliContext): number {
  const kge = openKge(flags, ctx);
  const query = {
    r: requireFlag(flags, 'r'),
    ...(flagNumber(flags, 'k') !== undefined && { k: flagNumber(flags, 'k') }),
  };
  ctx.out(JSON.stringify(kge.similarRelations(query)));
  return 0;
}

/** `optimize --model model.json [--campaign c.json]` */
export function runOptimize(flags: ParsedArgs['flags'], ctx: CliContext): number {
  const kge = openKge(flags, ctx);
  const campaignPath = flags.campaign;
  const campaign =
    typeof campaignPath === 'string'
      ? readJsonFile<Record<string, unknown>>(campaignPath)
      : {};
  ctx.out(JSON.stringify(kge.optimize(campaign)));
  return 0;
}
