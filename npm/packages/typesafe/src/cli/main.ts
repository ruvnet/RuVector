/**
 * CLI dispatch. `main(argv, deps)` never calls `process.exit` and never blocks:
 * it returns a `MainResult` with an exit code, and for `serve` a live handle so
 * a caller (or a test) can read the bound port and close the server. The `bench`
 * subcommand is handled by `bin/cli.js` (it must use a dynamic `import()` that
 * survives CommonJS transpilation), so it is only described here.
 */

import { TypesafeError } from '../errors';
import type { ServeHandle } from '../serve';
import { CliContext, MainDeps, parseArgs } from './args';
import { runDecide } from './decide';
import { runEval } from './eval';
import { HELP, VERSION } from './help';
import { runOptimize } from './optimize';
import { runServe } from './serveCmd';
import { runTrain } from './train';

export interface MainResult {
  code: number;
  serve?: ServeHandle;
}

export async function main(
  argv: readonly string[],
  deps: MainDeps = {},
): Promise<MainResult> {
  const out = deps.stdout ?? ((s: string) => process.stdout.write(s + '\n'));
  const err = deps.stderr ?? ((s: string) => process.stderr.write(s + '\n'));
  const ctx: CliContext = { binding: deps.binding, out, err };
  const parsed = parseArgs(argv);
  const command = parsed._[0];

  if (!command) {
    if (parsed.flags.version) out(VERSION);
    else out(HELP);
    return { code: 0 };
  }
  if (parsed.flags.help) {
    out(HELP);
    return { code: 0 };
  }

  try {
    switch (command) {
      case 'decide':
        return { code: await runDecide(parsed.flags, ctx) };
      case 'train':
        return { code: await runTrain(parsed.flags, ctx) };
      case 'eval':
        return { code: await runEval(parsed.flags, ctx) };
      case 'optimize':
        return { code: await runOptimize(parsed.flags, ctx) };
      case 'serve':
        return { code: 0, serve: await runServe(parsed.flags, ctx) };
      case 'bench':
        err('bench is run by the `typesafe` binary; it delegates to bench/run.mjs');
        return { code: 2 };
      case 'version':
        out(VERSION);
        return { code: 0 };
      case 'help':
        out(HELP);
        return { code: 0 };
      default:
        err(`unknown command: ${command}`);
        out(HELP);
        return { code: 2 };
    }
  } catch (e) {
    if (e instanceof TypesafeError) {
      err(`error [${e.kind}]: ${e.message}`);
    } else if (e instanceof Error) {
      err(`error: ${e.message}`);
    } else {
      err('error: unknown failure');
    }
    return { code: 1 };
  }
}
