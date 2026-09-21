/** `typesafe serve` — start the Jev-compatible HTTP server. */

import { serve, ServeHandle } from '../serve';
import { CliContext, flagBool, flagString, ParsedArgs } from './args';
import { openTypesafe } from './engine';

export async function runServe(
  flags: ParsedArgs['flags'],
  ctx: CliContext,
): Promise<ServeHandle> {
  const ts = openTypesafe(flags, ctx);
  const portRaw = flagString(flags, 'port');
  const port = portRaw !== undefined ? Number(portRaw) : 8787;
  const host = flagString(flags, 'host');
  const handle = await serve(ts, {
    port,
    host,
    jevShapeOnly: flagBool(flags, 'jev'),
    log: (line) => ctx.out(line),
  });
  ctx.out(`typesafe serve listening on http://${handle.host}:${handle.port}`);
  return handle;
}
