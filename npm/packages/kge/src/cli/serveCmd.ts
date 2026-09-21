/** `kge serve` — start the HTTP query server (POST /v1/predict, /v1/compose). */

import { serve, ServeHandle } from '../serve';
import { CliContext, flagNumber, flagString, ParsedArgs } from './args';
import { openKge } from './engine';

export async function runServe(
  flags: ParsedArgs['flags'],
  ctx: CliContext,
): Promise<ServeHandle> {
  const kge = openKge(flags, ctx);
  const port = flagNumber(flags, 'port') ?? 8788;
  const host = flagString(flags, 'host');
  const handle = await serve(kge, {
    port,
    host,
    log: (line) => ctx.out(line),
  });
  ctx.out(`kge serve listening on http://${handle.host}:${handle.port}`);
  return handle;
}
