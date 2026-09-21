/** Build a Typesafe client from CLI flags. */

import { createTypesafe, Typesafe } from '../client';
import type { EngineOptions } from '../binding';
import { CliContext, flagString, ParsedArgs } from './args';

export function engineOptionsFromFlags(flags: ParsedArgs['flags']): EngineOptions {
  const embedder = flagString(flags, 'embedder');
  if (embedder === 'onnx') {
    const modelDir = flagString(flags, 'model-dir');
    const manifest = flagString(flags, 'manifest');
    if (!modelDir || !manifest) {
      throw new Error('--embedder onnx requires --model-dir and --manifest');
    }
    return { embedder: { kind: 'onnx', modelDir, manifest } };
  }
  if (embedder !== undefined && embedder !== 'hash') {
    throw new Error(`unknown --embedder "${embedder}" (expected hash or onnx)`);
  }
  const dimsRaw = flagString(flags, 'dims');
  const opts: EngineOptions = { embedder: 'hash' };
  if (dimsRaw !== undefined) opts.dims = Number(dimsRaw);
  return opts;
}

export function openTypesafe(flags: ParsedArgs['flags'], ctx: CliContext): Typesafe {
  return createTypesafe({ ...engineOptionsFromFlags(flags), binding: ctx.binding });
}
