/** Resolves the installed forge version without baking it into two places. */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

let cached: string | undefined;

/** Version from the package manifest, or `0.0.0-unknown` if it cannot be read. */
export function forgeVersion(): string {
  if (cached) return cached;
  try {
    const raw = readFileSync(join(__dirname, '..', 'package.json'), 'utf8');
    const parsed = JSON.parse(raw) as { version?: string };
    cached = typeof parsed.version === 'string' ? parsed.version : '0.0.0-unknown';
  } catch {
    cached = '0.0.0-unknown';
  }
  return cached;
}

export const FORGE_PACKAGE_NAME = '@ruvector/rvforge';
