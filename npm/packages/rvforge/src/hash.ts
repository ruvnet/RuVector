/** SHA256 helpers and the canonical JSON encoding forge hashes over. */

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { pipeline } from 'node:stream/promises';

export const sha256Buffer = (data: Uint8Array): string =>
  createHash('sha256').update(data).digest('hex');

export const sha256String = (text: string): string =>
  createHash('sha256').update(text, 'utf8').digest('hex');

/** Stream a file through SHA256 so a multi-gigabyte RVF never lands in memory. */
export async function sha256File(path: string): Promise<string> {
  const hash = createHash('sha256');
  await pipeline(createReadStream(path), hash);
  return hash.digest('hex');
}

/**
 * Serialise `value` deterministically: object keys sorted, no insignificant
 * whitespace, `undefined` members dropped. The same logical input always
 * produces byte-identical output, which is what makes a manifest hash a stable
 * identity rather than an artifact of insertion order.
 */
export function canonicalJson(value: unknown): string {
  return encode(value);
}

function encode(value: unknown): string {
  if (value === null) return 'null';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError('cannot canonicalise non-finite number');
    return JSON.stringify(value);
  }
  if (typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'bigint') return JSON.stringify(value.toString());
  if (Array.isArray(value)) return `[${value.map(encode).join(',')}]`;
  if (typeof value === 'object') {
    const obj = value as Record<string, unknown>;
    const parts: string[] = [];
    for (const key of Object.keys(obj).sort()) {
      const encoded = obj[key];
      if (encoded === undefined) continue;
      parts.push(`${JSON.stringify(key)}:${encode(encoded)}`);
    }
    return `{${parts.join(',')}}`;
  }
  throw new TypeError(`cannot canonicalise value of type ${typeof value}`);
}

/** Canonical JSON with a trailing newline — the exact bytes forge writes to disk. */
export const canonicalJsonFile = (value: unknown): string => `${canonicalJson(value)}\n`;

/** SHA256 over the canonical JSON encoding of `value`. */
export const sha256Canonical = (value: unknown): string => sha256String(canonicalJson(value));
