import type { KgeErrorKind, KgeErrorShape } from './types';

/**
 * The binding returns `{"error":{"kind","message"}}` instead of a result for a
 * rejected request; the client turns that envelope into this thrown error. The
 * message never contains triple/label text (ADR-005: no PII in errors/logs).
 */
export class KgeError extends Error {
  readonly kind: KgeErrorKind;

  constructor(message: string, kind: KgeErrorKind) {
    super(message);
    this.name = 'KgeError';
    this.kind = kind;
    // Restore the prototype chain across the TS/ES target downlevel.
    Object.setPrototypeOf(this, KgeError.prototype);
  }
}

/** The closed kind set — MUST match the Rust `err_json` callers and index.d.ts. */
const KINDS: ReadonlySet<string> = new Set([
  'limit',
  'invalid',
  'unavailable',
  'unsupported',
  'scorer',
]);

/** Type guard for the binding's error envelope. */
export function isErrorShape(value: unknown): value is KgeErrorShape {
  if (typeof value !== 'object' || value === null) return false;
  const err = (value as { error?: unknown }).error;
  if (typeof err !== 'object' || err === null) return false;
  const kind = (err as { kind?: unknown }).kind;
  return typeof kind === 'string' && KINDS.has(kind);
}
