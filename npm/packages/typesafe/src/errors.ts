import type { ErrorKind, TypesafeErrorShape } from './types';

/**
 * The binding returns `{"error":{"kind","message"}}` instead of a response for
 * a rejected request; the client turns that envelope into this thrown error.
 * The message never contains `state` text (ADR-005: no PII in errors/logs).
 */
export class TypesafeError extends Error {
  readonly kind: ErrorKind;

  constructor(message: string, kind: ErrorKind) {
    super(message);
    this.name = 'TypesafeError';
    this.kind = kind;
    // Restore the prototype chain across the TS/ES target downlevel.
    Object.setPrototypeOf(this, TypesafeError.prototype);
  }
}

/** Type guard for the binding's error envelope. */
export function isErrorShape(value: unknown): value is TypesafeErrorShape {
  if (typeof value !== 'object' || value === null) return false;
  const err = (value as { error?: unknown }).error;
  if (typeof err !== 'object' || err === null) return false;
  const kind = (err as { kind?: unknown }).kind;
  return kind === 'limit' || kind === 'invalid' || kind === 'embedder';
}
