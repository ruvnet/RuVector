/**
 * Stable, machine-readable error codes for `@ruvector/rvforge`.
 *
 * Every failure path in the CLI returns one of these codes and one of the
 * exit codes in {@link EXIT_CODES}. Both are part of the public contract:
 * unattended callers branch on them, so values are append-only.
 */

export enum ForgeErrorCode {
  /** Argument parsing, missing operands, mutually exclusive flags. */
  USAGE = 'FORGE_E_USAGE',
  /** The `.rvf` is missing, truncated, has bad magic, or fails its checksum. */
  INVALID_RVF = 'FORGE_E_INVALID_RVF',
  /** An executable segment (kernel/eBPF/WASM) is present without a signature. */
  UNSIGNED_SEGMENT = 'FORGE_E_UNSIGNED_SEGMENT',
  /** A requested build target is not in the supported matrix. */
  UNSUPPORTED_TARGET = 'FORGE_E_UNSUPPORTED_TARGET',
  /** The build manifest is malformed, incomplete, or internally inconsistent. */
  MANIFEST = 'FORGE_E_MANIFEST',
  /** The hosted service is unreachable or returned a transport-level failure. */
  NETWORK = 'FORGE_E_NETWORK',
  /** The service rejected the credentials, or none were supplied. */
  AUTH = 'FORGE_E_AUTH',
  /** A recomputed digest did not match the provenance record. */
  VERIFY_FAILED = 'FORGE_E_VERIFY_FAILED',
  /** Filesystem failure: unreadable input, unwritable output directory. */
  IO = 'FORGE_E_IO',
  /** A referenced build, artifact, or provenance record does not exist. */
  NOT_FOUND = 'FORGE_E_NOT_FOUND',
  /** `forge.config.json` is missing, malformed, or fails schema validation. */
  CONFIG = 'FORGE_E_CONFIG',
  /** A required local build tool (Tauri, cargo) is unavailable. */
  TOOLCHAIN = 'FORGE_E_TOOLCHAIN',
  /** The capability policy is absent, vague, or contradicts itself. */
  POLICY = 'FORGE_E_POLICY',
  /** The listing declares no license, or one forge cannot reconcile. */
  LICENSE = 'FORGE_E_LICENSE',
  /** A signing key is missing, unusable, over-permissioned, or unregistered. */
  KEY = 'FORGE_E_KEY',
  /** The local registry directory is unreadable, malformed, or inconsistent. */
  REGISTRY = 'FORGE_E_REGISTRY',
  /** A release's predecessor chain does not resolve to a coherent lineage. */
  LINEAGE = 'FORGE_E_LINEAGE',
  /** An unexpected condition; always a bug in forge itself. */
  INTERNAL = 'FORGE_E_INTERNAL',
}

/** Process exit code per error code. Success is always 0. */
export const EXIT_CODES: Readonly<Record<ForgeErrorCode, number>> = Object.freeze({
  [ForgeErrorCode.USAGE]: 2,
  [ForgeErrorCode.INVALID_RVF]: 3,
  [ForgeErrorCode.UNSIGNED_SEGMENT]: 4,
  [ForgeErrorCode.UNSUPPORTED_TARGET]: 5,
  [ForgeErrorCode.MANIFEST]: 6,
  [ForgeErrorCode.NETWORK]: 7,
  [ForgeErrorCode.AUTH]: 8,
  [ForgeErrorCode.VERIFY_FAILED]: 9,
  [ForgeErrorCode.IO]: 10,
  [ForgeErrorCode.NOT_FOUND]: 11,
  [ForgeErrorCode.CONFIG]: 12,
  [ForgeErrorCode.TOOLCHAIN]: 13,
  [ForgeErrorCode.POLICY]: 14,
  [ForgeErrorCode.LICENSE]: 15,
  [ForgeErrorCode.KEY]: 16,
  [ForgeErrorCode.REGISTRY]: 17,
  [ForgeErrorCode.LINEAGE]: 18,
  [ForgeErrorCode.INTERNAL]: 20,
});

/** Structured, secret-free detail bag attached to a {@link ForgeError}. */
export type ForgeErrorDetails = Record<string, unknown>;

export interface ForgeErrorJson {
  code: ForgeErrorCode;
  message: string;
  details?: ForgeErrorDetails;
}

const SECRET_KEY_PATTERN = /(token|secret|password|passphrase|credential|api[-_]?key|private[-_]?key|signing[-_]?key)/i;
const BEARER_PATTERN = /\bBearer\s+[A-Za-z0-9._~+/-]+=*/gi;
const URL_USERINFO_PATTERN = /(\b[a-z][a-z0-9+.-]*:\/\/)[^/@\s]+@/gi;
const LONG_OPAQUE_PATTERN = /\b(?=[A-Za-z0-9_-]*[0-9])(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{32,}\b/g;

/** Lengths of the lowercase-hex digests forge reports: file id, git rev, sha256. */
const HEX_DIGEST_LENGTHS = new Set([32, 40, 64]);
const HEX_ONLY = /^[0-9a-f]+$/;

export const REDACTED = '[redacted]';

/**
 * Strip anything that looks like credential material out of a string bound
 * for a log line, an error message, or a JSON envelope.
 *
 * This is a defence in depth measure. Forge never reads private key material
 * in the first place — it carries signing *references* only — but tokens can
 * reach a message by way of a URL or an upstream error string.
 */
export function redact(text: string): string {
  return text
    .replace(BEARER_PATTERN, `Bearer ${REDACTED}`)
    .replace(URL_USERINFO_PATTERN, `$1${REDACTED}@`)
    .replace(LONG_OPAQUE_PATTERN, (match: string, offset: number, whole: string): string => {
      // A lowercase-hex run of digest length is a content address — an RVF
      // identity, a manifest hash, a source revision — not a credential.
      // Reporting those is most of what forge's errors are *for*, and the
      // tokens this pattern exists to catch are mixed-case base64url with
      // `-` and `_`, never pure hex.
      if (HEX_DIGEST_LENGTHS.has(match.length) && HEX_ONLY.test(match)) return match;

      // A long opaque run bordered by a path separator is a directory or file
      // name, not a credential. Redacting those would make the most common
      // error messages — "cannot open <path>" — unreadable, and forge never
      // puts a token in a path: credentials travel in headers only.
      const before = whole[offset - 1];
      const after = whole[offset + match.length];
      const pathish = (c: string | undefined): boolean => c === '/' || c === '\\';
      return pathish(before) || pathish(after) ? match : REDACTED;
    });
}

/** Recursively redact a detail bag: secret-named keys are dropped entirely. */
export function redactDetails(details: ForgeErrorDetails): ForgeErrorDetails {
  const out: ForgeErrorDetails = {};
  for (const key of Object.keys(details).sort()) {
    if (SECRET_KEY_PATTERN.test(key)) {
      out[key] = REDACTED;
      continue;
    }
    out[key] = redactValue(details[key]);
  }
  return out;
}

function redactValue(value: unknown): unknown {
  if (typeof value === 'string') return redact(value);
  if (Array.isArray(value)) return value.map(redactValue);
  if (value && typeof value === 'object') {
    return redactDetails(value as ForgeErrorDetails);
  }
  return value;
}

/** The only error type forge raises across a command boundary. */
export class ForgeError extends Error {
  readonly code: ForgeErrorCode;
  readonly details: ForgeErrorDetails;

  constructor(code: ForgeErrorCode, message: string, details: ForgeErrorDetails = {}) {
    super(redact(message));
    this.name = 'ForgeError';
    this.code = code;
    this.details = redactDetails(details);
    Object.setPrototypeOf(this, ForgeError.prototype);
  }

  /** Process exit code this error maps to. */
  get exitCode(): number {
    return EXIT_CODES[this.code] ?? EXIT_CODES[ForgeErrorCode.INTERNAL];
  }

  toJSON(): ForgeErrorJson {
    const json: ForgeErrorJson = { code: this.code, message: this.message };
    if (Object.keys(this.details).length > 0) json.details = this.details;
    return json;
  }
}

/** Coerce an unknown thrown value into a {@link ForgeError}. */
export function toForgeError(err: unknown): ForgeError {
  if (err instanceof ForgeError) return err;
  const message = err instanceof Error ? err.message : String(err);
  return new ForgeError(ForgeErrorCode.INTERNAL, message);
}

export const exitCodeFor = (code: ForgeErrorCode): number =>
  EXIT_CODES[code] ?? EXIT_CODES[ForgeErrorCode.INTERNAL];
