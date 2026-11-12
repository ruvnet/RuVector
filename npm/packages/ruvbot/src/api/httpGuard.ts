/**
 * HTTP guard helpers shared by the RuvBot API servers.
 *
 * - Bounded JSON body reading (prevents unbounded memory use per request)
 * - Constant-time API authentication (bearer / apikey / basic)
 * - Session summaries that never expose message contents
 */

import type { IncomingHttpHeaders, IncomingMessage } from 'node:http';
import { createHash, timingSafeEqual } from 'node:crypto';
import type { Session } from '../core/types.js';

export const DEFAULT_MAX_BODY_BYTES = 1024 * 1024; // 1 MiB

export class PayloadTooLargeError extends Error {
  readonly statusCode = 413;
  constructor(limit: number) {
    super(`Request body exceeds ${limit} bytes`);
    this.name = 'PayloadTooLargeError';
  }
}

export class InvalidJsonError extends Error {
  readonly statusCode = 400;
  constructor() {
    super('Invalid JSON');
    this.name = 'InvalidJsonError';
  }
}

/**
 * Read and parse a JSON request body, rejecting once `maxBytes` is exceeded.
 * Resolves to null for an empty body.
 */
export function readJsonBody(
  req: IncomingMessage,
  maxBytes: number = DEFAULT_MAX_BODY_BYTES
): Promise<Record<string, unknown> | null> {
  return new Promise((resolve, reject) => {
    const declared = Number(req.headers['content-length']);
    if (Number.isFinite(declared) && declared > maxBytes) {
      req.resume();
      reject(new PayloadTooLargeError(maxBytes));
      return;
    }

    const chunks: Buffer[] = [];
    let size = 0;
    let settled = false;
    const fail = (err: Error): void => {
      if (settled) return;
      settled = true;
      reject(err);
    };

    req.on('data', (chunk: Buffer) => {
      if (settled) return;
      size += chunk.length;
      if (size > maxBytes) {
        chunks.length = 0;
        // Stop buffering; drain the rest so the response can still be written.
        req.resume();
        fail(new PayloadTooLargeError(maxBytes));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (settled) return;
      settled = true;
      if (chunks.length === 0) {
        resolve(null);
        return;
      }
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString('utf-8')) as Record<string, unknown>);
      } catch {
        reject(new InvalidJsonError());
      }
    });
    req.on('error', fail);
  });
}

export interface ApiAuthConfig {
  enabled: boolean;
  type: 'bearer' | 'basic' | 'apikey';
  secret?: string;
}

function safeEqual(a: string, b: string): boolean {
  // Hash both sides so the comparison is constant-time regardless of length.
  const ha = createHash('sha256').update(a).digest();
  const hb = createHash('sha256').update(b).digest();
  return timingSafeEqual(ha, hb);
}

function headerValue(headers: IncomingHttpHeaders, name: string): string | undefined {
  const v = headers[name];
  return Array.isArray(v) ? v[0] : v;
}

/**
 * Check a request's credentials against the configured API auth.
 * Fails closed when auth is enabled but no secret is configured.
 */
export function isRequestAuthorized(headers: IncomingHttpHeaders, auth: ApiAuthConfig | undefined): boolean {
  if (!auth || !auth.enabled) return true;
  if (!auth.secret) return false;

  const authorization = headerValue(headers, 'authorization') ?? '';
  switch (auth.type) {
    case 'bearer': {
      const m = /^Bearer\s+(.+)$/i.exec(authorization);
      return !!m && safeEqual(m[1].trim(), auth.secret);
    }
    case 'apikey': {
      const key = headerValue(headers, 'x-api-key');
      return typeof key === 'string' && safeEqual(key, auth.secret);
    }
    case 'basic': {
      const m = /^Basic\s+(.+)$/i.exec(authorization);
      if (!m) return false;
      const decoded = Buffer.from(m[1].trim(), 'base64').toString('utf-8');
      return safeEqual(decoded, auth.secret);
    }
    default:
      return false;
  }
}

const PUBLIC_PATHS = new Set(['/', '/health', '/healthz', '/ready', '/readyz']);

/** Paths that never require authentication (landing page and probes). */
export function isPublicPath(pathname: string): boolean {
  return PUBLIC_PATHS.has(pathname);
}

export interface SessionSummary {
  id: string;
  agentId: string;
  userId?: string;
  channelId?: string;
  platform: Session['platform'];
  messageCount: number;
  createdAt: Date;
  updatedAt: Date;
  expiresAt?: Date;
}

/** Listing view of a session: identifiers and counts only, never message bodies or metadata. */
export function summarizeSession(session: Session): SessionSummary {
  return {
    id: session.id,
    agentId: session.agentId,
    userId: session.userId,
    channelId: session.channelId,
    platform: session.platform,
    messageCount: session.messages.length,
    createdAt: session.createdAt,
    updatedAt: session.updatedAt,
    expiresAt: session.expiresAt,
  };
}
