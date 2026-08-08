/**
 * Typed client for the hosted RVForge build service (requirements §7).
 *
 * Status: client-side only. The service does not exist yet, so every call here
 * fails with `FORGE_E_NETWORK` against a live endpoint. The request and
 * response shapes are the contract the service will be built to; nothing in
 * this file fakes a successful build.
 *
 * The bearer token is read from the environment and is never logged, echoed
 * into an error message, or written to a provenance record.
 */

import { ForgeError, ForgeErrorCode } from './errors';
import type { BuildManifest, BuildTarget } from './types';

export const DEFAULT_API_URL = 'https://forge.ruvector.dev';
export const API_URL_ENV = 'FORGE_API_URL';
export const API_TOKEN_ENV = 'FORGE_API_TOKEN';

export type BuildState = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface BuildArtifact {
  name: string;
  target: BuildTarget;
  sizeBytes: number;
  sha256: string;
  /** Time-limited download URL issued by the service. */
  url: string;
}

export interface BuildRecord {
  id: string;
  state: BuildState;
  createdAt: string;
  updatedAt: string;
  targets: BuildTarget[];
  rvfIdentity: string;
  manifestSha256: string;
  /** Present once the build reaches a terminal state. */
  message?: string;
}

export interface SubmitRequest {
  manifest: BuildManifest;
  /** SHA256 of the RVF the service should expect on the upload channel. */
  rvfSha256: string;
  rvfSizeBytes: number;
}

export interface VerifyResponse {
  id: string;
  verified: boolean;
  checks: Array<{ name: string; passed: boolean; detail?: string }>;
}

export interface ForgeApiOptions {
  baseUrl?: string;
  token?: string;
  /** Per-request timeout in milliseconds. */
  timeoutMs?: number;
  /** Injected for tests; defaults to the global `fetch`. */
  fetchImpl?: typeof fetch;
}

export class ForgeApiClient {
  private readonly baseUrl: string;
  private readonly token: string | undefined;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ForgeApiOptions = {}) {
    this.baseUrl = (options.baseUrl ?? process.env[API_URL_ENV] ?? DEFAULT_API_URL).replace(/\/+$/, '');
    this.token = options.token ?? process.env[API_TOKEN_ENV];
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch;
  }

  /** Endpoint the client is pointed at, for display. Carries no credentials. */
  get endpoint(): string {
    return this.baseUrl;
  }

  get hasToken(): boolean {
    return typeof this.token === 'string' && this.token.length > 0;
  }

  submitBuild(request: SubmitRequest): Promise<BuildRecord> {
    return this.request<BuildRecord>('POST', '/v1/builds', request);
  }

  getBuild(id: string): Promise<BuildRecord> {
    return this.request<BuildRecord>('GET', `/v1/builds/${encodeURIComponent(id)}`);
  }

  listArtifacts(id: string): Promise<{ artifacts: BuildArtifact[] }> {
    return this.request<{ artifacts: BuildArtifact[] }>(
      'GET',
      `/v1/builds/${encodeURIComponent(id)}/artifacts`,
    );
  }

  cancelBuild(id: string): Promise<BuildRecord> {
    return this.request<BuildRecord>('POST', `/v1/builds/${encodeURIComponent(id)}/cancel`);
  }

  verifyBuild(id: string): Promise<VerifyResponse> {
    return this.request<VerifyResponse>('POST', `/v1/builds/${encodeURIComponent(id)}/verify`);
  }

  deleteBuild(id: string): Promise<{ id: string; deleted: boolean }> {
    return this.request<{ id: string; deleted: boolean }>(
      'DELETE',
      `/v1/builds/${encodeURIComponent(id)}`,
    );
  }

  /** Fetch an artifact body as bytes. Callers verify the digest themselves. */
  async fetchArtifact(url: string): Promise<Buffer> {
    const response = await this.send('GET', url, undefined, { accept: 'application/octet-stream' });
    const buffer = Buffer.from(await response.arrayBuffer());
    return buffer;
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const response = await this.send(method, `${this.baseUrl}${path}`, body, {
      accept: 'application/json',
    });
    try {
      return (await response.json()) as T;
    } catch {
      throw new ForgeError(
        ForgeErrorCode.NETWORK,
        `Build service returned a non-JSON response for ${method} ${path}.`,
        { method, path, status: response.status },
      );
    }
  }

  private async send(
    method: string,
    url: string,
    body: unknown,
    headers: { accept: string },
  ): Promise<Response> {
    if (typeof this.fetchImpl !== 'function') {
      throw new ForgeError(
        ForgeErrorCode.NETWORK,
        'No fetch implementation available; Node.js 20 or later is required.',
      );
    }

    const requestHeaders: Record<string, string> = { accept: headers.accept };
    if (body !== undefined) requestHeaders['content-type'] = 'application/json';
    if (this.token) requestHeaders.authorization = `Bearer ${this.token}`;

    let response: Response;
    try {
      response = await this.fetchImpl(url, {
        method,
        headers: requestHeaders,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (err) {
      // Deliberately reports the method and path only — never the headers,
      // which hold the bearer token.
      throw new ForgeError(
        ForgeErrorCode.NETWORK,
        `Cannot reach the build service at ${this.baseUrl}: ${(err as Error).message}`,
        { method, endpoint: this.baseUrl },
      );
    }

    if (response.ok) return response;
    throw this.httpError(response, method, url);
  }

  private httpError(response: Response, method: string, url: string): ForgeError {
    const path = safePath(url);
    if (response.status === 401 || response.status === 403) {
      return new ForgeError(
        ForgeErrorCode.AUTH,
        `Build service rejected the credentials (HTTP ${response.status}). Set ${API_TOKEN_ENV}.`,
        { status: response.status, method, path },
      );
    }
    if (response.status === 404) {
      return new ForgeError(ForgeErrorCode.NOT_FOUND, `Not found: ${method} ${path}.`, {
        status: 404,
        method,
        path,
      });
    }
    return new ForgeError(
      ForgeErrorCode.NETWORK,
      `Build service returned HTTP ${response.status} for ${method} ${path}.`,
      { status: response.status, method, path },
    );
  }
}

/** Strip query strings, which can carry signed-URL credentials. */
function safePath(url: string): string {
  try {
    const parsed = new URL(url);
    return parsed.pathname;
  } catch {
    return url.split('?')[0];
  }
}
