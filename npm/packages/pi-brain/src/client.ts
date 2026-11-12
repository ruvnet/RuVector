/**
 * π Brain SDK Client
 *
 * Communicates with the π shared brain at pi.ruv.io
 */

const DEFAULT_URL = 'https://pi.ruv.io';
const DEFAULT_TIMEOUT_MS = 30_000;
const MAX_ERROR_BODY = 500;
const LOOPBACK = new Set(['localhost', '127.0.0.1', '[::1]']);

/**
 * Parse and validate the base URL. The bearer token is attached to every
 * request, so plain http is only allowed for loopback development servers.
 */
function normalizeBaseUrl(raw: string): string {
  let u: URL;
  try {
    u = new URL(raw);
  } catch {
    throw new Error(`Invalid π base URL: ${raw}`);
  }
  if (u.protocol !== 'https:' && !(u.protocol === 'http:' && LOOPBACK.has(u.hostname))) {
    throw new Error(
      `Refusing π base URL ${u.origin}: https is required (plain http only for localhost)`,
    );
  }
  return u.toString().replace(/\/+$/, '');
}

export interface ShareOptions {
  category: string;
  title: string;
  content: string;
  tags?: string[];
  code_snippet?: string;
}

export interface SearchOptions {
  query: string;
  category?: string;
  tags?: string;
  limit?: number;
  min_quality?: number;
}

export interface Memory {
  id: string;
  category: string;
  title: string;
  content: string;
  tags: string[];
  quality_score: number;
  contributor_id: string;
  created_at: string;
}

export interface ConsciousnessComputeOptions {
  /** Transition probability matrix (flattened n×n row-major) */
  tpm: number[];
  /** Number of states (must be power of 2) */
  n: number;
  /** Current state index */
  state: number;
  /** Algorithm: iit4_phi, ces, phi_id, pid, bounds, auto (default: auto) */
  algorithm?: string;
  /** Min φ for CES distinctions (default: 1e-6) */
  phi_threshold?: number;
  /** Bitmask for ΦID/PID source partition */
  partition_mask?: number;
}

export interface ConsciousnessComputeResult {
  algorithm: string;
  phi: number;
  num_elements: number;
  num_states: number;
  elapsed_us: number;
  details: Record<string, unknown>;
}

export class PiBrainClient {
  private baseUrl: string;
  private apiKey: string;
  private timeoutMs: number;

  constructor(options?: { url?: string; apiKey?: string; timeoutMs?: number }) {
    this.baseUrl = normalizeBaseUrl(options?.url ?? process.env.BRAIN_URL ?? DEFAULT_URL);
    this.timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.apiKey =
      options?.apiKey ??
      process.env.PI ??
      process.env.BRAIN_API_KEY ??
      'anonymous';
  }

  private async request(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<unknown> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      Authorization: `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
    };

    let res: Response;
    try {
      res = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (e) {
      const name = (e as Error).name;
      if (name === 'TimeoutError' || name === 'AbortError') {
        throw new Error(`π request timed out after ${this.timeoutMs}ms: ${method} ${path}`);
      }
      throw e;
    }

    if (!res.ok) {
      const text = await res.text();
      const shown = text.length > MAX_ERROR_BODY ? `${text.slice(0, MAX_ERROR_BODY)}…` : text;
      throw new Error(`π error (${res.status}): ${shown}`);
    }

    // 204 No Content (e.g. DELETE /v1/memories/:id) has no body to parse.
    const text = await res.text();
    if (res.status === 204 || text.length === 0) return null;
    try {
      return JSON.parse(text);
    } catch {
      throw new Error(`π returned non-JSON response (${res.status}) for ${method} ${path}`);
    }
  }

  async health(): Promise<unknown> {
    return this.request('GET', '/v1/health');
  }

  async share(opts: ShareOptions): Promise<unknown> {
    return this.request('POST', '/v1/memories', opts);
  }

  async search(opts: SearchOptions): Promise<unknown> {
    const params = new URLSearchParams();
    params.set('q', opts.query);
    if (opts.category) params.set('category', opts.category);
    if (opts.tags) params.set('tags', opts.tags);
    if (opts.limit) params.set('limit', String(opts.limit));
    if (opts.min_quality) params.set('min_quality', String(opts.min_quality));
    return this.request('GET', `/v1/memories/search?${params}`);
  }

  async get(id: string): Promise<unknown> {
    return this.request('GET', `/v1/memories/${encodeURIComponent(id)}`);
  }

  async list(category?: string, limit?: number): Promise<unknown> {
    const params = new URLSearchParams();
    if (category) params.set('category', category);
    if (limit) params.set('limit', String(limit));
    return this.request('GET', `/v1/memories/list?${params}`);
  }

  async vote(id: string, direction: 'up' | 'down'): Promise<unknown> {
    return this.request('POST', `/v1/memories/${encodeURIComponent(id)}/vote`, { direction });
  }

  async delete(id: string): Promise<unknown> {
    return this.request('DELETE', `/v1/memories/${encodeURIComponent(id)}`);
  }

  async transfer(source: string, target: string): Promise<unknown> {
    return this.request('POST', '/v1/transfer', {
      source_domain: source,
      target_domain: target,
    });
  }

  async drift(domain?: string): Promise<unknown> {
    const params = new URLSearchParams();
    if (domain) params.set('domain', domain);
    return this.request('GET', `/v1/drift?${params}`);
  }

  async partition(domain?: string): Promise<unknown> {
    const params = new URLSearchParams();
    if (domain) params.set('domain', domain);
    return this.request('GET', `/v1/partition?${params}`);
  }

  async status(): Promise<unknown> {
    return this.request('GET', '/v1/status');
  }

  async consciousnessCompute(
    opts: ConsciousnessComputeOptions,
  ): Promise<ConsciousnessComputeResult> {
    return this.request('POST', '/v1/consciousness/compute', opts) as Promise<ConsciousnessComputeResult>;
  }

  async consciousnessStatus(): Promise<unknown> {
    return this.request('GET', '/v1/consciousness/status');
  }
}
