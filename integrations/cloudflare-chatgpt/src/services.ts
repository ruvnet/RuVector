import type { Principal } from './auth.ts';

export type ServiceName = 'METAHARNESS' | 'AUTOGENOUS' | 'RUVLLM';
export interface ServiceEnv {
  METAHARNESS?: Fetcher;
  AUTOGENOUS?: Fetcher;
  RUVLLM?: Fetcher;
  SERVICE_SIGNING_KEY?: string;
}

export function serviceReady(env: ServiceEnv, name: ServiceName): boolean {
  return Boolean(env[name] && env.SERVICE_SIGNING_KEY && env.SERVICE_SIGNING_KEY.length >= 32);
}

// Internal service bindings have a fixed destination. The downstream Worker must
// verify the HMAC, timestamp, tenant, operation and idempotency key before work.
export async function invokeService(
  env: ServiceEnv, name: ServiceName, principal: Principal, tenantId: string,
  operation: string, input: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  if (!serviceReady(env, name)) throw new Error('service_unavailable');
  const body = JSON.stringify({
    version: 1, tenantId, issuer: principal.issuer, actor: principal.subject,
    operation, input, issuedAt: Date.now(), requestId: crypto.randomUUID(),
  });
  const encoder = new TextEncoder();
  const bodyBytes = encoder.encode(body);
  if (bodyBytes.byteLength > 8192) throw new RangeError('service_payload_too_large');
  const key = await crypto.subtle.importKey('raw', encoder.encode(env.SERVICE_SIGNING_KEY!),
    { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const signature = await crypto.subtle.sign('HMAC', key, bodyBytes);
  const encoded = Array.from(new Uint8Array(signature), b => b.toString(16).padStart(2, '0')).join('');
  const response = await env[name]!.fetch('https://ruvector.internal/ruvector-internal/v1', {
    method: 'POST', headers: { 'content-type': 'application/json', 'x-ruvector-signature': encoded },
    body, signal: AbortSignal.timeout(10000),
  });
  if (!response.ok) throw new Error('service_error');
  const text = await response.text();
  if (encoder.encode(text).byteLength > 65536) throw new Error('service_response_too_large');
  const result = JSON.parse(text);
  if (!result || typeof result !== 'object' || Array.isArray(result)) throw new Error('invalid_service_response');
  return result as Record<string, unknown>;
}
