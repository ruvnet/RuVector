import { createRemoteJWKSet, jwtVerify, type JWTVerifyGetKey } from 'jose';

export type Role = 'viewer' | 'editor' | 'admin';
export type Membership = { tenant_id: string; role: Role };
export type Principal = { issuer: string; subject: string; scopes: Set<string>; memberships: Membership[] };

export interface AuthEnv {
  DB: D1Database;
  OAUTH_ISSUER: string;
  OAUTH_AUDIENCE: string;
  OAUTH_JWKS_URL: string;
  MAX_DAILY_CALLS?: string;
}

export class AccessError extends Error {
  constructor(public status: number, public code: string) {
    super(code);
  }
}

const keySets = new Map<string, JWTVerifyGetKey>();

export function authConfigured(env: AuthEnv): boolean {
  try {
    const issuer = new URL(env.OAUTH_ISSUER);
    const jwks = new URL(env.OAUTH_JWKS_URL);
    return issuer.protocol === 'https:' && jwks.protocol === 'https:' &&
      !issuer.hostname.includes('replace_with') && !jwks.hostname.includes('replace_with') &&
      Boolean(env.OAUTH_AUDIENCE);
  } catch {
    return false;
  }
}

export async function authenticate(request: Request, env: AuthEnv, testKey?: JWTVerifyGetKey): Promise<Principal> {
  if (!authConfigured(env)) throw new AccessError(503, 'identity_provider_unconfigured');
  const match = /^Bearer ([A-Za-z0-9._~+/-]+=*)$/.exec(request.headers.get('authorization') ?? '');
  if (!match) throw new AccessError(401, 'authentication_required');

  let subject: string;
  let scopes: Set<string>;
  try {
    let keys = testKey ?? keySets.get(env.OAUTH_JWKS_URL);
    if (!keys) {
      keys = createRemoteJWKSet(new URL(env.OAUTH_JWKS_URL), { timeoutDuration: 3000 });
      keySets.set(env.OAUTH_JWKS_URL, keys);
    }
    const { payload } = await jwtVerify(match[1], keys, {
      issuer: env.OAUTH_ISSUER,
      audience: env.OAUTH_AUDIENCE,
      clockTolerance: 30,
    });
    if (!payload.sub || !payload.exp || !payload.iat) throw new Error('required_claim_missing');
    subject = payload.sub;
    scopes = new Set([
      ...(typeof payload.scope === 'string' ? payload.scope.split(/\s+/) : []),
      ...(Array.isArray(payload.scp) ? payload.scp.filter((s): s is string => typeof s === 'string') : []),
    ]);
    if (!scopes.has('ruvector.read')) throw new AccessError(403, 'read_scope_required');
  } catch (error) {
    if (error instanceof AccessError) throw error;
    throw new AccessError(401, 'invalid_access_token');
  }
  try {
    const { results } = await env.DB.prepare(
      'SELECT tenant_id, role FROM memberships WHERE issuer = ? AND subject = ? AND enabled = 1 LIMIT 21',
    ).bind(env.OAUTH_ISSUER, subject).all<Membership>();
    if (results.length > 20) throw new AccessError(403, 'membership_limit_exceeded');
    const memberships = results.filter(m => /^[A-Za-z0-9_-]{1,64}$/.test(m.tenant_id) &&
      ['viewer', 'editor', 'admin'].includes(m.role));
    if (memberships.length === 0) throw new AccessError(403, 'no_tenant_membership');
    return { issuer: env.OAUTH_ISSUER, subject, scopes, memberships };
  } catch (error) {
    if (error instanceof AccessError) throw error;
    throw new AccessError(503, 'membership_store_unavailable');
  }
}

export function requireTenant(principal: Principal, tenantId: string, write = false): Membership {
  const membership = principal.memberships.find(m => m.tenant_id === tenantId);
  if (!membership || (write && (membership.role === 'viewer' || !principal.scopes.has('ruvector.write')))) {
    throw new AccessError(403, 'tenant_access_denied');
  }
  return membership;
}

export async function chargeTenant(env: AuthEnv, tenantId: string): Promise<number> {
  const configured = Number(env.MAX_DAILY_CALLS ?? '1000');
  const limit = Number.isSafeInteger(configured) && configured > 0 ? Math.min(configured, 100000) : 1000;
  const day = new Date().toISOString().slice(0, 10);
  const result = await env.DB.prepare(
    `INSERT INTO usage_daily (tenant_id, day, used) VALUES (?, ?, 1)
     ON CONFLICT(tenant_id, day) DO UPDATE SET used = used + 1 WHERE used < ?
     RETURNING used`,
  ).bind(tenantId, day, limit).first<{ used: number }>();
  if (!result) throw new AccessError(429, 'tenant_daily_limit_reached');
  return result.used;
}
