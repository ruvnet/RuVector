import { McpServer } from '@modelcontextprotocol/server';
import { createMcpHandler } from 'agents/mcp/server';
import { z } from 'zod';
import rvfModule from './vendor/rvf.wasm';
import mincutModule from './vendor/mincut.wasm';
import { AccessError, authenticate, authConfigured, chargeTenant, requireTenant, type AuthEnv, type Principal } from './auth.ts';
import { calculateMinCut } from './mincut.ts';
import { loadVectors, storeVectors } from './storage.ts';
import { searchExact, validateVector } from './vector.ts';
import { consoleHtml, UI_URI } from './ui.ts';
import { invokeService, serviceReady, type ServiceEnv } from './services.ts';
import { requestTooLarge } from './limits.ts';

export interface Env extends AuthEnv, ServiceEnv {}

const oauth = [{ type: 'oauth2', scopes: ['ruvector.read'] }];
const tenantSchema = z.string().regex(/^[A-Za-z0-9_-]{1,64}$/);
const collectionSchema = z.string().regex(/^[A-Za-z0-9_-]{1,64}$/);
const vectorSchema = z.array(z.number().finite()).min(1).max(1536);
const recordSchema = z.object({
  id: z.string().regex(/^[A-Za-z0-9_-]{1,64}$/),
  vector: vectorSchema,
  metadata: z.record(z.string(), z.unknown()).optional(),
});

const coreCapabilities = [
  { name: 'RVF exact vector search', available: true, runtime: 'WASM', limit: '500 vectors per query' },
  { name: 'RVF action ranking', available: true, runtime: 'WASM', limit: 'numeric vectors only' },
  { name: 'RuVector weighted mincut', available: true, runtime: 'WASM', limit: '100 edges per call' },
  { name: 'HNSW persistent indexes', available: false, reason: 'requires a tenant partitioned index service' },
] as const;

function capabilityList(env: Env) {
  return [...coreCapabilities,
    { name: 'MetaHarness Darwin', available: serviceReady(env, 'METAHARNESS'), reason: 'requires isolated evaluation Worker and acceptance gates' },
    { name: 'Autogenous', available: serviceReady(env, 'AUTOGENOUS'), reason: 'requires tenant scoped orchestration Worker' },
    { name: 'ruvLLM inference', available: serviceReady(env, 'RUVLLM'), reason: 'requires separately validated inference Worker; local wrapper is a placeholder' },
  ];
}

function reply(data: Record<string, unknown>) {
  return { content: [{ type: 'text' as const, text: JSON.stringify(data) }], structuredContent: data };
}

async function inTenant(
  env: Env, principal: Principal, tenantId: string, write: boolean,
  work: () => Promise<Record<string, unknown>>,
) {
  try {
    requireTenant(principal, tenantId, write);
    const used = await chargeTenant(env, tenantId);
    return reply({ ...await work(), usageToday: used });
  } catch (error) {
    const code = error instanceof AccessError || error instanceof RangeError ? error.message : 'internal_error';
    return { ...reply({ error: code }), isError: true };
  }
}

export function createServer(env: Env, principal: Principal) {
  const server = new McpServer({ name: 'RuVector Cloudflare', version: '0.1.0' });

  server.registerResource('ruvector-console', UI_URI,
    { title: 'RuVector Console', mimeType: 'text/html;profile=mcp-app' },
    async uri => ({ contents: [{ uri: uri.href, mimeType: 'text/html;profile=mcp-app', text: consoleHtml,
      _meta: { ui: { prefersBorder: true, csp: { connectDomains: [], resourceDomains: [] } } },
    }] }),
  );

  server.registerTool('open_console', {
    title: 'Open RuVector Console',
    description: 'Show the tenant scoped vector search console and capability status.',
    inputSchema: z.object({}),
    annotations: { readOnlyHint: true },
    _meta: { ui: { resourceUri: UI_URI }, 'openai/outputTemplate': UI_URI, securitySchemes: oauth },
  }, async () => reply({ tenants: principal.memberships, capabilities: capabilityList(env) }));

  server.registerTool('list_tenants', {
    title: 'List RuVector tenants', description: 'List only tenants granted to this authenticated identity.',
    inputSchema: z.object({}), annotations: { readOnlyHint: true },
    _meta: { securitySchemes: oauth },
  }, async () => reply({ tenants: principal.memberships }));

  server.registerTool('capability_catalog', {
    title: 'RuVector capabilities', description: 'Report deployed capability status and exact runtime limits.',
    inputSchema: z.object({}), annotations: { readOnlyHint: true },
    _meta: { securitySchemes: oauth },
  }, async () => reply({ capabilities: capabilityList(env) }));

  server.registerTool('search_vectors', {
    title: 'Search tenant vectors', description: 'Exact RVF WASM nearest neighbor search over one tenant collection. Input is numeric vectors, not text.',
    inputSchema: z.object({ tenant_id: tenantSchema, collection: collectionSchema, query: vectorSchema, k: z.number().int().min(1).max(20).default(5), metric: z.enum(['cosine', 'l2']).default('cosine') }),
    annotations: { readOnlyHint: true }, _meta: { securitySchemes: oauth },
  }, async ({ tenant_id, collection, query, k, metric }) => inTenant(env, principal, tenant_id, false, async () => {
    const records = await loadVectors(env, tenant_id, collection);
    if (records.length && records[0].dimension !== query.length) throw new RangeError('dimension_mismatch');
    const hits = await searchExact(rvfModule, records, query, k, metric);
    return { tenant_id, collection, metric, hits, scanned: records.length, algorithm: 'rvf_exact_wasm' };
  }));

  server.registerTool('upsert_vectors', {
    title: 'Upsert tenant vectors', description: 'Add or replace up to 24 numeric vectors in one authorized tenant collection.',
    inputSchema: z.object({ tenant_id: tenantSchema, collection: collectionSchema, records: z.array(recordSchema).min(1).max(24) }),
    annotations: { readOnlyHint: false, destructiveHint: false },
    _meta: { securitySchemes: [{ type: 'oauth2', scopes: ['ruvector.write'] }] },
  }, async ({ tenant_id, collection, records }) => inTenant(env, principal, tenant_id, true, async () => {
    const count = await storeVectors(env, tenant_id, collection, records);
    return { tenant_id, collection, upserted: count };
  }));

  server.registerTool('rank_actions', {
    title: 'Rank numeric actions', description: 'Use RVF WASM to rank explicit action vectors. Distance is not a calibrated probability.',
    inputSchema: z.object({ tenant_id: tenantSchema, state: vectorSchema,
      actions: z.array(recordSchema).min(1).max(100), k: z.number().int().min(1).max(20).default(5) }),
    annotations: { readOnlyHint: true }, _meta: { securitySchemes: oauth },
  }, async ({ tenant_id, state, actions, k }) => inTenant(env, principal, tenant_id, false, async () => {
    validateVector(state);
    const ranked = await searchExact(rvfModule, actions, state, k);
    return { tenant_id, ranked, algorithm: 'rvf_exact_wasm', calibrated: false };
  }));

  server.registerTool('analyze_graph', {
    title: 'Analyze weighted graph', description: 'Compute a weighted mincut in the RuVector WASM module for up to 100 edges.',
    inputSchema: z.object({ tenant_id: tenantSchema, edges: z.array(z.tuple([
      z.number().int().nonnegative().max(1000), z.number().int().nonnegative().max(1000),
      z.number().positive().max(10000),
    ])).min(1).max(100) }),
    annotations: { readOnlyHint: true }, _meta: { securitySchemes: oauth },
  }, async ({ tenant_id, edges }) => inTenant(env, principal, tenant_id, false, async () => ({
    tenant_id, ...calculateMinCut(mincutModule, edges), algorithm: 'ruvector_mincut_wasm',
  })));

  if (serviceReady(env, 'METAHARNESS')) server.registerTool('evaluate_candidate', {
    title: 'Evaluate a MetaHarness candidate',
    description: 'Submit bounded numeric parameters to a tenant isolated evaluation Worker. It cannot promote a candidate.',
    inputSchema: z.object({ tenant_id: tenantSchema, metric: z.enum(['recall', 'latency', 'cost']),
      parameters: z.record(z.string().regex(/^[A-Za-z0-9_]{1,40}$/), z.number().finite()).refine(o => Object.keys(o).length <= 16), }),
    _meta: { securitySchemes: [{ type: 'oauth2', scopes: ['ruvector.write'] }] },
  }, async ({ tenant_id, metric, parameters }) => inTenant(env, principal, tenant_id, true, async () => ({
    tenant_id, ...await invokeService(env, 'METAHARNESS', principal, tenant_id, 'evaluate_candidate', { metric, parameters, promote: false }),
  })));

  if (serviceReady(env, 'AUTOGENOUS')) server.registerTool('propose_learning', {
    title: 'Propose an Autogenous learning run',
    description: 'Request a tenant scoped dry run proposal. No weights or acceptance criteria are changed.',
    inputSchema: z.object({ tenant_id: tenantSchema, objective: z.string().min(5).max(500) }),
    _meta: { securitySchemes: [{ type: 'oauth2', scopes: ['ruvector.write'] }] },
  }, async ({ tenant_id, objective }) => inTenant(env, principal, tenant_id, true, async () => ({
    tenant_id, ...await invokeService(env, 'AUTOGENOUS', principal, tenant_id, 'propose_learning', { objective, dryRun: true }),
  })));

  if (serviceReady(env, 'RUVLLM')) server.registerTool('generate_text', {
    title: 'Generate with tenant ruvLLM',
    description: 'Generate text through a tenant scoped validated inference Worker.',
    inputSchema: z.object({ tenant_id: tenantSchema, prompt: z.string().min(1).max(2000), max_tokens: z.number().int().min(1).max(256).default(128) }),
    _meta: { securitySchemes: oauth },
  }, async ({ tenant_id, prompt, max_tokens }) => inTenant(env, principal, tenant_id, false, async () => ({
    tenant_id, ...await invokeService(env, 'RUVLLM', principal, tenant_id, 'generate_text', { prompt, maxTokens: max_tokens }),
  })));

  return server;
}

function json(data: Record<string, unknown>, status: number, extra: HeadersInit = {}) {
  return new Response(JSON.stringify(data), { status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store', ...extra } });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === '/health' && request.method === 'GET') {
      return json({ name: 'RuVector Cloudflare', version: '0.1.0', authConfigured: authConfigured(env) }, 200);
    }
    if (url.pathname === '/.well-known/oauth-protected-resource' && request.method === 'GET') {
      return json({ resource: `${url.origin}/mcp`, authorization_servers: [env.OAUTH_ISSUER], scopes_supported: ['ruvector.read', 'ruvector.write'] }, 200);
    }
    if (url.pathname !== '/mcp') return json({ error: 'not_found' }, 404);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: { allow: 'GET, POST, DELETE, OPTIONS' } });
    const metadata = `${url.origin}/.well-known/oauth-protected-resource`;
    try {
      const principal = await authenticate(request, env);
      if (await requestTooLarge(request)) return json({ error: 'request_too_large' }, 413);
      const handler = createMcpHandler(() => createServer(env, principal), { route: '/mcp', corsOptions: false });
      return handler.fetch(request);
    } catch (error) {
      if (error instanceof AccessError) {
        const headers: Record<string, string> = error.status === 401 ? { 'www-authenticate': `Bearer resource_metadata="${metadata}", scope="ruvector.read"` } : {};
        return json({ error: error.code }, error.status, headers);
      }
      return json({ error: 'internal_error' }, 500);
    }
  },
} satisfies ExportedHandler<Env>;
