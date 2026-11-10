/**
 * Behaviour pinned across the zod 3 -> 4 migration.
 */

import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { BotConfigSchema, ConfigManager } from '../../../src/core/BotConfig.js';
import { TenantSchema, MessageSchema, DomainEventSchema } from '../../../src/core/entities/types.js';
import { PluginManifestSchema, createPluginManifest } from '../../../src/plugins/PluginManager.js';
import { SkillEntity } from '../../../src/core/entities/Skill.js';
import { AIDefenceConfigSchema } from '../../../src/security/AIDefenceGuard.js';

const uuid = '0b6f3c4e-2a1d-4c8e-9f7a-1d2e3f4a5b6c';

describe('BotConfigSchema (zod 4)', () => {
  it('fills nested section defaults when sections are omitted', () => {
    const cfg = BotConfigSchema.parse({});
    expect(cfg.memory.dimensions).toBe(384);
    expect(cfg.llm.provider).toBe('anthropic');
    expect(cfg.api.port).toBe(3000);
    expect(cfg.api.rateLimit).toEqual({ max: 100, timeWindow: 60000 });
    expect(cfg.api.auth.type).toBe('bearer');
    expect(cfg.skills.enabled).toContain('search');
    expect(cfg.session.maxPerUser).toBe(10);
    expect(cfg.workers.poolSize).toBe(4);
  });

  it('reports validation issues as path: message strings', () => {
    const mgr = new ConfigManager();
    (mgr as unknown as { config: Record<string, unknown> }).config = {
      ...mgr.getConfig(),
      api: { ...mgr.get('api'), port: 70000 },
    };
    const res = mgr.validate();
    expect(res.valid).toBe(false);
    expect(res.errors.some((e) => e.startsWith('api.port:'))).toBe(true);
  });

  it('rejects a malformed llm.baseUrl', () => {
    expect(() => BotConfigSchema.parse({ llm: { baseUrl: 'not a url' } })).toThrow();
  });
});

describe('entity schemas (zod 4 records)', () => {
  it('parses string-keyed records', () => {
    const now = new Date();
    const tenant = TenantSchema.parse({ id: uuid, name: 'Acme', slug: 'acme', createdAt: now, updatedAt: now });
    expect(tenant.settings).toEqual({});
    const msg = MessageSchema.safeParse({
      id: uuid, sessionId: uuid, role: 'assistant', content: 'hi',
      toolCalls: [{ id: 't1', name: 'search', arguments: { q: 'x' } }],
      createdAt: now,
    });
    expect(msg.success).toBe(true);
  });

  it('rejects a non-uuid aggregate id', () => {
    const res = DomainEventSchema.safeParse({
      id: uuid, type: 'x', tenantId: uuid, aggregateId: 'nope', aggregateType: 'agent', payload: {}, timestamp: new Date(),
    });
    expect(res.success).toBe(false);
    if (!res.success) expect(res.error.issues[0].path).toEqual(['aggregateId']);
  });
});

describe('PluginManifestSchema (zod 4)', () => {
  it('applies nested defaults and string records', () => {
    const m = createPluginManifest({ name: 'p', version: '1.0.0', description: 'd' });
    expect(m.engines).toEqual({ ruvbot: '>=0.1.0', node: '>=18.0.0' });
    expect(m.hooks).toEqual({});
    expect(m.dependencies).toEqual({});
  });

  it('rejects non-string dependency versions', () => {
    const res = PluginManifestSchema.safeParse({ name: 'p', version: '1.0.0', description: 'd', dependencies: { a: 1 } });
    expect(res.success).toBe(false);
    if (!res.success) expect(res.error.issues.length).toBeGreaterThan(0);
  });
});

describe('SkillEntity validation (zod 4 issues)', () => {
  it('returns path-qualified errors', () => {
    const skill = new SkillEntity({
      definition: { name: 's', description: 'd', version: '1.0.0' } as never,
      executor: async () => 'ok',
      inputSchema: z.object({ query: z.string() }),
    });
    const res = skill.validateInput({ query: 5 });
    expect(res.valid).toBe(false);
    expect(res.errors?.[0]).toMatch(/^query: /);
    expect(skill.validateInput({ query: 'x' }).valid).toBe(true);
  });
});

describe('AIDefenceConfigSchema (zod 4)', () => {
  it('validates serverUrl with z.url()', () => {
    expect(AIDefenceConfigSchema.safeParse({ serverUrl: 'https://aidefence.example' }).success).toBe(true);
    expect(AIDefenceConfigSchema.safeParse({ serverUrl: 'nope' }).success).toBe(false);
  });
});
