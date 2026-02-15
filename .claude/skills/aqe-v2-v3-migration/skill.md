---
name: aqe-v2-v3-migration
version: 1.0.0
description: Migrate Agentic QE projects from v2 to v3 with zero data loss
author: Agentic QE Team
tags:
  - migration
  - upgrade
  - v2-to-v3
  - agentic-qe
trigger:
  - migrate from v2 to v3
  - upgrade agentic-qe
  - aqe migration
  - v2 to v3 migration
trust_tier: 1
validation:
  schema_path: schemas/output.json

---

# AQE v2 to v3 Migration Skill

<default_to_action>
When migrating from v2 to v3:
1. ANALYZE current v2 installation
2. BACKUP all data before any changes
3. MIGRATE configuration, memory, and patterns
4. VALIDATE migration success
5. PROVIDE rollback instructions

**Never delete v2 data without explicit user confirmation.**
</default_to_action>

## Quick Reference

### Migration Command
```bash
# When v3 becomes main release, just update the package
npm install agentic-qe@latest

# Run migration
aqe migrate

# Or use this skill
/aqe-v2-v3-migration
```

### What Gets Migrated

| Component | v2 Location | v3 Location | Auto-Migrate |
|-----------|-------------|-------------|--------------|
| Memory DB | `.agentic-qe/memory.db` | `.aqe/agentdb/` | Yes |
| Config | `.agentic-qe/config.json` | `.aqe/config.json` | Yes |
| Patterns | `.agentic-qe/patterns/` | `.aqe/reasoning-bank/` | Yes |
| Cache | `.agentic-qe/cache/` | `.aqe/cache/` | Optional |
| Logs | `.agentic-qe/logs/` | `.aqe/logs/` | No (fresh start) |

---

## Migration Checklist

### Pre-Migration
- [ ] Verify v2 installation exists (`.agentic-qe/` directory)
- [ ] Check v2 version: `aqe --version` (should be 2.x.x)
- [ ] Backup current data: `npm run backup` (in v2 project)
- [ ] Note any custom configurations
- [ ] Document current test counts and coverage

### During Migration
- [ ] Update to v3: `npm install agentic-qe@latest`
- [ ] Run migration: `aqe migrate`
- [ ] Review migration report
- [ ] Verify data transferred correctly

### Post-Migration
- [ ] Run v3 tests: `aqe test`
- [ ] Check coverage: `aqe coverage`
- [ ] Verify patterns loaded: `aqe patterns list`
- [ ] Test MCP integration with Claude Code

---

## Architecture Changes (v2 → v3)

### From Monolithic to DDD

```
v2 Structure:                    v3 Structure:
├── src/mcp/tools/              ├── src/domains/
│   ├── test-*.ts (40+ tools)   │   ├── test-generation/
│   └── ...                     │   ├── test-execution/
├── src/core/agents/            │   ├── coverage-analysis/
│   ├── mixed agents            │   ├── quality-assessment/
│   └── ...                     │   ├── defect-intelligence/
└── src/core/memory/            │   ├── requirements-validation/
    └── scattered impls         │   ├── code-intelligence/
                                │   ├── security-compliance/
                                │   ├── contract-testing/
                                │   ├── visual-accessibility/
                                │   ├── chaos-resilience/
                                │   └── learning-optimization/
                                ├── src/kernel/
                                │   ├── event-bus.ts
                                │   └── coordinator.ts
                                └── src/mcp/
                                    └── domain-handlers.ts
```

### Key API Changes

| v2 API | v3 API | Notes |
|--------|--------|-------|
| `aqe init` | `aqe init` | Different binary |
| `aqe.generateTests()` | `testGeneration.generate()` | Domain-based |
| `aqe.analyzeGaps()` | `coverageAnalysis.findGaps()` | O(log n) now |
| `memory.store()` | `agentDB.store()` | HNSW-indexed |
| `patterns.learn()` | `reasoningBank.record()` | With verdicts |

---

## Configuration Migration

### v2 Config Format
```json
{
  "version": "2.8.2",
  "memory": {
    "path": ".agentic-qe/memory.db",
    "type": "sqlite"
  },
  "agents": {
    "enabled": ["test-generator", "coverage-analyzer"]
  }
}
```

### v3 Config Format
```json
{
  "version": "3.0.0",
  "kernel": {
    "eventBus": "in-memory",
    "coordinator": "queen"
  },
  "domains": {
    "test-generation": { "enabled": true },
    "test-execution": { "enabled": true },
    "coverage-analysis": {
      "enabled": true,
      "algorithm": "hnsw",
      "dimensions": 128
    }
  },
  "memory": {
    "backend": "agentdb",
    "path": ".aqe/agentdb/",
    "hnsw": {
      "M": 16,
      "efConstruction": 200
    }
  },
  "learning": {
    "reasoningBank": true,
    "sona": true
  }
}
```

---

## Memory Migration

### SQLite to AgentDB

```typescript
// v2: Direct SQLite access
import Database from 'better-sqlite3';
const db = new Database('.agentic-qe/memory.db');
const patterns = db.prepare('SELECT * FROM patterns').all();

// v3: AgentDB with HNSW
import { AgentDB } from 'agentic-qe';
const db = new AgentDB('.aqe/agentdb/');
await db.initialize({ dimensions: 128, M: 16 });

// Migration script transfers and indexes
for (const pattern of v2Patterns) {
  await db.store({
    key: pattern.id,
    value: pattern.data,
    embedding: await generateEmbedding(pattern.data),
    metadata: { migratedFrom: 'v2', originalId: pattern.id }
  });
}
```

---

## Breaking Changes

### Must Update

1. **Import Paths**
   ```typescript
   // v2
   import { AgenticQE } from 'agentic-qe';

   // v3 (when v3 becomes main release, package name is still 'agentic-qe')
   import { TestGenerationDomain } from 'agentic-qe/domains';
   ```

2. **CLI Commands**
   ```bash
   # v2
   aqe test --parallel

   # v3
   aqe test --workers=4 --topology=mesh
   ```

3. **MCP Server**
   ```bash
   # v2
   claude mcp add aqe -- npx aqe-mcp

   # v3 (same CLI name, enhanced capabilities)
   claude mcp add aqe -- npx aqe mcp
   ```

### Deprecated (Will Warn)

- `aqe.runTests()` → Use domain-specific methods
- Direct memory access → Use AgentDB API
- Flat agent list → Use domain coordinators

---

## Rollback Instructions

If migration fails or you need to revert:

```bash
# 1. v3 does NOT modify v2 data
# Your .agentic-qe/ folder is untouched

# 2. Downgrade to v2
npm install agentic-qe@2.x
rm -rf .aqe/

# 3. Continue using v2
aqe --version  # Should show 2.x.x
```

---

## Agent Coordination Examples

### Spawning Migration Agents

```typescript
// Use Task tool to spawn migration agents in parallel
Task({
  prompt: "Analyze v2 memory.db and extract all patterns",
  subagent_type: "researcher",
  description: "Analyze v2 patterns"
});

Task({
  prompt: "Convert v2 config to v3 format",
  subagent_type: "coder",
  description: "Convert config"
});

Task({
  prompt: "Validate migration results",
  subagent_type: "tester",
  description: "Validate migration"
});
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Cannot find .agentic-qe/" | No v2 installation | Run `aqe init` first |
| "Memory migration failed" | Corrupted SQLite | Use backup: `npm run backup:restore` |
| "HNSW index error" | Dimension mismatch | Set `dimensions: 128` in config |
| "Pattern not found" | Not migrated | Re-run: `aqe migrate --patterns` |

### Debug Mode

```bash
# Run migration with debug output
DEBUG=aqe:migrate aqe migrate

# Check migration logs
cat .aqe/logs/migration.log
```

---

## Support

- **Migration Issues**: Open issue with `[v2-v3-migration]` tag
- **Documentation**: [Migration Guide](../../docs/plans/V2-TO-V3-MIGRATION-PLAN.md)
- **Discord**: #v3-migration channel

---

## Version Compatibility Matrix

| v2 Version | v3 Version | Migration Support |
|------------|------------|-------------------|
| 2.8.x | 3.0.x | Full |
| 2.7.x | 3.0.x | Full |
| 2.6.x | 3.0.x | Partial (config only) |
| 2.5.x and below | 3.0.x | Manual migration |

---

*Skill Version: 1.0.0 | Last Updated: 2026-01-11*
