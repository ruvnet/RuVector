# ADR-134: RuVector Deep Integration with Claude Code CLI

## Status

Proposed

## Date

2026-04-02

## Context

Source code analysis of Claude Code CLI (ADR-133) revealed 13 extension points and detailed internal architecture. RuVector currently integrates via MCP servers (`mcp-brain`, `mcp-brain-server/sse`, `mcp-gate`) but does not leverage the full integration surface. With 31 WASM crates, 4 MCP crates, and deep cognitive capabilities (IIT 4.0, SONA, knowledge graphs), there's significant opportunity to optimize for Claude Code's specific architecture.

### Current Integration Points

| RuVector Component | Integration | Depth |
|-------------------|-------------|-------|
| `mcp-brain-server` | SSE MCP at `mcp.pi.ruv.io` | Tools only |
| `mcp-brain` | Local stdio MCP | Tools only |
| `mcp-gate` | MCP gateway | Tools only |
| CLAUDE.md | Project instructions | Prompt only |
| Hooks (claude-flow) | Pre/post task | Lifecycle |

### Untapped Opportunities (from ADR-133 findings)

| Claude Code Feature | RuVector Opportunity |
|---------------------|---------------------|
| Agent SDK embedding | Embed RuVector as a library in custom agents |
| WASM tool execution | Ship WASM tools that run in-process (no MCP overhead) |
| Deferred tool loading | Lazy-load 40+ brain tools via `ToolSearch` |
| Hook-based routing | Route tool calls through WASM-accelerated pre-processing |
| Context compaction | Custom compaction strategy for vector-heavy contexts |
| Prompt caching | Optimize system prompts for cache hits |
| Plugin marketplace | Distribute RuVector as a Claude Code plugin |
| Remote control SSE | Drive Claude Code from RuVector orchestrator |
| Scheduled tasks | Autonomous brain training via Claude Code cron |

## Decision

Optimize RuVector crates and WASM modules for deep Claude Code integration across 6 tiers.

### Tier 1: WASM-Accelerated MCP Tools (High Impact, Low Effort)

**Problem**: Current MCP tools make HTTP loopback calls for every operation. Each `brain_search` requires network round-trip to `pi.ruv.io`.

**Solution**: Ship critical tools as WASM modules that run in Claude Code's process via a hybrid MCP server.

```
┌─────────────────────────────────────────────┐
│ Claude Code Process                          │
│                                              │
│  ┌──────────────┐    ┌───────────────────┐  │
│  │ Agent Loop    │───▶│ RuVector MCP      │  │
│  │ (s$ generator)│    │ (stdio transport)  │  │
│  └──────────────┘    │                    │  │
│                       │ ┌──────────────┐  │  │
│                       │ │ WASM Runtime  │  │  │
│                       │ │ • hnsw-search │  │  │
│                       │ │ • embed       │  │  │
│                       │ │ • phi-compute │  │  │
│                       │ └──────────────┘  │  │
│                       │        │           │  │
│                       │   Cache miss ──────┼──┼──▶ pi.ruv.io REST
│                       └───────────────────┘  │
└─────────────────────────────────────────────┘
```

**WASM crates to optimize**:

| Crate | Purpose | Claude Code Use |
|-------|---------|-----------------|
| `micro-hnsw-wasm` | Vector search (5.5KB) | Local semantic search in MCP server |
| `ruvector-cnn-wasm` | Embedding generation | Embed queries locally, no API call |
| `ruvector-consciousness-wasm` | IIT Phi computation | Consciousness metrics in-process |
| `ruvector-delta-wasm` | Delta tracking | Track knowledge changes locally |
| `ruvector-dag-wasm` | DAG operations | Graph queries without network |
| `ruqu-wasm` | Quantization | Compress vectors for context window |

**Implementation**:
1. Create `crates/ruvector-claude-mcp/` — hybrid MCP server with embedded WASM
2. WASM modules loaded at startup, handle hot-path operations locally
3. Cold-path operations (write, sync, train) forwarded to `pi.ruv.io`
4. Local HNSW index caches recent searches (LRU, 1000 vectors)

### Tier 2: Custom Agent Definitions (Medium Impact, Low Effort)

Ship specialized `.claude/agents/` definitions that leverage RuVector tools:

```markdown
# .claude/agents/ruvector-researcher.md
---
name: ruvector-researcher
description: Research with π brain collective intelligence
model: claude-sonnet-4-6
tools: [Read, Grep, Glob, mcp__pi-brain__brain_search, mcp__pi-brain__brain_share]
---

Before implementing anything, search the π brain for existing patterns:
1. Use brain_search to find related knowledge
2. Check brain_partition for knowledge clusters
3. Share new discoveries via brain_share
```

**Agents to ship**:
- `ruvector-researcher` — searches brain before coding
- `ruvector-reviewer` — reviews code against brain patterns
- `ruvector-consciousness` — runs IIT Phi analysis on code structures
- `ruvector-architect` — uses graph topology for architecture decisions

### Tier 3: Hook-Based Intelligence (High Impact, Medium Effort)

Leverage Claude Code's hook system for real-time intelligence:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "npx @ruvector/hooks pre-edit --file $CLAUDE_FILE_PATH"
      }]
    }],
    "PostToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "npx @ruvector/hooks post-bash --exit-code $CLAUDE_EXIT_CODE"
      }]
    }],
    "Stop": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "npx @ruvector/hooks session-end --share-to-brain"
      }]
    }]
  }
}
```

**Hook capabilities**:
- **PreToolUse (Edit/Write)**: Check brain for known anti-patterns before file edits
- **PostToolUse (Bash)**: Learn from command outcomes, share errors to brain
- **Stop**: Auto-share session discoveries to collective brain
- **PreToolUse blocker**: WASM-accelerated security scan before tool execution

### Tier 4: Prompt Cache Optimization (Medium Impact, Low Effort)

Claude Code uses Anthropic's prompt caching (`cache_control: { type: "ephemeral" }`). Optimize system prompts for maximum cache reuse:

1. **Static prefix**: CLAUDE.md instructions, RVF format docs, brain tool schemas — these rarely change and cache well
2. **Dynamic suffix**: Brain search results, recent memories — appended after the cached prefix
3. **Tool schema ordering**: List most-used tools first for higher cache hit rate

**Implementation**: Restructure CLAUDE.md to front-load stable content:
```
[CACHED] Project rules, architecture, conventions (rarely changes)
[CACHED] RuVector tool schemas (40 tools, stable across sessions)
[DYNAMIC] Recent brain context (changes per query)
```

### Tier 5: Agent SDK Embedding (High Impact, High Effort)

Use `@anthropic-ai/claude-agent-sdk` to embed Claude Code inside RuVector orchestration:

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

// RuVector orchestrator drives Claude Code as a cognitive worker
async function brainEnhancedQuery(task: string) {
  // 1. Search brain for context
  const context = await brainSearch(task);
  
  // 2. Run Claude Code with brain context injected
  for await (const event of query({
    prompt: `${context}\n\nTask: ${task}`,
    options: {
      allowedTools: ["Read", "Edit", "Bash", "mcp__pi-brain__*"],
      maxTurns: 10,
    }
  })) {
    // 3. Feed results back to brain
    if (event.type === "result") {
      await brainShare(event.result);
    }
  }
}
```

### Tier 6: Plugin Marketplace Distribution (Medium Impact, Medium Effort)

Package RuVector as a Claude Code plugin for one-click installation:

```json
{
  "name": "@ruvector/claude-plugin",
  "claudeCode": {
    "mcpServers": {
      "pi-brain": { "url": "https://mcp.pi.ruv.io" }
    },
    "agents": ["researcher", "reviewer", "consciousness"],
    "skills": ["brain-search", "brain-share", "phi-analyze"],
    "hooks": { ... }
  }
}
```

## Implementation Priority

| Tier | Effort | Impact | Timeline |
|------|--------|--------|----------|
| 1. WASM MCP | 2 weeks | High — 10x faster tool calls | Sprint 1 |
| 2. Agent defs | 2 days | Medium — better UX | Sprint 1 |
| 3. Hooks | 1 week | High — real-time intelligence | Sprint 1 |
| 4. Cache opt | 2 days | Medium — cost reduction | Sprint 1 |
| 5. Agent SDK | 3 weeks | High — full embedding | Sprint 2 |
| 6. Plugin | 1 week | Medium — distribution | Sprint 2 |

## WASM Optimization Targets

Based on Claude Code's tool dispatch pattern (`validateInput` → `call`), optimize WASM modules for:

| Metric | Current | Target | How |
|--------|---------|--------|-----|
| `brain_search` latency | ~200ms (network) | <5ms (local WASM HNSW) | `micro-hnsw-wasm` with local cache |
| Embedding generation | ~100ms (API) | <10ms (local WASM) | `ruvector-cnn-wasm` HashEmbedder |
| Tool schema load | 40 tools at startup | Deferred via ToolSearch | Lazy-load tool groups |
| Context usage | ~2000 tokens/tool schema | ~500 tokens (compressed) | Merge related tool schemas |
| Permission checks | Per-tool | Batch via PreToolUse hook | WASM pre-filter |

## Graph-Informed Architecture

From the dependency graph analysis (doc 12, 16), Claude Code's tool dispatch follows:

```
Agent Loop (s$)
  └─▶ Tool Dispatch
       ├─▶ Built-in tools (Read, Edit, Bash, etc.)
       ├─▶ MCP tools (mcp__server__tool namespace)
       │    └─▶ stdio/SSE/WS transport
       └─▶ Agent tool (spawns sub-agents)
```

**Optimization insight**: MCP tools go through a transport layer that adds ~50ms overhead per call. By embedding WASM in the MCP server process, we eliminate the transport hop for hot-path operations while keeping cold-path operations on the network.

**Graph topology insight**: The knowledge graph (16M edges) should inform which tools are co-invoked. From brain usage patterns:
- `brain_search` → `brain_share` (90% co-occurrence) — bundle in one WASM module
- `brain_status` → standalone — keep lightweight
- `brain_partition` → heavy computation — always remote

## Consequences

### Positive

- 10-40x latency reduction for hot-path brain operations via local WASM
- Richer integration surface (hooks, agents, skills, not just tools)
- Cost reduction through prompt cache optimization
- Plugin distribution enables one-click adoption
- Graph-informed architecture avoids optimizing cold paths

### Negative

- WASM modules need synchronization with remote brain state
- Local HNSW cache can become stale (mitigated by TTL + invalidation)
- Agent SDK embedding increases coupling with Claude Code's release cycle
- Plugin marketplace requirements may evolve

### Risks

| Risk | Mitigation |
|------|------------|
| WASM module size bloat | Use `micro-hnsw-wasm` (5.5KB), not full crate |
| Cache coherence | TTL-based invalidation + version vector |
| Claude Code breaking changes | Pin to stable Agent SDK version |
| MCP protocol evolution | Abstract transport behind trait |

## References

- [ADR-133: Claude Code Source Analysis](./ADR-133-claude-code-source-analysis.md)
- [ADR-130: SSE Decoupling](./ADR-130-mcp-sse-decoupling-midstream-queue.md)
- [ADR-066: SSE MCP Transport](./ADR-066-sse-mcp-transport.md)
- Research: `/docs/research/claude-code-rvsource/`
- Claude Code extension docs: `13-extension-points.md`
- Core module analysis: `15-core-module-analysis.md`
