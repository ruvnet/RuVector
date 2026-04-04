# RuVector + Claude Code Integration Guide

## Quick Start

### 1. Connect to Shared Brain (30 seconds)

```bash
claude mcp add π --url https://mcp.pi.ruv.io
```

This gives you 40+ brain tools: `brain_search`, `brain_share`, `brain_status`, etc.

### 2. Install Full Local Stack (2 minutes)

```bash
# MCP brain (local, 91 tools including vector DB, hooks, SONA)
claude mcp add claude-flow -- npx -y @claude-flow/cli@latest

# Brain server (shared intelligence, 40 tools)
claude mcp add π --url https://mcp.pi.ruv.io
```

### 3. Add RuVector Agents

Copy to `.claude/agents/`:

```bash
cp examples/claude-agents/*.md .claude/agents/
```

---

## Integration Tiers

### Tier 1: MCP Tools (You Have This)

**What**: Claude Code calls RuVector tools via MCP protocol.

**Architecture**:
```
Claude Code ──stdio──▶ claude-flow MCP ──▶ local tools (91)
Claude Code ──SSE────▶ mcp.pi.ruv.io  ──▶ brain tools (40)
```

**Key tools**:
| Tool | Use Case |
|------|----------|
| `brain_search` | Find existing patterns before coding |
| `brain_share` | Contribute discoveries to collective |
| `brain_partition` | See knowledge topology |
| `brain_consciousness_compute` | IIT Phi on transition systems |
| `brain_reason` | Neural-symbolic inference |

**Optimization**: Search brain BEFORE implementing. Add to CLAUDE.md:
```markdown
Before implementing any feature, search the π brain:
brain_search("your feature description")
```

### Tier 2: WASM-Accelerated Local Tools

**What**: Run vector search, embedding, and graph ops locally via WASM — no network round-trip.

**Architecture**:
```
Claude Code ──stdio──▶ RuVector Hybrid MCP
                         ├──▶ WASM (hot path): search, embed, phi
                         └──▶ Network (cold path): write, train, sync
```

**WASM modules available** (31 crates):

| Module | Size | Capability |
|--------|------|------------|
| `micro-hnsw-wasm` | 5.5KB | Vector nearest-neighbor search |
| `ruvector-cnn-wasm` | ~50KB | CNN inference + embeddings |
| `ruvector-consciousness-wasm` | ~30KB | IIT 4.0 Phi computation |
| `ruvector-delta-wasm` | ~10KB | Delta/change tracking |
| `ruvector-dag-wasm` | ~15KB | DAG graph operations |
| `ruqu-wasm` | ~8KB | Vector quantization (4-32x compression) |
| `ruvector-attention-wasm` | ~20KB | Attention mechanisms |

**Build a hybrid MCP server**:

```rust
// crates/ruvector-claude-mcp/src/main.rs
use micro_hnsw_wasm::HnswIndex;
use ruvector_cnn_wasm::Embedder;

struct HybridMcpServer {
    // Local WASM-powered operations
    local_index: HnswIndex,      // cached vectors for search
    embedder: Embedder,           // local embedding generation
    
    // Remote brain for writes
    brain_url: String,            // https://pi.ruv.io
}

impl HybridMcpServer {
    async fn brain_search(&self, query: &str) -> Vec<Memory> {
        // 1. Embed query locally (WASM, <5ms)
        let embedding = self.embedder.embed(query);
        
        // 2. Search local cache first (WASM HNSW, <1ms)
        let local_results = self.local_index.search(&embedding, 10);
        
        // 3. If cache miss or stale, fall back to remote
        if local_results.is_empty() || self.is_stale() {
            return self.remote_search(query).await;
        }
        local_results
    }
}
```

### Tier 3: Hooks Integration

**What**: React to Claude Code events in real-time.

**Hook events mapped to RuVector actions**:

| Event | Trigger | RuVector Action |
|-------|---------|-----------------|
| `PreToolUse(Edit)` | Before file edit | Check brain for anti-patterns |
| `PreToolUse(Bash)` | Before command | Security scan via WASM |
| `PostToolUse(Bash)` | After command | Learn from errors, share to brain |
| `PostToolUse(Edit)` | After file edit | Track delta, update knowledge graph |
| `Stop` | Session ends | Share session discoveries to brain |
| `Notification` | Agent notification | Route to brain voice system |

**Setup** (add to `.claude/settings.json`):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{
          "type": "command",
          "command": "npx @ruvector/hooks pre-edit --file \"$CLAUDE_FILE_PATH\" --brain https://pi.ruv.io"
        }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "npx @ruvector/hooks post-bash --exit-code \"$CLAUDE_EXIT_CODE\" --learn"
        }]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "npx @ruvector/hooks session-end --share-discoveries"
        }]
      }
    ]
  }
}
```

### Tier 4: Custom Agents

**What**: Specialized AI personas that use RuVector tools natively.

**Agent: Brain-First Researcher**
```markdown
# .claude/agents/brain-researcher.md
---
name: brain-researcher
description: Research with collective brain intelligence before coding
model: claude-sonnet-4-6
tools: [Read, Grep, Glob, WebSearch, mcp__pi-brain__brain_search, mcp__pi-brain__brain_share, mcp__pi-brain__brain_partition]
---

You are a researcher with access to the π collective brain (6,600+ memories, 100+ contributors).

ALWAYS search the brain before starting research:
1. brain_search("topic") to find existing knowledge
2. brain_partition() to see knowledge clusters
3. Only then do your own research
4. brain_share() any new discoveries back to the collective
```

**Agent: Consciousness Analyzer**
```markdown
# .claude/agents/consciousness-analyst.md
---
name: consciousness-analyst
description: Analyze code structures using IIT 4.0 consciousness metrics
model: claude-opus-4-6
tools: [Read, Grep, mcp__pi-brain__brain_consciousness_compute, mcp__pi-brain__brain_consciousness_status]
---

You analyze software systems through the lens of Integrated Information Theory.
Use brain_consciousness_compute to measure Phi (integrated information) of:
- Module dependency graphs (as transition probability matrices)
- State machines (as TPMs)
- Data flow networks
Higher Phi suggests more tightly integrated, conscious-like systems.
```

### Tier 5: Prompt Cache Optimization

**What**: Structure CLAUDE.md for maximum Anthropic prompt cache hits.

**Principle**: Claude Code caches system prompt prefixes. Put stable content first:

```markdown
# CLAUDE.md (optimized for cache)

## [STABLE - cached across sessions]
### Project Rules
- Use TypeScript strict mode
- Follow TDD London School
- Keep files under 500 lines

### RuVector Tool Reference
brain_search(query) - semantic search across 6,600+ shared memories
brain_share(category, title, content) - contribute knowledge
brain_status() - system health check
[... all 40 tool descriptions ...]

## [DYNAMIC - changes per session]
### Current Sprint
- Working on SSE proxy decoupling (ADR-130)
- Brain has 6,628 memories, 101 contributors
```

**Impact**: ~60-80% cache hit rate on system prompt tokens, significant cost reduction.

### Tier 6: Agent SDK + Remote Control

**What**: Embed Claude Code inside RuVector orchestration.

**Use case**: RuVector swarm coordinator drives multiple Claude Code instances.

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

// Brain-enhanced autonomous coding
async function brainAssistedTask(task: string) {
  // Pre-fetch brain context
  const context = await fetch("https://pi.ruv.io/v1/memories/search?q=" + encodeURIComponent(task));
  const memories = await context.json();
  
  // Inject brain context into Claude Code session
  const enrichedPrompt = `
    Brain context (${memories.length} relevant memories):
    ${memories.map(m => `- [${m.category}] ${m.title}`).join('\n')}
    
    Task: ${task}
  `;
  
  for await (const event of query({
    prompt: enrichedPrompt,
    options: {
      allowedTools: ["Read", "Edit", "Bash", "Grep"],
      maxTurns: 20,
    }
  })) {
    if (event.type === "result") {
      // Share discoveries back to brain
      await fetch("https://pi.ruv.io/v1/memories", {
        method: "POST",
        headers: { "Authorization": "Bearer " + apiKey },
        body: JSON.stringify({
          category: "solution",
          title: `Auto-discovered: ${task.slice(0, 80)}`,
          content: event.result,
        })
      });
    }
  }
}
```

---

## Performance Comparison

| Operation | Current (MCP+Network) | Optimized (WASM+Local) | Speedup |
|-----------|----------------------|----------------------|---------|
| `brain_search` | ~200ms | <5ms | 40x |
| Embedding | ~100ms (API) | <10ms (WASM) | 10x |
| Phi compute | ~500ms (network) | <20ms (WASM) | 25x |
| Tool schema load | 40 tools at once | Deferred groups | 4x less tokens |
| Security check | N/A | <1ms (WASM hook) | New capability |

## Architecture: Full Integration Stack

```
┌────────────────────────────────────────────────────────────┐
│                    Claude Code CLI                          │
│                                                            │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────────┐  │
│  │ Agent    │  │ Hooks    │  │ MCP Servers              │  │
│  │ Loop     │──│ Engine   │──│                          │  │
│  │ (s$)     │  │          │  │  ┌─────────────────────┐ │  │
│  └──────────┘  │ Pre/Post │  │  │ RuVector Hybrid     │ │  │
│                │ Edit/Bash│  │  │ ┌─────────────────┐ │ │  │
│  ┌──────────┐  │ Stop     │  │  │ │ WASM Runtime    │ │ │  │
│  │ Custom   │  └──────────┘  │  │ │ • hnsw-search   │ │ │  │
│  │ Agents   │                │  │ │ • embed         │ │ │  │
│  │ brain-*  │                │  │ │ • phi-compute   │ │ │  │
│  └──────────┘                │  │ │ • quantize      │ │ │  │
│                              │  │ └────────┬────────┘ │ │  │
│  ┌──────────┐                │  │          │ cache    │ │  │
│  │ Skills   │                │  │     miss ▼          │ │  │
│  │ /brain   │                │  │  ┌──────────────┐  │ │  │
│  │ /phi     │                │  │  │ pi.ruv.io    │  │ │  │
│  └──────────┘                │  │  │ (REST API)   │  │ │  │
│                              │  │  └──────────────┘  │ │  │
│                              │  └─────────────────────┘ │  │
│                              │                          │  │
│                              │  ┌─────────────────────┐ │  │
│                              │  │ mcp.pi.ruv.io       │ │  │
│                              │  │ (SSE proxy)         │ │  │
│                              │  └─────────────────────┘ │  │
│                              └─────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

## Next Steps

1. Build `crates/ruvector-claude-mcp/` hybrid server (Tier 1)
2. Ship agent definitions in `examples/claude-agents/` (Tier 2)
3. Create `@ruvector/hooks` npm package (Tier 3)
4. Restructure CLAUDE.md for cache optimization (Tier 4)
5. Prototype Agent SDK embedding (Tier 5)
6. Package as Claude Code plugin (Tier 6)

## References

- [ADR-134: RuVector Deep Integration](../../adr/ADR-134-ruvector-claude-code-deep-integration.md)
- [ADR-133: Claude Code Source Analysis](../../adr/ADR-133-claude-code-source-analysis.md)
- [13-extension-points.md](./13-extension-points.md)
- [15-core-module-analysis.md](./15-core-module-analysis.md)
- [16-call-graphs.md](./16-call-graphs.md)
