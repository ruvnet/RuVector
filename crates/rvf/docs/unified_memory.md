---
RVF Unified Memory Architecture
---

AgentDB is the unified memory abstraction layer that all agent systems use,
with RVF as the storage backend.

```
┌─────────────────────────────────────────────────────┐
│  Agent Applications                                  │
│  (Claude-Flow, Agentic-Flow, OSPipe, SONA)          │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  AgentDB Unified Memory API                          │
│  ├─ Episodes & Trajectories (RL)                    │
│  ├─ Semantic Search (HNSW 150x-12,500x faster)     │
│  ├─ Pattern Learning (rewards, critiques)           │
│  └─ Memory Types (Working, Episodic, Semantic)      │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  RVF Adapters (one per system)                       │
│  ├─ rvf-adapter-agentdb                             │
│  ├─ rvf-adapter-claude-flow                         │
│  ├─ rvf-adapter-agentic-flow                        │
│  ├─ rvf-adapter-sona                                │
│  └─ rvf-adapter-ospipe                              │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  RVF Storage (single .rvf file)                      │
│  VEC_SEG → INDEX_SEG → META_SEG → WITNESS_SEG      │
└─────────────────────────────────────────────────────┘
```


Layer Map
=========

LAYER 4: APPLICATION
  Claude-Flow CLI: npx @claude-flow/cli@latest memory store/search/retrieve
    Default backend: sql.js (WASM SQLite) + JSON flat files
    RVF backend: v3/@claude-flow/memory/src/rvf-backend.ts (implemented, not default)

LAYER 3: BRIDGES (Rust, in ruvllm)
  ClaudeFlowMemoryBridge (claude_flow_bridge.rs)
    Shells out to CLI via std::process::Command
    Has caching (5-min TTL), hive mind sync, routing suggestions
    Does NOT call RVF directly

  AgenticMemory (agentic_memory.rs)
    Uses ruvector-core HnswIndex directly (not RVF)
    4 memory types: Working, Episodic, Semantic, Procedural

LAYER 2B: N-API BRIDGE (cross-repo, live)
  claude-flow repo:
    rvf-backend.ts → import('@ruvector/rvf') → RvfDatabase class
      IMemoryBackend implementation against RVF
      In-memory entry cache for rich MemoryEntry fields
      Auto-compaction on deadSpaceRatio > 0.3
      RVCOW derive for agent sandboxing
      Dual-path: native HNSW if available, brute-force fallback

  ruvector repo:
    rvf-node/index.js → SDK adapter (flattenVectors, convertMetadata, FNV-1a hashing)
    rvf-node/src/lib.rs → N-API bridge (RvfDatabase class with Mutex<Option<RvfStore>>)
      16 segment types, derive(), embed_kernel(), embed_ebpf()

  Binary verified identical (SHA-256 match) between:
    ruvector:     crates/rvf/rvf-node/rvf-node.darwin-arm64.node
    claude-flow:  v3/node_modules/@ruvector/rvf-node/rvf-node.darwin-arm64.node

LAYER 2A: RVF ADAPTERS (Rust, all working and tested)
  rvf-adapter-agentdb
    RvfVectorStore: add_vectors → ingest_batch, search → query, delete
    RvfPatternStore: store_pattern, search_patterns (min_reward), search_failures, stats
    RvfIndexAdapter: build HNSW → Layer A/B/C, progressive search

  rvf-adapter-claude-flow
    RvfMemoryStore: ingest_memory (key/value/ns/embedding), search_memories, get/delete
    WitnessWriter: record_action/ingest/search/session, verify chain integrity
    SwarmCoordination: record_state, consensus_vote
    LearningPatternStore: store_pattern, get_top_patterns

  rvf-adapter-agentic-flow
    SwarmStore, Coordination + Learning

  rvf-adapter-sona
    Neural trajectories + replay

LAYER 2B-ALT: FAST JS BRIDGE (TypeScript, in npm/packages/ruvector)
  FastAgentDB (50-200x faster than CLI)
    Episode/Trajectory storage (in-memory + LRU)
    Vector search via @ruvector/core (not RVF)
  SONA, OnnxEmbedder, ParallelIntelligence

LAYER 1: RVF RUNTIME (Rust)
  RvfStore: create/open/close, ingest_batch/query/delete/compact
  Segments: VEC_SEG + INDEX_SEG + META_SEG + WITNESS_SEG
  Progressive HNSW (Layer A/B/C)

LAYER 0: STORAGE
  Single .rvf binary file on disk


Key Adapters
============

rvf-adapter-agentdb (crates/rvf/rvf-adapters/agentdb)
  VectorStore   vector_store.rs   Maps agentdb CRUD → RvfStore.ingest_batch/query/delete
  PatternStore  pattern_store.rs  Memory patterns with rewards, critiques, success rates
  IndexAdapter  index_adapter.rs  Maps HNSW operations to RVF's 3-layer progressive indexing

ClaudeFlowMemoryBridge (crates/ruvllm/src/context/claude_flow_bridge.rs)
  Executes npx @claude-flow/cli@latest memory commands
  Maps claude-flow patterns to AgentDB namespaces
  Syncs memory across agent swarms (hive mind)
  Caches search results (5-min TTL)

FastAgentDB (npm/packages/ruvector/src/core/agentdb-fast.ts)
  50-200x faster in-process operations vs CLI


Integration Status
==================

Per ADR-029, all systems converge on RVF:

  System            Target                      Status
  ────────────────  ──────────────────────────  ──────
  claude-flow       RVF with WITNESS_SEG        Done (rvf-backend.ts + rvf-node N-API)
  agentdb           RVF with RVText profile     Done (rvf-adapter-agentdb)
  agentic-flow      RVF streaming protocol      Done (rvf-adapter-agentic-flow)
  ospipe            RVF with META_SEG           Done (rvf-adapter-ospipe)
  sona              RVF with SKETCH_SEG         Done (rvf-adapter-sona)


Remaining Gaps
==============

Gap A — Claude-Flow CLI default backend (claude-flow repo)
  rvf-backend.ts exists but sql.js is still the default.
  Config/wiring change in claude-flow, not a missing feature.

Gap B — ClaudeFlowMemoryBridge (ruvector: crates/ruvllm)
  Shells out to CLI instead of calling RVF directly.
  Fix: replace Command("npx ...") with rvf-adapter-claude-flow calls.

Gap D — AgenticMemory (ruvector: crates/ruvllm)
  Uses ruvector-core HNSW, not RVF files.
  Fix: swap HnswIndex for rvf-adapter-agentdb VectorStore.

Gap E — No MCP server (ruvector: new crate)
  No way for Claude Code agents to call RVF adapter ops as MCP tools.
  Fix: build rvf-mcp-server composing both adapters.

Closed:
  Gap C — FastAgentDB/IntelligenceEngine N-API bridge
    Closed. rvf-backend.ts (claude-flow) → RvfDatabase (rvf-node) → RvfStore → .rvf
    Binary deployed and verified (identical SHA-256 across repos).


Recommended Path
================

Build an MCP server (Gap E) as a new crate in ruvector:

  Claude Code agents
      │ MCP tools (memory_store, memory_search, pattern_store,
      │            pattern_search, witness_log, coordination_state)
      ▼
  rvf-mcp-server
      │ composes both adapters:
      │  rvf-adapter-claude-flow (memory + witness + coordination)
      │  rvf-adapter-agentdb (patterns: rewards, critiques, success tracking)
      ▼
  .rvf files (persistent, with audit trails)

Why this is highest impact:
  1. Claude Code agents already use MCP tools for everything
  2. Bridges claude-flow and AgentDB simultaneously
  3. Persists to .rvf files with witness chains
  4. No modification needed to upstream claude-flow or ruvllm
  5. Sidesteps Gap A entirely — agents bypass the CLI
