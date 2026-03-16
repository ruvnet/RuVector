# ADR-108: rvAgent–ruvbot Integration Architecture

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | In Progress                                     |
| **Date**    | 2026-03-15                                      |
| **Updated** | 2026-03-15                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion), ADR-107 (Native Swarm) |
| **Related** | ruvbot ADR-001, ADR-011, ADR-007                |

## Context

Two parallel agent frameworks exist in the RuVector ecosystem:

1. **rvAgent** (`crates/rvAgent/`): Native Rust agent framework
   - High-performance tool execution with witness chains
   - Typed AgentState with Arc O(1) cloning
   - Filesystem, sandbox, and state backends
   - Middleware pipeline (9 layers)
   - Missing: SONA learning, HNSW retrieval, swarm coordination

2. **ruvbot** (`npm/packages/ruvbot/`): TypeScript enterprise assistant
   - Multi-platform integration (Slack, Discord, webhooks)
   - SwarmCoordinator with 12 background workers
   - Byzantine consensus (PBFT implementation)
   - Multi-tenancy with PostgreSQL + pgvector
   - SONA learning system (ADR-007)

Both frameworks share conceptual overlap but execute in different runtimes. This ADR defines an integration architecture to:
- Avoid duplication of swarm/learning logic
- Enable rvAgent as the execution backend for ruvbot
- Share WASM-compiled RuVector primitives

---

## Decision

### Integration Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Unified Agent Execution Layer                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────────┐        ┌─────────────────────────┐               │
│  │        ruvbot           │        │        rvAgent           │               │
│  │   (TypeScript/Node)     │        │     (Native Rust)        │               │
│  │                         │        │                          │               │
│  │  • Multi-platform I/O   │ ─────► │  • Tool execution        │               │
│  │  • User sessions        │  MCP   │  • Witness chains        │               │
│  │  • Swarm coordination   │ ◄───── │  • File operations       │               │
│  │  • Byzantine consensus  │        │  • Sandbox backend       │               │
│  │  • Background workers   │        │  • State management      │               │
│  └─────────────────────────┘        └─────────────────────────┘               │
│              │                                   │                             │
│              │              ┌────────────────────┘                             │
│              │              │                                                  │
│              ▼              ▼                                                  │
│  ┌──────────────────────────────────────────────────────────────┐             │
│  │              @ruvector/* WASM Modules (Shared)                │             │
│  │                                                                │             │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │             │
│  │   │  HNSW    │  │   SONA   │  │  LoRA    │  │  RVF     │     │             │
│  │   │  Index   │  │ Learning │  │ Adapter  │  │ Runtime  │     │             │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘     │             │
│  └──────────────────────────────────────────────────────────────┘             │
│                                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Integration Points

#### 1. MCP Bridge (rvAgent ↔ ruvbot)

rvAgent exposes an MCP server that ruvbot can consume:

```rust
// crates/rvAgent/rvagent-mcp/src/lib.rs
pub struct RvAgentMcpServer {
    runtime: ToolRuntime,
    witness_builder: WitnessBuilder,
    state: AgentState,
}

impl McpServer for RvAgentMcpServer {
    // Tool execution with witness chain
    async fn execute_tool(&self, name: &str, args: Value) -> Result<McpToolResult> {
        let result = self.runtime.execute(name, args).await?;
        self.witness_builder.add_entry(name, &args, &result);
        Ok(McpToolResult::from(result))
    }
}
```

ruvbot consumes via MCP client:

```typescript
// npm/packages/ruvbot/src/integration/rvagent.ts
import { McpClient } from '@modelcontextprotocol/sdk';

export class RvAgentBridge {
  private client: McpClient;

  async executeTool(name: string, args: Record<string, unknown>) {
    return this.client.callTool(name, args);
  }
}
```

#### 2. Shared Swarm Coordination

Both systems support the same topologies. Unify via shared config:

| Topology | ruvbot Implementation | rvAgent Implementation |
|----------|----------------------|------------------------|
| hierarchical | `SwarmCoordinator.ts` | `rvagent-swarm::HierarchicalSwarm` |
| mesh | `SwarmCoordinator.ts` | `rvagent-swarm::MeshSwarm` |
| hierarchical-mesh | `SwarmCoordinator.ts` | `rvagent-swarm::HybridSwarm` |
| adaptive | `SwarmCoordinator.ts` | `rvagent-swarm::AdaptiveSwarm` |

Shared configuration schema:

```yaml
# Unified swarm config (YAML/JSON)
swarm:
  topology: hierarchical
  maxAgents: 8
  strategy: specialized
  consensus: raft
  heartbeatInterval: 5000
  taskTimeout: 60000
```

#### 3. Background Worker Delegation

ruvbot's 12 workers can delegate compute-heavy tasks to rvAgent:

| Worker | ruvbot Role | rvAgent Delegation |
|--------|-------------|-------------------|
| ultralearn | Coordination | Pattern learning via SONA |
| optimize | Task dispatch | Profiling via Criterion |
| consolidate | Scheduling | EWC++ memory consolidation |
| audit | Security orchestration | Local file/command audit |
| map | Codebase indexing trigger | AST parsing, symbol extraction |
| deepdive | Analysis coordination | Code flow analysis |
| benchmark | Result aggregation | Criterion benchmark execution |
| testgaps | Coverage coordination | Test runner execution |

#### 4. SONA Learning Integration

Both systems use SONA for adaptive learning. Shared via WASM:

```rust
// crates/ruvllm/ruvllm-sona/src/lib.rs (already exists)
pub struct SonaEngine {
    trajectory_buffer: TrajectoryBuffer,
    reasoning_bank: ReasoningBank,
    ewc_plus_plus: EwcPlusPlus,
}

// WASM export for ruvbot
#[wasm_bindgen]
pub fn create_sona_engine() -> SonaEngine { ... }
```

rvAgent middleware:

```rust
// crates/rvAgent/rvagent-middleware/src/sona.rs (to implement)
pub struct SonaMiddleware {
    engine: Arc<SonaEngine>,
}

impl Middleware for SonaMiddleware {
    fn wrap_model_call(&self, request: ModelRequest, handler: ...) -> ModelResponse {
        let trajectory_id = self.engine.start_trajectory(&request);
        let response = handler(request);
        self.engine.record_step(trajectory_id, &response);
        response
    }
}
```

#### 5. HNSW Memory Retrieval

Shared HNSW index for semantic search:

```rust
// crates/rvAgent/rvagent-middleware/src/hnsw.rs (to implement)
use ruvector_hyperbolic_hnsw::{HnswIndex, HnswConfig};

pub struct HnswMiddleware {
    skill_index: HnswIndex<f32>,
    memory_index: HnswIndex<f32>,
}

impl HnswMiddleware {
    /// Retrieve top-k relevant skills instead of injecting all
    pub fn retrieve_skills(&self, query_embedding: &[f32], k: usize) -> Vec<SkillMetadata> {
        self.skill_index.search(query_embedding, k)
            .iter()
            .map(|id| self.skill_registry.get(*id))
            .collect()
    }
}
```

#### 6. Resource Budget Enforcement

Shared budget types (from rvf-types):

```rust
// crates/rvAgent/rvagent-core/src/budget.rs (to implement)
use rvf_types::agi_container::ResourceBudget;

pub struct BudgetEnforcer {
    budget: ResourceBudget,
    consumed: ResourceBudget,
}

impl BudgetEnforcer {
    pub fn check_tool_call(&mut self) -> Result<(), BudgetError> {
        if self.consumed.tool_calls >= self.budget.max_tool_calls {
            return Err(BudgetError::ToolCallLimitExceeded);
        }
        self.consumed.tool_calls += 1;
        Ok(())
    }
}
```

#### 7. Witness Chain Sharing

rvAgent's witness chains provide provenance for ruvbot's audit trail:

```typescript
// npm/packages/ruvbot/src/audit/witness.ts
import { WitnessChain } from '@ruvector/rvf-runtime';

export class AuditTrail {
  private chain: WitnessChain;

  async recordToolExecution(tool: string, args: unknown, result: unknown) {
    const entry = this.chain.addEntry({
      toolName: tool,
      argumentsHash: await sha3_256(JSON.stringify(args)),
      timestamp: Date.now(),
    });
    return entry.id;
  }
}
```

### Implementation Phases

| Phase | Timeline | Deliverables |
|-------|----------|--------------|
| 1. HNSW Middleware | Week 1 | `rvagent-middleware/src/hnsw.rs` |
| 2. SONA Middleware | Week 2 | `rvagent-middleware/src/sona.rs` |
| 3. Resource Budget | Week 2 | `rvagent-core/src/budget.rs` |
| 4. MCP Server | Week 3-4 | `rvagent-mcp` crate |
| 5. ruvbot Bridge | Week 4-5 | `ruvbot/src/integration/rvagent.ts` |
| 6. Swarm Unification | Week 5-6 | Shared swarm config schema |

---

## Consequences

### Positive

1. **Unified execution backend**: ruvbot leverages rvAgent's Rust performance
2. **Shared WASM modules**: No duplication of vector/learning code
3. **Cross-platform consistency**: Same algorithms in Node.js and native contexts
4. **Provenance**: rvAgent witness chains provide audit trail for ruvbot
5. **Resource governance**: Shared budget enforcement

### Negative

1. **MCP latency**: Cross-process calls add ~1-5ms overhead
2. **Deployment complexity**: Two processes to manage
3. **Version coupling**: Must keep rvAgent/ruvbot in sync

### Risks

1. **WASM performance variance**: Different platforms may have different WASM execution speeds
2. **MCP stability**: MCP protocol still evolving
3. **Migration effort**: Existing ruvbot deployments need rvAgent sidecar

---

## Implementation Status

**Last Updated:** 2026-03-15

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| rvAgent core | ✅ Implemented | `crates/rvAgent/rvagent-core/` | Full state management |
| rvAgent middleware | ✅ Complete | `crates/rvAgent/rvagent-middleware/` | 20 middleware modules |
| WitnessMiddleware | ✅ Implemented | `rvagent-middleware/src/witness.rs` | Chain tracking |
| **HNSW Middleware** | ✅ Implemented | `rvagent-middleware/src/hnsw.rs` | **NEW** - Vector search |
| **SONA Middleware** | ✅ Implemented | `rvagent-middleware/src/sona.rs` | **NEW** - Adaptive learning |
| **Resource Budget** | ✅ Implemented | `rvagent-core/src/budget.rs` | **NEW** - Budget enforcement |
| **AGI Container** | ✅ Implemented | `rvagent-core/src/agi_container.rs` | **NEW** - Resource governance |
| **Session Crypto** | ✅ Implemented | `rvagent-core/src/session_crypto.rs` | **NEW** - Encrypted sessions |
| **Unicode Security** | ✅ Implemented | `rvagent-middleware/src/unicode_security.rs` | **NEW** - Input sanitization |
| MCP Server | ❌ Missing | `rvagent-mcp` crate needed | Use π brain server as bridge |
| ruvbot SwarmCoordinator | ✅ Implemented | `ruvbot/src/swarm/` | 12 background workers |
| ruvbot ByzantineConsensus | ✅ Implemented | `ruvbot/src/swarm/` | PBFT implementation |
| ruvbot rvAgent Bridge | ❌ Missing | To implement | Blocked on rvagent-mcp |

### Implementation Progress

| Phase | Timeline | Status | Deliverables |
|-------|----------|--------|--------------|
| 1. HNSW Middleware | Week 1 | ✅ DONE | `rvagent-middleware/src/hnsw.rs` |
| 2. SONA Middleware | Week 2 | ✅ DONE | `rvagent-middleware/src/sona.rs` |
| 3. Resource Budget | Week 2 | ✅ DONE | `rvagent-core/src/budget.rs` |
| 4. MCP Server | Week 3-4 | ❌ TODO | `rvagent-mcp` crate |
| 5. ruvbot Bridge | Week 4-5 | ❌ TODO | `ruvbot/src/integration/rvagent.ts` |
| 6. Swarm Unification | Week 5-6 | ❌ TODO | Shared swarm config schema |

### Workaround: π Brain Server as MCP Bridge

Until `rvagent-mcp` is implemented, π.ruv.io serves as the MCP bridge:

```
RuVocal UI ──MCP──▶ π.ruv.io/v1/mcp ──REST──▶ π Brain API
                         │
                         └── 91 MCP tools available
```

This provides functional integration while native rvAgent MCP is developed.

---

## Related ADRs

- **ADR-093**: DeepAgents Rust Conversion Overview
- **ADR-103**: Review Amendments (A1-A9, B1-B7, C1-C13)
- **ADR-107**: rvAgent Native Swarm Orchestration with WASM
- **ruvbot ADR-001**: Architecture Overview
- **ruvbot ADR-007**: Learning System (SONA)
- **ruvbot ADR-011**: Swarm Coordination
