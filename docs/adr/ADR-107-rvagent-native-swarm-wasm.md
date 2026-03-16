# ADR-107: rvAgent Native Swarm Orchestration with WASM Integration

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Proposed                                        |
| **Date**    | 2026-03-15                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |
| **Depends** | ADR-094, ADR-097, ADR-100, ADR-104, ADR-106     |
| **Crates**  | `rvagent-swarm` (new), `rvagent-wasm` (new), `rvagent-core`, `rvagent-backends` |

## Context

The rvAgent framework currently supports single-agent execution with stub or API-backed models (Anthropic, Gemini). However, enterprise AI orchestration requires multi-agent swarm coordination capabilities natively in Rust, with WASM bindings for JavaScript/TypeScript consumption in `agentic-flow`, `agentdb`, and `ruflo` npm packages.

### Current State

1. **ruflo/claude-flow** provides swarm orchestration in pure TypeScript:
   - 60+ agent types
   - Hierarchical, mesh, pipeline, and adaptive topologies
   - HNSW vector memory (150x-12,500x faster)
   - Self-learning hooks with neural patterns
   - MCP integration

2. **rvAgent** provides Rust agent primitives:
   - AgentGraph execution engine
   - Tool system (filesystem, execute, grep, glob)
   - Backend abstraction (Anthropic, Gemini)
   - Memory system (in-progress)

3. **RuVector** provides 90+ WASM-ready crates:
   - Vector operations (`ruvector-core`, `ruvector-wasm`)
   - HNSW indexing (`micro-hnsw-wasm`, `ruvector-hyperbolic-hnsw-wasm`)
   - Learning algorithms (`ruvector-learning-wasm`)
   - Consensus (`ruvector-raft`, `ruvector-delta-consensus`)
   - Attention mechanisms (`ruvector-attention-wasm`)
   - Graph neural networks (`ruvector-gnn-wasm`)
   - Task routing (`ruvector-router-wasm`)

### Gap

Native Rust swarm orchestration with WASM export is missing. TypeScript packages must re-implement coordination logic instead of calling optimized Rust code via WASM.

---

## Decision

### 1. New `rvagent-swarm` Crate

Create a native Rust swarm orchestration crate that integrates existing RuVector primitives.

#### 1.1 Crate Structure

```
crates/rvAgent/rvagent-swarm/
  Cargo.toml
  src/
    lib.rs              # Public API surface
    topology/
      mod.rs            # Topology trait
      hierarchical.rs   # Queen → Workers (anti-drift)
      mesh.rs           # Peer-to-peer (fully connected)
      pipeline.rs       # Sequential stages
      star.rs           # Central hub with spokes
      adaptive.rs       # Dynamic switching
    coordinator/
      mod.rs            # SwarmCoordinator trait
      queen.rs          # Queen coordination logic
      worker.rs         # Worker agent behavior
      consensus.rs      # Consensus protocol adapter
    routing/
      mod.rs            # Task routing
      semantic.rs       # HNSW-based semantic routing
      load_balance.rs   # Round-robin, weighted, adaptive
      complexity.rs     # 3-tier model routing (ADR-026)
    memory/
      mod.rs            # Shared memory interface
      hnsw.rs           # HNSW index wrapper
      vector_store.rs   # Vector storage
    learning/
      mod.rs            # Pattern learning
      sona.rs           # SONA integration
      ewc.rs            # EWC++ memory consolidation
    hooks/
      mod.rs            # Pre/post hooks
      pre_task.rs       # Pre-task intelligence
      post_task.rs      # Post-task learning
      route.rs          # Task-to-agent routing
  tests/
    topology_tests.rs
    coordinator_tests.rs
    routing_tests.rs
```

#### 1.2 Core Traits

```rust
// crates/rvAgent/rvagent-swarm/src/lib.rs

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Swarm topology enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SwarmTopology {
    Hierarchical,       // Queen → Workers
    Mesh,               // Peer-to-peer
    Pipeline,           // Sequential stages
    Star,               // Central hub
    HierarchicalMesh,   // V3 hybrid
    Adaptive,           // Dynamic switching
}

/// Agent role within a swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmRole {
    Queen { workers: Vec<AgentId> },
    Worker { queen: Option<AgentId> },
    Peer { neighbors: Vec<AgentId> },
    Stage { position: usize, next: Option<AgentId> },
}

/// Swarm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    pub topology: SwarmTopology,
    pub max_agents: usize,
    pub strategy: CoordinationStrategy,
    pub consensus: ConsensusAlgorithm,
    pub memory_backend: MemoryBackend,
}

/// Coordination strategy for task distribution
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    Specialized,    // Clear role boundaries
    Balanced,       // Even distribution
    Adaptive,       // Dynamic adjustment
}

/// Consensus algorithm for distributed state
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConsensusAlgorithm {
    Raft,           // Leader-based (f < n/2 failures)
    Byzantine,      // BFT (f < n/3 failures)
    Gossip,         // Eventual consistency
    CRDT,           // Conflict-free replicated data types
}

/// Memory backend selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryBackend {
    InMemory,       // Ephemeral
    HNSW,           // Vector-indexed
    Hybrid,         // HNSW + persistent storage
}

/// Swarm coordinator trait
#[async_trait]
pub trait SwarmCoordinator: Send + Sync {
    /// Initialize the swarm with configuration
    async fn init(&mut self, config: SwarmConfig) -> Result<SwarmId>;

    /// Spawn a new agent in the swarm
    async fn spawn_agent(&mut self, agent_type: &str, role: SwarmRole) -> Result<AgentId>;

    /// Route a task to the optimal agent
    async fn route_task(&self, task: &TaskDescription) -> Result<AgentId>;

    /// Execute a task with swarm coordination
    async fn execute(&mut self, task: &TaskDescription) -> Result<TaskResult>;

    /// Broadcast a message to all agents
    async fn broadcast(&self, message: SwarmMessage) -> Result<()>;

    /// Get swarm status
    async fn status(&self) -> Result<SwarmStatus>;

    /// Shutdown the swarm gracefully
    async fn shutdown(&mut self, graceful: bool) -> Result<()>;
}
```

#### 1.3 HNSW Integration

```rust
// crates/rvAgent/rvagent-swarm/src/memory/hnsw.rs

use micro_hnsw::HnswIndex;

/// HNSW-backed vector memory for semantic routing
pub struct HnswMemory {
    index: HnswIndex<f32>,
    dimension: usize,
    ef_construction: usize,
    m: usize,
}

impl HnswMemory {
    pub fn new(dimension: usize) -> Self {
        Self {
            index: HnswIndex::new(dimension, 16, 200),
            dimension,
            ef_construction: 200,
            m: 16,
        }
    }

    /// Store a pattern with vector embedding
    pub fn store(&mut self, key: &str, embedding: &[f32]) -> Result<()>;

    /// Semantic search using HNSW
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>>;
}
```

---

### 2. New `rvagent-wasm` Crate

WASM bindings for consuming rvAgent swarm functionality from JavaScript/TypeScript.

#### 2.1 Crate Structure

```
crates/rvAgent/rvagent-wasm/
  Cargo.toml
  src/
    lib.rs              # wasm-bindgen entry point
    backends.rs         # In-memory virtual filesystem
    bridge.rs           # JS ↔ Rust message bridge
    mcp.rs              # MCP server bindings (JSON-RPC 2.0)
    rvf.rs              # RVF container builder/parser
    tools.rs            # Tool execution system
  pkg/                  # Generated WASM package
```

#### 2.2 RVF Container Support (rvf.rs)

The `rvf.rs` module provides WASM bindings for building and parsing RVF (RuVector Format) cognitive containers. RVF containers package tools, prompts, skills, orchestrator configs, MCP tools, and Ruvix capabilities into a single verifiable binary format.

**AGI Segment Tags:**

| Tag | Hex | Description |
|-----|-----|-------------|
| `TOOL_REGISTRY` | `0x0105` | Tool definitions with parameters |
| `AGENT_PROMPTS` | `0x0106` | System prompts for agents |
| `SKILL_LIBRARY` | `0x0109` | Skill definitions with triggers |
| `ORCHESTRATOR` | `0x0108` | Multi-agent topology config |
| `MIDDLEWARE_CONFIG` | `0x010A` | Middleware settings |
| `MCP_TOOLS` | `0x010B` | MCP tool entries (new) |
| `CAPABILITY_SET` | `0x010C` | Ruvix capability definitions (new) |

**RVF Container Binary Format:**

```
┌─────────────────────────────────────────────────────┐
│ Magic: "RVF\x01" (4 bytes)                          │
├─────────────────────────────────────────────────────┤
│ Segment Count: u32 LE (4 bytes)                     │
├─────────────────────────────────────────────────────┤
│ Segment 1: type(1) + tag(2) + len(4) + data(len)    │
├─────────────────────────────────────────────────────┤
│ Segment 2: ...                                       │
├─────────────────────────────────────────────────────┤
│ ...                                                  │
├─────────────────────────────────────────────────────┤
│ SHA3-256 Checksum (32 bytes)                        │
└─────────────────────────────────────────────────────┘
```

**JavaScript Usage:**

```javascript
import { WasmRvfBuilder } from '@ruvector/rvagent/wasm';

// Build an RVF container
const builder = new WasmRvfBuilder();

// Add tools
builder.addTool(JSON.stringify({
  name: "web_search",
  description: "Search the web",
  parameters: { query: "string" },
  returns: "results"
}));

// Add MCP tools
builder.addMcpTools(JSON.stringify([{
  name: "read_file",
  description: "Read a file",
  input_schema: { path: { type: "string" } },
  group: "file"
}]));

// Add Ruvix capabilities
builder.addCapabilities(JSON.stringify([{
  name: "file_read",
  rights: ["read"],
  scope: "sandbox",
  delegation_depth: 2
}]));

// Add agent prompts
builder.addPrompt(JSON.stringify({
  name: "coder",
  system_prompt: "You are a coding assistant",
  version: "1.0.0"
}));

// Build container (returns Uint8Array)
const container = builder.build();

// Parse existing container
const parsed = WasmRvfBuilder.parse(container);
console.log(parsed.tools);        // Tool definitions
console.log(parsed.mcp_tools);    // MCP tool entries
console.log(parsed.capabilities); // Ruvix capabilities
console.log(parsed.prompts);      // Agent prompts
console.log(parsed.skills);       // Skill definitions
console.log(parsed.orchestrator); // Orchestrator config

// Validate container integrity
const isValid = WasmRvfBuilder.validate(container);
```

#### 2.3 MCP Server Support (mcp.rs)

The `mcp.rs` module implements an MCP (Model Context Protocol) server that runs entirely in WASM. It uses JSON-RPC 2.0 over a virtual transport.

**Available Tools:**

| Tool | Description |
|------|-------------|
| `read_file` | Read file from virtual filesystem |
| `write_file` | Write file to virtual filesystem |
| `edit_file` | Edit file with string replacement |
| `list_files` | List files in virtual filesystem |
| `write_todos` | Manage todo list |

**JavaScript Usage:**

```javascript
import { WasmMcpServer } from '@ruvector/rvagent/wasm';

const mcp = new WasmMcpServer();

// Initialize (returns server info)
const initResponse = await mcp.handle_message(JSON.stringify({
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: { protocolVersion: "2024-11-05" }
}));

// List available tools
const toolsResponse = await mcp.handle_message(JSON.stringify({
  jsonrpc: "2.0",
  id: 2,
  method: "tools/list"
}));

// Execute a tool
const execResponse = await mcp.handle_message(JSON.stringify({
  jsonrpc: "2.0",
  id: 3,
  method: "tools/call",
  params: {
    name: "write_file",
    arguments: { path: "test.txt", content: "Hello WASM" }
  }
}));
```

#### 2.4 Ruvix Capability Integration

Ruvix capabilities provide a fine-grained security model for AI agents, based on object-capability theory. Each capability defines:

| Field | Type | Description |
|-------|------|-------------|
| `name` | String | Capability identifier |
| `rights` | Vec<String> | Allowed operations (read, write, execute, etc.) |
| `scope` | String | Scope boundary (sandbox, local, network) |
| `delegation_depth` | u8 | Max delegation hops (0 = no delegation) |

**Capability-Based Tool Access:**

```rust
// In rvf.rs
pub struct CapabilityDef {
    pub name: String,
    pub rights: Vec<String>,
    pub scope: String,
    pub delegation_depth: u8,
}
```

**Integration with MCP Tools:**

```javascript
// RVF container with capabilities
const builder = new WasmRvfBuilder();

// Define capability
builder.addCapabilities(JSON.stringify([{
  name: "file_read",
  rights: ["read"],
  scope: "sandbox",
  delegation_depth: 0  // No delegation
}, {
  name: "file_write",
  rights: ["write", "create"],
  scope: "sandbox",
  delegation_depth: 1  // Can delegate once
}]));

// MCP tools reference capabilities
builder.addMcpTools(JSON.stringify([{
  name: "read_file",
  description: "Read file (requires file_read capability)",
  input_schema: { path: { type: "string" } },
  group: "file"
}]));
```

#### 2.2 WASM Bindings

```rust
// crates/rvAgent/rvagent-wasm/src/lib.rs

use wasm_bindgen::prelude::*;
use rvagent_swarm::{SwarmCoordinator, SwarmConfig, SwarmTopology};

/// WASM-exported swarm coordinator
#[wasm_bindgen]
pub struct WasmSwarmCoordinator {
    inner: Box<dyn SwarmCoordinator>,
}

#[wasm_bindgen]
impl WasmSwarmCoordinator {
    /// Create a new swarm coordinator
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<WasmSwarmCoordinator, JsValue> {
        let config: SwarmConfig = serde_wasm_bindgen::from_value(config)?;
        Ok(Self {
            inner: Box::new(DefaultSwarmCoordinator::new(config)),
        })
    }

    /// Spawn an agent
    #[wasm_bindgen]
    pub async fn spawn_agent(&mut self, agent_type: &str) -> Result<JsValue, JsValue> {
        let id = self.inner.spawn_agent(agent_type, SwarmRole::Worker { queen: None }).await?;
        Ok(serde_wasm_bindgen::to_value(&id)?)
    }

    /// Route a task
    #[wasm_bindgen]
    pub async fn route_task(&self, task: JsValue) -> Result<JsValue, JsValue> {
        let task: TaskDescription = serde_wasm_bindgen::from_value(task)?;
        let agent_id = self.inner.route_task(&task).await?;
        Ok(serde_wasm_bindgen::to_value(&agent_id)?)
    }

    /// Execute a task
    #[wasm_bindgen]
    pub async fn execute(&mut self, task: JsValue) -> Result<JsValue, JsValue> {
        let task: TaskDescription = serde_wasm_bindgen::from_value(task)?;
        let result = self.inner.execute(&task).await?;
        Ok(serde_wasm_bindgen::to_value(&result)?)
    }

    /// Get swarm status
    #[wasm_bindgen]
    pub async fn status(&self) -> Result<JsValue, JsValue> {
        let status = self.inner.status().await?;
        Ok(serde_wasm_bindgen::to_value(&status)?)
    }
}

/// WASM-exported HNSW memory
#[wasm_bindgen]
pub struct WasmHnswMemory {
    inner: HnswMemory,
}

#[wasm_bindgen]
impl WasmHnswMemory {
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        Self {
            inner: HnswMemory::new(dimension),
        }
    }

    #[wasm_bindgen]
    pub fn store(&mut self, key: &str, embedding: &[f32]) -> Result<(), JsValue> {
        self.inner.store(key, embedding).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn search(&self, query: &[f32], k: usize) -> Result<JsValue, JsValue> {
        let results = self.inner.search(query, k)?;
        Ok(serde_wasm_bindgen::to_value(&results)?)
    }
}
```

#### 2.3 npm Package Structure

The WASM output will be published as part of the `@ruvector/rvagent` npm package:

```json
{
  "name": "@ruvector/rvagent",
  "version": "0.1.0",
  "description": "Native Rust swarm orchestration for AI agents via WASM",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "files": [
    "dist/",
    "pkg/"
  ],
  "exports": {
    ".": {
      "import": "./dist/index.mjs",
      "require": "./dist/index.js",
      "types": "./dist/index.d.ts"
    },
    "./wasm": {
      "import": "./pkg/rvagent_wasm.js",
      "types": "./pkg/rvagent_wasm.d.ts"
    }
  },
  "dependencies": {
    "@ruvector/core": "^0.1.0",
    "@ruvector/hnsw": "^0.1.0"
  }
}
```

---

### 3. Integration with Existing npm Packages

#### 3.1 agentic-flow Integration

```typescript
// In @agentic-flow/core

import { WasmSwarmCoordinator, WasmHnswMemory } from '@ruvector/rvagent/wasm';

export class SwarmEngine {
  private coordinator: WasmSwarmCoordinator;
  private memory: WasmHnswMemory;

  constructor(config: SwarmConfig) {
    this.coordinator = new WasmSwarmCoordinator(config);
    this.memory = new WasmHnswMemory(384); // MiniLM embedding dimension
  }

  async routeTask(task: TaskDescription): Promise<AgentId> {
    // Use native Rust HNSW for 150x faster semantic routing
    return await this.coordinator.route_task(task);
  }
}
```

#### 3.2 agentdb Integration

```typescript
// In @agentdb/core

import { WasmHnswMemory } from '@ruvector/rvagent/wasm';

export class VectorStore {
  private hnsw: WasmHnswMemory;

  constructor(dimension: number = 384) {
    this.hnsw = new WasmHnswMemory(dimension);
  }

  async store(key: string, embedding: Float32Array): Promise<void> {
    this.hnsw.store(key, Array.from(embedding));
  }

  async search(query: Float32Array, k: number): Promise<SearchResult[]> {
    return this.hnsw.search(Array.from(query), k);
  }
}
```

#### 3.3 ruflo Integration

```typescript
// In ruflo CLI

import { WasmSwarmCoordinator } from '@ruvector/rvagent/wasm';

export async function initSwarm(config: SwarmConfig): Promise<SwarmHandle> {
  const coordinator = new WasmSwarmCoordinator({
    topology: config.topology,
    max_agents: config.maxAgents,
    strategy: 'specialized',
    consensus: 'raft',
    memory_backend: 'hnsw',
  });

  return {
    spawnAgent: (type) => coordinator.spawn_agent(type),
    routeTask: (task) => coordinator.route_task(task),
    execute: (task) => coordinator.execute(task),
    status: () => coordinator.status(),
  };
}
```

---

### 4. Topology Implementations

#### 4.1 Hierarchical (Queen-Led)

```rust
// crates/rvAgent/rvagent-swarm/src/topology/hierarchical.rs

pub struct HierarchicalTopology {
    queen_id: AgentId,
    workers: Vec<AgentId>,
    task_queue: VecDeque<TaskDescription>,
}

impl HierarchicalTopology {
    /// Anti-drift: Queen maintains authoritative state
    pub async fn coordinate(&mut self, task: TaskDescription) -> Result<TaskResult> {
        // 1. Queen analyzes task
        let subtasks = self.queen_decompose(&task).await?;

        // 2. Assign to workers based on specialization
        let assignments: Vec<_> = subtasks.iter()
            .zip(self.workers.iter().cycle())
            .map(|(subtask, worker)| (worker.clone(), subtask.clone()))
            .collect();

        // 3. Execute in parallel with coordination
        let results = futures::future::join_all(
            assignments.iter().map(|(worker, subtask)| {
                self.execute_on_worker(worker, subtask)
            })
        ).await;

        // 4. Queen synthesizes results
        self.queen_synthesize(&task, &results).await
    }
}
```

#### 4.2 Mesh (Peer-to-Peer)

```rust
// crates/rvAgent/rvagent-swarm/src/topology/mesh.rs

pub struct MeshTopology {
    peers: HashMap<AgentId, PeerInfo>,
    consensus: Box<dyn Consensus>,
}

impl MeshTopology {
    /// All peers participate in decision-making
    pub async fn coordinate(&mut self, task: TaskDescription) -> Result<TaskResult> {
        // 1. Broadcast task to all peers
        self.broadcast(SwarmMessage::NewTask(task.clone())).await?;

        // 2. Each peer contributes perspective
        let perspectives = self.gather_perspectives(&task).await?;

        // 3. Reach consensus on approach
        let consensus = self.consensus.propose(perspectives).await?;

        // 4. Execute agreed approach
        self.execute_consensus(&task, &consensus).await
    }
}
```

#### 4.3 Pipeline (Sequential)

```rust
// crates/rvAgent/rvagent-swarm/src/topology/pipeline.rs

pub struct PipelineTopology {
    stages: Vec<(AgentId, StageConfig)>,
}

impl PipelineTopology {
    /// Sequential processing through stages
    pub async fn coordinate(&mut self, task: TaskDescription) -> Result<TaskResult> {
        let mut current_input = task.input.clone();

        for (stage_idx, (agent_id, config)) in self.stages.iter().enumerate() {
            let stage_task = TaskDescription {
                input: current_input,
                stage: Some(stage_idx),
                ..task.clone()
            };

            let result = self.execute_stage(agent_id, &stage_task).await?;
            current_input = result.output;
        }

        Ok(TaskResult { output: current_input, ..Default::default() })
    }
}
```

---

### 5. 3-Tier Model Routing Integration

Integrate ADR-026 model routing for cost optimization:

```rust
// crates/rvAgent/rvagent-swarm/src/routing/complexity.rs

#[derive(Debug, Clone, Copy)]
pub enum ModelTier {
    Tier1Booster,   // WASM transform, <1ms, $0
    Tier2Haiku,     // Simple tasks, ~500ms, $0.0002
    Tier3Sonnet,    // Complex reasoning, ~2s, $0.003
    Tier3Opus,      // Architecture/security, ~5s, $0.015
}

pub struct ComplexityRouter {
    booster_intents: HashSet<&'static str>,
}

impl ComplexityRouter {
    pub fn route(&self, task: &TaskDescription) -> ModelTier {
        // Check if task can be handled by Tier 1 (no LLM needed)
        if let Some(intent) = self.detect_booster_intent(task) {
            return ModelTier::Tier1Booster;
        }

        // Analyze complexity for Tier 2 vs Tier 3
        let complexity = self.analyze_complexity(task);

        match complexity {
            c if c < 0.3 => ModelTier::Tier2Haiku,
            c if c < 0.7 => ModelTier::Tier3Sonnet,
            _ => ModelTier::Tier3Opus,
        }
    }

    fn detect_booster_intent(&self, task: &TaskDescription) -> Option<&'static str> {
        // Fast pattern matching for booster-eligible transforms
        for intent in &["var-to-const", "add-types", "remove-console", "add-logging"] {
            if task.description.contains(intent) {
                return Some(intent);
            }
        }
        None
    }
}
```

---

### 6. Performance Targets

| Metric | TypeScript (Current) | Rust/WASM (Target) | Improvement |
|--------|---------------------|-------------------|-------------|
| HNSW Search (1M vectors) | 15ms | 0.1ms | 150x |
| Task Routing | 50ms | 0.5ms | 100x |
| Swarm Init | 200ms | 20ms | 10x |
| Memory Footprint | 500MB | 50MB | 10x |
| WASM Bundle Size | N/A | 2MB | - |

---

### 7. Migration Path

#### Phase 1: Core Crates (Week 1-2)
- Create `rvagent-swarm` with topology implementations
- Create `rvagent-wasm` with basic bindings
- Unit tests for all topologies

#### Phase 2: HNSW Integration (Week 3)
- Integrate `micro-hnsw-wasm` for vector memory
- Implement semantic routing
- Benchmark against TypeScript baseline

#### Phase 3: npm Package (Week 4)
- Publish `@ruvector/rvagent` to npm
- Update `agentic-flow` to use WASM bindings
- Update `agentdb` to use WASM HNSW

#### Phase 4: ruflo Integration (Week 5)
- Update ruflo CLI to use native coordinator
- Deprecate pure-TypeScript swarm engine
- Performance validation

---

## Consequences

### Positive
1. **Performance**: 10-150x faster swarm operations via native Rust
2. **Memory efficiency**: 10x smaller footprint
3. **Single source of truth**: One Rust implementation, multiple language bindings
4. **Type safety**: Rust's type system catches errors at compile time
5. **Witness chains**: RVF audit trail for all swarm operations

### Negative
1. **Build complexity**: WASM compilation adds build step
2. **Debugging**: Stack traces cross WASM boundary
3. **Bundle size**: ~2MB WASM adds to npm package

### Neutral
1. **Learning curve**: Developers must understand WASM interop
2. **Async handling**: Promise/Future bridging required

---

## References

- ADR-097: DeepAgents Subagent Orchestration
- ADR-100: DeepAgents RVF Integration Crate Structure
- ADR-104: rvAgent MCP Skills Topology
- ADR-106: Ruvix Kernel RVF Integration
- ruflo npm package: https://www.npmjs.com/package/ruflo
- micro-hnsw: https://crates.io/crates/micro-hnsw
