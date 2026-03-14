# ADR-100: RVF Integration & Crate Structure

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-14                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |

## Context

The Rust conversion must integrate with RuVector's existing workspace of 100+ crates and leverage the RVF (RuVector Format) for serialization, cognitive containers, and WASM deployment.

## Decision

### Workspace Layout

```
crates/
├── ruvector-deep-core/           # Core types, agent factory, graph
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                # create_deep_agent(), BASE_AGENT_PROMPT
│       ├── models.rs             # resolve_model(), ChatModel trait
│       ├── graph.rs              # Agent state machine (replaces LangGraph)
│       ├── config.rs             # DeepAgentConfig
│       └── messages.rs           # Message types (System, Human, AI, Tool)
│
├── ruvector-deep-backends/       # Backend protocol + all implementations
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                # Re-exports
│       ├── protocol.rs           # Backend, SandboxBackend traits
│       ├── state.rs              # StateBackend
│       ├── filesystem.rs         # FilesystemBackend
│       ├── local_shell.rs        # LocalShellBackend
│       ├── composite.rs          # CompositeBackend
│       ├── sandbox.rs            # BaseSandbox trait
│       ├── store.rs              # StoreBackend (persistent)
│       └── utils.rs              # format_content_with_line_numbers, etc.
│
├── ruvector-deep-middleware/      # Middleware trait + all implementations
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                # Middleware trait, MiddlewarePipeline
│       ├── todolist.rs           # TodoListMiddleware
│       ├── filesystem.rs         # FilesystemMiddleware (tool injection)
│       ├── subagents.rs          # SubAgentMiddleware
│       ├── summarization.rs      # SummarizationMiddleware
│       ├── memory.rs             # MemoryMiddleware
│       ├── skills.rs             # SkillsMiddleware
│       ├── patch_tool_calls.rs   # PatchToolCallsMiddleware
│       ├── prompt_caching.rs     # PromptCachingMiddleware
│       ├── hitl.rs               # HumanInTheLoopMiddleware
│       └── utils.rs              # append_to_system_message
│
├── ruvector-deep-tools/          # Tool trait + all tool implementations
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                # Tool trait, ToolRuntime, ToolResult
│       ├── ls.rs
│       ├── read_file.rs
│       ├── write_file.rs
│       ├── edit_file.rs
│       ├── glob.rs
│       ├── grep.rs
│       ├── execute.rs
│       ├── write_todos.rs
│       └── task.rs               # SubAgent task tool
│
├── ruvector-deep-subagents/      # SubAgent types and orchestration
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                # SubAgentSpec, CompiledSubAgent
│       ├── builder.rs            # compile_subagents()
│       └── prompts.rs            # TASK_TOOL_DESCRIPTION, TASK_SYSTEM_PROMPT
│
├── ruvector-deep-cli/            # Terminal UI application
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs               # Entry point
│       ├── app.rs                # TUI application
│       ├── agent.rs              # CLI agent creation
│       ├── config.rs             # Settings management
│       ├── sessions.rs           # Session persistence
│       ├── hooks.rs              # Execution hooks
│       ├── mcp.rs                # MCP client integration
│       ├── skills/               # Skill loading and slash commands
│       ├── widgets/              # ratatui widgets (15+ modules)
│       ├── integrations/         # Modal, Runloop, Daytona
│       └── ...                   # 20+ additional modules
│
├── ruvector-deep-acp/            # ACP server
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── server.rs             # ACP agent implementation
│       └── utils.rs              # Content block conversions
│
├── ruvector-deep-providers/      # LLM provider clients
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                # ChatModel trait
│       ├── anthropic.rs          # Anthropic Claude client
│       ├── openai.rs             # OpenAI client (Responses API support)
│       └── init_chat_model.rs    # "provider:model" resolution
│
└── ruvector-deep-wasm/           # WASM build targets
    ├── Cargo.toml
    └── src/
        ├── lib.rs                # WASM entry points
        ├── state_backend.rs      # StateBackend for browser
        └── agent.rs              # Browser-compatible agent
```

### Crate Dependency Graph

```
ruvector-deep-cli
├── ruvector-deep-core
│   ├── ruvector-deep-middleware
│   │   ├── ruvector-deep-tools
│   │   ├── ruvector-deep-subagents
│   │   └── ruvector-deep-backends
│   ├── ruvector-deep-providers
│   └── ruvector-deep-backends
├── ruvector-deep-acp
│   └── ruvector-deep-core
└── ruvector-deep-providers
```

### RVF Integration Points

#### 1. Agent Configuration as RVF Cognitive Containers

```rust
// Agent configs serialize to RVF for portable agent definitions
use ruvector_rvf::{RvfContainer, CognitiveLayer};

impl DeepAgentConfig {
    /// Serialize agent configuration to RVF cognitive container.
    /// Enables portable agent definitions across Rust/WASM/Python.
    pub fn to_rvf(&self) -> RvfContainer {
        RvfContainer::new()
            .with_layer(CognitiveLayer::AgentConfig {
                model: self.model.identifier(),
                system_prompt: self.system_prompt.clone(),
                tools: self.tool_names(),
                middleware: self.middleware_names(),
                subagents: self.subagent_specs(),
            })
    }

    /// Deserialize from RVF cognitive container.
    pub fn from_rvf(container: &RvfContainer) -> Result<Self, RvfError> { ... }
}
```

#### 2. State Serialization via RVF

```rust
// Agent state checkpoints use RVF format for persistence
impl StateBackend {
    /// Checkpoint state to RVF.
    pub fn checkpoint_to_rvf(&self) -> RvfContainer {
        let state = self.state.read().unwrap();
        RvfContainer::new()
            .with_layer(CognitiveLayer::AgentState {
                files: state.files.clone(),
                messages: state.messages.clone(),
                todos: state.todos.clone(),
            })
    }
}
```

#### 3. WASM Backend via ruvector-wasm

```rust
// Browser deployment uses StateBackend + WASM-compiled agent
#[cfg(target_arch = "wasm32")]
pub fn create_wasm_agent(config_rvf: &[u8]) -> WasmAgent {
    let config = DeepAgentConfig::from_rvf_bytes(config_rvf).unwrap();
    let agent = create_deep_agent(config);
    WasmAgent { inner: agent }
}
```

#### 4. Graph Operations via ruvector-graph

```rust
// Agent topology maps to RuVector graph primitives
use ruvector_graph::Graph;

impl AgentGraph {
    /// Export agent graph topology for visualization.
    pub fn to_ruvector_graph(&self) -> Graph {
        let mut g = Graph::new();
        // Nodes: agent, subagents, tools
        // Edges: tool calls, state transitions
        ...
    }
}
```

### Workspace Cargo.toml Addition

```toml
# Added to /home/user/RuVector/Cargo.toml [workspace.members]
members = [
    # ... existing crates ...
    "crates/ruvector-deep-core",
    "crates/ruvector-deep-backends",
    "crates/ruvector-deep-middleware",
    "crates/ruvector-deep-tools",
    "crates/ruvector-deep-subagents",
    "crates/ruvector-deep-cli",
    "crates/ruvector-deep-acp",
    "crates/ruvector-deep-providers",
    "crates/ruvector-deep-wasm",
]
```

### Existing RuVector Crate Integration

| Existing Crate | Usage in Deep-* |
|---|---|
| `ruvector-math` | Token counting, vector operations |
| `ruvector-graph` | Agent topology visualization |
| `ruvector-wasm` | WASM compilation targets |
| `ruvector-solver` | Optimization in agent scheduling |
| `ruvector-replication` | Multi-agent state sync |
| `ruvector-hnsw` (via graph) | Semantic search in memory/skills |

## Consequences

- 9 new crates added to workspace with clean dependency boundaries
- RVF serialization enables agent portability (Rust ↔ WASM ↔ Python)
- WASM compilation via `ruvector-deep-wasm` for browser deployment
- Existing RuVector crates provide math, graph, and search capabilities
- Clear separation: backends → tools → middleware → core → cli
