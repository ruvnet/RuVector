# ADR-104: rvAgent MCP Tools and Resources System

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-15                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |
| **Depends** | ADR-095, ADR-096, ADR-097, ADR-098, ADR-099, ADR-100, ADR-101, ADR-102, ADR-103 |
| **Crates**  | `rvagent-mcp` (new), `rvagent-core`, `rvagent-tools`, `rvagent-skills` |

---

## Context

### Current State of rvAgent

The rvAgent framework comprises 8 crates that collectively provide a full-featured agentic system:

| Crate | ADR | Responsibility |
|-------|-----|----------------|
| `ruvector-deep-core` | ADR-100 | Agent factory, state machine, config |
| `ruvector-deep-backends` | ADR-094 | Backend protocol traits, sandbox, filesystem |
| `ruvector-deep-middleware` | ADR-095 | Middleware pipeline (10-layer stack) |
| `ruvector-deep-tools` | ADR-096 | Tool system (filesystem, execute, grep, glob) |
| `ruvector-deep-subagents` | ADR-097 | SubAgent orchestration, task tool |
| `ruvector-deep-middleware` | ADR-098 | Memory, skills, summarization middleware |
| `ruvector-deep-cli` | ADR-099 | CLI and ACP server |
| `ruvector-deep-rvf` | ADR-100 | RVF integration, cognitive containers |

The middleware pipeline (ADR-095) processes requests through a 10-layer stack. The tool system (ADR-096) provides 9 built-in tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`, `write_todos`, `task`). SubAgent orchestration (ADR-097) supports ephemeral subagent spawning with state isolation.

### Gaps Identified

1. **No MCP (Model Context Protocol) exposure.** The tool system (ADR-096) exposes tools internally through the `Tool` trait and `ToolSet` abstraction, but there is no way for external MCP clients -- Claude Code, OpenAI Codex CLI, VS Code extensions, or Claude Desktop -- to discover, invoke, or observe these tools via the MCP standard. The SSE transport work in ADR-066 covers the brain server but not the agent framework itself.

2. **No resource observability.** Agent state (messages, todos, files, memory contents, skills metadata) is locked inside the `AgentState` struct (ADR-103 amendment A1). External clients cannot read agent state, query the skills catalog, or observe the topology graph without direct code access.

3. **No topology-aware routing for tool calls.** SubAgent orchestration (ADR-097) operates in a single-process model. When agents are deployed across nodes in hierarchical, mesh, or adaptive topologies, tool calls must be routed to the correct node based on capabilities, load, and topology rules. This routing layer does not exist.

4. **Skills system is not portable.** The skills middleware (ADR-098) loads `SKILL.md` files with YAML frontmatter, but the format is rvAgent-specific. There is no bridge to OpenAI Codex CLI task definitions or Anthropic Claude Code slash command manifests. Teams adopting rvAgent must maintain separate skill definitions for each platform.

5. **Testing gaps for MCP and distributed scenarios.** ADR-101 defines unit and integration test strategies but does not cover MCP protocol compliance, transport-level fuzz testing, topology-specific failure modes, or cross-platform skill format round-trips.

### Driving Requirements

- Claude Code users expect to add rvAgent via `claude mcp add` and immediately access all tools
- Codex CLI users expect skills to appear as task definitions with `codex --skill` flags
- Multi-node deployments require tool calls to reach the correct agent regardless of topology
- Operations teams need real-time observability into agent state without instrumenting application code

---

## Decision

Create a new `rvagent-mcp` crate that provides four subsystems: a tool registry, a resource system, a transport layer, and a topology-aware router. Enhance the existing skills system with a cross-platform bridge.

### 1. MCP Tool Registry

The `McpToolRegistry` wraps every `AnyTool` from `rvagent-tools` (ADR-096) into MCP-compatible tool definitions with JSON Schema parameter validation.

#### 1.1 Crate Structure

```
crates/rvagent-mcp/
  Cargo.toml
  src/
    lib.rs                  # Public API surface, re-exports
    registry.rs             # McpToolRegistry: register, discover, invoke
    resources.rs            # McpResourceProvider: state, skills, topology
    transport/
      mod.rs                # McpTransport trait definition
      stdio.rs              # StdioTransport (JSON-RPC over stdin/stdout)
      sse.rs                # SseTransport (HTTP + Server-Sent Events)
    router.rs               # TopologyRouter: Hierarchical, Mesh, Adaptive
    adapter.rs              # AnyToolAdapter: bridges rvagent-tools -> MCP
    schema.rs               # JSON Schema generation from Tool trait
    protocol.rs             # JSON-RPC 2.0 message types
    capabilities.rs         # Server/client capability negotiation
    skills_bridge.rs        # SkillBridge: rvAgent <-> Codex/Claude Code
    error.rs                # McpError type hierarchy
  tests/
    registry_tests.rs
    resource_tests.rs
    transport_tests.rs
    adapter_tests.rs
    router_tests.rs
    skills_bridge_tests.rs
    protocol_compliance.rs
```

#### 1.2 Registry Design

```rust
// crates/rvagent-mcp/src/registry.rs

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Central registry for MCP tools. Wraps rvagent-tools into MCP-compatible
/// definitions with JSON Schema parameters and tool annotations.
pub struct McpToolRegistry {
    tools: Arc<RwLock<HashMap<String, RegisteredMcpTool>>>,
    schema_validator: Arc<SchemaValidator>,
    change_subscribers: Arc<RwLock<Vec<tokio::sync::broadcast::Sender<ToolListChanged>>>>,
}

/// A tool registered in the MCP registry.
pub struct RegisteredMcpTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,  // JSON Schema draft 2020-12
    pub handler: Arc<dyn McpToolHandler>,
    pub annotations: ToolAnnotations,
}

/// Tool behavior annotations per MCP specification.
#[derive(Debug, Clone, Default)]
pub struct ToolAnnotations {
    /// Tool performs destructive/non-idempotent operations.
    pub destructive: bool,
    /// Tool reads state outside its own scope.
    pub reads_external: bool,
    /// Tool writes state outside its own scope.
    pub writes_external: bool,
    /// Expected latency category for client-side UX hints.
    pub latency_hint: LatencyHint,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum LatencyHint {
    #[default]
    Fast,       // <100ms (ls, glob, read_file)
    Medium,     // 100ms-1s (grep, edit_file)
    Slow,       // >1s (execute, task)
}

#[async_trait::async_trait]
pub trait McpToolHandler: Send + Sync {
    async fn call(
        &self,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult, McpError>;
}

pub struct ToolCallResult {
    pub content: Vec<ContentBlock>,
    pub is_error: bool,
}

pub enum ContentBlock {
    Text { text: String },
    Image { data: String, mime_type: String },
    Resource { uri: String, mime_type: Option<String>, text: Option<String> },
}

impl McpToolRegistry {
    pub fn new() -> Self { /* ... */ }

    /// Register a single tool with schema validation.
    pub async fn register(&self, tool: RegisteredMcpTool) -> Result<(), McpError> {
        // Validate input_schema is valid JSON Schema
        // Insert into tools map
        // Notify change subscribers
    }

    /// Unregister a tool by name.
    pub async fn unregister(&self, name: &str) -> Result<(), McpError> { /* ... */ }

    /// List all registered tools (for tools/list MCP method).
    pub async fn list_tools(&self) -> Vec<ToolDescription> { /* ... */ }

    /// Invoke a tool by name with arguments (for tools/call MCP method).
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult, McpError> {
        // Validate arguments against input_schema
        // Dispatch to handler
        // Return result or error
    }

    /// Subscribe to tool list changes (for notifications/tools/list_changed).
    pub fn subscribe_changes(&self) -> tokio::sync::broadcast::Receiver<ToolListChanged> {
        /* ... */
    }
}
```

#### 1.3 Built-in Tool Mapping

All 9 built-in tools from ADR-096 plus dynamically registered tools are exposed:

| rvAgent Tool | MCP Tool Name | Annotations | JSON Schema Parameters |
|-------------|---------------|-------------|----------------------|
| `ls` | `rvagent_ls` | reads_external | `{ path: string }` |
| `read_file` | `rvagent_read_file` | reads_external | `{ file_path: string, offset?: int, limit?: int }` |
| `write_file` | `rvagent_write_file` | destructive, writes_external | `{ file_path: string, content: string }` |
| `edit_file` | `rvagent_edit_file` | destructive, writes_external | `{ file_path: string, old_string: string, new_string: string, replace_all?: bool }` |
| `glob` | `rvagent_glob` | reads_external | `{ pattern: string, path?: string }` |
| `grep` | `rvagent_grep` | reads_external | `{ pattern: string, path?: string, include?: string }` |
| `execute` | `rvagent_execute` | destructive, reads_external, writes_external, Slow | `{ command: string, timeout?: int }` |
| `write_todos` | `rvagent_write_todos` | writes_external | `{ todos: TodoItem[] }` |
| `task` | `rvagent_task` | writes_external, Slow | `{ description: string, subagent_type?: string }` |

---

### 2. MCP Resource System

The `McpResourceProvider` exposes agent state, the skills catalog, and topology status as MCP resources with URI templates. This enables external clients to observe agent internals without direct code access.

#### 2.1 Resource URI Scheme

```
rvagent://state/{agent_id}                    Agent state snapshot
rvagent://state/{agent_id}/messages           Conversation history
rvagent://state/{agent_id}/todos              Active todo list
rvagent://state/{agent_id}/files              Tracked files manifest
rvagent://state/{agent_id}/memory             Memory contents

rvagent://skills                              Skills catalog (all skills)
rvagent://skills/{skill_name}                 Single skill definition
rvagent://skills/{skill_name}/versions        Version history

rvagent://topology                            Current topology graph
rvagent://topology/nodes                      All nodes with health
rvagent://topology/nodes/{node_id}            Single node detail
rvagent://topology/nodes/{node_id}/tools      Tools available on a node
rvagent://topology/leader                     Current leader (hierarchical only)
rvagent://topology/metrics                    Topology-level metrics

rvagent://config                              Agent configuration
rvagent://config/middleware                   Middleware stack
```

#### 2.2 Resource Provider Implementation

```rust
// crates/rvagent-mcp/src/resources.rs

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};

/// Static resource descriptor.
pub struct McpResource {
    pub uri: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}

/// URI template for dynamic resources.
pub struct ResourceTemplate {
    pub uri_template: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}

/// Content returned when reading a resource.
pub enum ResourceContent {
    Text { uri: String, mime_type: Option<String>, text: String },
    Blob { uri: String, mime_type: Option<String>, blob: Vec<u8> },
}

/// Notification emitted when a resource changes.
#[derive(Debug, Clone)]
pub struct ResourceChanged {
    pub uri: String,
}

/// Trait for dynamic resource providers.
#[async_trait::async_trait]
pub trait ResourceProvider: Send + Sync {
    /// List all resources this provider can serve.
    async fn list(&self) -> Result<Vec<McpResource>, McpError>;

    /// List URI templates for parameterized resources.
    async fn list_templates(&self) -> Result<Vec<ResourceTemplate>, McpError>;

    /// Read a resource by URI.
    async fn read(&self, uri: &str) -> Result<Vec<ResourceContent>, McpError>;

    /// Subscribe to changes for a specific URI.
    async fn subscribe(
        &self,
        uri: &str,
    ) -> Result<broadcast::Receiver<ResourceChanged>, McpError>;

    /// Unsubscribe from changes.
    async fn unsubscribe(&self, uri: &str) -> Result<(), McpError>;
}

/// Central resource manager that aggregates multiple providers.
pub struct ResourceManager {
    providers: Vec<Arc<dyn ResourceProvider>>,
    subscriptions: Arc<RwLock<HashMap<String, Vec<broadcast::Sender<ResourceChanged>>>>>,
}

impl ResourceManager {
    pub fn new() -> Self { /* ... */ }

    /// Register a resource provider.
    pub fn add_provider(&mut self, provider: Arc<dyn ResourceProvider>) { /* ... */ }

    /// List all resources across all providers (for resources/list).
    pub async fn list_resources(&self) -> Result<Vec<McpResource>, McpError> { /* ... */ }

    /// List all templates across all providers (for resources/templates/list).
    pub async fn list_templates(&self) -> Result<Vec<ResourceTemplate>, McpError> { /* ... */ }

    /// Read a resource by URI, routing to the correct provider (for resources/read).
    pub async fn read_resource(
        &self,
        uri: &str,
    ) -> Result<Vec<ResourceContent>, McpError> { /* ... */ }

    /// Subscribe to a resource URI (for resources/subscribe).
    pub async fn subscribe(
        &self,
        uri: &str,
    ) -> Result<broadcast::Receiver<ResourceChanged>, McpError> { /* ... */ }
}
```

#### 2.3 Built-in Resource Providers

Three providers ship with `rvagent-mcp`:

```rust
/// Serves AgentState fields as resources.
/// Reads from the typed AgentState (ADR-103 amendment A1).
pub struct AgentStateProvider {
    state: Arc<RwLock<AgentState>>,
    agent_id: String,
}

/// Serves the skills catalog and individual skill definitions.
/// Reads from SkillLoader (ADR-098).
pub struct SkillsCatalogProvider {
    loader: Arc<SkillLoader>,
}

/// Serves topology graph, node health, and metrics.
/// Reads from the active Topology implementation.
pub struct TopologyProvider {
    topology: Arc<dyn Topology>,
}
```

---

### 3. Transport Layer

Two transport implementations cover the primary deployment scenarios. Both implement the `McpTransport` trait.

#### 3.1 Transport Trait

```rust
// crates/rvagent-mcp/src/transport/mod.rs

use crate::protocol::{JsonRpcMessage, JsonRpcRequest, JsonRpcResponse};

/// Bidirectional transport for MCP JSON-RPC messages.
#[async_trait::async_trait]
pub trait McpTransport: Send + Sync {
    /// Start the transport (bind ports, open streams).
    async fn start(&mut self) -> Result<(), McpError>;

    /// Receive the next incoming message (request or notification).
    async fn recv(&mut self) -> Result<JsonRpcMessage, McpError>;

    /// Send an outgoing message (response or notification).
    async fn send(&self, message: JsonRpcMessage) -> Result<(), McpError>;

    /// Gracefully shut down.
    async fn shutdown(&self) -> Result<(), McpError>;
}
```

#### 3.2 Stdio Transport (for Claude Code)

```
+------------------+         stdin (JSON-RPC)         +------------------+
|                  | ----------------------------------> |                  |
|   Claude Code    |                                    |   rvagent-mcp    |
|   (MCP client)   | <---------------------------------- |   StdioTransport |
|                  |         stdout (JSON-RPC)         |                  |
+------------------+                                    +------------------+
```

```rust
// crates/rvagent-mcp/src/transport/stdio.rs

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Reads JSON-RPC messages from stdin, writes responses to stdout.
/// Designed for Claude Code's `claude mcp add` integration.
pub struct StdioTransport {
    reader: BufReader<tokio::io::Stdin>,
    writer: tokio::io::Stdout,
}

impl StdioTransport {
    pub fn new() -> Self {
        Self {
            reader: BufReader::new(tokio::io::stdin()),
            writer: tokio::io::stdout(),
        }
    }
}

#[async_trait::async_trait]
impl McpTransport for StdioTransport {
    async fn start(&mut self) -> Result<(), McpError> {
        // No-op for stdio; streams are immediately available.
        Ok(())
    }

    async fn recv(&mut self) -> Result<JsonRpcMessage, McpError> {
        let mut line = String::new();
        self.reader.read_line(&mut line).await
            .map_err(|e| McpError::Transport(format!("stdin read: {}", e)))?;
        serde_json::from_str(&line)
            .map_err(|e| McpError::Protocol(format!("JSON parse: {}", e)))
    }

    async fn send(&self, message: JsonRpcMessage) -> Result<(), McpError> {
        let mut out = serde_json::to_string(&message)
            .map_err(|e| McpError::Protocol(format!("JSON serialize: {}", e)))?;
        out.push('\n');
        let mut writer = tokio::io::stdout();
        writer.write_all(out.as_bytes()).await
            .map_err(|e| McpError::Transport(format!("stdout write: {}", e)))?;
        writer.flush().await
            .map_err(|e| McpError::Transport(format!("stdout flush: {}", e)))?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), McpError> { Ok(()) }
}
```

#### 3.3 SSE Transport (for Remote Clients)

Builds on the SSE transport pattern from ADR-066 but adapted for the agent framework rather than the brain server:

```
+------------------+     HTTP POST (JSON-RPC request)     +------------------+
|                  | -------------------------------------> |                  |
|  Remote Client   |                                       |   rvagent-mcp    |
|  (browser, CLI)  | <------------------------------------- |   SseTransport   |
|                  |     SSE stream (JSON-RPC responses)   |                  |
+------------------+                                       +------------------+
```

```rust
// crates/rvagent-mcp/src/transport/sse.rs

use axum::{Router, routing::{get, post}};
use tokio::sync::broadcast;

/// HTTP server with SSE for server-to-client push and POST for
/// client-to-server requests. Supports multiple concurrent clients.
pub struct SseTransport {
    bind_addr: std::net::SocketAddr,
    incoming: tokio::sync::mpsc::Receiver<JsonRpcMessage>,
    outgoing: broadcast::Sender<JsonRpcMessage>,
}

impl SseTransport {
    pub fn new(bind_addr: std::net::SocketAddr) -> Self { /* ... */ }

    /// Build the axum router with /sse (GET) and /message (POST) endpoints.
    fn build_router(&self) -> Router {
        Router::new()
            .route("/sse", get(Self::handle_sse))
            .route("/message", post(Self::handle_message))
    }

    /// SSE endpoint: streams JSON-RPC notifications and responses.
    async fn handle_sse(/* ... */) -> impl axum::response::IntoResponse { /* ... */ }

    /// POST endpoint: receives JSON-RPC requests from clients.
    async fn handle_message(/* ... */) -> impl axum::response::IntoResponse { /* ... */ }
}
```

---

### 4. Topology-Aware Routing

The `TopologyRouter` intercepts tool calls and routes them to the appropriate node based on the active topology (hierarchical, mesh, or adaptive). This bridges the MCP tool registry with the topology system.

#### 4.1 Architecture Diagram

```
                          +-------------------+
                          |   MCP Client      |
                          | (Claude Code /    |
                          |  Codex / VS Code) |
                          +--------+----------+
                                   |
                          JSON-RPC | (stdio or SSE)
                                   v
                          +--------+----------+
                          |   McpTransport    |
                          +--------+----------+
                                   |
                                   v
                          +--------+----------+
                          | McpToolRegistry   |
                          | (tools/call)      |
                          +--------+----------+
                                   |
                    +--------------+--------------+
                    |                             |
              local tool?                  remote tool?
                    |                             |
                    v                             v
           +-------+--------+          +---------+---------+
           | Direct Handler |          |  TopologyRouter   |
           | (AnyToolAdapter)|         |                   |
           +----------------+          +--+------+------+--+
                                          |      |      |
                              +-----------+  +---+  +---+-----------+
                              |              |              |
                              v              v              v
                       +------+----+  +------+----+  +-----+------+
                       |Hierarchical|  |   Mesh    |  | Adaptive   |
                       | (queen     |  | (gossip   |  | (dynamic   |
                       |  assigns)  |  |  routes)  |  |  switch)   |
                       +-----------+  +-----------+  +------------+
                              |              |              |
                              v              v              v
                       +------+----+  +------+----+  +-----+------+
                       |  Worker   |  |   Peer    |  |   Node     |
                       |  Node     |  |   Node    |  |   (active) |
                       +-----------+  +-----------+  +------------+
```

#### 4.2 Router Implementation

```rust
// crates/rvagent-mcp/src/router.rs

use crate::registry::{McpToolRegistry, McpToolHandler, ToolCallResult};

/// Determines whether a tool call should be handled locally or routed
/// to a remote node based on topology and capability matching.
pub struct TopologyRouter {
    topology: Arc<dyn Topology>,
    local_node_id: NodeId,
    local_registry: Arc<McpToolRegistry>,
    strategy: RoutingStrategy,
}

#[derive(Debug, Clone)]
pub enum RoutingStrategy {
    /// Route to the node with the matching capability and lowest load.
    /// Used in hierarchical topology where the queen assigns work.
    Hierarchical,

    /// Route to the nearest peer with the capability using consistent hashing.
    /// Used in mesh topology for even distribution.
    Mesh,

    /// Delegate to the active topology's routing rules.
    /// Switches between Hierarchical and Mesh strategies automatically.
    Adaptive,
}

impl TopologyRouter {
    pub fn new(
        topology: Arc<dyn Topology>,
        local_node_id: NodeId,
        local_registry: Arc<McpToolRegistry>,
        strategy: RoutingStrategy,
    ) -> Self { /* ... */ }

    /// Route a tool call to the correct node.
    /// Returns the result from whichever node handles the call.
    pub async fn route_tool_call(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult, McpError> {
        // 1. Check if the tool is available locally
        if self.local_registry.has_tool(tool_name).await {
            return self.local_registry.call_tool(tool_name, arguments).await;
        }

        // 2. Find a remote node with the capability
        let target = self.find_capable_node(tool_name).await?;

        // 3. Route via topology message
        let request = TopologyMessage {
            id: uuid::Uuid::new_v4(),
            from: self.local_node_id.clone(),
            to: MessageTarget::Node(target),
            payload: serde_json::json!({
                "type": "tool_call",
                "tool": tool_name,
                "arguments": arguments,
            }),
            timestamp: chrono::Utc::now(),
            ttl: 3,
        };
        self.topology.route(request).await?;

        // 4. Await response via topology recv
        let response = self.await_tool_response(request.id).await?;
        Ok(response)
    }

    /// Find a node capable of handling the given tool.
    async fn find_capable_node(
        &self,
        tool_name: &str,
    ) -> Result<NodeId, McpError> {
        let nodes = self.topology.discover().await
            .map_err(|e| McpError::Routing(format!("discovery failed: {}", e)))?;

        let capable: Vec<_> = nodes.iter()
            .filter(|n| n.status == HealthStatus::Healthy)
            .filter(|n| n.capabilities.contains(&tool_name.to_string()))
            .collect();

        match self.strategy {
            RoutingStrategy::Hierarchical => {
                // Prefer least-loaded worker
                capable.iter()
                    .min_by(|a, b| a.load.partial_cmp(&b.load).unwrap())
                    .map(|n| n.node_id.clone())
                    .ok_or(McpError::Routing(
                        format!("no node has capability '{}'", tool_name)
                    ))
            }
            RoutingStrategy::Mesh => {
                // Consistent hash based on tool name for even distribution
                let hash = consistent_hash(tool_name, capable.len());
                Ok(capable[hash].node_id.clone())
            }
            RoutingStrategy::Adaptive => {
                // Delegate to whichever sub-strategy the adaptive topology is using
                if self.topology.leader().await.is_some() {
                    // Currently hierarchical
                    capable.iter()
                        .min_by(|a, b| a.load.partial_cmp(&b.load).unwrap())
                        .map(|n| n.node_id.clone())
                        .ok_or(McpError::Routing(
                            format!("no node has capability '{}'", tool_name)
                        ))
                } else {
                    // Currently mesh
                    let hash = consistent_hash(tool_name, capable.len());
                    Ok(capable[hash].node_id.clone())
                }
            }
        }
    }
}
```

---

### 5. Skills Compatibility Bridge

The `SkillBridge` converts between rvAgent `SkillMetadata` (ADR-098), OpenAI Codex CLI task definitions, and Claude Code slash command manifests.

#### 5.1 Format Mapping

```
+----------------------------+     +----------------------------+
|   rvAgent Skill (YAML)     |     |   Codex Task Definition    |
|                            |     |                            |
|  name: "deploy-service"   | --> |  name: "deploy-service"    |
|  triggers:                 |     |  input: "{service} {env}"  |
|    - pattern: "/deploy"    |     |  instructions: "..."       |
|    - type: slash-command   |     |  tools: ["shell", "file"]  |
|  dependencies:             |     +----------------------------+
|    - name: "check-health"  |
|  codex_compatible: true    |     +----------------------------+
|  claude_code_compatible:   |     |   Claude Code Skill        |
|    true                    |     |                            |
+----------------------------+ --> |  skill: "deploy-service"   |
                                   |  args: "{service} {env}"   |
                                   |  description: "..."        |
                                   +----------------------------+
```

#### 5.2 Bridge Implementation

```rust
// crates/rvagent-mcp/src/skills_bridge.rs

use rvagent_skills::SkillMetadata;

/// Converts between rvAgent skill format and external skill formats.
pub struct SkillBridge;

/// Codex CLI task definition format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexTaskDefinition {
    pub name: String,
    pub input: String,
    pub instructions: String,
    pub tools: Vec<String>,
    pub model_hint: Option<String>,
}

/// Claude Code slash command manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCodeSkill {
    pub skill: String,
    pub args: Option<String>,
    pub description: String,
}

impl SkillBridge {
    /// Convert rvAgent skill metadata to Codex task definition.
    pub fn to_codex(meta: &SkillMetadata, body: &str) -> Option<CodexTaskDefinition> {
        if !meta.codex_compatible {
            return None;
        }

        let input = meta.triggers.iter()
            .find(|t| t.trigger_type == TriggerType::Regex)
            .map(|t| t.pattern.clone())
            .unwrap_or_default();

        Some(CodexTaskDefinition {
            name: meta.name.clone(),
            input,
            instructions: body.to_string(),
            tools: infer_tools_from_body(body),
            model_hint: match meta.model_routing.preferred_tier {
                1 => Some("fast".into()),
                2 => Some("balanced".into()),
                3 => Some("reasoning".into()),
                _ => None,
            },
        })
    }

    /// Convert rvAgent skill metadata to Claude Code slash command.
    pub fn to_claude_code(meta: &SkillMetadata) -> Option<ClaudeCodeSkill> {
        if !meta.claude_code_compatible {
            return None;
        }

        let args = meta.triggers.iter()
            .find(|t| t.trigger_type == TriggerType::Regex)
            .map(|t| t.pattern.clone());

        Some(ClaudeCodeSkill {
            skill: meta.name.clone(),
            args,
            description: meta.description.clone(),
        })
    }

    /// Parse a Codex task definition into rvAgent skill metadata.
    pub fn from_codex(task: &CodexTaskDefinition) -> SkillMetadata {
        SkillMetadata {
            name: task.name.clone(),
            version: semver::Version::new(1, 0, 0),
            description: task.instructions.lines().next()
                .unwrap_or(&task.name).to_string(),
            triggers: vec![Trigger {
                pattern: task.input.clone(),
                trigger_type: TriggerType::Regex,
            }],
            codex_compatible: true,
            claude_code_compatible: false,
            dependencies: vec![],
            timeout_seconds: 300,
            retry_policy: None,
            model_routing: ModelRoutingHint::default(),
        }
    }

    /// Parse a Claude Code skill manifest into rvAgent skill metadata.
    pub fn from_claude_code(skill: &ClaudeCodeSkill) -> SkillMetadata {
        SkillMetadata {
            name: skill.skill.clone(),
            version: semver::Version::new(1, 0, 0),
            description: skill.description.clone(),
            triggers: vec![Trigger {
                pattern: format!("/{}", skill.skill),
                trigger_type: TriggerType::SlashCommand,
            }],
            codex_compatible: false,
            claude_code_compatible: true,
            dependencies: vec![],
            timeout_seconds: 300,
            retry_policy: None,
            model_routing: ModelRoutingHint::default(),
        }
    }
}
```

---

### 6. MCP Server Lifecycle

The `McpServer` orchestrates the registry, resources, transport, and router into a single runnable server:

```rust
// crates/rvagent-mcp/src/lib.rs

pub struct McpServer {
    registry: Arc<McpToolRegistry>,
    resources: Arc<ResourceManager>,
    router: Option<TopologyRouter>,
    transport: Box<dyn McpTransport>,
    capabilities: ServerCapabilities,
}

impl McpServer {
    pub fn builder() -> McpServerBuilder { McpServerBuilder::new() }

    /// Run the server loop: recv -> dispatch -> send.
    pub async fn run(&mut self) -> Result<(), McpError> {
        self.transport.start().await?;

        loop {
            let message = self.transport.recv().await?;
            match message {
                JsonRpcMessage::Request(req) => {
                    let response = self.dispatch(req).await;
                    self.transport.send(JsonRpcMessage::Response(response)).await?;
                }
                JsonRpcMessage::Notification(notif) => {
                    self.handle_notification(notif).await;
                }
            }
        }
    }

    /// Dispatch a JSON-RPC request to the correct handler.
    async fn dispatch(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        match req.method.as_str() {
            "initialize" => self.handle_initialize(req).await,
            "tools/list" => self.handle_tools_list(req).await,
            "tools/call" => self.handle_tools_call(req).await,
            "resources/list" => self.handle_resources_list(req).await,
            "resources/read" => self.handle_resources_read(req).await,
            "resources/templates/list" => self.handle_templates_list(req).await,
            "resources/subscribe" => self.handle_resources_subscribe(req).await,
            "resources/unsubscribe" => self.handle_resources_unsubscribe(req).await,
            "ping" => self.handle_ping(req).await,
            _ => JsonRpcResponse::error(
                req.id,
                -32601,
                format!("Method not found: {}", req.method),
            ),
        }
    }
}

pub struct McpServerBuilder {
    transport: Option<Box<dyn McpTransport>>,
    tool_registry: Option<Arc<McpToolRegistry>>,
    resource_manager: Option<Arc<ResourceManager>>,
    topology_router: Option<TopologyRouter>,
}

impl McpServerBuilder {
    pub fn transport(mut self, transport: impl McpTransport + 'static) -> Self { /* ... */ }
    pub fn registry(mut self, registry: Arc<McpToolRegistry>) -> Self { /* ... */ }
    pub fn resources(mut self, manager: Arc<ResourceManager>) -> Self { /* ... */ }
    pub fn router(mut self, router: TopologyRouter) -> Self { /* ... */ }
    pub fn build(self) -> Result<McpServer, McpError> { /* ... */ }
}
```

#### 6.1 Claude Code Integration

```bash
# Register rvagent-mcp as a Claude Code MCP server
claude mcp add rvagent -- cargo run -p rvagent-mcp --bin rvagent-mcp-stdio

# Or via npx for the npm package
claude mcp add rvagent -- npx ruvector mcp serve --transport stdio
```

#### 6.2 SSE Deployment

```bash
# Start the SSE transport for remote clients
cargo run -p rvagent-mcp --bin rvagent-mcp-sse -- --bind 0.0.0.0:9200

# Or via npx
npx ruvector mcp serve --transport sse --port 9200
```

---

### 7. Protocol Messages

The full JSON-RPC 2.0 protocol layer (see ADR-104 sister document for complete type definitions):

```rust
// crates/rvagent-mcp/src/protocol.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsonRpcMessage {
    Request(JsonRpcRequest),
    Response(JsonRpcResponse),
    Notification(JsonRpcNotification),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,  // "2.0"
    pub id: RequestId,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: RequestId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    Number(i64),
    String(String),
}
```

---

### 8. Testing Strategy

#### 8.1 MCP Protocol Compliance Tests

```rust
#[cfg(test)]
mod protocol_compliance {
    /// initialize handshake: client sends initialize, server responds with
    /// capabilities and protocol_version, client sends initialized notification.
    #[tokio::test]
    async fn test_initialize_handshake() { /* ... */ }

    /// tools/list returns all registered tools with valid JSON Schema.
    #[tokio::test]
    async fn test_tools_list_schema_validity() { /* ... */ }

    /// tools/call validates arguments against schema before dispatch.
    #[tokio::test]
    async fn test_tools_call_schema_validation() { /* ... */ }

    /// tools/call returns isError=true for tool execution failures.
    #[tokio::test]
    async fn test_tools_call_error_propagation() { /* ... */ }

    /// resources/list returns all static resources and templates.
    #[tokio::test]
    async fn test_resources_list_completeness() { /* ... */ }

    /// resources/read with parameterized URI resolves template variables.
    #[tokio::test]
    async fn test_resources_read_template_resolution() { /* ... */ }

    /// resources/subscribe + notification/resources/updated flow.
    #[tokio::test]
    async fn test_resource_subscription_notifications() { /* ... */ }

    /// Unknown method returns JSON-RPC -32601 error.
    #[tokio::test]
    async fn test_unknown_method_error() { /* ... */ }

    /// Malformed JSON returns JSON-RPC -32700 parse error.
    #[tokio::test]
    async fn test_malformed_json_parse_error() { /* ... */ }
}
```

#### 8.2 AnyTool Adapter Tests

```rust
#[cfg(test)]
mod adapter_tests {
    /// All 9 built-in tools register successfully via AnyToolAdapter.
    #[tokio::test]
    async fn test_register_all_builtin_tools() { /* ... */ }

    /// Tool call through adapter produces correct ToolCallResult.
    #[tokio::test]
    async fn test_adapter_call_passthrough() { /* ... */ }

    /// Tool annotations (destructive, reads_external, etc.) are set correctly.
    #[tokio::test]
    async fn test_tool_annotations_mapping() { /* ... */ }

    /// Dynamic tool registration after server start triggers list_changed notification.
    #[tokio::test]
    async fn test_dynamic_tool_registration_notification() { /* ... */ }

    /// Concurrent calls to the same tool do not deadlock.
    #[tokio::test]
    async fn test_concurrent_same_tool_calls() { /* ... */ }
}
```

#### 8.3 Topology Router Tests

```rust
#[cfg(test)]
mod router_tests {
    /// Local tool call bypasses topology routing.
    #[tokio::test]
    async fn test_local_tool_call_direct() { /* ... */ }

    /// Remote tool call routes through hierarchical topology to least-loaded worker.
    #[tokio::test]
    async fn test_hierarchical_routing_least_loaded() { /* ... */ }

    /// Remote tool call routes through mesh topology via consistent hash.
    #[tokio::test]
    async fn test_mesh_routing_consistent_hash() { /* ... */ }

    /// Adaptive router delegates to correct sub-strategy.
    #[tokio::test]
    async fn test_adaptive_routing_delegation() { /* ... */ }

    /// Routing fails gracefully when no node has the requested capability.
    #[tokio::test]
    async fn test_routing_no_capable_node() { /* ... */ }

    /// TTL prevents infinite routing loops in mesh topology.
    #[tokio::test]
    async fn test_ttl_prevents_routing_loops() { /* ... */ }
}
```

#### 8.4 Skills Bridge Tests

```rust
#[cfg(test)]
mod skills_bridge_tests {
    /// rvAgent skill -> Codex task definition round-trip preserves semantics.
    #[test]
    fn test_codex_roundtrip() { /* ... */ }

    /// rvAgent skill -> Claude Code slash command round-trip preserves semantics.
    #[test]
    fn test_claude_code_roundtrip() { /* ... */ }

    /// Skill with codex_compatible=false returns None from to_codex.
    #[test]
    fn test_codex_incompatible_returns_none() { /* ... */ }

    /// Skill with claude_code_compatible=false returns None from to_claude_code.
    #[test]
    fn test_claude_code_incompatible_returns_none() { /* ... */ }

    /// Model routing tier maps correctly to Codex model_hint.
    #[test]
    fn test_model_routing_tier_mapping() { /* ... */ }

    /// Codex task definition -> rvAgent skill preserves name, description, triggers.
    #[test]
    fn test_from_codex_preserves_fields() { /* ... */ }
}
```

#### 8.5 Transport Integration Tests

```rust
#[cfg(test)]
mod transport_tests {
    /// Stdio transport: write request to stdin, read response from stdout.
    #[tokio::test]
    async fn test_stdio_request_response_cycle() { /* ... */ }

    /// SSE transport: POST request to /message, receive response via /sse stream.
    #[tokio::test]
    async fn test_sse_post_and_stream() { /* ... */ }

    /// SSE transport: multiple concurrent clients each receive their own responses.
    #[tokio::test]
    async fn test_sse_multiple_clients() { /* ... */ }

    /// Transport shutdown closes gracefully without dropping in-flight messages.
    #[tokio::test]
    async fn test_graceful_shutdown() { /* ... */ }
}
```

#### 8.6 Property-Based and Stress Tests

```rust
#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    proptest! {
        /// Any valid JSON-RPC request round-trips through serialize/deserialize.
        #[test]
        fn json_rpc_roundtrip(
            method in "[a-z/]{1,30}",
            id in prop_oneof![
                any::<i64>().prop_map(RequestId::Number),
                "[a-z0-9]{1,20}".prop_map(RequestId::String),
            ],
        ) {
            // Construct request, serialize, deserialize, assert equality
        }

        /// Tool call with valid arguments always returns a result (not a transport error).
        #[test]
        fn valid_tool_call_never_transport_errors(
            tool_index in 0..9usize,
        ) {
            // Select a built-in tool, generate valid arguments, call, assert no McpError::Transport
        }
    }
}

#[cfg(test)]
mod stress_tests {
    /// 1000 concurrent tool calls through the registry without deadlock.
    #[tokio::test]
    async fn stress_concurrent_registry_calls() { /* ... */ }

    /// 500 resource reads interleaved with state mutations.
    #[tokio::test]
    async fn stress_resource_reads_during_mutations() { /* ... */ }

    /// Rapid tool register/unregister cycles while serving calls.
    #[tokio::test]
    async fn stress_registry_churn() { /* ... */ }
}
```

---

## Architecture: End-to-End Flow

```
+------------------------------------------------------------------+
|                        MCP Client                                 |
|  (Claude Code / Codex CLI / VS Code / Claude Desktop)            |
+---+---------------------------+---------------------------+------+
    |                           |                           |
    | tools/list                | tools/call                | resources/read
    | tools/call                | "rvagent_execute"         | "rvagent://topology"
    v                           v                           v
+---+---------------------------+---------------------------+------+
|                      McpTransport (stdio or SSE)                  |
+---+---------------------------+---------------------------+------+
    |                           |                           |
    v                           v                           v
+---+----------+    +-----------+-----------+    +----------+------+
| McpServer    |    | McpToolRegistry       |    | ResourceManager |
| (dispatch)   |--->| (validate + invoke)   |    | (list + read)   |
+--------------+    +-----------+-----------+    +---------+-------+
                                |                          |
                    +-----------+-----------+    +----------+------+
                    |                       |    |                 |
               local?                  remote?   | AgentState     |
                    |                       |    | SkillsCatalog  |
                    v                       v    | Topology       |
            +-------+------+    +-----------+--+ +---------+------+
            | AnyTool      |    | Topology     |
            | Adapter      |    | Router       |
            | (ADR-096)    |    | (route msg)  |
            +--------------+    +------+-------+
                                       |
                          +------------+------------+
                          |            |            |
                          v            v            v
                    Hierarchical    Mesh       Adaptive
                    (queen/worker) (gossip)   (auto-switch)
```

---

## Consequences

### Positive

1. **Full MCP compliance.** The `rvagent-mcp` crate implements the MCP specification for tools, resources, and transports. Any MCP-compatible client (Claude Code, VS Code with MCP extension, Claude Desktop) can connect and use rvAgent tools without custom integration code.

2. **Zero-rewrite tool integration.** The `AnyToolAdapter` (section 1.3) bridges all existing `rvagent-tools` implementations into MCP without modifying them. New tools added via ADR-096's `Tool` trait are automatically available through MCP. No dual maintenance burden.

3. **Agent state observability.** Operations teams can monitor agent state, conversation history, todo lists, and topology health through standard MCP resource reads. No custom dashboards or instrumentation required -- any MCP client serves as an observation tool.

4. **Topology transparency.** Tool calls are routed to the correct node regardless of which transport the client connected through. The client does not need to know about the topology -- it calls `tools/call` and the router handles the rest.

5. **Cross-platform skills portability.** A single YAML skill definition works with rvAgent, OpenAI Codex CLI, and Claude Code slash commands. Teams author skills once and use them across all platforms. The `SkillBridge` handles format translation automatically.

6. **Incremental adoption.** The `McpServer::builder()` pattern allows teams to start with just tools (no resources, no routing) and add capabilities incrementally. The topology router is optional -- single-node deployments skip it entirely.

### Negative

1. **New crate dependency.** Adding `rvagent-mcp` increases the workspace by one crate and introduces dependencies on `axum` (for SSE), `jsonschema` (for validation), and `uuid` (for message IDs). Mitigated by making SSE transport feature-gated (`feature = "sse"`) so stdio-only deployments avoid the HTTP stack.

2. **Topology router complexity.** Routing tool calls across nodes introduces failure modes (network partitions, stale capability caches, split-brain in adaptive mode). Mitigated by TTL on routed messages, fallback to local execution, and the topology-level fault tolerance from Raft/gossip protocols.

3. **Skills bridge maintenance.** Both Codex and Claude Code may change their skill formats. The `SkillBridge` must track upstream changes. Mitigated by the compatibility flags (`codex_compatible`, `claude_code_compatible`) being opt-in -- skills that do not set these flags are unaffected by external format changes.

4. **Testing surface area.** MCP protocol compliance, transport reliability, topology routing, and skills bridge each require dedicated test suites. The testing strategy in section 8 adds approximately 40 new test functions. Mitigated by the test pyramid -- most tests are fast unit tests, with a small number of integration and stress tests.

### Migration Path

| Phase | Scope | Estimated Effort | Dependencies |
|-------|-------|-----------------|--------------|
| **Phase 1** | Protocol types, registry, stdio transport | 2 weeks | ADR-096 tool system |
| **Phase 2** | AnyToolAdapter, schema generation, all 9 built-in tools | 1 week | Phase 1 |
| **Phase 3** | Resource system, AgentState/Skills/Topology providers | 1.5 weeks | Phase 1, ADR-098, ADR-103 |
| **Phase 4** | SSE transport, multi-client support | 1 week | Phase 1, ADR-066 |
| **Phase 5** | TopologyRouter with Hierarchical/Mesh/Adaptive strategies | 2 weeks | Phase 2, topology module |
| **Phase 6** | SkillBridge: Codex + Claude Code format converters | 1 week | Phase 3, ADR-098 |
| **Phase 7** | Protocol compliance tests, stress tests, property tests | 1.5 weeks | All phases |
| **Phase 8** | Documentation, CLI integration (`npx ruvector mcp serve`) | 1 week | All phases |

### Test Coverage Targets

| Component | Target | Rationale |
|-----------|--------|-----------|
| Protocol layer (JSON-RPC types) | 95% | Serialization correctness is critical for interop |
| McpToolRegistry | 90% | Core MCP functionality, concurrent access |
| McpTransport (stdio) | 90% | Primary integration path for Claude Code |
| McpTransport (SSE) | 85% | I/O-heavy, harder to unit test |
| AnyToolAdapter | 95% | Bridge correctness ensures all tools work |
| ResourceManager + providers | 85% | Read-heavy, lower risk than writes |
| TopologyRouter | 90% | Routing correctness is critical for multi-node |
| SkillBridge | 90% | Format fidelity across platforms |

### Crate Dependency Graph (Updated)

```
rvagent-mcp (new)
  +-- rvagent-tools       (AnyTool adapter, ADR-096)
  +-- rvagent-core        (Topology trait, ADR-097/103)
  +-- rvagent-skills      (SkillLoader, ADR-098)
  +-- serde / serde_json  (serialization)
  +-- tokio               (async runtime)
  +-- async-trait          (trait async methods)
  +-- jsonschema          (tool argument validation)
  +-- uuid                (message IDs)
  +-- axum [optional]     (SSE transport, feature = "sse")
  +-- semver              (skill version resolution)

rvagent-skills (enhanced)
  +-- rvagent-mcp         (resource-based skill loading)
  +-- rvagent-core
  +-- serde_yaml          (YAML frontmatter parsing)
  +-- semver              (version constraint resolution)

rvagent-core (enhanced)
  +-- topology/           (new module, see ADR-104 sister document)
      +-- mod.rs
      +-- hierarchical.rs
      +-- mesh.rs
      +-- adaptive.rs
```

---

## Cross-References

| ADR | Relationship |
|-----|-------------|
| [ADR-095](ADR-095-deepagents-middleware-pipeline.md) | MCP tool calls pass through the middleware pipeline; typed AgentState (amendment A1 from ADR-103) is served as resources |
| [ADR-096](ADR-096-deepagents-tool-system.md) | AnyToolAdapter bridges the `Tool` trait and `ToolSet` into MCP tool definitions |
| [ADR-097](ADR-097-deepagents-subagent-orchestration.md) | SubAgent `task` tool exposed as MCP tool; topology routing extends subagent model to multi-node |
| [ADR-098](ADR-098-deepagents-memory-skills-summarization.md) | Skills middleware provides `SkillLoader` and `SkillMetadata` consumed by `SkillBridge` and `SkillsCatalogProvider` |
| [ADR-099](ADR-099-deepagents-cli-acp-server.md) | CLI gains `mcp serve` subcommand; ACP server can optionally expose MCP alongside ACP |
| [ADR-100](ADR-100-deepagents-rvf-integration-crate-structure.md) | `rvagent-mcp` added to workspace layout; crate dependency graph updated |
| [ADR-101](ADR-101-deepagents-testing-strategy.md) | Testing strategy extended with MCP protocol compliance, transport, and topology router test suites |
| [ADR-102](ADR-102-deepagents-implementation-roadmap.md) | Roadmap updated with 8-phase MCP integration plan |
| [ADR-103](ADR-103-deepagents-review-amendments.md) | Typed AgentState (A1), parallel tool execution (A2), and security hardening feed into MCP resource providers and tool handlers |

---

## References

- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [MCP Tool Annotations](https://spec.modelcontextprotocol.io/specification/2025-03-26/server/tools/)
- [MCP Resources](https://spec.modelcontextprotocol.io/specification/2025-03-26/server/resources/)
- [Raft Consensus Algorithm](https://raft.github.io/)
- [SWIM Gossip Protocol](https://www.cs.cornell.edu/projects/Quicksilver/public_pdfs/SWIM.pdf)
- ADR-066: SSE MCP Transport (brain server precedent)
- ADR-093: DeepAgents Rust Conversion Overview (series root)
