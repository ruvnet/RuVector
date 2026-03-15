# ADR-104: rvAgent MCP Tools/Resources, Enhanced Skills, and Topology-Aware Deployment

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-15                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |
| **Depends** | ADR-095, ADR-096, ADR-098, ADR-100              |
| **Crates**  | `rvagent-mcp` (new), `rvagent-core`, `rvagent-tools`, `rvagent-skills` |

## Context

The rvAgent framework currently comprises 8 crates covering backend protocols, middleware pipelines, tool systems, sub-agent orchestration, memory, skills, CLI/ACP server, and RVF integration. Three gaps remain before the framework can operate as a fully autonomous, topology-aware multi-agent system:

1. **No native MCP (Model Context Protocol) support.** The existing tool system (ADR-096) handles filesystem/execute/grep/glob tools, but cannot expose or consume tools via the MCP standard. Agents cannot discover remote tools, serve their capabilities to external MCP clients, or negotiate capabilities with MCP-compliant hosts. The SSE transport work in ADR-066 covers the brain server but not the agent framework itself.

2. **Skills system is single-format.** The current skills middleware (ADR-098) loads skills from the filesystem but uses a proprietary format. It cannot interoperate with OpenAI Codex task definitions or Anthropic Claude Code skill manifests. Skill composition (one skill invoking another) and versioned dependency resolution are unsupported.

3. **No topology awareness.** Sub-agent orchestration (ADR-097) assumes a single-machine, single-process model. There is no support for hierarchical (queen/worker), mesh (peer-to-peer), or adaptive (dynamic switching) deployment topologies. Message routing, node discovery, consensus, and fault tolerance are absent.

4. **Testing gaps for distributed scenarios.** ADR-101 defines unit and integration test strategies but does not cover topology-specific failure modes, chaos testing, or cross-topology property-based invariants.

---

## Decision

### 1. MCP Integration: New `rvagent-mcp` Crate

#### 1.1 Crate Structure

```
crates/rvagent-mcp/
  src/
    lib.rs              # Public API surface
    registry.rs         # Tool registry and discovery
    resources.rs        # Resource system (templates, static, dynamic)
    transport/
      mod.rs            # Transport trait
      stdio.rs          # Stdio JSON-RPC transport
      sse.rs            # Server-Sent Events transport
      websocket.rs      # WebSocket transport
    protocol.rs         # JSON-RPC 2.0 message types
    capabilities.rs     # Server/client capability negotiation
    adapter.rs          # AnyTool adapter bridging rvagent-tools
    schema.rs           # Tool schema validation (JSON Schema)
    uri.rs              # Resource URI parsing (mcp://resources/*)
  tests/
    registry_tests.rs
    transport_tests.rs
    resource_tests.rs
    adapter_tests.rs
```

#### 1.2 JSON-RPC 2.0 Protocol Layer

```rust
// crates/rvagent-mcp/src/protocol.rs

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,  // Always "2.0"
    pub id: RequestId,
    pub method: String,
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
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    Number(i64),
    String(String),
}
```

#### 1.3 Tool Registry

```rust
// crates/rvagent-mcp/src/registry.rs

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Central registry for MCP tools. Supports dynamic registration,
/// discovery, and schema validation.
pub struct ToolRegistry {
    tools: Arc<RwLock<HashMap<String, RegisteredTool>>>,
    validators: Arc<SchemaValidator>,
}

pub struct RegisteredTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub handler: Arc<dyn McpToolHandler>,
    pub annotations: ToolAnnotations,
}

/// Annotations per MCP spec: hints about tool behavior.
pub struct ToolAnnotations {
    /// Whether the tool has side effects (non-idempotent)
    pub destructive: bool,
    /// Whether the tool reads external state
    pub reads_external: bool,
    /// Whether the tool writes external state
    pub writes_external: bool,
    /// Estimated latency category
    pub latency_hint: LatencyHint,
}

#[derive(Debug, Clone, Copy)]
pub enum LatencyHint {
    Fast,       // <100ms
    Medium,     // 100ms-1s
    Slow,       // >1s
}

#[async_trait::async_trait]
pub trait McpToolHandler: Send + Sync {
    async fn call(
        &self,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult, McpError>;
}

impl ToolRegistry {
    pub fn new() -> Self { /* ... */ }

    pub async fn register(&self, tool: RegisteredTool) -> Result<(), McpError> { /* ... */ }

    pub async fn unregister(&self, name: &str) -> Result<(), McpError> { /* ... */ }

    pub async fn list_tools(&self) -> Vec<ToolDescription> { /* ... */ }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult, McpError> { /* ... */ }

    pub async fn get_schema(&self, name: &str) -> Option<serde_json::Value> { /* ... */ }
}
```

#### 1.4 Resource System

```rust
// crates/rvagent-mcp/src/resources.rs

use std::collections::HashMap;

/// A resource exposed via MCP.
pub struct McpResource {
    pub uri: String,           // e.g., "mcp://resources/config/agent.yaml"
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}

/// A resource template with URI patterns.
pub struct ResourceTemplate {
    pub uri_template: String,  // e.g., "mcp://resources/agents/{agent_id}/state"
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}

/// Dynamic resource provider trait.
#[async_trait::async_trait]
pub trait ResourceProvider: Send + Sync {
    /// List available resources under this provider.
    async fn list(&self) -> Result<Vec<McpResource>, McpError>;

    /// Read a specific resource by URI.
    async fn read(&self, uri: &str) -> Result<ResourceContent, McpError>;

    /// Subscribe to resource changes (optional).
    async fn subscribe(&self, uri: &str) -> Option<tokio::sync::broadcast::Receiver<ResourceChanged>>;
}

pub enum ResourceContent {
    Text { uri: String, mime_type: Option<String>, text: String },
    Blob { uri: String, mime_type: Option<String>, blob: Vec<u8> },
}

pub struct ResourceChanged {
    pub uri: String,
}

/// Central resource manager.
pub struct ResourceManager {
    static_resources: HashMap<String, McpResource>,
    templates: Vec<ResourceTemplate>,
    providers: Vec<Box<dyn ResourceProvider>>,
}
```

#### 1.5 Transport Abstraction

```rust
// crates/rvagent-mcp/src/transport/mod.rs

#[async_trait::async_trait]
pub trait McpTransport: Send + Sync {
    /// Start the transport, returning a handle for sending responses.
    async fn start(&mut self) -> Result<(), McpError>;

    /// Receive the next incoming request or notification.
    async fn recv(&mut self) -> Result<JsonRpcMessage, McpError>;

    /// Send a response or notification.
    async fn send(&self, message: JsonRpcMessage) -> Result<(), McpError>;

    /// Gracefully shut down the transport.
    async fn shutdown(&self) -> Result<(), McpError>;
}

/// Stdio transport: reads JSON-RPC from stdin, writes to stdout.
pub struct StdioTransport { /* ... */ }

/// SSE transport: HTTP server with Server-Sent Events for server-to-client,
/// HTTP POST for client-to-server.
pub struct SseTransport {
    pub bind_addr: std::net::SocketAddr,
    /* ... */
}

/// WebSocket transport: full-duplex JSON-RPC over WebSocket.
pub struct WebSocketTransport {
    pub bind_addr: std::net::SocketAddr,
    /* ... */
}
```

#### 1.6 Server Capabilities Negotiation

```rust
// crates/rvagent-mcp/src/capabilities.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub tools: Option<ToolsCapability>,
    pub resources: Option<ResourcesCapability>,
    pub prompts: Option<PromptsCapability>,
    pub logging: Option<LoggingCapability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    /// Server supports tool list change notifications.
    pub list_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcesCapability {
    /// Server supports resource subscriptions.
    pub subscribe: bool,
    /// Server supports resource list change notifications.
    pub list_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptsCapability {
    pub list_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingCapability {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    pub protocol_version: String,  // "2025-03-26"
    pub capabilities: ServerCapabilities,
    pub server_info: ServerInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}
```

#### 1.7 AnyTool Adapter

Bridges existing `rvagent-tools::Tool` implementations (ADR-096) into the MCP tool registry without rewriting them:

```rust
// crates/rvagent-mcp/src/adapter.rs

use rvagent_tools::Tool as AgentTool;

/// Wraps any rvagent-tools::Tool as an McpToolHandler.
pub struct AnyToolAdapter {
    inner: Arc<dyn AgentTool>,
    runtime: Arc<ToolRuntime>,
}

#[async_trait::async_trait]
impl McpToolHandler for AnyToolAdapter {
    async fn call(
        &self,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult, McpError> {
        let result = self.inner.ainvoke(arguments, &self.runtime).await;
        match result {
            ToolResult::Text(s) => Ok(ToolCallResult::text(s)),
            ToolResult::Command(cmd) => Ok(ToolCallResult::text(
                format!("State update applied: {:?}", cmd)
            )),
        }
    }
}

impl AnyToolAdapter {
    /// Register all tools from an rvagent-tools ToolSet into the MCP registry.
    pub async fn register_all(
        toolset: &dyn ToolSet,
        registry: &ToolRegistry,
        runtime: Arc<ToolRuntime>,
    ) -> Result<(), McpError> {
        for tool in toolset.tools() {
            let adapter = Arc::new(AnyToolAdapter {
                inner: tool.clone(),
                runtime: runtime.clone(),
            });
            registry.register(RegisteredTool {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                input_schema: tool.parameters_schema(),
                handler: adapter,
                annotations: ToolAnnotations::default(),
            }).await?;
        }
        Ok(())
    }
}
```

---

### 2. Enhanced Skills System

#### 2.1 Unified Skill Format

Skills use a YAML frontmatter block followed by a markdown body. The format is designed to be compatible with both OpenAI Codex task definitions and Anthropic Claude Code skill manifests:

```yaml
---
# Required fields
name: "deploy-service"
version: "1.2.0"
description: "Deploy a service to the target environment"

# Triggers: when this skill should be invoked
triggers:
  - pattern: "deploy {service} to {env}"
    type: regex
  - pattern: "/deploy"
    type: slash-command
  - event: "ci.pipeline.success"
    type: event

# Model routing hints (maps to ADR-026 3-tier routing)
model_routing:
  complexity_hint: "medium"      # low | medium | high
  preferred_tier: 2              # 1=WASM, 2=Haiku, 3=Sonnet/Opus
  max_tier: 3                    # Escalation ceiling
  requires_reasoning: false

# Compatibility
codex_compatible: true           # Can be used as an OpenAI Codex task
claude_code_compatible: true     # Can be used as a Claude Code skill

# Composition: skills this skill may invoke
dependencies:
  - name: "check-health"
    version: ">=1.0.0"
  - name: "run-tests"
    version: "^2.0.0"
    optional: true

# Runtime metadata
timeout_seconds: 300
retry_policy:
  max_retries: 2
  backoff_ms: 1000
---

## Instructions

Deploy the service `{{service}}` to the `{{env}}` environment.

### Steps

1. Run health check: `!invoke check-health --target {{service}}`
2. If tests skill is available: `!invoke run-tests --suite integration`
3. Execute deployment command
4. Verify post-deployment health

### Constraints

- Never deploy to production without passing health checks
- Always create a rollback plan before deploying
```

#### 2.2 Skill Loader

```rust
// In rvagent-skills crate

pub struct SkillLoader {
    /// Filesystem paths to search for skills.
    search_paths: Vec<PathBuf>,
    /// MCP resource providers for remote skills.
    mcp_providers: Vec<Arc<dyn ResourceProvider>>,
    /// Cached, parsed skills indexed by name.
    cache: Arc<RwLock<HashMap<String, ParsedSkill>>>,
}

pub struct ParsedSkill {
    pub metadata: SkillMetadata,
    pub body: String,
    pub source: SkillSource,
}

pub enum SkillSource {
    Filesystem(PathBuf),
    McpResource(String),   // URI
    Inline,
}

pub struct SkillMetadata {
    pub name: String,
    pub version: semver::Version,
    pub description: String,
    pub triggers: Vec<Trigger>,
    pub model_routing: ModelRoutingHint,
    pub codex_compatible: bool,
    pub claude_code_compatible: bool,
    pub dependencies: Vec<SkillDependency>,
    pub timeout_seconds: u64,
    pub retry_policy: Option<RetryPolicy>,
}

impl SkillLoader {
    /// Load all skills from configured paths and MCP providers.
    pub async fn load_all(&self) -> Result<Vec<ParsedSkill>, SkillError> { /* ... */ }

    /// Resolve a skill by name, respecting version constraints.
    pub async fn resolve(
        &self,
        name: &str,
        version_req: Option<&semver::VersionReq>,
    ) -> Result<ParsedSkill, SkillError> { /* ... */ }

    /// Resolve a full dependency graph for a skill, detecting cycles.
    pub async fn resolve_dependencies(
        &self,
        skill: &ParsedSkill,
    ) -> Result<Vec<ParsedSkill>, SkillError> { /* ... */ }
}
```

#### 2.3 Skill Composition Runtime

```rust
/// Executes a skill, recursively invoking dependent skills as needed.
pub struct SkillExecutor {
    loader: Arc<SkillLoader>,
    tool_registry: Arc<ToolRegistry>,
    max_depth: usize,  // Prevent infinite recursion, default 10
}

impl SkillExecutor {
    pub async fn execute(
        &self,
        skill_name: &str,
        params: HashMap<String, String>,
        context: &ExecutionContext,
    ) -> Result<SkillResult, SkillError> {
        self.execute_inner(skill_name, params, context, 0).await
    }

    async fn execute_inner(
        &self,
        skill_name: &str,
        params: HashMap<String, String>,
        context: &ExecutionContext,
        depth: usize,
    ) -> Result<SkillResult, SkillError> {
        if depth >= self.max_depth {
            return Err(SkillError::MaxDepthExceeded(self.max_depth));
        }

        let skill = self.loader.resolve(skill_name, None).await?;
        let deps = self.loader.resolve_dependencies(&skill).await?;

        // Execute required dependencies first
        for dep in &deps {
            if !dep.metadata.dependencies.iter().any(|d| d.optional) {
                self.execute_inner(
                    &dep.metadata.name,
                    params.clone(),
                    context,
                    depth + 1,
                ).await?;
            }
        }

        // Execute the skill body via the appropriate model tier
        let tier = self.route_to_tier(&skill.metadata.model_routing);
        tier.execute(&skill.body, &params, context).await
    }
}
```

---

### 3. Topology-Aware Deployment

#### 3.1 Topology Trait

```rust
// crates/rvagent-core/src/topology/mod.rs

pub mod hierarchical;
pub mod mesh;
pub mod adaptive;

use std::collections::HashMap;

/// Unique identifier for a node in the topology.
pub type NodeId = String;

/// A message routed between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyMessage {
    pub id: uuid::Uuid,
    pub from: NodeId,
    pub to: MessageTarget,
    pub payload: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub ttl: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageTarget {
    /// Direct message to a specific node.
    Node(NodeId),
    /// Broadcast to all nodes.
    Broadcast,
    /// Send to the current leader/queen.
    Leader,
    /// Send to any node matching a role.
    Role(String),
}

/// Health status for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealth {
    pub node_id: NodeId,
    pub status: HealthStatus,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub load: f64,       // 0.0 - 1.0
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unreachable,
}

/// Core topology trait. All topologies implement this interface.
#[async_trait::async_trait]
pub trait Topology: Send + Sync {
    /// Human-readable name of this topology.
    fn name(&self) -> &str;

    /// Join the topology as a node.
    async fn join(&self, node_id: NodeId, metadata: NodeMetadata) -> Result<(), TopologyError>;

    /// Leave the topology gracefully.
    async fn leave(&self, node_id: &str) -> Result<(), TopologyError>;

    /// Discover all known nodes.
    async fn discover(&self) -> Result<Vec<NodeHealth>, TopologyError>;

    /// Route a message according to topology rules.
    async fn route(&self, message: TopologyMessage) -> Result<(), TopologyError>;

    /// Receive the next message for a given node.
    async fn recv(&self, node_id: &str) -> Result<TopologyMessage, TopologyError>;

    /// Get health of a specific node.
    async fn health(&self, node_id: &str) -> Result<NodeHealth, TopologyError>;

    /// Get the current leader, if the topology has one.
    async fn leader(&self) -> Option<NodeId>;
}

pub struct NodeMetadata {
    pub role: String,
    pub capabilities: Vec<String>,
    pub max_concurrent_tasks: usize,
}
```

#### 3.2 Hierarchical Topology (Queen/Worker)

```rust
// crates/rvagent-core/src/topology/hierarchical.rs

/// Hierarchical topology with a single queen (leader) and N workers.
/// Uses Raft consensus for leader election and log replication.
pub struct HierarchicalTopology {
    queen: Arc<RwLock<Option<NodeId>>>,
    workers: Arc<RwLock<HashMap<NodeId, NodeHealth>>>,
    raft_state: Arc<RwLock<RaftState>>,
    message_queue: Arc<RwLock<HashMap<NodeId, VecDeque<TopologyMessage>>>>,
    heartbeat_interval: Duration,
    election_timeout: Duration,
}

struct RaftState {
    current_term: u64,
    voted_for: Option<NodeId>,
    log: Vec<LogEntry>,
    commit_index: u64,
    role: RaftRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RaftRole {
    Follower,
    Candidate,
    Leader,
}

impl HierarchicalTopology {
    pub fn new(config: HierarchicalConfig) -> Self { /* ... */ }

    /// Queen assigns a task to the least-loaded worker.
    pub async fn assign_task(
        &self,
        task: TopologyMessage,
    ) -> Result<NodeId, TopologyError> {
        let workers = self.workers.read().await;
        let target = workers.values()
            .filter(|w| w.status == HealthStatus::Healthy)
            .min_by(|a, b| a.load.partial_cmp(&b.load).unwrap())
            .ok_or(TopologyError::NoHealthyWorkers)?;
        self.route(TopologyMessage {
            to: MessageTarget::Node(target.node_id.clone()),
            ..task
        }).await?;
        Ok(target.node_id.clone())
    }

    /// Start the Raft election timer. Called on each worker.
    async fn start_election_timer(&self, node_id: &str) { /* ... */ }

    /// Process heartbeats from queen. Resets election timer.
    async fn handle_heartbeat(&self, from: &str, term: u64) { /* ... */ }
}
```

#### 3.3 Mesh Topology (Peer-to-Peer)

```rust
// crates/rvagent-core/src/topology/mesh.rs

/// Mesh topology where all nodes are equal peers.
/// Uses gossip protocol for state dissemination and node discovery.
pub struct MeshTopology {
    nodes: Arc<RwLock<HashMap<NodeId, NodeHealth>>>,
    gossip_state: Arc<RwLock<GossipState>>,
    message_queue: Arc<RwLock<HashMap<NodeId, VecDeque<TopologyMessage>>>>,
    gossip_interval: Duration,
    fanout: usize,  // Number of peers to gossip to per round
}

struct GossipState {
    /// Crdt-based membership set (add-wins).
    membership: HashMap<NodeId, (NodeMetadata, lamport::Clock)>,
    /// Vector clock for causal ordering.
    vector_clock: HashMap<NodeId, u64>,
}

impl MeshTopology {
    pub fn new(config: MeshConfig) -> Self { /* ... */ }

    /// Gossip local state to `fanout` random peers.
    async fn gossip_round(&self) { /* ... */ }

    /// Merge received gossip state with local state.
    async fn merge_gossip(&self, remote: &GossipState) { /* ... */ }

    /// Route message using consistent hashing when target is Role-based.
    async fn route_by_role(
        &self,
        role: &str,
        message: TopologyMessage,
    ) -> Result<(), TopologyError> { /* ... */ }
}
```

#### 3.4 Adaptive Topology (Dynamic Switching)

```rust
// crates/rvagent-core/src/topology/adaptive.rs

/// Adaptive topology that switches between hierarchical and mesh
/// based on cluster size, failure rate, and load characteristics.
pub struct AdaptiveTopology {
    current: Arc<RwLock<ActiveTopology>>,
    hierarchical: HierarchicalTopology,
    mesh: MeshTopology,
    switch_policy: SwitchPolicy,
    metrics: Arc<RwLock<TopologyMetrics>>,
}

enum ActiveTopology {
    Hierarchical,
    Mesh,
}

pub struct SwitchPolicy {
    /// Switch to mesh when node count exceeds this threshold.
    pub mesh_threshold_nodes: usize,
    /// Switch to hierarchical when failure rate exceeds this (0.0-1.0).
    pub hierarchical_threshold_failure_rate: f64,
    /// Minimum time between topology switches.
    pub cooldown: Duration,
    /// Switch to mesh when leader latency exceeds this.
    pub leader_latency_threshold: Duration,
}

struct TopologyMetrics {
    node_count: usize,
    failure_rate_1m: f64,
    avg_leader_latency: Duration,
    last_switch: Option<chrono::DateTime<chrono::Utc>>,
}

impl AdaptiveTopology {
    pub fn new(
        hierarchical_config: HierarchicalConfig,
        mesh_config: MeshConfig,
        switch_policy: SwitchPolicy,
    ) -> Self { /* ... */ }

    /// Evaluate whether a topology switch is warranted.
    async fn evaluate_switch(&self) -> Option<ActiveTopology> {
        let metrics = self.metrics.read().await;

        // Enforce cooldown
        if let Some(last) = metrics.last_switch {
            if chrono::Utc::now() - last < self.switch_policy.cooldown {
                return None;
            }
        }

        let current = self.current.read().await;
        match *current {
            ActiveTopology::Hierarchical => {
                if metrics.node_count > self.switch_policy.mesh_threshold_nodes {
                    return Some(ActiveTopology::Mesh);
                }
                if metrics.avg_leader_latency > self.switch_policy.leader_latency_threshold {
                    return Some(ActiveTopology::Mesh);
                }
            }
            ActiveTopology::Mesh => {
                if metrics.failure_rate_1m > self.switch_policy.hierarchical_threshold_failure_rate {
                    return Some(ActiveTopology::Hierarchical);
                }
                if metrics.node_count <= self.switch_policy.mesh_threshold_nodes / 2 {
                    return Some(ActiveTopology::Hierarchical);
                }
            }
        }
        None
    }

    /// Perform a topology switch with state migration.
    async fn switch_to(&self, target: ActiveTopology) -> Result<(), TopologyError> { /* ... */ }
}

#[async_trait::async_trait]
impl Topology for AdaptiveTopology {
    fn name(&self) -> &str { "adaptive" }

    async fn route(&self, message: TopologyMessage) -> Result<(), TopologyError> {
        // Check if we should switch before routing
        if let Some(target) = self.evaluate_switch().await {
            self.switch_to(target).await?;
        }

        let current = self.current.read().await;
        match *current {
            ActiveTopology::Hierarchical => self.hierarchical.route(message).await,
            ActiveTopology::Mesh => self.mesh.route(message).await,
        }
    }

    // ... delegate other methods similarly
    # // remaining trait methods delegate to active topology
    async fn join(&self, node_id: NodeId, metadata: NodeMetadata) -> Result<(), TopologyError> {
        let current = self.current.read().await;
        match *current {
            ActiveTopology::Hierarchical => self.hierarchical.join(node_id, metadata).await,
            ActiveTopology::Mesh => self.mesh.join(node_id, metadata).await,
        }
    }

    async fn leave(&self, node_id: &str) -> Result<(), TopologyError> {
        let current = self.current.read().await;
        match *current {
            ActiveTopology::Hierarchical => self.hierarchical.leave(node_id).await,
            ActiveTopology::Mesh => self.mesh.leave(node_id).await,
        }
    }

    async fn discover(&self) -> Result<Vec<NodeHealth>, TopologyError> {
        let current = self.current.read().await;
        match *current {
            ActiveTopology::Hierarchical => self.hierarchical.discover().await,
            ActiveTopology::Mesh => self.mesh.discover().await,
        }
    }

    async fn recv(&self, node_id: &str) -> Result<TopologyMessage, TopologyError> {
        let current = self.current.read().await;
        match *current {
            ActiveTopology::Hierarchical => self.hierarchical.recv(node_id).await,
            ActiveTopology::Mesh => self.mesh.recv(node_id).await,
        }
    }

    async fn health(&self, node_id: &str) -> Result<NodeHealth, TopologyError> {
        let current = self.current.read().await;
        match *current {
            ActiveTopology::Hierarchical => self.hierarchical.health(node_id).await,
            ActiveTopology::Mesh => self.mesh.health(node_id).await,
        }
    }

    async fn leader(&self) -> Option<NodeId> {
        let current = self.current.read().await;
        match *current {
            ActiveTopology::Hierarchical => self.hierarchical.leader().await,
            ActiveTopology::Mesh => None,  // Mesh has no leader
        }
    }
}
```

#### 3.5 Deployment Descriptors

```yaml
# deploy/hierarchical-3node.yaml
topology: hierarchical
nodes:
  - id: queen-01
    role: queen
    capabilities: [orchestrate, assign, monitor]
    resources:
      cpu: 4
      memory_gb: 8
    mcp:
      transport: sse
      port: 9100

  - id: worker-01
    role: worker
    capabilities: [code, test, review]
    resources:
      cpu: 8
      memory_gb: 16
    mcp:
      transport: websocket
      port: 9101

  - id: worker-02
    role: worker
    capabilities: [code, deploy, security]
    resources:
      cpu: 8
      memory_gb: 16
    mcp:
      transport: websocket
      port: 9102

consensus:
  type: raft
  heartbeat_interval_ms: 150
  election_timeout_ms: 1500

health:
  check_interval_ms: 5000
  unhealthy_threshold: 3
```

---

### 4. Testing Strategy

#### 4.1 MCP Integration Tests

```rust
#[cfg(test)]
mod mcp_integration_tests {
    /// Test tool registration, discovery, and invocation via stdio transport.
    #[tokio::test]
    async fn test_tool_lifecycle_stdio() { /* ... */ }

    /// Test resource listing, reading, and subscription notifications.
    #[tokio::test]
    async fn test_resource_crud_and_subscribe() { /* ... */ }

    /// Test capabilities negotiation: client requests tools+resources,
    /// server responds with supported capabilities.
    #[tokio::test]
    async fn test_capabilities_negotiation() { /* ... */ }

    /// Test AnyTool adapter bridges all rvagent-tools correctly.
    #[tokio::test]
    async fn test_anytool_adapter_full_toolset() { /* ... */ }

    /// Test JSON Schema validation rejects malformed tool arguments.
    #[tokio::test]
    async fn test_schema_validation_rejects_invalid() { /* ... */ }

    /// Test concurrent tool calls do not deadlock the registry.
    #[tokio::test]
    async fn test_concurrent_tool_calls() { /* ... */ }
}
```

#### 4.2 Topology Integration Tests

```rust
#[cfg(test)]
mod topology_integration_tests {
    /// Hierarchical: queen assigns tasks, workers report results.
    #[tokio::test]
    async fn test_hierarchical_task_assignment() { /* ... */ }

    /// Hierarchical: queen crashes, new queen elected via Raft.
    #[tokio::test]
    async fn test_hierarchical_leader_election_on_failure() { /* ... */ }

    /// Mesh: all nodes discover each other via gossip.
    #[tokio::test]
    async fn test_mesh_gossip_convergence() { /* ... */ }

    /// Mesh: messages routed to correct node by role.
    #[tokio::test]
    async fn test_mesh_role_based_routing() { /* ... */ }

    /// Adaptive: topology switches from hierarchical to mesh when
    /// node count exceeds threshold.
    #[tokio::test]
    async fn test_adaptive_switch_on_scale() { /* ... */ }

    /// Adaptive: topology switches from mesh to hierarchical on
    /// high failure rate.
    #[tokio::test]
    async fn test_adaptive_switch_on_failure_rate() { /* ... */ }

    /// Adaptive: switch cooldown prevents rapid oscillation.
    #[tokio::test]
    async fn test_adaptive_cooldown_prevents_flapping() { /* ... */ }
}
```

#### 4.3 Property-Based Tests

```rust
#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    proptest! {
        /// Any message sent to a healthy node must eventually be received.
        #[test]
        fn message_delivery_to_healthy_node(
            topology_type in prop_oneof!["hierarchical", "mesh", "adaptive"],
            node_count in 2..20usize,
            message_count in 1..100usize,
        ) {
            // Setup topology with node_count nodes
            // Send message_count messages to random healthy nodes
            // Assert all messages are received
        }

        /// Node discovery must return all healthy nodes.
        #[test]
        fn discovery_completeness(
            node_count in 2..50usize,
            failed_count in 0..5usize,
        ) {
            // Setup topology, mark `failed_count` as unreachable
            // Assert discover() returns exactly (node_count - failed_count) healthy nodes
        }

        /// Raft leader election must converge to exactly one leader.
        #[test]
        fn raft_single_leader(
            node_count in 3..10usize,
            partition_at in 0..5usize,
        ) {
            // Setup hierarchical topology
            // Simulate network partition at step `partition_at`
            // Assert at most one leader exists in each partition
        }
    }
}
```

#### 4.4 Stress and Chaos Tests

```rust
#[cfg(test)]
mod chaos_tests {
    /// Stress: 1000 concurrent tool calls across all three topologies.
    #[tokio::test]
    async fn stress_concurrent_tool_execution() {
        for topology in [hierarchical(), mesh(), adaptive()] {
            let handles: Vec<_> = (0..1000).map(|i| {
                let topo = topology.clone();
                tokio::spawn(async move {
                    topo.route(make_tool_call(i)).await
                })
            }).collect();

            let results = futures::future::join_all(handles).await;
            let failures: Vec<_> = results.iter()
                .filter(|r| r.as_ref().map(|r| r.is_err()).unwrap_or(true))
                .collect();
            assert!(
                failures.len() as f64 / 1000.0 < 0.01,
                "Failure rate exceeded 1%"
            );
        }
    }

    /// Chaos: randomly kill nodes during active task execution.
    #[tokio::test]
    async fn chaos_random_node_failures() {
        let topology = adaptive();
        let nodes = spawn_nodes(&topology, 10).await;

        // Start continuous message flow
        let message_task = tokio::spawn(continuous_messages(topology.clone()));

        // Randomly kill nodes at intervals
        let chaos_task = tokio::spawn(async move {
            let mut rng = rand::thread_rng();
            for _ in 0..5 {
                let victim = &nodes[rng.gen_range(1..nodes.len())]; // Never kill node 0
                topology.leave(&victim.id).await.ok();
                tokio::time::sleep(Duration::from_millis(200)).await;
                topology.join(victim.id.clone(), victim.metadata.clone()).await.ok();
            }
        });

        let (msg_result, _) = tokio::join!(message_task, chaos_task);
        let stats = msg_result.unwrap();
        assert!(stats.delivery_rate > 0.95, "Delivery rate below 95%");
    }

    /// Chaos: simulate network partitions during adaptive switch.
    #[tokio::test]
    async fn chaos_partition_during_topology_switch() { /* ... */ }
}
```

#### 4.5 Skills System Tests

```rust
#[cfg(test)]
mod skills_tests {
    /// Parse YAML frontmatter and validate all fields.
    #[test]
    fn test_skill_yaml_parsing() { /* ... */ }

    /// Resolve a skill with version constraints.
    #[tokio::test]
    async fn test_skill_version_resolution() { /* ... */ }

    /// Detect and reject circular skill dependencies.
    #[tokio::test]
    async fn test_circular_dependency_detection() { /* ... */ }

    /// Execute a composed skill that invokes two sub-skills.
    #[tokio::test]
    async fn test_skill_composition_execution() { /* ... */ }

    /// Load skills from both filesystem and MCP resource provider.
    #[tokio::test]
    async fn test_mixed_source_skill_loading() { /* ... */ }

    /// Verify Codex-compatible skills can be exported to Codex format.
    #[tokio::test]
    async fn test_codex_export_roundtrip() { /* ... */ }

    /// Verify Claude Code compatible skills can be exported to slash-command format.
    #[tokio::test]
    async fn test_claude_code_export_roundtrip() { /* ... */ }

    /// Max depth guard prevents runaway recursion.
    #[tokio::test]
    async fn test_max_depth_guard() { /* ... */ }
}
```

---

## Consequences

### Positive

1. **MCP compliance.** The `rvagent-mcp` crate enables any rvAgent instance to act as both an MCP server and client, making the framework interoperable with the broader MCP ecosystem (VS Code, Claude Desktop, third-party tools).

2. **Zero-rewrite tool integration.** The `AnyToolAdapter` bridges all existing `rvagent-tools` implementations into MCP without modifying them. New tools only need to implement one trait.

3. **Cross-platform skills.** A single skill definition works with OpenAI Codex, Claude Code, and native rvAgent. This eliminates vendor lock-in for skill authoring.

4. **Topology flexibility.** Teams can start with a simple hierarchical deployment and seamlessly transition to mesh or adaptive topologies as their agent clusters grow, without code changes.

5. **Resilience.** Raft consensus for hierarchical and gossip protocol for mesh provide well-understood fault tolerance guarantees. Adaptive topology adds automatic response to changing conditions.

6. **Comprehensive test coverage.** Property-based and chaos tests catch edge cases that unit tests miss, particularly around distributed system invariants.

### Negative

1. **New crate overhead.** Adding `rvagent-mcp` increases the workspace size. Mitigated by keeping it optional (feature-gated in dependent crates).

2. **Complexity increase.** Three topology implementations with different consensus mechanisms increase the surface area for bugs. Mitigated by the shared `Topology` trait and extensive testing.

3. **Raft implementation risk.** Implementing Raft correctly is non-trivial. Consider using an existing crate (`openraft` or `async-raft`) rather than a from-scratch implementation. Decision on this is deferred to implementation phase.

4. **Skill format maintenance burden.** Supporting two external formats (Codex and Claude Code) means tracking upstream format changes. Mitigated by the compatibility flags being optional -- skills can opt out of cross-format support.

### Migration Path

| Phase | Scope | Estimated Effort |
|-------|-------|-----------------|
| **Phase 1** | `rvagent-mcp` crate: protocol, registry, stdio transport | 2 weeks |
| **Phase 2** | SSE + WebSocket transports, AnyTool adapter, resource system | 1.5 weeks |
| **Phase 3** | Enhanced skills: YAML parser, composition, loader | 1.5 weeks |
| **Phase 4** | Topology module: hierarchical with Raft | 2 weeks |
| **Phase 5** | Topology module: mesh with gossip, adaptive wrapper | 2 weeks |
| **Phase 6** | Integration tests, property tests, chaos tests | 1.5 weeks |
| **Phase 7** | Documentation, deployment descriptors, examples | 1 week |

### Test Coverage Targets

| Component | Target | Rationale |
|-----------|--------|-----------|
| `rvagent-mcp` protocol layer | 95% | Serialization correctness is critical |
| `rvagent-mcp` registry | 90% | Core MCP functionality |
| `rvagent-mcp` transports | 85% | I/O-heavy, harder to unit test |
| Skills parser + loader | 95% | Must handle all YAML edge cases |
| Skills executor | 90% | Composition logic is complex |
| Topology trait + hierarchical | 90% | Raft correctness is critical |
| Topology mesh | 85% | Gossip is eventually consistent |
| Topology adaptive | 90% | Switching logic must be correct |

### Crate Dependency Graph (Updated)

```
rvagent-mcp (new)
  ├── rvagent-tools    (AnyTool adapter)
  ├── rvagent-core     (topology module)
  ├── serde / serde_json
  ├── tokio
  ├── async-trait
  └── jsonschema       (schema validation)

rvagent-skills (enhanced)
  ├── rvagent-mcp      (resource-based skill loading)
  ├── rvagent-core
  ├── serde_yaml
  └── semver           (version resolution)

rvagent-core (enhanced)
  └── topology/        (new module)
      ├── mod.rs
      ├── hierarchical.rs
      ├── mesh.rs
      └── adaptive.rs
```

---

## References

- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [Raft Consensus Algorithm](https://raft.github.io/)
- [SWIM Gossip Protocol](https://www.cs.cornell.edu/projects/Quicksilver/public_pdfs/SWIM.pdf)
- ADR-093: DeepAgents Rust Conversion Overview
- ADR-095: Middleware Pipeline
- ADR-096: Tool System
- ADR-097: SubAgent Orchestration
- ADR-098: Memory, Skills, Summarization
- ADR-100: RVF Integration & Crate Structure
- ADR-101: Testing Strategy
- ADR-066: SSE MCP Transport
