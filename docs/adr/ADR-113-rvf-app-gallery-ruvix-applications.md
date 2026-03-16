# ADR-113: RVF App Gallery and Ruvix-Powered Applications

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-15                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |
| **Depends** | ADR-087, ADR-100, ADR-106, ADR-107, ADR-112     |
| **Crates**  | `rvagent-wasm`                                  |

## Context

The rvAgent WASM module now supports RVF containers, MCP tools, and Ruvix capabilities. However, users need pre-built agent templates and applications that leverage these capabilities without building containers from scratch.

### Inspiration: Claude Flow

Claude Flow provides 60+ agent types with specialized configurations for various tasks. Similarly, the RVF App Gallery provides ready-to-use agent templates packaged as RVF containers, with:

1. **Pre-built prompts** for specific roles (coder, researcher, tester, etc.)
2. **Tool configurations** for each agent type
3. **Skill definitions** with triggers
4. **MCP tool bindings** for standardized interfaces
5. **Ruvix capabilities** for security-first execution
6. **Orchestrator configs** for multi-agent swarms

---

## Decision

### 1. RVF App Gallery Module

Create a gallery module in `rvagent-wasm` that provides pre-built templates.

#### 1.1 Module Structure

```
crates/rvAgent/rvagent-wasm/src/
  gallery.rs          # Gallery module
    - TemplateCategory enum
    - GalleryTemplate struct
    - Built-in templates (6 initial)
    - WasmGallery class (WASM-exported)
```

#### 1.2 Template Categories

```rust
pub enum TemplateCategory {
    Development,    // Coding, debugging, refactoring
    Research,       // Analysis, information gathering
    Testing,        // QA, test generation, coverage
    Documentation,  // Docs, API specs, comments
    DevOps,         // CI/CD, deployment, monitoring
    Security,       // Vulnerability scanning, audits
    Orchestration,  // Multi-agent coordination
    Custom,         // User-defined templates
}
```

#### 1.3 GalleryTemplate Structure

```rust
pub struct GalleryTemplate {
    pub id: String,                         // Unique identifier
    pub name: String,                       // Display name
    pub description: String,                // Template description
    pub category: TemplateCategory,         // Category for organization
    pub version: String,                    // Semantic version
    pub author: String,                     // Template author
    pub tags: Vec<String>,                  // Searchable tags
    pub tools: Vec<ToolDefinition>,         // Tool definitions
    pub prompts: Vec<AgentPrompt>,          // System prompts
    pub skills: Vec<SkillDefinition>,       // Skills with triggers
    pub mcp_tools: Vec<McpToolEntry>,       // MCP tool bindings
    pub capabilities: Vec<CapabilityDef>,   // Ruvix capabilities
    pub orchestrator: Option<OrchestratorConfig>,  // Multi-agent config
    pub builtin: bool,                      // Built-in vs custom
}
```

---

### 2. Built-in Templates

#### 2.1 Coder Agent

| Field | Value |
|-------|-------|
| ID | `coder` |
| Category | Development |
| Tools | `analyze_code` |
| Skills | `/refactor`, `/explain` |
| MCP Tools | `read_file`, `write_file`, `edit_file` |
| Capabilities | `file_read` (sandbox), `file_write` (sandbox) |

**System Prompt:**
```
You are an expert software engineer. Write clean, efficient, and
well-documented code. Follow best practices and design patterns.
Always consider edge cases and error handling.
```

#### 2.2 Research Agent

| Field | Value |
|-------|-------|
| ID | `researcher` |
| Category | Research |
| Tools | `web_search`, `summarize` |
| Skills | `/deepdive` |
| MCP Tools | `read_file` |
| Capabilities | `file_read` (sandbox, delegation:1), `web_access` (network) |

**System Prompt:**
```
You are a meticulous research assistant. Gather comprehensive
information from multiple sources, verify facts, identify patterns,
and synthesize findings into clear, well-organized reports.
Always cite sources and acknowledge limitations.
```

#### 2.3 Testing Agent

| Field | Value |
|-------|-------|
| ID | `tester` |
| Category | Testing |
| Tools | `generate_tests` |
| Skills | `/coverage` |
| MCP Tools | `read_file`, `write_file` |
| Capabilities | `file_read` (sandbox), `file_write` (sandbox) |

**System Prompt:**
```
You are a thorough QA engineer. Write comprehensive tests covering
edge cases, error conditions, and happy paths. Analyze code coverage
and identify untested paths. Follow testing best practices and TDD principles.
```

#### 2.4 Code Review Agent

| Field | Value |
|-------|-------|
| ID | `reviewer` |
| Category | Development |
| Tools | `review_diff` |
| Skills | `/security` |
| MCP Tools | `read_file` |
| Capabilities | `file_read` (sandbox, delegation:2) |

**System Prompt:**
```
You are a senior code reviewer. Analyze code for quality, security
vulnerabilities, performance issues, and adherence to best practices.
Provide constructive feedback with specific suggestions for improvement.
```

#### 2.5 Security Agent

| Field | Value |
|-------|-------|
| ID | `security` |
| Category | Security |
| Tools | `scan_vulnerabilities` |
| Skills | `/threatmodel` |
| MCP Tools | `read_file` |
| Capabilities | `file_read` (sandbox) |

**System Prompt:**
```
You are a security expert. Identify vulnerabilities, analyze attack
vectors, and recommend mitigations. Follow OWASP guidelines and
security best practices. Be thorough and prioritize findings by severity.
```

#### 2.6 Swarm Orchestrator

| Field | Value |
|-------|-------|
| ID | `swarm-orchestrator` |
| Category | Orchestration |
| Topology | Hierarchical |
| Agents | queen, coder-1, tester-1, reviewer-1 |
| Connections | queen→coder, queen→tester, queen→reviewer, coder→tester, tester→reviewer |
| MCP Tools | `read_file`, `write_file` |
| Capabilities | `file_read` (delegation:3), `file_write` (delegation:2) |

**Orchestrator Config:**
```json
{
  "topology": "hierarchical",
  "agents": [
    { "id": "queen", "agent_type": "coordinator", "prompt_ref": "queen" },
    { "id": "coder-1", "agent_type": "coder", "prompt_ref": "coder" },
    { "id": "tester-1", "agent_type": "tester", "prompt_ref": "tester" },
    { "id": "reviewer-1", "agent_type": "reviewer", "prompt_ref": "reviewer" }
  ],
  "connections": [
    ["queen", "coder-1"],
    ["queen", "tester-1"],
    ["queen", "reviewer-1"],
    ["coder-1", "tester-1"],
    ["tester-1", "reviewer-1"]
  ]
}
```

---

### 3. WasmGallery API

#### 3.1 WASM-Exported Methods

```rust
#[wasm_bindgen]
impl WasmGallery {
    /// List all templates
    pub fn list(&self) -> Result<JsValue, JsValue>;

    /// List by category
    pub fn list_by_category(&self, category: &str) -> Result<JsValue, JsValue>;

    /// Search templates by query
    pub fn search(&self, query: &str) -> Result<JsValue, JsValue>;

    /// Get template by ID
    pub fn get(&self, id: &str) -> Result<JsValue, JsValue>;

    /// Load template as RVF container (returns Uint8Array)
    pub fn load_rvf(&self, id: &str) -> Result<js_sys::Uint8Array, JsValue>;

    /// Set active template
    pub fn set_active(&mut self, id: &str) -> Result<(), JsValue>;

    /// Get active template ID
    pub fn get_active(&self) -> Option<String>;

    /// Configure active template
    pub fn configure(&mut self, config_json: &str) -> Result<(), JsValue>;

    /// Add custom template
    pub fn add_custom(&mut self, template_json: &str) -> Result<(), JsValue>;

    /// Remove custom template
    pub fn remove_custom(&mut self, id: &str) -> Result<(), JsValue>;

    /// Get categories with counts
    pub fn get_categories(&self) -> Result<JsValue, JsValue>;

    /// Export custom templates
    pub fn export_custom(&self) -> Result<JsValue, JsValue>;

    /// Import custom templates
    pub fn import_custom(&mut self, templates_json: &str) -> Result<u32, JsValue>;
}
```

#### 3.2 JavaScript Usage

```javascript
import { WasmGallery, WasmMcpServer } from '@ruvector/rvagent/wasm';

// Create gallery instance
const gallery = new WasmGallery();

// List all templates
const templates = gallery.list();
console.log(templates);
// [
//   { id: "coder", name: "Coder Agent", category: "development", ... },
//   { id: "researcher", name: "Research Agent", category: "research", ... },
//   ...
// ]

// Search templates
const securityAgents = gallery.search("security vulnerability");

// Get template details
const coderTemplate = gallery.get("coder");
console.log(coderTemplate.tools);
console.log(coderTemplate.capabilities);

// Load as RVF container
const rvfBytes = gallery.loadRvf("coder");
console.log(`RVF size: ${rvfBytes.length} bytes`);

// Set active template
gallery.setActive("coder");

// Configure active template
gallery.configure(JSON.stringify({ maxTurns: 100 }));

// Add custom template
gallery.addCustom(JSON.stringify({
  id: "my-agent",
  name: "My Custom Agent",
  description: "A custom agent for my workflow",
  category: "custom",
  version: "1.0.0",
  author: "user",
  tags: ["custom", "workflow"],
  tools: [],
  prompts: [{
    name: "custom",
    system_prompt: "You are a helpful assistant.",
    version: "1.0.0"
  }],
  skills: [],
  mcp_tools: [],
  capabilities: []
}));
```

---

### 4. MCP Integration

#### 4.1 Gallery MCP Tools

The MCP server exposes gallery operations:

| Method | Description |
|--------|-------------|
| `gallery/list` | List all templates |
| `gallery/search` | Search templates by query |
| `gallery/get` | Get template by ID |
| `gallery/load` | Load template as active |
| `gallery/configure` | Configure active template |
| `gallery/categories` | Get categories with counts |

#### 4.2 MCP Tool Definitions

```json
{
  "name": "gallery_list",
  "description": "List all available gallery templates",
  "inputSchema": {
    "type": "object",
    "properties": {
      "category": { "type": "string", "description": "Filter by category" }
    }
  }
}
```

```json
{
  "name": "gallery_load",
  "description": "Load a gallery template by ID",
  "inputSchema": {
    "type": "object",
    "properties": {
      "id": { "type": "string", "description": "Template ID" }
    },
    "required": ["id"]
  }
}
```

#### 4.3 Usage via MCP

```javascript
const mcp = new WasmMcpServer();

// List templates via MCP
const listResponse = mcp.handleMessage(JSON.stringify({
  jsonrpc: "2.0",
  id: 1,
  method: "gallery/list"
}));

// Load a template
const loadResponse = mcp.handleMessage(JSON.stringify({
  jsonrpc: "2.0",
  id: 2,
  method: "gallery/load",
  params: { id: "coder" }
}));

// Get prompts from active template
const promptsResponse = mcp.handleMessage(JSON.stringify({
  jsonrpc: "2.0",
  id: 3,
  method: "prompts/list"
}));
```

---

### 5. Ruvix Capability Model

Each template defines capabilities following the Ruvix kernel security model:

#### 5.1 Capability Definition

```rust
pub struct CapabilityDef {
    pub name: String,           // Unique capability name
    pub rights: Vec<String>,    // Allowed operations
    pub scope: String,          // Boundary (sandbox, local, network)
    pub delegation_depth: u8,   // Max delegation hops (0 = no delegation)
}
```

#### 5.2 Scope Hierarchy

| Scope | Description | Example Operations |
|-------|-------------|-------------------|
| `sandbox` | Isolated virtual filesystem | read_file, write_file |
| `local` | Host filesystem (restricted) | read_config, write_logs |
| `network` | Network access | web_fetch, api_call |
| `system` | System operations | execute_command |

#### 5.3 Delegation Depth

- **0**: No delegation (capability cannot be passed to sub-agents)
- **1**: Single hop (capability can be delegated once)
- **2+**: Multi-hop (capability chains limited to N hops)

**Example:**
```rust
CapabilityDef {
    name: "file_read".to_string(),
    rights: vec!["read".to_string()],
    scope: "sandbox".to_string(),
    delegation_depth: 2,  // Can be delegated twice
}
```

---

### 6. Security Hardening

#### 6.1 Gallery Limits

```rust
/// Maximum number of custom templates
pub const MAX_CUSTOM_TEMPLATES: usize = 100;

/// Maximum template name length
pub const MAX_TEMPLATE_NAME_LENGTH: usize = 64;

/// Maximum template description length
pub const MAX_TEMPLATE_DESC_LENGTH: usize = 512;
```

#### 6.2 Input Validation

- Template IDs validated for allowed characters
- JSON payloads size-limited
- Search queries sanitized and length-limited

---

### 7. Future Applications

#### 7.1 Additional Templates (Planned)

| ID | Category | Description |
|----|----------|-------------|
| `api-designer` | Documentation | OpenAPI/Swagger spec generation |
| `db-architect` | Development | Database schema design |
| `perf-engineer` | DevOps | Performance profiling and optimization |
| `data-analyst` | Research | Data exploration and visualization |
| `ux-reviewer` | Documentation | UX/accessibility analysis |
| `cicd-pipeline` | DevOps | CI/CD workflow generation |
| `security-swarm` | Orchestration | Multi-agent security scanning |
| `tdd-london` | Testing | London School TDD with mocks |

#### 7.2 Template Marketplace

Future versions will support:
- **IPFS-backed distribution** for decentralized template sharing
- **Template versioning** with semantic versioning
- **Template ratings** and community reviews
- **Template dependencies** for composition

---

### 8. Integration with Claude Flow

The RVF App Gallery complements Claude Flow's TypeScript agents:

| Claude Flow Agent | RVF Gallery Template | Notes |
|-------------------|---------------------|-------|
| `coder` | `coder` | Same capabilities, WASM execution |
| `researcher` | `researcher` | Same capabilities, WASM execution |
| `tester` | `tester` | Same capabilities, WASM execution |
| `reviewer` | `reviewer` | Same capabilities, WASM execution |
| `security-architect` | `security` | Focused security template |
| `hierarchical-coordinator` | `swarm-orchestrator` | Multi-agent coordination |

---

## Consequences

### Positive

1. **Faster Agent Setup**: Pre-built templates reduce configuration time
2. **Consistent Patterns**: Templates enforce best practices
3. **Security-First**: Ruvix capabilities define clear boundaries
4. **Portable**: RVF containers work across environments
5. **Extensible**: Custom templates for specialized workflows

### Negative

1. **Template Maintenance**: Templates require updates for new patterns
2. **Learning Curve**: Users must understand capability model
3. **Size Overhead**: Built-in templates add to WASM bundle

### Neutral

1. **Template Selection**: Users must choose appropriate templates
2. **Customization**: Some users may prefer building from scratch

---

## Implementation Status

| Component | Status |
|-----------|--------|
| `gallery.rs` module | Implemented |
| 6 built-in templates | Implemented |
| WasmGallery API | Implemented |
| MCP gallery tools | Implemented |
| Security limits | Implemented |
| Tests | 61 tests passing |

---

## References

- ADR-087: Ruvix Cognition Kernel
- ADR-100: DeepAgents RVF Integration Crate Structure
- ADR-106: Ruvix Kernel RVF Integration
- ADR-107: rvAgent Native Swarm Orchestration with WASM Integration
- ADR-112: rvAgent MCP Server
- Claude Flow: https://github.com/ruvnet/claude-flow
