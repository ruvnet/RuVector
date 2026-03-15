# rvAgent API Reference

High-level reference for rvAgent's public types, traits, and modules.

## Core Types (`rvagent-core`)

### AgentState

Typed agent state using `Arc`-wrapped fields for O(1) clone. Defined in `rvagent-core/src/state.rs`.

```rust
pub struct AgentState {
    pub messages: Arc<Vec<Message>>,
    pub todos: Arc<Vec<TodoItem>>,
    pub files: Arc<HashMap<String, FileData>>,
    pub memory_contents: Option<Arc<HashMap<String, String>>>,
    pub skills_metadata: Option<Arc<Vec<SkillMetadata>>>,
    extensions: HashMap<String, Box<dyn Any + Send + Sync>>,
}
```

Key methods:

| Method | Description |
|---|---|
| `new()` | Create empty state |
| `with_system_message(content)` | Create state with initial system message |
| `push_message(msg)` | Append message (copy-on-write) |
| `push_todo(item)` | Append todo item |
| `set_file(path, data)` | Insert/update file entry |
| `get_extension::<T>(key)` | Get typed extension value |
| `set_extension(key, value)` | Set extension value |
| `merge_subagent(child)` | Merge child state into parent |
| `clone()` | O(1) clone via Arc (extensions not shared) |

### Message

Unified message enum for agent communication. Defined in `rvagent-core/src/messages.rs`.

```rust
pub enum Message {
    System(SystemMessage),
    Human(HumanMessage),
    Ai(AiMessage),
    Tool(ToolMessage),
}
```

Constructors: `Message::system(content)`, `Message::human(content)`, `Message::ai(content)`, `Message::ai_with_tools(content, tool_calls)`, `Message::tool(tool_call_id, content)`.

### ToolCall

```rust
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}
```

### RvAgentConfig

Top-level agent configuration. Defined in `rvagent-core/src/config.rs`.

```rust
pub struct RvAgentConfig {
    pub model: String,                          // "provider:model" format
    pub name: Option<String>,                   // agent name for logging
    pub instructions: String,                   // system prompt
    pub middleware: Vec<MiddlewareConfig>,       // ordered pipeline
    pub tools: Vec<ToolConfig>,                 // additional tools
    pub backend: BackendConfig,                 // backend settings
    pub security_policy: SecurityPolicy,        // security controls
    pub resource_budget: Option<ResourceBudget>, // cost/time limits
}
```

### SecurityPolicy

```rust
pub struct SecurityPolicy {
    pub virtual_mode: bool,                     // default: true
    pub command_allowlist: Vec<String>,          // default: empty
    pub sensitive_env_patterns: Vec<String>,     // default: 10 patterns
    pub max_response_length: usize,             // default: 100KB
    pub trust_agents_md: bool,                  // default: false
}
```

### ResourceBudget

```rust
pub struct ResourceBudget {
    pub max_time_secs: u32,          // default: 300
    pub max_tokens: u64,             // default: 200_000
    pub max_cost_microdollars: u64,  // default: 5_000_000
    pub max_tool_calls: u32,         // default: 500
    pub max_external_writes: u32,    // default: 100
}
```

### ModelConfig and ChatModel Trait

Model resolution and the async chat model trait. Defined in `rvagent-core/src/models.rs`.

```rust
pub fn resolve_model(model_str: &str) -> ModelConfig;

pub struct ModelConfig {
    pub provider: Provider,           // Anthropic, OpenAi, Google, Bedrock, Fireworks, Other
    pub model_id: String,
    pub api_key_source: ApiKeySource, // Env(name), File(path), None
    pub max_tokens: u32,              // default: 16_384
    pub temperature: f32,             // default: 0.0
}

#[async_trait]
pub trait ChatModel: Send + Sync {
    async fn complete(&self, messages: &[Message]) -> Result<Message>;
    async fn stream(&self, messages: &[Message]) -> Result<Vec<Message>>;
}
```

### SystemPromptBuilder

Efficient deferred string concatenation. Defined in `rvagent-core/src/prompt.rs`.

```rust
pub struct SystemPromptBuilder {
    segments: SmallVec<[Cow<'static, str>; 8]>,
}
```

| Method | Description |
|---|---|
| `new()` | Empty builder |
| `with_base_prompt()` | Pre-loaded with `BASE_AGENT_PROMPT` |
| `append(text)` | Add segment |
| `append_section(text)` | Add segment with `\n\n` separator |
| `build()` | Single-allocation concatenation |

### RvAgentError

```rust
pub enum RvAgentError {
    Config(String),
    Model(String),
    Tool(String),
    Backend(String),
    Middleware(String),
    State(String),
    Security(String),
    Timeout(String),
    Json(serde_json::Error),
    Io(std::io::Error),
}
```

---

## Backend Trait and Implementations (`rvagent-backends`)

### Backend Trait

```rust
#[async_trait]
pub trait Backend: Send + Sync {
    async fn ls_info(&self, path: &str) -> Vec<FileInfo>;
    async fn read_file(&self, file_path: &str, offset: usize, limit: usize)
        -> Result<String, FileOperationError>;
    async fn write_file(&self, file_path: &str, content: &str) -> WriteResult;
    async fn edit_file(&self, file_path: &str, old_string: &str, new_string: &str,
        replace_all: bool) -> EditResult;
    async fn glob_info(&self, pattern: &str, path: &str) -> Vec<FileInfo>;
    async fn grep(&self, pattern: &str, path: Option<&str>, include_glob: Option<&str>)
        -> Result<Vec<GrepMatch>, String>;
    async fn download_files(&self, paths: &[String]) -> Vec<FileDownloadResponse>;
    async fn upload_files(&self, files: &[(String, Vec<u8>)]) -> Vec<FileUploadResponse>;
}
```

### SandboxBackend Trait

```rust
#[async_trait]
pub trait SandboxBackend: Backend {
    async fn execute(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse;
    fn id(&self) -> &str;
    fn sandbox_root(&self) -> &Path;
}
```

### Response Types

| Type | Fields |
|---|---|
| `FileInfo` | `path`, `is_dir`, `size`, `modified_at` |
| `FileOperationError` | `FileNotFound`, `PermissionDenied`, `IsDirectory`, `InvalidPath`, `SecurityViolation(String)` |
| `GrepMatch` | `path`, `line`, `text` |
| `WriteResult` | `error`, `path`, `files_update` |
| `EditResult` | `error`, `path`, `files_update`, `occurrences` |
| `ExecuteResponse` | `output`, `exit_code`, `truncated` |

### Backend Implementations

| Struct | Trait | Storage |
|---|---|---|
| `StateBackend` | `Backend` | `Arc<RwLock<HashMap<String, FileData>>>` |
| `FilesystemBackend` | `Backend` | Local disk with `virtual_mode` |
| `LocalShellBackend` | `SandboxBackend` | Local disk + shell |
| `CompositeBackend` | `Backend` | Routes to sub-backends by prefix |

### Utility Functions

```rust
pub fn format_content_with_line_numbers(content: &str, start_line: usize, max_line_len: usize) -> String;
pub fn is_safe_path_component(component: &str) -> bool;
pub fn contains_traversal(path: &str) -> bool;
```

### Unicode Security Functions

```rust
pub fn detect_dangerous_unicode(text: &str) -> Vec<UnicodeIssue>;
pub fn strip_dangerous_unicode(text: &str) -> String;
pub fn check_url_safety(url: &str) -> UrlSafetyResult;
pub fn detect_confusables(text: &str) -> Vec<(usize, char, char, &'static str)>;
pub fn validate_ascii_identifier(name: &str) -> bool;
```

---

## Middleware Trait and Implementations (`rvagent-middleware`)

### Middleware Trait

```rust
#[async_trait]
pub trait Middleware: Send + Sync {
    fn before_agent(&self, state: &AgentState, runtime: &Runtime, config: &RunnableConfig)
        -> Option<AgentState> { None }
    async fn abefore_agent(&self, state: &AgentState, runtime: &Runtime, config: &RunnableConfig)
        -> Option<AgentState> { self.before_agent(state, runtime, config) }
    fn wrap_model_call(&self, request: ModelRequest<()>,
        handler: &dyn Fn(ModelRequest<()>) -> ModelResponse<()>) -> ModelResponse<()> { handler(request) }
    fn modify_request(&self, request: ModelRequest<()>) -> ModelRequest<()> { request }
    fn tools(&self) -> Vec<Box<dyn Tool>> { vec![] }
    fn state_keys(&self) -> Vec<&str> { vec![] }
}
```

### MiddlewarePipeline

```rust
pub struct MiddlewarePipeline {
    middlewares: Vec<Box<dyn Middleware>>,
}

impl MiddlewarePipeline {
    pub fn new(middlewares: Vec<Box<dyn Middleware>>) -> Self;
    pub async fn run_before_agent(&self, state: &mut AgentState, runtime: &Runtime, config: &RunnableConfig);
    pub fn collect_tools(&self) -> Vec<Box<dyn Tool>>;
    pub async fn wrap_model_call(&self, request: ModelRequest<()>, base_handler: impl Fn(...)) -> ModelResponse<()>;
}
```

### Built-in Middleware

| Middleware | Tools Provided | State Keys | Hook |
|---|---|---|---|
| `TodoListMiddleware` | `write_todos` | `todos` | `before_agent` |
| `MemoryMiddleware` | -- | `memory_contents` | `before_agent`, `wrap_model_call` |
| `SkillsMiddleware` | -- | `skills_metadata` | `before_agent`, `wrap_model_call` |
| `FilesystemMiddleware` | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute` | -- | `tools` |
| `SubAgentMiddleware` | `task` | -- | `tools`, `wrap_model_call` |
| `SummarizationMiddleware` | `compact_conversation` | -- | `wrap_model_call` |
| `PromptCachingMiddleware` | -- | -- | `wrap_model_call` |
| `PatchToolCallsMiddleware` | -- | `messages` | `before_agent` |
| `WitnessMiddleware` | -- | -- | `wrap_model_call` |
| `ToolResultSanitizerMiddleware` | -- | -- | `wrap_model_call` |
| `HumanInTheLoopMiddleware` | -- | -- | `wrap_model_call` |

---

## Tool Trait and Enum Dispatch (`rvagent-tools`)

### Tool Trait

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult;
    async fn ainvoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult;
}
```

### ToolResult

```rust
pub enum ToolResult {
    Text(String),
    Command(StateUpdate),
}
```

### Enum Dispatch (Built-in Tools)

```rust
pub enum BuiltinTool { Ls, ReadFile, WriteFile, EditFile, Glob, Grep, Execute, WriteTodos, Task }
pub enum AnyTool { Builtin(BuiltinTool), Dynamic(Box<dyn Tool>) }
```

Built-in tools use enum dispatch to avoid vtable indirection. User-defined tools use `Box<dyn Tool>`.

### Built-in Tool Parameters

| Tool | Parameters |
|---|---|
| `ls` | `path: String` |
| `read_file` | `file_path: String`, `offset?: usize` (default 0), `limit?: usize` (default 100) |
| `write_file` | `file_path: String`, `content: String` |
| `edit_file` | `file_path: String`, `old_string: String`, `new_string: String`, `replace_all?: bool` (default false) |
| `glob` | `pattern: String`, `path?: String` (default "/") |
| `grep` | `pattern: String`, `path?: String`, `include?: String` |
| `execute` | `command: String`, `timeout?: u32` |
| `write_todos` | `todos: Vec<TodoItem>` |
| `task` | `description: String`, `subagent_type: String` |

---

## SubAgent Orchestration (`rvagent-subagents`)

### SubAgentSpec

```rust
pub struct SubAgentSpec {
    pub name: String,
    pub model: Option<String>,
    pub instructions: String,
    pub tools: Vec<String>,
    pub handoff_description: Option<String>,
    pub can_read: bool,      // default: true
    pub can_write: bool,     // default: false
    pub can_execute: bool,   // default: false
}
```

Factory methods: `SubAgentSpec::new(name, instructions)`, `SubAgentSpec::general_purpose()`.

### CompiledSubAgent

```rust
pub struct CompiledSubAgent {
    pub spec: SubAgentSpec,
    pub graph: Vec<String>,
    pub middleware_pipeline: Vec<String>,
    pub backend: String,
}
```

### Orchestration Functions

```rust
pub fn compile_subagents(specs: &[SubAgentSpec], parent_config: &RvAgentConfig) -> Vec<CompiledSubAgent>;
pub fn prepare_subagent_state(parent_state: &AgentState, task_description: &str) -> AgentState;
pub fn extract_result_message(result_state: &AgentState) -> Option<String>;
pub fn merge_subagent_state(parent: &mut AgentState, subagent_result: &AgentState);
pub fn resolve_tools(spec: &SubAgentSpec, parent_config: &RvAgentConfig) -> Vec<String>;
```

### State Isolation

Excluded keys (never passed to/from subagents):
`messages`, `remaining_steps`, `task_completion`, `todos`, `structured_response`, `skills_metadata`, `memory_contents`

---

## ACP Server Types (`rvagent-acp`)

### Request/Response Types

```rust
pub enum ContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
    ToolResult { tool_use_id: String, content: String, is_error: bool },
}

pub struct PromptRequest {
    pub session_id: Option<String>,
    pub content: Vec<ContentBlock>,
}

pub struct PromptResponse {
    pub session_id: String,
    pub messages: Vec<ResponseMessage>,
}

pub struct SessionInfo {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub message_count: usize,
}

pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    pub status: u16,
}
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/prompt` | Submit prompt to agent |
| `POST` | `/sessions` | Create new session |
| `GET` | `/sessions` | List active sessions |

---

## Configuration Options Summary

| Option | Type | Default | Crate |
|---|---|---|---|
| `model` | `String` | `"anthropic:claude-sonnet-4-20250514"` | `rvagent-core` |
| `instructions` | `String` | `BASE_AGENT_PROMPT` | `rvagent-core` |
| `backend.backend_type` | `String` | `"local_shell"` | `rvagent-core` |
| `backend.cwd` | `Option<String>` | `None` | `rvagent-core` |
| `security_policy.virtual_mode` | `bool` | `true` | `rvagent-core` |
| `security_policy.command_allowlist` | `Vec<String>` | `[]` | `rvagent-core` |
| `security_policy.max_response_length` | `usize` | `102400` | `rvagent-core` |
| `security_policy.trust_agents_md` | `bool` | `false` | `rvagent-core` |
| `resource_budget.max_time_secs` | `u32` | `300` | `rvagent-core` |
| `resource_budget.max_tokens` | `u64` | `200_000` | `rvagent-core` |
| `resource_budget.max_cost_microdollars` | `u64` | `5_000_000` | `rvagent-core` |
| `resource_budget.max_tool_calls` | `u32` | `500` | `rvagent-core` |
| `resource_budget.max_external_writes` | `u32` | `100` | `rvagent-core` |
