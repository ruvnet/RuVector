//! rvAgent middleware pipeline — core trait, types, and concrete middleware implementations.
//!
//! Provides the `Middleware` trait and `MiddlewarePipeline` for composing middleware
//! in the DeepAgents architecture (ADR-095, ADR-103).
//!
//! ## ADR-103 Learning Middleware (B5, B6)
//!
//! - [`sona`] — SONA Adaptive Learning with three loops (instant, background, deep)
//! - [`hnsw`] — HNSW Semantic Retrieval for skills and memory (150x-12,500x faster)

pub mod filesystem;
pub mod hitl;
pub mod hnsw;
pub mod mcp_bridge;
pub mod memory;
pub mod patch_tool_calls;
pub mod prompt_caching;
pub mod retry;
pub mod rvf_manifest;
pub mod skills;
pub mod sona;
pub mod subagents;
pub mod summarization;
pub mod todolist;
pub mod tool_sanitizer;
pub mod unicode_security;
pub mod unicode_security_middleware;
pub mod utils;
pub mod witness;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// Re-exports
pub use unicode_security::{UnicodeIssue, UnicodeSecurityChecker, UnicodeSecurityConfig};
pub use unicode_security_middleware::UnicodeSecurityMiddleware;
pub use utils::{append_to_system_message, SystemPromptBuilder};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Message role in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single tool call within an assistant message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

/// A conversation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
            tool_name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
            tool_name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
            tool_name: None,
        }
    }

    pub fn tool(
        content: impl Into<String>,
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: Some(tool_call_id.into()),
            tool_name: Some(name.into()),
        }
    }
}

/// Cache control hint for prompt caching (Anthropic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    pub cache_type: String,
}

/// Agent state — typed structure (ADR-103 A1) with extension map.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentState {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub todos: Vec<TodoItem>,
    #[serde(default)]
    pub extensions: HashMap<String, serde_json::Value>,
}

/// A single todo item managed by TodoListMiddleware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub id: String,
    pub content: String,
    #[serde(default)]
    pub status: TodoStatus,
}

/// Status of a todo item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

impl Default for TodoStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// State update returned by `before_agent`. Merged into `AgentState`.
#[derive(Debug, Clone, Default)]
pub struct AgentStateUpdate {
    pub messages: Option<Vec<Message>>,
    pub todos: Option<Vec<TodoItem>>,
    pub extensions: HashMap<String, serde_json::Value>,
}

/// Model request wrapping messages and configuration.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub system_message: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub cache_control: HashMap<String, CacheControl>,
    pub extensions: HashMap<String, serde_json::Value>,
}

impl ModelRequest {
    /// Create a new model request.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            system_message: None,
            messages,
            tools: vec![],
            cache_control: HashMap::new(),
            extensions: HashMap::new(),
        }
    }

    /// Return a copy with a different system message.
    pub fn with_system(mut self, system_message: Option<String>) -> Self {
        self.system_message = system_message;
        self
    }

    /// Return a copy with different messages.
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }
}

/// Model response from an LLM call.
#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub message: Message,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
}

impl ModelResponse {
    /// Create a simple text response.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            message: Message::assistant(content),
            tool_calls: vec![],
            usage: None,
        }
    }
}

/// Token usage information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_tokens: u64,
    #[serde(default)]
    pub cache_creation_tokens: u64,
}

/// Tool definition for model requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Runtime context passed to middleware hooks.
pub struct Runtime {
    pub context: serde_json::Value,
    pub config: RunnableConfig,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            context: serde_json::Value::Null,
            config: RunnableConfig::default(),
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for a runnable (thread/run IDs, metadata).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunnableConfig {
    #[serde(default)]
    pub thread_id: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Model handler traits
// ---------------------------------------------------------------------------

/// Synchronous model handler — called by `wrap_model_call`.
pub trait ModelHandler: Send + Sync {
    fn call(&self, request: ModelRequest) -> ModelResponse;
}

/// Async model handler — called by `awrap_model_call`.
#[async_trait]
pub trait AsyncModelHandler: Send + Sync {
    async fn call(&self, request: ModelRequest) -> ModelResponse;
}

/// Tool trait — tools injected by middleware.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn invoke(&self, args: serde_json::Value) -> Result<String, String>;
}

impl fmt::Debug for dyn Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tool").field("name", &self.name()).finish()
    }
}

// ---------------------------------------------------------------------------
// Middleware trait (ADR-095)
// ---------------------------------------------------------------------------

/// Core middleware trait — mirrors Python's `AgentMiddleware`.
///
/// Each method has a default no-op implementation, so concrete middleware
/// only needs to override the hooks it uses.
#[async_trait]
pub trait Middleware: Send + Sync {
    /// Called before agent execution. Returns state update or None.
    fn before_agent(
        &self,
        _state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        None
    }

    /// Async version of `before_agent`.
    async fn abefore_agent(
        &self,
        state: &AgentState,
        runtime: &Runtime,
        config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        self.before_agent(state, runtime, config)
    }

    /// Wrap a synchronous model call — intercept request/response.
    fn wrap_model_call(&self, request: ModelRequest, handler: &dyn ModelHandler) -> ModelResponse {
        handler.call(request)
    }

    /// Wrap an async model call.
    async fn awrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn AsyncModelHandler,
    ) -> ModelResponse {
        handler.call(request).await
    }

    /// Transform request before model call.
    fn modify_request(&self, request: ModelRequest) -> ModelRequest {
        request
    }

    /// Additional tools provided by this middleware.
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![]
    }

    /// Human-readable name of this middleware.
    fn name(&self) -> &str;
}

impl fmt::Debug for dyn Middleware {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Middleware")
            .field("name", &self.name())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Middleware Pipeline (ADR-095)
// ---------------------------------------------------------------------------

/// Executes the middleware pipeline in order.
/// Mirrors LangChain's `create_agent` middleware composition.
pub struct MiddlewarePipeline {
    middlewares: Vec<Box<dyn Middleware>>,
}

impl MiddlewarePipeline {
    /// Create a new pipeline from an ordered list of middlewares.
    pub fn new(middlewares: Vec<Box<dyn Middleware>>) -> Self {
        Self { middlewares }
    }

    /// Create an empty pipeline.
    pub fn empty() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Add a middleware to the end of the pipeline.
    pub fn push(&mut self, middleware: Box<dyn Middleware>) {
        self.middlewares.push(middleware);
    }

    /// Number of middlewares in the pipeline.
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Get middleware names in order.
    pub fn names(&self) -> Vec<&str> {
        self.middlewares.iter().map(|mw| mw.name()).collect()
    }

    /// Run `before_agent` hooks in order, accumulating state updates.
    pub async fn run_before_agent(
        &self,
        state: &mut AgentState,
        runtime: &Runtime,
        config: &RunnableConfig,
    ) {
        for mw in &self.middlewares {
            if let Some(update) = mw.abefore_agent(state, runtime, config).await {
                // Merge update into state
                if let Some(messages) = update.messages {
                    state.messages = messages;
                }
                if let Some(todos) = update.todos {
                    state.todos = todos;
                }
                for (k, v) in update.extensions {
                    state.extensions.insert(k, v);
                }
            }
        }
    }

    /// Collect all tools from all middlewares.
    pub fn collect_tools(&self) -> Vec<Box<dyn Tool>> {
        self.middlewares.iter().flat_map(|mw| mw.tools()).collect()
    }

    /// Run `modify_request` through all middlewares in order.
    pub fn run_modify_request(&self, mut request: ModelRequest) -> ModelRequest {
        for mw in &self.middlewares {
            request = mw.modify_request(request);
        }
        request
    }

    /// Run `wrap_model_call` through the pipeline.
    /// Middlewares are chained so the outermost (first) wraps the innermost (last).
    pub fn run_wrap_model_call(
        &self,
        request: ModelRequest,
        base_handler: &dyn ModelHandler,
    ) -> ModelResponse {
        if self.middlewares.is_empty() {
            return base_handler.call(request);
        }

        // Build chain from inside out using recursive approach.
        fn chain_call<'a>(
            middlewares: &'a [Box<dyn Middleware>],
            request: ModelRequest,
            handler: &'a dyn ModelHandler,
        ) -> ModelResponse {
            if middlewares.is_empty() {
                return handler.call(request);
            }
            let (first, rest) = middlewares.split_first().unwrap();
            let inner = ChainedInner { rest, handler };
            first.wrap_model_call(request, &inner)
        }

        struct ChainedInner<'a> {
            rest: &'a [Box<dyn Middleware>],
            handler: &'a dyn ModelHandler,
        }
        impl<'a> ModelHandler for ChainedInner<'a> {
            fn call(&self, request: ModelRequest) -> ModelResponse {
                chain_call(self.rest, request, self.handler)
            }
        }

        chain_call(&self.middlewares, request, base_handler)
    }

    /// Full pipeline run: before_agent, collect tools, modify_request, wrap_model_call.
    pub async fn run(
        &self,
        state: &mut AgentState,
        runtime: &Runtime,
        config: &RunnableConfig,
        mut request: ModelRequest,
        handler: &dyn ModelHandler,
    ) -> ModelResponse {
        // 1. Run before_agent hooks
        self.run_before_agent(state, runtime, config).await;

        // 2. Collect tools from all middlewares
        let tools: Vec<Box<dyn Tool>> = self.collect_tools();
        for tool in &tools {
            request.tools.push(ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters_schema(),
            });
        }

        // 3. Run modify_request
        request = self.run_modify_request(request);

        // 4. Run wrap_model_call chain
        self.run_wrap_model_call(request, handler)
    }
}

// ---------------------------------------------------------------------------
// Default pipeline builder (ADR-095)
// ---------------------------------------------------------------------------

/// Configuration for building the default middleware pipeline.
#[derive(Debug, Clone, Default)]
pub struct PipelineConfig {
    pub memory_sources: Option<Vec<String>>,
    pub skill_sources: Option<Vec<String>>,
    pub interrupt_on: Option<Vec<String>>,
    pub enable_witness: bool,
    /// Enable SONA adaptive learning middleware (ADR-103 B5).
    pub enable_sona: bool,
    /// Enable HNSW semantic retrieval middleware (ADR-103 B6).
    pub enable_hnsw: bool,
    /// Enable Unicode security middleware (C7 - CVE mitigation).
    pub enable_unicode_security: bool,
    /// Custom SONA configuration.
    pub sona_config: Option<sona::SonaMiddlewareConfig>,
    /// Custom HNSW configuration.
    pub hnsw_config: Option<hnsw::HnswMiddlewareConfig>,
    /// Custom Unicode security configuration.
    pub unicode_security_config: Option<UnicodeSecurityConfig>,
}

/// Build the default middleware pipeline per ADR-095 ordering:
/// Todo -> HNSW -> Memory -> Skills -> Filesystem -> SubAgent -> Summarization
/// -> PromptCaching -> PatchToolCalls -> UnicodeSecurityMiddleware -> SONA -> Witness -> ToolSanitizer -> HITL
///
/// HNSW is early in the pipeline to augment context before other middleware.
/// UnicodeSecurityMiddleware runs before SONA to sanitize inputs/outputs (C7).
/// SONA wraps model calls late to capture full request/response context.
pub fn build_default_pipeline(config: &PipelineConfig) -> MiddlewarePipeline {
    let mut middlewares: Vec<Box<dyn Middleware>> =
        vec![Box::new(todolist::TodoListMiddleware::new())];

    // HNSW early for context augmentation (ADR-103 B6)
    if config.enable_hnsw {
        let hnsw_config = config
            .hnsw_config
            .clone()
            .unwrap_or_else(hnsw::HnswMiddlewareConfig::default);
        middlewares.push(Box::new(hnsw::HnswMiddleware::new(hnsw_config)));
    }

    if let Some(sources) = &config.memory_sources {
        middlewares.push(Box::new(memory::MemoryMiddleware::new(sources.clone())));
    }

    if let Some(sources) = &config.skill_sources {
        middlewares.push(Box::new(skills::SkillsMiddleware::new(sources.clone())));
    }

    middlewares.push(Box::new(filesystem::FilesystemMiddleware::new()));
    middlewares.push(Box::new(subagents::SubAgentMiddleware::new()));
    middlewares.push(Box::new(summarization::SummarizationMiddleware::new(
        100_000, 0.85, 0.10,
    )));
    middlewares.push(Box::new(prompt_caching::PromptCachingMiddleware::new()));
    middlewares.push(Box::new(patch_tool_calls::PatchToolCallsMiddleware::new()));

    // Unicode security before SONA to sanitize inputs (C7 - CVE mitigation)
    if config.enable_unicode_security {
        let unicode_config = config
            .unicode_security_config
            .clone()
            .unwrap_or_else(UnicodeSecurityConfig::strict);
        middlewares.push(Box::new(
            UnicodeSecurityMiddleware::new(unicode_config)
                .with_input_sanitization(true)
                .with_output_sanitization(false), // Log only by default
        ));
    }

    // SONA late to capture full context (ADR-103 B5)
    if config.enable_sona {
        let sona_config = config
            .sona_config
            .clone()
            .unwrap_or_else(sona::SonaMiddlewareConfig::default);
        middlewares.push(Box::new(sona::SonaMiddleware::new(sona_config)));
    }

    if config.enable_witness {
        middlewares.push(Box::new(witness::WitnessMiddleware::new()));
    }

    middlewares.push(Box::new(
        tool_sanitizer::ToolResultSanitizerMiddleware::new(),
    ));

    if let Some(patterns) = &config.interrupt_on {
        middlewares.push(Box::new(hitl::HumanInTheLoopMiddleware::new(
            patterns.clone(),
        )));
    }

    MiddlewarePipeline::new(middlewares)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A passthrough test handler.
    struct EchoHandler;
    impl ModelHandler for EchoHandler {
        fn call(&self, request: ModelRequest) -> ModelResponse {
            ModelResponse::text(format!("echo: {}", request.messages.len()))
        }
    }

    /// A test middleware that prepends to system message.
    struct PrependMiddleware {
        text: String,
    }
    impl PrependMiddleware {
        fn new(text: &str) -> Self {
            Self {
                text: text.to_string(),
            }
        }
    }
    #[async_trait]
    impl Middleware for PrependMiddleware {
        fn name(&self) -> &str {
            "prepend"
        }
        fn wrap_model_call(
            &self,
            request: ModelRequest,
            handler: &dyn ModelHandler,
        ) -> ModelResponse {
            let new_sys = append_to_system_message(&request.system_message, &self.text);
            handler.call(request.with_system(new_sys))
        }
    }

    /// A test middleware that injects a tool.
    struct ToolInjector;
    struct DummyTool;
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            "dummy_tool"
        }
        fn description(&self) -> &str {
            "A dummy tool"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        fn invoke(&self, _args: serde_json::Value) -> Result<String, String> {
            Ok("ok".into())
        }
    }
    #[async_trait]
    impl Middleware for ToolInjector {
        fn name(&self) -> &str {
            "tool_injector"
        }
        fn tools(&self) -> Vec<Box<dyn Tool>> {
            vec![Box::new(DummyTool)]
        }
    }

    #[test]
    fn test_message_constructors() {
        let sys = Message::system("sys");
        assert_eq!(sys.role, Role::System);
        let usr = Message::user("hi");
        assert_eq!(usr.role, Role::User);
        let asst = Message::assistant("resp");
        assert_eq!(asst.role, Role::Assistant);
        let tool = Message::tool("result", "tc-1", "my_tool");
        assert_eq!(tool.role, Role::Tool);
        assert_eq!(tool.tool_call_id.as_deref(), Some("tc-1"));
    }

    #[test]
    fn test_model_request_with_system() {
        let req = ModelRequest::new(vec![Message::user("hi")]);
        assert!(req.system_message.is_none());
        let req2 = req.with_system(Some("system".into()));
        assert_eq!(req2.system_message, Some("system".into()));
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline = MiddlewarePipeline::empty();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
        assert!(pipeline.collect_tools().is_empty());
    }

    #[test]
    fn test_pipeline_ordering() {
        let mut pipeline = MiddlewarePipeline::empty();
        pipeline.push(Box::new(PrependMiddleware::new("first")));
        pipeline.push(Box::new(PrependMiddleware::new("second")));
        let names = pipeline.names();
        assert_eq!(names, vec!["prepend", "prepend"]);
        assert_eq!(pipeline.len(), 2);
    }

    #[test]
    fn test_pipeline_wrap_model_call_chaining() {
        // Two prepend middlewares should chain: first wraps second wraps handler
        let pipeline = MiddlewarePipeline::new(vec![
            Box::new(PrependMiddleware::new("A")),
            Box::new(PrependMiddleware::new("B")),
        ]);

        let request = ModelRequest::new(vec![Message::user("hi")]).with_system(Some("base".into()));

        // Track what system message the handler receives
        struct CaptureHandler;
        impl ModelHandler for CaptureHandler {
            fn call(&self, request: ModelRequest) -> ModelResponse {
                ModelResponse::text(request.system_message.unwrap_or_default())
            }
        }

        let response = pipeline.run_wrap_model_call(request, &CaptureHandler);
        // First middleware appends A, second appends B
        assert!(response.message.content.contains("A"));
        assert!(response.message.content.contains("B"));
        assert!(response.message.content.contains("base"));
    }

    #[test]
    fn test_pipeline_tool_collection() {
        let pipeline =
            MiddlewarePipeline::new(vec![Box::new(ToolInjector), Box::new(ToolInjector)]);
        let tools = pipeline.collect_tools();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name(), "dummy_tool");
    }

    #[tokio::test]
    async fn test_pipeline_run_full() {
        let pipeline = MiddlewarePipeline::new(vec![
            Box::new(PrependMiddleware::new("injected")),
            Box::new(ToolInjector),
        ]);

        let mut state = AgentState::default();
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        let request = ModelRequest::new(vec![Message::user("test")]);

        let response = pipeline
            .run(&mut state, &runtime, &config, request, &EchoHandler)
            .await;
        assert!(response.message.content.contains("echo"));
    }

    #[test]
    fn test_build_default_pipeline_minimal() {
        let config = PipelineConfig::default();
        let pipeline = build_default_pipeline(&config);
        // Should have: todo, filesystem, subagent, summarization, prompt_caching,
        // patch_tool_calls, tool_sanitizer = 7
        assert!(pipeline.len() >= 7);
    }

    #[test]
    fn test_build_default_pipeline_full() {
        let config = PipelineConfig {
            memory_sources: Some(vec!["AGENTS.md".into()]),
            skill_sources: Some(vec![".skills".into()]),
            interrupt_on: Some(vec!["execute".into()]),
            enable_witness: true,
            enable_sona: false,
            enable_hnsw: false,
            enable_unicode_security: false,
            sona_config: None,
            hnsw_config: None,
            unicode_security_config: None,
        };
        let pipeline = build_default_pipeline(&config);
        // todo + memory + skills + filesystem + subagent + summarization + prompt_caching
        // + patch_tool_calls + witness + tool_sanitizer + hitl = 11
        assert_eq!(pipeline.len(), 11);
    }

    #[test]
    fn test_agent_state_default() {
        let state = AgentState::default();
        assert!(state.messages.is_empty());
        assert!(state.todos.is_empty());
        assert!(state.extensions.is_empty());
    }

    #[test]
    fn test_todo_status_default() {
        let status = TodoStatus::default();
        assert_eq!(status, TodoStatus::Pending);
    }

    #[test]
    fn test_model_response_text() {
        let resp = ModelResponse::text("hello");
        assert_eq!(resp.message.content, "hello");
        assert_eq!(resp.message.role, Role::Assistant);
        assert!(resp.tool_calls.is_empty());
    }

    #[test]
    fn test_runtime_default() {
        let rt = Runtime::default();
        assert_eq!(rt.context, serde_json::Value::Null);
    }
}
