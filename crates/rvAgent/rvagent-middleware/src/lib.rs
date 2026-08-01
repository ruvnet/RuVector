//! rvAgent middleware pipeline — core trait, types, and concrete middleware implementations.
//!
//! Provides the `Middleware` trait and `MiddlewarePipeline` for composing middleware
//! in the DeepAgents architecture (ADR-095, ADR-103).
//!
//! ## Unified type system (P0.1)
//!
//! All conversation/state types (`Message`, `ToolCall`, `AgentState`, `TodoItem`,
//! `TodoStatus`, `RunnableConfig`, `ToolDefinition`) are the canonical
//! `rvagent-core` definitions, re-exported here for convenience.
//!
//! ## Async model-call chain (P0.3)
//!
//! `Middleware::wrap_model_call` and `ModelHandler::call` are async, so a real
//! HTTP model call can run inside the pipeline. Use [`PipelineModel`] to wrap
//! any `rvagent_core::models::ChatModel` with a pipeline.
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
pub mod pipeline;
pub mod pipeline_model;
pub mod prompt_caching;
pub mod retry;
pub mod rvf_manifest;
pub mod skills;
pub mod sona;
pub mod subagents;
pub mod summarization;
pub mod todolist;
pub mod tool_sanitizer;
pub mod types;
pub mod unicode_security;
pub mod unicode_security_middleware;
pub mod utils;
pub mod witness;

use async_trait::async_trait;
use std::fmt;

// Re-exports
pub use pipeline::MiddlewarePipeline;
pub use pipeline_model::PipelineModel;
pub use types::{
    json_extension, AgentState, AgentStateUpdate, AiMessage, CacheControl, FileData, HumanMessage,
    Message, ModelRequest, ModelResponse, RunnableConfig, Runtime, SystemMessage, TodoItem,
    TodoStatus, ToolCall, ToolDefinition, ToolMessage, Usage,
};
pub use unicode_security::{UnicodeIssue, UnicodeSecurityChecker, UnicodeSecurityConfig};
pub use unicode_security_middleware::UnicodeSecurityMiddleware;
pub use utils::{append_to_system_message, SystemPromptBuilder};

// ---------------------------------------------------------------------------
// Model handler trait (async — P0.3)
// ---------------------------------------------------------------------------

/// Async model handler — the "next" link called by `wrap_model_call`.
#[async_trait]
pub trait ModelHandler: Send + Sync {
    async fn call(&self, request: ModelRequest) -> ModelResponse;
}

// ---------------------------------------------------------------------------
// Tool trait (aligned with rvagent_core::models::ToolDefinition)
// ---------------------------------------------------------------------------

/// Tool trait — tools injected by middleware.
///
/// Schema exposure aligns with `rvagent_core::models::ToolDefinition`
/// (`input_schema`); `definition()` produces the canonical form.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema for the tool's arguments (ToolDefinition::input_schema).
    fn input_schema(&self) -> serde_json::Value;

    /// Canonical schema form advertised to models.
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }

    /// Execute the tool asynchronously.
    async fn invoke(&self, args: serde_json::Value) -> Result<String, String>;
}

impl fmt::Debug for dyn Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tool").field("name", &self.name()).finish()
    }
}

// ---------------------------------------------------------------------------
// Middleware trait (ADR-095) — async hooks (P0.3)
// ---------------------------------------------------------------------------

/// Core middleware trait — mirrors Python's `AgentMiddleware`.
///
/// Each method has a default no-op implementation, so concrete middleware
/// only needs to override the hooks it uses. The former sync/async duplicate
/// hook pairs (`before_agent`/`abefore_agent`, `wrap_model_call`/
/// `awrap_model_call`) are merged into single async hooks.
#[async_trait]
pub trait Middleware: Send + Sync {
    /// Called before agent execution. Returns state update or None.
    async fn before_agent(
        &self,
        _state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        None
    }

    /// Wrap a model call — intercept request/response. Async so real HTTP
    /// calls can run inside the chain.
    async fn wrap_model_call(
        &self,
        request: ModelRequest,
        handler: &dyn ModelHandler,
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
// Name-based middleware resolution (CLI wiring)
// ---------------------------------------------------------------------------

/// Resolve a middleware name (as used in `RvAgentConfig::middleware` /
/// the CLI `DEFAULT_MIDDLEWARE` list) into a middleware instance.
///
/// Returns `None` for unknown names — callers should warn and skip.
pub fn middleware_by_name(name: &str) -> Option<Box<dyn Middleware>> {
    match name {
        "todo" | "todos" | "todolist" => Some(Box::new(todolist::TodoListMiddleware::new())),
        "memory" => Some(Box::new(memory::MemoryMiddleware::new(vec![
            "AGENTS.md".into()
        ]))),
        "skills" => Some(Box::new(skills::SkillsMiddleware::new(vec![
            ".skills".into()
        ]))),
        "filesystem" => Some(Box::new(filesystem::FilesystemMiddleware::new())),
        "subagent" | "subagents" => Some(Box::new(subagents::SubAgentMiddleware::new())),
        "summarization" => Some(Box::new(summarization::SummarizationMiddleware::new(
            100_000, 0.85, 0.10,
        ))),
        "prompt_caching" => Some(Box::new(prompt_caching::PromptCachingMiddleware::new())),
        "patch_tool_calls" => Some(Box::new(patch_tool_calls::PatchToolCallsMiddleware::new())),
        "witness" => Some(Box::new(witness::WitnessMiddleware::new())),
        "tool_result_sanitizer" | "tool_sanitizer" => Some(Box::new(
            tool_sanitizer::ToolResultSanitizerMiddleware::new(),
        )),
        // HITL with no interrupt patterns configured never interrupts.
        "hitl" => Some(Box::new(hitl::HumanInTheLoopMiddleware::new(Vec::new()))),
        "retry" => Some(Box::new(retry::RetryMiddleware::default())),
        "hnsw" => Some(Box::new(hnsw::HnswMiddleware::default_config())),
        "sona" => Some(Box::new(sona::SonaMiddleware::default_config())),
        "unicode_security" => Some(Box::new(UnicodeSecurityMiddleware::strict())),
        "mcp_bridge" => Some(Box::new(mcp_bridge::McpBridgeMiddleware::new())),
        _ => None,
    }
}

/// Build a pipeline from an ordered list of middleware names.
/// Unknown names are logged (warn) and skipped.
pub fn build_pipeline_from_names<S: AsRef<str>>(names: &[S]) -> MiddlewarePipeline {
    let mut pipeline = MiddlewarePipeline::empty();
    for name in names {
        match middleware_by_name(name.as_ref()) {
            Some(mw) => pipeline.push(mw),
            None => tracing::warn!(
                "unknown middleware '{}' — skipping (see rvagent_middleware::middleware_by_name)",
                name.as_ref()
            ),
        }
    }
    pipeline
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A passthrough test handler.
    struct EchoHandler;

    #[async_trait]
    impl ModelHandler for EchoHandler {
        async fn call(&self, request: ModelRequest) -> ModelResponse {
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
        async fn wrap_model_call(
            &self,
            request: ModelRequest,
            handler: &dyn ModelHandler,
        ) -> ModelResponse {
            let new_sys = append_to_system_message(&request.system_message, &self.text);
            handler.call(request.with_system(new_sys)).await
        }
    }

    /// A test middleware that injects a tool.
    struct ToolInjector;
    struct DummyTool;

    #[async_trait]
    impl Tool for DummyTool {
        fn name(&self) -> &str {
            "dummy_tool"
        }
        fn description(&self) -> &str {
            "A dummy tool"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn invoke(&self, _args: serde_json::Value) -> Result<String, String> {
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

    #[tokio::test]
    async fn test_pipeline_wrap_model_call_chaining() {
        // Two prepend middlewares should chain: first wraps second wraps handler
        let pipeline = MiddlewarePipeline::new(vec![
            Box::new(PrependMiddleware::new("A")),
            Box::new(PrependMiddleware::new("B")),
        ]);

        let request =
            ModelRequest::new(vec![Message::human("hi")]).with_system(Some("base".into()));

        // Track what system message the handler receives
        struct CaptureHandler;

        #[async_trait]
        impl ModelHandler for CaptureHandler {
            async fn call(&self, request: ModelRequest) -> ModelResponse {
                ModelResponse::text(request.system_message.unwrap_or_default())
            }
        }

        let response = pipeline.run_wrap_model_call(request, &CaptureHandler).await;
        // First middleware appends A, second appends B
        assert!(response.content().contains("A"));
        assert!(response.content().contains("B"));
        assert!(response.content().contains("base"));
    }

    #[test]
    fn test_pipeline_tool_collection() {
        let pipeline =
            MiddlewarePipeline::new(vec![Box::new(ToolInjector), Box::new(ToolInjector)]);
        let tools = pipeline.collect_tools();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name(), "dummy_tool");
        let def = tools[0].definition();
        assert_eq!(def.name, "dummy_tool");
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
        let request = ModelRequest::new(vec![Message::human("test")]);

        let response = pipeline
            .run(&mut state, &runtime, &config, request, &EchoHandler)
            .await;
        assert!(response.content().contains("echo"));
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
    }

    #[test]
    fn test_middleware_by_name_known() {
        for name in [
            "todo",
            "memory",
            "skills",
            "filesystem",
            "subagent",
            "summarization",
            "prompt_caching",
            "patch_tool_calls",
            "witness",
            "tool_result_sanitizer",
            "hitl",
            "retry",
        ] {
            assert!(middleware_by_name(name).is_some(), "should resolve {name}");
        }
    }

    #[test]
    fn test_middleware_by_name_unknown() {
        assert!(middleware_by_name("does_not_exist").is_none());
    }

    #[test]
    fn test_build_pipeline_from_names_skips_unknown() {
        let pipeline = build_pipeline_from_names(&["todo", "bogus", "filesystem"]);
        assert_eq!(pipeline.len(), 2);
    }
}
