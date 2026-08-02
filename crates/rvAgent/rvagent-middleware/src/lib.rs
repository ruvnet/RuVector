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
//! - [`hnsw`] — experimental in-process retrieval index (hash-embedding placeholder)

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
    /// Enable LLM summarization as the compaction strategy (ADR-274).
    ///
    /// **Off by default.** Observation masking in the agent loop is the default
    /// strategy; measured comparisons put simple masking at or above LLM
    /// summarization on solve rate at roughly half the cost, and show
    /// summarization inflating trajectories 13–15% by destroying the stopping
    /// signals an agent uses to notice it has finished. Enable only as a
    /// deliberate fallback, and supply a preservation rubric when you do — the
    /// rubric is the load-bearing part, not the summarizer.
    pub enable_summarization: bool,
    /// Custom SONA configuration.
    pub sona_config: Option<sona::SonaMiddlewareConfig>,
    /// Custom HNSW configuration.
    pub hnsw_config: Option<hnsw::HnswMiddlewareConfig>,
    /// Custom Unicode security configuration.
    pub unicode_security_config: Option<UnicodeSecurityConfig>,
}

/// Build the default middleware pipeline (ADR-095 ordering, amended by ADR-274):
/// Todo -> HNSW -> Memory -> Skills -> Filesystem -> SubAgent
/// -> [Summarization, opt-in] -> PromptCaching -> PatchToolCalls
/// -> UnicodeSecurityMiddleware -> SONA -> Witness -> ToolSanitizer -> HITL
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

    // Summarization is OFF by default (ADR-274). Observation masking in the
    // agent loop is the default compaction strategy; this remains available as
    // an explicit fallback. Trigger lowered from 0.85 to 0.75 when enabled:
    // context degrades well before the nominal limit, so compacting at 85%
    // leaves too little headroom to be selective rather than desperate.
    if config.enable_summarization {
        middlewares.push(Box::new(summarization::SummarizationMiddleware::new(
            100_000, 0.75, 0.10,
        )));
    }

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

    // Unconditional, and via the same fallback the by-name path uses: an
    // unconfigured caller (the ACP agent reaches this with a default config)
    // must not silently get an agent with no approval gate at all. An explicit
    // `Some(vec![])` still opts out — it matches nothing.
    middlewares.push(Box::new(hitl::HumanInTheLoopMiddleware::new(
        interrupt_patterns(config),
    )));

    MiddlewarePipeline::new(middlewares)
}

// ---------------------------------------------------------------------------
// Name-based middleware resolution (CLI wiring)
// ---------------------------------------------------------------------------

/// Tool-name patterns gated by HITL when no `interrupt_on` is configured.
///
/// Every pipeline builder falls back to this set: an approval gate that
/// approves everything — or a pipeline with no gate at all — is worse than a
/// real one, because it reads as protection that isn't there. Shell execution
/// and file mutation are the irreversible operations, so those are what a
/// caller who never configured `interrupt_on` gets gated on.
///
/// `write_todos` is deliberately absent — gating the agent's own scratchpad
/// would interrupt every turn without protecting anything.
pub const DEFAULT_INTERRUPT_PATTERNS: &[&str] = &[
    // Shell / arbitrary command execution.
    "execute",
    "execute_command",
    "shell*",
    "bash*",
    "run_command*",
    // File mutation.
    "write_file",
    "edit_file",
    "apply_patch",
    "delete_file",
    "move_file",
];

fn default_interrupt_patterns() -> Vec<String> {
    DEFAULT_INTERRUPT_PATTERNS
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

/// The HITL patterns `config` asks for, or the conservative built-in set.
///
/// Both pipeline builders resolve gating through here. The two paths having
/// separate fallbacks is what produced an ACP agent with no gate at all while
/// the CLI had one.
fn interrupt_patterns(config: &PipelineConfig) -> Vec<String> {
    config
        .interrupt_on
        .clone()
        .unwrap_or_else(default_interrupt_patterns)
}

/// A middleware name that could not be resolved to an implementation.
///
/// Unknown names are fatal rather than skipped: a typo in the middleware list
/// silently drops whatever that entry was supposed to do, and the entries most
/// worth typo-ing are the security ones (`hitl`, `unicode_security`,
/// `tool_result_sanitizer`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownMiddlewareError {
    pub name: String,
}

impl fmt::Display for UnknownMiddlewareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown middleware '{}' (see rvagent_middleware::middleware_by_name for valid names)",
            self.name
        )
    }
}

impl std::error::Error for UnknownMiddlewareError {}

/// Resolve a middleware name (as used in `RvAgentConfig::middleware` /
/// the CLI `DEFAULT_MIDDLEWARE` list) into a middleware instance.
///
/// `config` supplies the same settings [`build_default_pipeline`] uses, so the
/// two construction paths produce equivalently configured middleware.
///
/// Returns `Err` for unknown names.
pub fn middleware_by_name(
    name: &str,
    config: &PipelineConfig,
) -> std::result::Result<Box<dyn Middleware>, UnknownMiddlewareError> {
    let mw: Box<dyn Middleware> = match name {
        "todo" | "todos" | "todolist" => Box::new(todolist::TodoListMiddleware::new()),
        "memory" => Box::new(memory::MemoryMiddleware::new(
            config
                .memory_sources
                .clone()
                .unwrap_or_else(|| vec!["AGENTS.md".into()]),
        )),
        "skills" => Box::new(skills::SkillsMiddleware::new(
            config
                .skill_sources
                .clone()
                .unwrap_or_else(|| vec![".skills".into()]),
        )),
        "filesystem" => Box::new(filesystem::FilesystemMiddleware::new()),
        "subagent" | "subagents" => Box::new(subagents::SubAgentMiddleware::new()),
        "summarization" => Box::new(summarization::SummarizationMiddleware::new(
            100_000, 0.75, 0.10,
        )),
        "prompt_caching" => Box::new(prompt_caching::PromptCachingMiddleware::new()),
        "patch_tool_calls" => Box::new(patch_tool_calls::PatchToolCallsMiddleware::new()),
        "witness" => Box::new(witness::WitnessMiddleware::new()),
        "tool_result_sanitizer" | "tool_sanitizer" => {
            Box::new(tool_sanitizer::ToolResultSanitizerMiddleware::new())
        }
        "hitl" => Box::new(hitl::HumanInTheLoopMiddleware::new(interrupt_patterns(
            config,
        ))),
        "retry" => Box::new(retry::RetryMiddleware::default()),
        "hnsw" => Box::new(hnsw::HnswMiddleware::new(
            config.hnsw_config.clone().unwrap_or_default(),
        )),
        "sona" => Box::new(sona::SonaMiddleware::new(
            config.sona_config.clone().unwrap_or_default(),
        )),
        "unicode_security" => Box::new(
            UnicodeSecurityMiddleware::new(
                config
                    .unicode_security_config
                    .clone()
                    .unwrap_or_else(UnicodeSecurityConfig::strict),
            )
            .with_input_sanitization(true)
            .with_output_sanitization(false), // Log only by default
        ),
        "mcp_bridge" => Box::new(mcp_bridge::McpBridgeMiddleware::new()),
        _ => {
            return Err(UnknownMiddlewareError {
                name: name.to_string(),
            })
        }
    };
    Ok(mw)
}

/// Build a pipeline from an ordered list of middleware names.
///
/// An unknown name is an error, not a skip — see [`UnknownMiddlewareError`].
pub fn build_pipeline_from_names<S: AsRef<str>>(
    names: &[S],
    config: &PipelineConfig,
) -> std::result::Result<MiddlewarePipeline, UnknownMiddlewareError> {
    let mut pipeline = MiddlewarePipeline::empty();
    for name in names {
        pipeline.push(middleware_by_name(name.as_ref(), config)?);
    }
    Ok(pipeline)
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
        // todo, filesystem, subagent, prompt_caching, patch_tool_calls,
        // tool_sanitizer, hitl = 7. Summarization is opt-in (ADR-274).
        assert!(pipeline.len() >= 7);
        assert!(pipeline.names().contains(&"hitl"));
    }

    #[tokio::test]
    async fn test_build_default_pipeline_gates_execute_without_config() {
        // rvagent-acp builds this pipeline with a default config whenever the
        // agent config lists no middleware; it must not come out ungated.
        let pipeline = build_default_pipeline(&PipelineConfig::default());
        let response = pipeline
            .run_wrap_model_call(
                ModelRequest::new(vec![Message::human("go")]),
                &DangerousCallHandler,
            )
            .await;

        let survived: Vec<&str> = response
            .tool_calls
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(survived, vec!["read_file"]);
        assert!(response.content().contains("[HITL]"));
    }

    #[tokio::test]
    async fn test_build_default_pipeline_explicit_empty_interrupt_on_disables_gating() {
        // The documented opt-out for unattended runs must keep working.
        let pipeline = build_default_pipeline(&PipelineConfig {
            interrupt_on: Some(Vec::new()),
            ..PipelineConfig::default()
        });
        let response = pipeline
            .run_wrap_model_call(
                ModelRequest::new(vec![Message::human("go")]),
                &DangerousCallHandler,
            )
            .await;

        assert_eq!(response.tool_calls.len(), 3);
        assert!(!response.content().contains("[HITL]"));
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
            enable_summarization: false,
            sona_config: None,
            hnsw_config: None,
            unicode_security_config: None,
        };
        let pipeline = build_default_pipeline(&config);
        // todo + memory + skills + filesystem + subagent + prompt_caching
        // + patch_tool_calls + witness + tool_sanitizer + hitl = 10.
        // Summarization is absent by default (ADR-274).
        assert_eq!(pipeline.len(), 10);
    }

    #[test]
    fn test_summarization_is_absent_by_default_and_available_opt_in() {
        // The default path must not carry summarization: masking in the agent
        // loop is the decided strategy (ADR-274).
        let default_len = build_default_pipeline(&PipelineConfig::default()).len();
        let opted_in = build_default_pipeline(&PipelineConfig {
            enable_summarization: true,
            ..PipelineConfig::default()
        });
        assert_eq!(
            opted_in.len(),
            default_len + 1,
            "enabling summarization must add exactly one middleware"
        );
        // And it must still be constructible by name, so the fallback is real.
        assert!(middleware_by_name("summarization", &PipelineConfig::default()).is_ok());
    }

    #[test]
    fn test_agent_state_default() {
        let state = AgentState::default();
        assert!(state.messages.is_empty());
        assert!(state.todos.is_empty());
    }

    #[test]
    fn test_middleware_by_name_known() {
        let config = PipelineConfig::default();
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
            assert!(
                middleware_by_name(name, &config).is_ok(),
                "should resolve {name}"
            );
        }
    }

    #[test]
    fn test_middleware_by_name_unknown() {
        let err = middleware_by_name("does_not_exist", &PipelineConfig::default()).unwrap_err();
        assert_eq!(err.name, "does_not_exist");
    }

    #[test]
    fn test_build_pipeline_from_names_rejects_unknown() {
        // A typo must not silently drop a middleware — the ones most worth
        // typo-ing are the security ones.
        let err =
            build_pipeline_from_names(&["todo", "bogus", "filesystem"], &PipelineConfig::default())
                .err()
                .expect("an unknown middleware name must fail the build");
        assert_eq!(err.name, "bogus");
    }

    /// Returns one shell call and one read-only call, so an approval gate is
    /// observable by which calls survive.
    struct DangerousCallHandler;

    #[async_trait]
    impl ModelHandler for DangerousCallHandler {
        async fn call(&self, _request: ModelRequest) -> ModelResponse {
            let mut response = ModelResponse::text("");
            response.tool_calls = vec![
                ToolCall {
                    id: "c1".into(),
                    name: "execute".into(),
                    args: serde_json::json!({"command": "rm -rf /"}),
                },
                ToolCall {
                    id: "c2".into(),
                    name: "write_file".into(),
                    args: serde_json::json!({"path": "a.txt"}),
                },
                ToolCall {
                    id: "c3".into(),
                    name: "read_file".into(),
                    args: serde_json::json!({"path": "a.txt"}),
                },
            ];
            response
        }
    }

    #[tokio::test]
    async fn test_by_name_hitl_gates_shell_and_writes_without_config() {
        // The unconfigured by-name path is what the CLI runs; an approval gate
        // that approves everything would be worse than no gate at all.
        let pipeline = build_pipeline_from_names(&["hitl"], &PipelineConfig::default()).unwrap();
        let response = pipeline
            .run_wrap_model_call(
                ModelRequest::new(vec![Message::human("go")]),
                &DangerousCallHandler,
            )
            .await;

        let survived: Vec<&str> = response
            .tool_calls
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(survived, vec!["read_file"]);
        assert!(response.content().contains("[HITL]"));
    }

    #[tokio::test]
    async fn test_by_name_hitl_honours_configured_interrupt_on() {
        let config = PipelineConfig {
            interrupt_on: Some(vec!["read_file".into()]),
            ..PipelineConfig::default()
        };
        let pipeline = build_pipeline_from_names(&["hitl"], &config).unwrap();
        let response = pipeline
            .run_wrap_model_call(
                ModelRequest::new(vec![Message::human("go")]),
                &DangerousCallHandler,
            )
            .await;

        let survived: Vec<&str> = response
            .tool_calls
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        assert_eq!(survived, vec!["execute", "write_file"]);
    }
}
