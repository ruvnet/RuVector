//! Application core for the rvAgent CLI.
//!
//! `App` initializes configuration from CLI arguments, creates the backend
//! and middleware pipeline, builds the agent graph, and drives the run loop.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use tracing::{info, warn};

use rvagent_core::bootstrap::EnvironmentSnapshot;
use rvagent_core::config::{BackendConfig, MiddlewareConfig, RvAgentConfig, SecurityPolicy};
use rvagent_core::graph::{AgentGraph, ToolExecutor};
use rvagent_core::messages::{Message, ToolCall as CoreToolCall};
use rvagent_core::models::{resolve_model, ChatModel, ToolDefinition};
use rvagent_core::prompt::BASE_AGENT_PROMPT;
use rvagent_core::state::AgentState;

use rvagent_tools::Tool as _;

use crate::display;
use crate::mcp::McpRegistry;
use crate::session::{self, Session};
use crate::tui::Tui;

// ---------------------------------------------------------------------------
// Middleware names for the default pipeline (11 middlewares)
// ---------------------------------------------------------------------------

/// The full default middleware pipeline in execution order.
/// (ADR-103 B3 amended ordering)
const DEFAULT_MIDDLEWARE: &[&str] = &[
    "todo",
    "memory",
    "skills",
    "filesystem",
    "subagent",
    // "summarization" removed (ADR-274): observation masking in the agent loop
    // is the default compaction strategy. Still available opt-in via
    // PipelineConfig::enable_summarization.
    "prompt_caching",
    "patch_tool_calls",
    "witness",
    "tool_result_sanitizer",
    "hitl",
];

// ---------------------------------------------------------------------------
// StubModel — fallback when no API key is configured
// ---------------------------------------------------------------------------

/// A stub model that returns a helpful message when no API key is available.
///
/// Used as a fallback so the CLI can start and provide feedback to the user
/// even when credentials are not configured.
struct StubModel {
    model_name: String,
}

impl StubModel {
    fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
        }
    }
}

#[async_trait]
impl ChatModel for StubModel {
    async fn complete(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> rvagent_core::error::Result<Message> {
        Ok(Message::ai(format!(
            "No API key configured for model '{}'. \
             Set the appropriate environment variable (e.g. ANTHROPIC_API_KEY) \
             and restart rvAgent.",
            self.model_name
        )))
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> rvagent_core::error::Result<Vec<Message>> {
        let msg = self.complete(messages, tools).await?;
        Ok(vec![msg])
    }
}

// ---------------------------------------------------------------------------
// CliModel — enum wrapper for supported model backends
// ---------------------------------------------------------------------------

/// Enum wrapper for supported model backends.
/// This allows AgentGraph to work with multiple model types without trait objects.
enum CliModel {
    Stub(StubModel),
    Anthropic(rvagent_backends::anthropic::AnthropicClient),
    Gemini(rvagent_backends::gemini::GeminiClient),
}

#[async_trait]
impl ChatModel for CliModel {
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> rvagent_core::error::Result<Message> {
        match self {
            CliModel::Stub(m) => m.complete(messages, tools).await,
            CliModel::Anthropic(m) => m.complete(messages, tools).await,
            CliModel::Gemini(m) => m.complete(messages, tools).await,
        }
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> rvagent_core::error::Result<Vec<Message>> {
        match self {
            CliModel::Stub(m) => m.stream(messages, tools).await,
            CliModel::Anthropic(m) => m.stream(messages, tools).await,
            CliModel::Gemini(m) => m.stream(messages, tools).await,
        }
    }
}

// ---------------------------------------------------------------------------
// CliToolExecutor — dispatches tool calls to rvagent-tools
// ---------------------------------------------------------------------------

/// Tool executor that dispatches tool calls to the built-in tool registry
/// from `rvagent_tools`.
struct CliToolExecutor {
    tools: Vec<rvagent_tools::AnyTool>,
    backend: rvagent_tools::BackendRef,
}

impl CliToolExecutor {
    fn new(cwd: &Path) -> Self {
        // Confined to `cwd`: tool-supplied paths cannot escape the workspace.
        let backend: rvagent_tools::BackendRef = Arc::new(rvagent_tools::LocalFsBackend::new(cwd));
        Self {
            tools: rvagent_tools::builtin_tools(),
            backend,
        }
    }
}

#[async_trait]
impl ToolExecutor for CliToolExecutor {
    async fn execute(
        &self,
        call: &CoreToolCall,
        _state: &AgentState,
    ) -> rvagent_core::error::Result<String> {
        let runtime = rvagent_tools::ToolRuntime::new(Arc::clone(&self.backend));
        match rvagent_tools::resolve_tool(&call.name, &self.tools) {
            Some(tool) => {
                let result = tool.invoke(call.args.clone(), &runtime);
                Ok(result.to_string())
            }
            None => Ok(format!("Error: tool '{}' not found", call.name)),
        }
    }

    fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.parameters_schema(),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

/// Top-level application state for the rvAgent CLI.
pub struct App {
    /// Agent configuration.
    config: RvAgentConfig,
    /// Current session.
    session: Session,
    /// Working directory.
    cwd: PathBuf,
    /// System prompt used to initialize agent state.
    system_prompt: String,
    /// MCP tool registry for external tool servers (wired when MCP transport is implemented).
    #[allow(dead_code)]
    mcp_registry: McpRegistry,
}

impl App {
    /// Create a new `App` from CLI arguments.
    ///
    /// If `resume_id` is provided, the session is loaded from disk;
    /// otherwise a fresh session is created.
    pub fn new(model: &str, cwd: &Path, resume_id: Option<&str>) -> Result<Self> {
        let model_config = resolve_model(model);
        info!(
            provider = ?model_config.provider,
            model = %model_config.model_id,
            "resolved model"
        );

        // Build middleware pipeline config.
        let middleware: Vec<MiddlewareConfig> = DEFAULT_MIDDLEWARE
            .iter()
            .map(|name| MiddlewareConfig {
                name: name.to_string(),
                settings: serde_json::Value::Null,
            })
            .collect();

        // Backend: LocalShell with security defaults.
        let backend = BackendConfig {
            backend_type: "local_shell".into(),
            cwd: Some(cwd.to_string_lossy().into_owned()),
            settings: serde_json::Value::Null,
        };

        let config = RvAgentConfig {
            model: model.to_string(),
            name: Some("rvagent-cli".into()),
            middleware,
            backend,
            security_policy: SecurityPolicy::default(),
            ..Default::default()
        };

        // Resume or create session.
        let session = match resume_id {
            Some(id) => {
                info!(session_id = %id, "resuming session");
                session::load_session(id)
                    .with_context(|| format!("failed to resume session {}", id))?
            }
            None => Session::new(model),
        };

        // Environment bootstrap (ADR-273 §3.5): hand the agent the workspace
        // facts up front so it does not spend its first turns discovering
        // them. Filesystem-only, so this costs nothing measurable.
        let snapshot =
            EnvironmentSnapshot::collect(cwd, &rvagent_core::bootstrap::BootstrapConfig::default());
        let system_prompt = snapshot.augment_prompt(BASE_AGENT_PROMPT);

        Ok(Self {
            config,
            session,
            cwd: cwd.to_path_buf(),
            system_prompt,
            mcp_registry: McpRegistry::new(),
        })
    }

    /// Run a single prompt (non-interactive mode) and exit.
    pub async fn run_once(&mut self, prompt: &str) -> Result<()> {
        self.session.push_message(Message::human(prompt));

        let mut state = AgentState::with_system_message(&self.system_prompt);
        // Replay session messages into state.
        for msg in &self.session.messages {
            state.push_message(msg.clone());
        }

        let response = self.invoke_agent(&state).await?;

        self.session.push_message(response.clone());
        display::print_assistant_message(&response);

        // Persist session.
        session::save_session(&self.session)?;
        Ok(())
    }

    /// Run the interactive TUI loop.
    pub async fn run_interactive(&mut self) -> Result<()> {
        let mut tui = Tui::new(&self.config.model, &self.session.id)?;

        // Show existing messages if resuming.
        for msg in &self.session.messages {
            tui.add_message(msg);
        }

        loop {
            match tui.next_event().await? {
                TuiEvent::Input(text) => {
                    if text.trim().is_empty() {
                        continue;
                    }

                    // Check for quit commands.
                    let lower = text.trim().to_lowercase();
                    if lower == "/quit" || lower == "/exit" || lower == "/q" {
                        break;
                    }

                    self.session.push_message(Message::human(&text));
                    tui.add_message(&Message::human(&text));

                    tui.set_status("Thinking...");
                    let mut state = AgentState::with_system_message(&self.system_prompt);
                    for msg in &self.session.messages {
                        state.push_message(msg.clone());
                    }
                    let response = self.invoke_agent(&state).await?;

                    self.session.push_message(response.clone());
                    tui.add_message(&response);
                    tui.set_status("Ready");

                    // Auto-save after each exchange.
                    session::save_session(&self.session)?;
                }
                TuiEvent::Quit => break,
                TuiEvent::Resize => {
                    tui.redraw()?;
                }
            }
        }

        tui.shutdown()?;
        Ok(())
    }

    /// Invoke the agent pipeline with the given state.
    ///
    /// Creates the appropriate model (real Anthropic client or stub) and
    /// tool executor, wraps the model in the configured middleware pipeline
    /// (`PipelineModel`), builds an `AgentGraph`, and runs it to completion.
    /// Returns the final AI message from the completed state.
    async fn invoke_agent(&self, initial_state: &AgentState) -> Result<Message> {
        info!(
            messages = initial_state.message_count(),
            model = %self.config.model,
            "invoking agent"
        );

        let tool_executor = CliToolExecutor::new(&self.cwd);

        // Check if the appropriate API key is available.
        let model_config = resolve_model(&self.config.model);
        let has_api_key = match &model_config.api_key_source {
            rvagent_core::models::ApiKeySource::Env(var) => std::env::var(var).is_ok(),
            rvagent_core::models::ApiKeySource::File(path) => std::path::Path::new(path).exists(),
            rvagent_core::models::ApiKeySource::None => false,
        };

        // Use StubModel when no API key is configured.
        // When API key is available, use the real AnthropicClient.
        let model: CliModel = if has_api_key {
            match &model_config.provider {
                rvagent_core::models::Provider::Anthropic => {
                    info!(
                        provider = ?model_config.provider,
                        model_id = ?model_config.model_id,
                        "Using AnthropicClient with API key"
                    );
                    match rvagent_backends::anthropic::AnthropicClient::new(model_config.clone()) {
                        Ok(client) => CliModel::Anthropic(client),
                        Err(e) => {
                            warn!("Failed to create AnthropicClient: {e}; falling back to stub");
                            CliModel::Stub(StubModel::new(&format!(
                                "{} (client error: {})",
                                self.config.model, e
                            )))
                        }
                    }
                }
                rvagent_core::models::Provider::Google => {
                    info!(
                        provider = ?model_config.provider,
                        model_id = ?model_config.model_id,
                        "Using GeminiClient with API key"
                    );
                    match rvagent_backends::gemini::GeminiClient::new(model_config.clone()) {
                        Ok(client) => CliModel::Gemini(client),
                        Err(e) => {
                            warn!("Failed to create GeminiClient: {e}; falling back to stub");
                            CliModel::Stub(StubModel::new(&format!(
                                "{} (client error: {})",
                                self.config.model, e
                            )))
                        }
                    }
                }
                _ => {
                    info!(
                        provider = ?model_config.provider,
                        "Provider not yet implemented; using stub"
                    );
                    CliModel::Stub(StubModel::new(&self.config.model))
                }
            }
        } else {
            CliModel::Stub(StubModel::new(&self.config.model))
        };

        // Wire the middleware pipeline (P0.3): resolve the configured
        // middleware names (DEFAULT_MIDDLEWARE) into instances — an unknown
        // name is fatal — and run all model calls through it.
        //
        // The pipeline config carries the settings the middleware need to be
        // built correctly; leaving `interrupt_on` unset gives HITL its
        // conservative built-in gate rather than an empty (approve-everything)
        // pattern list. The CLI has no interactive approval prompt yet, so
        // gated calls fail closed; RVAGENT_AUTO_APPROVE=1 is the explicit,
        // logged opt-out for unattended use.
        let middleware_names: Vec<&str> = self
            .config
            .middleware
            .iter()
            .map(|m| m.name.as_str())
            .collect();
        let mut pipeline_config = rvagent_middleware::PipelineConfig::default();
        if matches!(
            std::env::var("RVAGENT_AUTO_APPROVE").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        ) {
            // Straight to stderr, not just tracing: the TUI installs no
            // subscriber and the non-TUI default is ERROR-only, so a `warn!`
            // here is invisible in exactly the modes people run. A security
            // downgrade the operator cannot see is one they cannot revoke.
            eprintln!(
                "warning: RVAGENT_AUTO_APPROVE set — HITL approval gate disabled; \
                 all tool calls (including shell execution and file writes) run unattended"
            );
            warn!("RVAGENT_AUTO_APPROVE set: HITL approval gate disabled; all tool calls run unattended");
            pipeline_config.interrupt_on = Some(Vec::new());
        }
        let pipeline = Arc::new(
            rvagent_middleware::build_pipeline_from_names(&middleware_names, &pipeline_config)
                .context("failed to build middleware pipeline")?,
        );
        info!(middlewares = ?pipeline.names(), "middleware pipeline wired");

        // Run before_agent hooks (state patching, context injection).
        let mut state = initial_state.clone();
        let mw_runtime = rvagent_middleware::Runtime::new();
        let run_config = rvagent_middleware::RunnableConfig::default();
        pipeline
            .run_before_agent(&mut state, &mw_runtime, &run_config)
            .await;

        let model = rvagent_middleware::PipelineModel::new(model, Arc::clone(&pipeline));

        let graph = AgentGraph::new(model, tool_executor);
        let completed_state = graph
            .run(state)
            .await
            .map_err(|e| anyhow::anyhow!("agent graph error: {}", e))?;

        // Extract the last AI message from the completed state.
        let last_ai = completed_state
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m, Message::Ai(_)))
            .cloned()
            .unwrap_or_else(|| {
                Message::ai("[rvAgent] Agent completed without producing a response.")
            });

        Ok(last_ai)
    }
}

/// Events produced by the TUI event loop.
pub enum TuiEvent {
    /// User submitted input text.
    Input(String),
    /// User requested quit.
    Quit,
    /// Terminal was resized.
    Resize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_app_new_creates_session() {
        let cwd = PathBuf::from("/tmp");
        let app = App::new("anthropic:claude-sonnet-4-20250514", &cwd, None).unwrap();
        assert_eq!(app.config.model, "anthropic:claude-sonnet-4-20250514");
        assert!(!app.session.id.is_empty());
        assert_eq!(app.config.middleware.len(), DEFAULT_MIDDLEWARE.len());
        assert_eq!(app.config.backend.backend_type, "local_shell");
    }

    #[test]
    fn test_app_config_has_security_defaults() {
        let cwd = PathBuf::from("/tmp");
        let app = App::new("openai:gpt-4o", &cwd, None).unwrap();
        assert!(app.config.security_policy.virtual_mode);
        assert!(!app.config.security_policy.sensitive_env_patterns.is_empty());
    }

    #[test]
    fn test_default_middleware_count() {
        // 10 since ADR-274 demoted summarization to opt-in.
        assert_eq!(DEFAULT_MIDDLEWARE.len(), 10);
    }

    #[test]
    fn test_summarization_is_not_on_the_default_path() {
        // The shipped default must match the decided strategy: masking in the
        // agent loop, not LLM summarization.
        assert!(
            !DEFAULT_MIDDLEWARE.contains(&"summarization"),
            "summarization is on the default path but ADR-274 decided against it"
        );
    }

    #[test]
    fn test_default_middleware_order() {
        // Verify critical ordering constraints from ADR-103.
        let todo_pos = DEFAULT_MIDDLEWARE
            .iter()
            .position(|m| *m == "todo")
            .expect("'todo' middleware must be in DEFAULT_MIDDLEWARE");
        let witness_pos = DEFAULT_MIDDLEWARE
            .iter()
            .position(|m| *m == "witness")
            .expect("'witness' middleware must be in DEFAULT_MIDDLEWARE");
        let hitl_pos = DEFAULT_MIDDLEWARE
            .iter()
            .position(|m| *m == "hitl")
            .expect("'hitl' middleware must be in DEFAULT_MIDDLEWARE");
        let patch_pos = DEFAULT_MIDDLEWARE
            .iter()
            .position(|m| *m == "patch_tool_calls")
            .expect("'patch_tool_calls' middleware must be in DEFAULT_MIDDLEWARE");

        // todo before witness; patch_tool_calls before witness; witness before hitl.
        assert!(todo_pos < witness_pos);
        assert!(patch_pos < witness_pos);
        assert!(witness_pos < hitl_pos);
    }
}
