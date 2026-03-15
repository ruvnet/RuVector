//! Application core for the rvAgent CLI.
//!
//! `App` initializes configuration from CLI arguments, creates the backend
//! and middleware pipeline, builds the agent graph, and drives the run loop.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use tracing::{info, warn};

use rvagent_core::config::{
    BackendConfig, MiddlewareConfig, RvAgentConfig, SecurityPolicy,
};
use rvagent_core::graph::{AgentGraph, ToolExecutor};
use rvagent_core::messages::{Message, ToolCall as CoreToolCall};
use rvagent_core::models::{ChatModel, resolve_model};
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
    "summarization",
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
    async fn complete(&self, _messages: &[Message]) -> rvagent_core::error::Result<Message> {
        Ok(Message::ai(format!(
            "No API key configured for model '{}'. \
             Set the appropriate environment variable (e.g. ANTHROPIC_API_KEY) \
             and restart rvAgent.",
            self.model_name
        )))
    }

    async fn stream(&self, messages: &[Message]) -> rvagent_core::error::Result<Vec<Message>> {
        let msg = self.complete(messages).await?;
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
}

#[async_trait]
impl ChatModel for CliModel {
    async fn complete(&self, messages: &[Message]) -> rvagent_core::error::Result<Message> {
        match self {
            CliModel::Stub(m) => m.complete(messages).await,
            CliModel::Anthropic(m) => m.complete(messages).await,
        }
    }

    async fn stream(&self, messages: &[Message]) -> rvagent_core::error::Result<Vec<Message>> {
        match self {
            CliModel::Stub(m) => m.stream(messages).await,
            CliModel::Anthropic(m) => m.stream(messages).await,
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
        let backend: rvagent_tools::BackendRef = Arc::new(LocalFsBackend {
            cwd: cwd.to_path_buf(),
        });
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
}

// ---------------------------------------------------------------------------
// LocalFsBackend — adapts the local filesystem for rvagent_tools::Backend
// ---------------------------------------------------------------------------

/// A minimal filesystem backend implementing `rvagent_tools::Backend` for CLI use.
///
/// Provides real filesystem and shell operations rooted at a working directory.
struct LocalFsBackend {
    cwd: PathBuf,
}

impl rvagent_tools::Backend for LocalFsBackend {
    fn ls_info(&self, path: &str) -> std::result::Result<Vec<rvagent_tools::FileInfo>, String> {
        let target = if path.is_empty() || path == "." {
            self.cwd.clone()
        } else {
            PathBuf::from(path)
        };
        let entries = std::fs::read_dir(&target)
            .map_err(|e| format!("ls failed on '{}': {}", target.display(), e))?;
        let mut infos = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("read_dir entry error: {}", e))?;
            let meta = entry
                .metadata()
                .map_err(|e| format!("metadata error: {}", e))?;
            let file_type = if meta.is_dir() {
                "directory"
            } else if meta.is_symlink() {
                "symlink"
            } else {
                "file"
            };
            infos.push(rvagent_tools::FileInfo {
                name: entry.file_name().to_string_lossy().into_owned(),
                file_type: file_type.to_string(),
                permissions: String::new(),
                size: meta.len(),
            });
        }
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(infos)
    }

    fn read(
        &self,
        path: &str,
        offset: usize,
        limit: usize,
    ) -> std::result::Result<String, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("read '{}': {}", path, e))?;
        let lines: Vec<&str> = content.lines().collect();
        if offset >= lines.len() {
            return Ok(String::new());
        }
        let end = (offset + limit).min(lines.len());
        Ok(lines[offset..end].join("\n"))
    }

    fn write(&self, path: &str, content: &str) -> rvagent_tools::WriteResult {
        if std::path::Path::new(path).exists() {
            return rvagent_tools::WriteResult {
                error: Some(format!(
                    "Error: file {} already exists. Use force flag to overwrite.",
                    path
                )),
                ..Default::default()
            };
        }
        if let Some(parent) = std::path::Path::new(path).parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return rvagent_tools::WriteResult {
                    error: Some(format!("mkdir failed: {}", e)),
                    ..Default::default()
                };
            }
        }
        match std::fs::write(path, content) {
            Ok(_) => rvagent_tools::WriteResult::default(),
            Err(e) => rvagent_tools::WriteResult {
                error: Some(format!("write '{}': {}", path, e)),
                ..Default::default()
            },
        }
    }

    fn edit(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> rvagent_tools::WriteResult {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                return rvagent_tools::WriteResult {
                    error: Some(format!("read '{}': {}", path, e)),
                    ..Default::default()
                }
            }
        };
        let count = content.matches(old_string).count();
        if count == 0 {
            return rvagent_tools::WriteResult {
                error: Some(format!("Error: old_string not found in {}", path)),
                ..Default::default()
            };
        }
        if count > 1 && !replace_all {
            return rvagent_tools::WriteResult {
                error: Some(format!(
                    "Error: old_string is not unique in {} ({} occurrences). Use replace_all=true.",
                    path, count
                )),
                ..Default::default()
            };
        }
        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            content.replacen(old_string, new_string, 1)
        };
        match std::fs::write(path, &new_content) {
            Ok(_) => rvagent_tools::WriteResult {
                error: None,
                occurrences: Some(if replace_all { count } else { 1 }),
                ..Default::default()
            },
            Err(e) => rvagent_tools::WriteResult {
                error: Some(format!("write '{}': {}", path, e)),
                ..Default::default()
            },
        }
    }

    fn glob_info(
        &self,
        pattern: &str,
        path: &str,
    ) -> std::result::Result<Vec<String>, String> {
        let base = if path.is_empty() || path == "." {
            self.cwd.clone()
        } else {
            PathBuf::from(path)
        };
        // Simple glob: walk directory and match by extension or name suffix.
        // This handles common patterns like "*.rs", "**/*.toml" without
        // requiring the `glob` crate.
        let suffix = pattern
            .trim_start_matches('*')
            .trim_start_matches('/')
            .trim_start_matches('*');
        let mut results = Vec::new();
        collect_glob_matches(&base, suffix, &mut results);
        results.sort();
        Ok(results)
    }

    fn grep_raw(
        &self,
        pattern: &str,
        path: Option<&str>,
        _include: Option<&str>,
    ) -> std::result::Result<Vec<rvagent_tools::GrepMatch>, String> {
        // Simple in-process grep implementation.
        let search_dir = match path {
            Some(p) if !p.is_empty() => PathBuf::from(p),
            _ => self.cwd.clone(),
        };
        let mut matches = Vec::new();
        if search_dir.is_file() {
            grep_file(&search_dir, pattern, &mut matches)?;
        } else if search_dir.is_dir() {
            grep_dir(&search_dir, pattern, &mut matches)?;
        }
        Ok(matches)
    }

    fn execute(
        &self,
        command: &str,
        timeout_secs: u32,
    ) -> std::result::Result<rvagent_tools::ExecuteResponse, String> {
        use std::process::Command;
        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&self.cwd)
            .output()
            .map_err(|e| format!("execute failed: {}", e))?;
        let _ = timeout_secs; // timeout handled at a higher level if needed
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = if stderr.is_empty() {
            stdout.into_owned()
        } else {
            format!("{}\n{}", stdout, stderr)
        };
        Ok(rvagent_tools::ExecuteResponse {
            output: combined,
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
}

/// Recursively collect files matching a name suffix (simple glob substitute).
fn collect_glob_matches(dir: &Path, suffix: &str, results: &mut Vec<String>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if path.is_file() && name.ends_with(suffix) {
            results.push(path.to_string_lossy().into_owned());
        } else if path.is_dir() && !name.starts_with('.') {
            collect_glob_matches(&path, suffix, results);
        }
    }
}

/// Grep a single file for a pattern.
fn grep_file(
    path: &Path,
    pattern: &str,
    matches: &mut Vec<rvagent_tools::GrepMatch>,
) -> std::result::Result<(), String> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()), // skip binary / unreadable files
    };
    for (i, line) in content.lines().enumerate() {
        if line.contains(pattern) {
            matches.push(rvagent_tools::GrepMatch {
                file: path.to_string_lossy().into_owned(),
                line_number: i + 1,
                text: line.to_string(),
            });
        }
    }
    Ok(())
}

/// Recursively grep a directory (limited depth).
fn grep_dir(
    dir: &Path,
    pattern: &str,
    matches: &mut Vec<rvagent_tools::GrepMatch>,
) -> std::result::Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir: {}", e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("entry: {}", e))?;
        let path = entry.path();
        if path.is_file() {
            grep_file(&path, pattern, matches)?;
        } else if path.is_dir() {
            // Skip hidden directories.
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if !name.starts_with('.') {
                grep_dir(&path, pattern, matches)?;
            }
        }
    }
    Ok(())
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

        Ok(Self {
            config,
            session,
            cwd: cwd.to_path_buf(),
            system_prompt: BASE_AGENT_PROMPT.to_string(),
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
        let mut tui = Tui::new(
            &self.config.model,
            &self.session.id,
        )?;

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
    /// tool executor, builds an `AgentGraph`, and runs it to completion.
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

        let graph = AgentGraph::new(model, tool_executor);
        let completed_state = graph
            .run(initial_state.clone())
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
        assert_eq!(DEFAULT_MIDDLEWARE.len(), 11);
    }

    #[test]
    fn test_default_middleware_order() {
        // Verify critical ordering constraints from ADR-103.
        let todo_pos = DEFAULT_MIDDLEWARE.iter().position(|m| *m == "todo")
            .expect("'todo' middleware must be in DEFAULT_MIDDLEWARE");
        let witness_pos = DEFAULT_MIDDLEWARE.iter().position(|m| *m == "witness")
            .expect("'witness' middleware must be in DEFAULT_MIDDLEWARE");
        let hitl_pos = DEFAULT_MIDDLEWARE.iter().position(|m| *m == "hitl")
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
