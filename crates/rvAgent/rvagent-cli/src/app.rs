//! Application core for the rvAgent CLI.
//!
//! `App` initializes configuration from CLI arguments, creates the backend
//! and middleware pipeline, builds the agent graph, and drives the run loop.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::info;

use rvagent_core::config::{
    BackendConfig, MiddlewareConfig, RvAgentConfig, SecurityPolicy,
};
use rvagent_core::messages::Message;
use rvagent_core::models::resolve_model;

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
    /// MCP tool registry for external tool servers.
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
            mcp_registry: McpRegistry::new(),
        })
    }

    /// Run a single prompt (non-interactive mode) and exit.
    pub async fn run_once(&mut self, prompt: &str) -> Result<()> {
        self.session.push_message(Message::human(prompt));

        let response = self.invoke_agent(prompt).await?;

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
                    let response = self.invoke_agent(&text).await?;

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

    /// Invoke the agent pipeline with a user prompt.
    ///
    /// In a full implementation this would run the `AgentGraph` with the
    /// configured middleware pipeline. For now it returns a placeholder
    /// response acknowledging the prompt.
    async fn invoke_agent(&self, prompt: &str) -> Result<Message> {
        // TODO: Wire up real AgentGraph from rvagent-core once the graph
        // module is implemented. For now, produce a stub response.
        info!(prompt_len = prompt.len(), "invoking agent");

        let response_text = format!(
            "[rvAgent stub] Received prompt ({} chars). \
             Model: {}. Pipeline: {} middlewares. CWD: {}",
            prompt.len(),
            self.config.model,
            self.config.middleware.len(),
            self.cwd.display(),
        );

        Ok(Message::ai(response_text))
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
        let todo_pos = DEFAULT_MIDDLEWARE.iter().position(|m| *m == "todo").unwrap();
        let witness_pos = DEFAULT_MIDDLEWARE.iter().position(|m| *m == "witness").unwrap();
        let hitl_pos = DEFAULT_MIDDLEWARE.iter().position(|m| *m == "hitl").unwrap();
        let patch_pos = DEFAULT_MIDDLEWARE
            .iter()
            .position(|m| *m == "patch_tool_calls")
            .unwrap();

        // todo before witness; patch_tool_calls before witness; witness before hitl.
        assert!(todo_pos < witness_pos);
        assert!(patch_pos < witness_pos);
        assert!(witness_pos < hitl_pos);
    }
}
