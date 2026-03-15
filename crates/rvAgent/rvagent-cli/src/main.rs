//! rvAgent CLI — terminal coding agent with TUI.
//!
//! Entry point for the `rvagent` binary. Parses CLI arguments via `clap`,
//! initializes tracing, and dispatches to the appropriate run mode
//! (interactive TUI, single-prompt, or session management).

mod app;
mod display;
mod mcp;
mod session;
mod tui;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use crate::app::App;
use crate::session::SessionAction;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "rvagent", about = "rvAgent \u{2014} AI coding agent", version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Model to use (provider:model format).
    #[arg(short, long, default_value = "anthropic:claude-sonnet-4-20250514")]
    model: String,

    /// Working directory.
    #[arg(short = 'd', long)]
    directory: Option<PathBuf>,

    /// Resume session by ID.
    #[arg(long)]
    resume: Option<String>,

    /// Non-interactive mode with prompt.
    #[arg(short, long)]
    prompt: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive agent session.
    Chat,
    /// Run a single prompt and exit.
    Run {
        /// The prompt to send to the agent.
        prompt: String,
    },
    /// List/manage sessions.
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present (supports ANTHROPIC_API_KEY, etc.)
    // Try current directory first, then project root
    if dotenvy::dotenv().is_err() {
        // Try loading from common locations
        let _ = dotenvy::from_filename(".env.local");
    }

    let cli = Cli::parse();

    // Determine if we're running in interactive TUI mode.
    // In TUI mode, we suppress console tracing to avoid corrupting the display.
    let is_tui_mode = match &cli.command {
        Some(Commands::Session { .. }) => false,
        Some(Commands::Run { .. }) => false,
        Some(Commands::Chat) | None => cli.prompt.is_none(),
    };

    // Initialize tracing only for non-TUI modes.
    // TUI mode uses its own display system.
    if !is_tui_mode {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .with_target(false)
            .init();
    }

    // Resolve working directory.
    let cwd = match &cli.directory {
        Some(d) => std::fs::canonicalize(d)?,
        None => std::env::current_dir()?,
    };

    match &cli.command {
        // Explicit session management sub-commands.
        Some(Commands::Session { action }) => {
            session::handle_session_action(action)?;
        }

        // Single-shot prompt execution.
        Some(Commands::Run { prompt }) => {
            let mut app = App::new(&cli.model, &cwd, cli.resume.as_deref())?;
            app.run_once(prompt).await?;
        }

        // Interactive TUI chat (default when no sub-command given).
        Some(Commands::Chat) | None => {
            // If --prompt is supplied without a sub-command, treat as non-interactive.
            if let Some(ref prompt) = cli.prompt {
                let mut app = App::new(&cli.model, &cwd, cli.resume.as_deref())?;
                app.run_once(prompt).await?;
            } else {
                let mut app = App::new(&cli.model, &cwd, cli.resume.as_deref())?;
                app.run_interactive().await?;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_parse_defaults() {
        let cli = Cli::parse_from(["rvagent"]);
        assert_eq!(cli.model, "anthropic:claude-sonnet-4-20250514");
        assert!(cli.directory.is_none());
        assert!(cli.resume.is_none());
        assert!(cli.prompt.is_none());
        assert!(cli.command.is_none());
    }

    #[test]
    fn test_cli_parse_model_flag() {
        let cli = Cli::parse_from(["rvagent", "-m", "openai:gpt-4o"]);
        assert_eq!(cli.model, "openai:gpt-4o");
    }

    #[test]
    fn test_cli_parse_run_subcommand() {
        let cli = Cli::parse_from(["rvagent", "run", "hello world"]);
        match cli.command {
            Some(Commands::Run { ref prompt }) => assert_eq!(prompt, "hello world"),
            _ => panic!("expected Run subcommand"),
        }
    }

    #[test]
    fn test_cli_parse_session_list() {
        let cli = Cli::parse_from(["rvagent", "session", "list"]);
        match cli.command {
            Some(Commands::Session {
                action: SessionAction::List,
            }) => {}
            _ => panic!("expected Session List"),
        }
    }

    #[test]
    fn test_cli_parse_directory() {
        let cli = Cli::parse_from(["rvagent", "-d", "/tmp"]);
        assert_eq!(cli.directory, Some(PathBuf::from("/tmp")));
    }

    #[test]
    fn test_cli_parse_resume() {
        let cli = Cli::parse_from(["rvagent", "--resume", "abc-123"]);
        assert_eq!(cli.resume.as_deref(), Some("abc-123"));
    }

    #[test]
    fn test_cli_parse_prompt_flag() {
        let cli = Cli::parse_from(["rvagent", "-p", "fix the bug"]);
        assert_eq!(cli.prompt.as_deref(), Some("fix the bug"));
    }

    #[test]
    fn test_cli_verify_app() {
        // Validates that the clap derive macros produce a valid command structure.
        Cli::command().debug_assert();
    }
}
