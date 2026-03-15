# ADR-099: CLI & ACP Server Conversion

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-14                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |
| **Crates**  | `ruvector-deep-cli`, `ruvector-deep-acp`        |

## Context

### CLI (`deepagents_cli/`) — 60+ Python modules

The CLI is a full terminal coding agent with:

- **Textual TUI** — Rich terminal UI with widgets (chat, approval, diff, model selector, etc.)
- **Session management** — Persist/resume conversations across sessions
- **MCP integration** — Connect to MCP servers for external tools
- **Sandbox providers** — Modal, Runloop, Daytona integrations
- **Skills system** — Custom slash commands from SKILL.md files
- **Hooks** — Pre/post execution hooks
- **Non-interactive mode** — Headless operation for CI/CD
- **Web search** — Built-in web search tool
- **Unicode security** — Dangerous unicode detection/stripping

### ACP Server (`deepagents_acp/`) — 2 Python modules

Agent Communication Protocol server for remote agent interaction:

- **ACP agent** — Implements `acp.Agent` interface
- **Session context** — Working directory and mode management
- **Content block conversion** — Text, image, audio, resource blocks

## Decision

### CLI Architecture (`ruvector-deep-cli`)

#### Core Application

```rust
// crates/ruvector-deep-cli/src/main.rs

use clap::Parser;
use ratatui::prelude::*;

/// DeepAgents CLI — Rust edition
/// Python: deepagents_cli/main.py
#[derive(Parser)]
#[command(name = "deep", version)]
struct Cli {
    /// Prompt to send to the agent
    prompt: Option<String>,

    /// Agent name to use
    #[arg(short = 'a', long)]
    agent: Option<String>,

    /// Model to use (provider:model format)
    #[arg(short = 'm', long)]
    model: Option<String>,

    /// Resume a previous session
    #[arg(short = 'r', long)]
    resume: Option<String>,

    /// Non-interactive mode
    #[arg(long)]
    headless: bool,

    /// Working directory
    #[arg(short = 'd', long)]
    directory: Option<String>,

    /// MCP server configs
    #[arg(long)]
    mcp: Vec<String>,

    /// Output format (text/json)
    #[arg(long, default_value = "text")]
    output: String,
}
```

#### TUI Application (Textual → ratatui)

| Python Widget (Textual) | Rust Widget (ratatui) |
|---|---|
| `ChatInput` | `ChatInputWidget` — Input with autocomplete |
| `Messages` | `MessagesWidget` — Scrollable message list |
| `Approval` | `ApprovalWidget` — Tool call approval dialog |
| `Diff` | `DiffWidget` — Unified diff display |
| `ModelSelector` | `ModelSelectorWidget` — Provider:model picker |
| `StatusBar` | `StatusWidget` — Token count, model, session |
| `Welcome` | `WelcomeWidget` — Initial greeting |
| `Loading` | `LoadingWidget` — Spinner/progress |
| `ToolRenderers` | `ToolRenderWidget` — Per-tool output formatting |
| `ThreadSelector` | `ThreadSelectorWidget` — Session picker |
| `McpViewer` | `McpViewerWidget` — MCP server status |
| `History` | `HistoryWidget` — Command history |
| `AskUser` | `AskUserWidget` — User input prompts |

```rust
// crates/ruvector-deep-cli/src/app.rs

pub struct App {
    agent: Box<dyn AgentRunnable>,
    session: Session,
    config: CliConfig,
    widgets: WidgetState,
    mcp_clients: Vec<McpClient>,
}

impl App {
    pub async fn run(&mut self, terminal: &mut Terminal<impl Backend>) -> Result<()> {
        loop {
            terminal.draw(|f| self.render(f))?;
            if let Some(event) = crossterm::event::poll(Duration::from_millis(100))? {
                self.handle_event(event).await?;
            }
        }
    }
}
```

#### Module Mapping

| Python Module | Rust Module | Purpose |
|---|---|---|
| `agent.py` | `agent.rs` | Agent creation with backend setup |
| `app.py` | `app.rs` | TUI application main loop |
| `config.py` | `config.rs` | Settings, colors, glyphs |
| `sessions.py` | `sessions.rs` | Session persistence (JSON files) |
| `hooks.py` | `hooks.rs` | Pre/post execution hooks |
| `tools.py` | `tools.rs` | CLI-specific tools |
| `mcp_tools.py` | `mcp.rs` | MCP server connection |
| `mcp_trust.py` | `mcp_trust.rs` | MCP trust management |
| `subagents.py` | `subagents.rs` | Subagent listing/management |
| `skills/load.py` | `skills/load.rs` | Skill discovery and loading |
| `skills/commands.py` | `skills/commands.rs` | Slash command dispatch |
| `input.py` | `input.rs` | Input parsing (slash commands, files) |
| `output.py` | `output.rs` | JSON/text output formatting |
| `file_ops.py` | `file_ops.rs` | File operation utilities |
| `clipboard.py` | `clipboard.rs` | System clipboard integration |
| `media_utils.py` | `media_utils.rs` | Image/media handling |
| `unicode_security.py` | `unicode_security.rs` | Dangerous unicode detection |
| `update_check.py` | `update_check.rs` | Version update notifications |
| `non_interactive.py` | `non_interactive.rs` | Headless mode |
| `remote_client.py` | `remote_client.rs` | Remote agent connection |
| `server.py` | `server.rs` | Local agent server |
| `server_graph.py` | `server_graph.rs` | Server graph management |
| `server_manager.py` | `server_manager.rs` | Server lifecycle |
| `model_config.py` | `model_config.rs` | Model configuration |
| `configurable_model.py` | `configurable_model.rs` | Runtime model switching |
| `local_context.py` | `local_context.rs` | Project context loading |
| `project_utils.py` | `project_utils.rs` | Project detection |
| `tool_display.py` | `tool_display.rs` | Tool output formatting |
| `textual_adapter.py` | — | N/A (ratatui native) |

#### Sandbox Integrations

```rust
// crates/ruvector-deep-cli/src/integrations/

/// Python: integrations/sandbox_factory.py
pub mod sandbox_factory {
    pub fn create_sandbox(provider: &str, config: &SandboxConfig) -> Box<dyn SandboxBackend>;
}

/// Python: integrations/modal.py
pub mod modal {
    pub struct ModalSandbox { /* Modal API client */ }
    impl SandboxBackend for ModalSandbox { ... }
}

/// Python: integrations/runloop.py
pub mod runloop {
    pub struct RunloopSandbox { /* Runloop API client */ }
    impl SandboxBackend for RunloopSandbox { ... }
}

/// Python: integrations/daytona.py
pub mod daytona {
    pub struct DaytonaSandbox { /* Daytona API client */ }
    impl SandboxBackend for DaytonaSandbox { ... }
}
```

### ACP Server (`ruvector-deep-acp`)

```rust
// crates/ruvector-deep-acp/src/server.rs

use axum::{Router, routing::post};

/// ACP agent session context.
/// Python: AgentSessionContext
#[derive(Debug, Clone)]
pub struct AgentSessionContext {
    pub cwd: String,
    pub mode: String,
}

/// ACP agent implementation.
/// Python: deepagents_acp server.py
pub struct AcpAgent {
    graph: Box<dyn AgentRunnable>,
    sessions: HashMap<String, AgentSessionContext>,
}

impl AcpAgent {
    /// Initialize agent with capabilities.
    /// Python: initialize() -> InitializeResponse
    pub async fn initialize(&self) -> InitializeResponse { ... }

    /// Create new session.
    /// Python: new_session() -> NewSessionResponse
    pub async fn new_session(&self, cwd: &str) -> NewSessionResponse { ... }

    /// Handle prompt.
    /// Python: prompt() -> PromptResponse
    pub async fn prompt(&self, session_id: &str, content: Vec<ContentBlock>) -> PromptResponse { ... }
}

/// Content block conversions (exact fidelity).
/// Python: utils.py — convert_*_block_to_content_blocks
pub mod utils {
    pub fn convert_text_block(block: &TextContentBlock) -> Vec<ContentBlock> { ... }
    pub fn convert_image_block(block: &ImageContentBlock) -> Vec<ContentBlock> { ... }
    pub fn convert_audio_block(block: &AudioContentBlock) -> Vec<ContentBlock> { ... }
    pub fn convert_resource_block(block: &ResourceContentBlock) -> Vec<ContentBlock> { ... }
    pub fn format_execute_result(response: &ExecuteResponse) -> String { ... }
    pub fn truncate_command_for_display(cmd: &str) -> String { ... }
}
```

### CLI Dependencies

```toml
[dependencies]
# TUI
ratatui = "0.29"
crossterm = "0.28"
tui-textarea = "0.7"

# CLI
clap = { version = "4", features = ["derive"] }

# Async
tokio = { version = "1", features = ["full"] }

# HTTP (for MCP, sandbox providers)
reqwest = { version = "0.12", features = ["json"] }

# Clipboard
arboard = "3"

# Config
dirs = "5"
toml = "0.8"
```

## Consequences

- Full TUI rewrite from Textual (Python) to ratatui (Rust) with identical UX
- All 30+ CLI modules ported with same argument parsing and behavior
- MCP integration via HTTP/stdio transports (same as Python)
- Session persistence uses same JSON format for cross-language compatibility
- ACP server uses axum (same HTTP semantics as Python's implementation)
- Sandbox providers (Modal, Runloop, Daytona) use reqwest HTTP clients
