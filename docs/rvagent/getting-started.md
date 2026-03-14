# Getting Started with rvAgent

This guide walks through installing rvAgent, building your first agent, adding custom tools and middleware, managing sessions, and deploying an ACP server.

## Prerequisites

- **Rust 1.75+** with the 2021 edition
- **Tokio** async runtime (pulled in as a dependency)
- **An LLM API key** (Anthropic, OpenAI, or other supported provider) set as an environment variable

For WASM targets:
- `wasm-pack` (`cargo install wasm-pack`)

For the CLI:
- A terminal supporting 256 colors (for ratatui TUI)

## Installation

rvAgent is part of the RuVector workspace. Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
# Core types (AgentState, Message, Config)
rvagent-core = { path = "crates/rvAgent/rvagent-core" }

# Backend implementations (StateBackend, FilesystemBackend, etc.)
rvagent-backends = { path = "crates/rvAgent/rvagent-backends" }

# Tool trait and built-in tools
rvagent-tools = { path = "crates/rvAgent/rvagent-tools" }

# Middleware pipeline
rvagent-middleware = { path = "crates/rvAgent/rvagent-middleware" }

# SubAgent orchestration
rvagent-subagents = { path = "crates/rvAgent/rvagent-subagents" }
```

To install the CLI binary:

```bash
cargo install --path crates/rvAgent/rvagent-cli
```

To install the ACP server binary:

```bash
cargo install --path crates/rvAgent/rvagent-acp
```

## First Agent

This example creates an agent with typed state, sends a message, and inspects the result.

```rust
use rvagent_core::{
    config::RvAgentConfig,
    messages::{Message, ToolCall},
    state::{AgentState, TodoItem, TodoStatus},
    models::resolve_model,
    prompt::SystemPromptBuilder,
};

#[tokio::main]
async fn main() {
    // 1. Configure the agent
    let config = RvAgentConfig {
        model: "anthropic:claude-sonnet-4-20250514".into(),
        name: Some("my-first-agent".into()),
        ..Default::default()
    };

    // 2. Resolve the model
    let model_config = resolve_model(&config.model);
    println!("Provider: {:?}, Model: {}", model_config.provider, model_config.model_id);

    // 3. Build agent state
    let mut state = AgentState::with_system_message(&config.instructions);
    state.push_message(Message::human("What files are in this directory?"));

    println!("Messages: {}", state.message_count());
    println!("Virtual mode: {}", config.security_policy.virtual_mode);

    // 4. Clone state for a subagent (O(1) operation)
    let subagent_state = state.clone();
    assert_eq!(state.message_count(), subagent_state.message_count());

    // 5. Build system prompt efficiently
    let mut prompt_builder = SystemPromptBuilder::with_base_prompt();
    prompt_builder.append_section("## Project Context\nThis is a Rust project.");
    prompt_builder.append_section("## Memory\nThe user prefers concise responses.");
    let system_prompt = prompt_builder.build();
    println!("System prompt length: {} chars", system_prompt.len());
}
```

## Using a Backend

Interact with files using one of the backend implementations:

```rust
use rvagent_backends::{
    protocol::{Backend, FileOperationError},
    state::StateBackend,
};

#[tokio::main]
async fn main() {
    // StateBackend stores files in memory (no filesystem access needed)
    let backend = StateBackend::new();

    // Write a file
    let result = backend.write_file("src/main.rs", "fn main() {\n    println!(\"hello\");\n}").await;
    assert!(result.error.is_none());

    // Read it back with line numbers
    let content = backend.read_file("src/main.rs", 0, 100).await.unwrap();
    println!("{}", content);
    // Output:
    //      1	fn main() {
    //      2	    println!("hello");
    //      3	}

    // Edit the file
    let edit = backend.edit_file("src/main.rs", "hello", "world", false).await;
    assert!(edit.error.is_none());
    assert_eq!(edit.occurrences, Some(1));

    // Search with grep (literal mode by default)
    let matches = backend.grep("println", None, None).await.unwrap();
    assert_eq!(matches.len(), 1);
    println!("Found at {}:{}", matches[0].path, matches[0].line);

    // List directory contents
    let entries = backend.ls_info("src").await;
    for entry in &entries {
        println!("{} (dir: {})", entry.path, entry.is_dir);
    }

    // Glob for files
    let rs_files = backend.glob_info("src/*.rs", "").await;
    println!("Rust files: {}", rs_files.len());
}
```

## Adding Custom Tools

Implement the `Tool` trait to create custom tools:

```rust
use async_trait::async_trait;
use rvagent_tools::{Tool, ToolRuntime, ToolResult};
use serde_json::Value;

struct CountLinesTool;

#[async_trait]
impl Tool for CountLinesTool {
    fn name(&self) -> &str { "count_lines" }

    fn description(&self) -> &str {
        "Count the number of lines in a file."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to count lines in"
                }
            },
            "required": ["file_path"]
        })
    }

    fn invoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        let file_path = args["file_path"].as_str().unwrap_or("");
        // In a real implementation, read the file via the backend
        ToolResult::Text(format!("File {} has N lines", file_path))
    }

    async fn ainvoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        self.invoke(args, runtime)
    }
}
```

Register the tool by adding it to your middleware pipeline or tool configuration.

## Adding Custom Middleware

Implement the `Middleware` trait to add custom behavior to the agent pipeline:

```rust
use async_trait::async_trait;
use rvagent_middleware::{Middleware, ModelRequest, ModelResponse};
use rvagent_core::state::AgentState;

/// Middleware that logs every model call.
struct LoggingMiddleware;

impl Middleware for LoggingMiddleware {
    fn wrap_model_call(
        &self,
        request: ModelRequest<()>,
        handler: &dyn Fn(ModelRequest<()>) -> ModelResponse<()>,
    ) -> ModelResponse<()> {
        let msg_count = request.messages.len();
        println!("[LoggingMiddleware] Model call with {} messages", msg_count);
        let response = handler(request);
        println!("[LoggingMiddleware] Response received");
        response
    }
}

/// Middleware that injects project context into the system prompt.
struct ProjectContextMiddleware {
    context: String,
}

impl Middleware for ProjectContextMiddleware {
    fn before_agent(
        &self,
        _state: &AgentState,
        _runtime: &rvagent_middleware::Runtime,
        _config: &rvagent_middleware::RunnableConfig,
    ) -> Option<AgentState> {
        // Return None to skip state modification, or Some(update) to inject state
        None
    }
}
```

## SubAgent Orchestration

Define and compile subagents for delegated task execution:

```rust
use rvagent_subagents::{
    SubAgentSpec, CompiledSubAgent, RvAgentConfig,
    prepare_subagent_state, extract_result_message, merge_subagent_state,
    builder::compile_subagents,
};

fn main() {
    // Define subagent specs
    let specs = vec![
        SubAgentSpec::general_purpose(),
        SubAgentSpec {
            name: "researcher".into(),
            instructions: "Search for information in the codebase.".into(),
            tools: vec!["grep".into(), "read_file".into(), "glob".into()],
            can_read: true,
            can_write: false,
            can_execute: false,
            ..SubAgentSpec::new("researcher", "Search for information")
        },
    ];

    // Compile specs into runnable subagents
    let parent_config = RvAgentConfig::default();
    let compiled = compile_subagents(&specs, &parent_config);

    println!("Compiled {} subagents:", compiled.len());
    for agent in &compiled {
        println!("  - {} (backend: {}, middleware: {:?})",
            agent.spec.name, agent.backend, agent.middleware_pipeline);
    }

    // Prepare isolated state for a subagent invocation
    let mut parent_state = std::collections::HashMap::new();
    parent_state.insert("messages".into(), serde_json::json!([]));
    parent_state.insert("custom_data".into(), serde_json::json!("shared"));

    let child_state = prepare_subagent_state(&parent_state, "Find all TODO comments in src/");
    // child_state has: messages=[{type: human, content: "Find all..."}], custom_data="shared"
    // parent's original messages, todos, etc. are NOT visible to the child

    println!("Child state keys: {:?}", child_state.keys().collect::<Vec<_>>());
}
```

## Session Management

The CLI provides session persistence for resuming conversations:

```bash
# Start a session (auto-saved)
rvagent

# List saved sessions
rvagent session list

# Resume a session by ID
rvagent --resume abc-123-def

# Delete a session
rvagent session delete abc-123-def
```

Sessions are stored as JSON files in the user's data directory (typically `~/.local/share/rvagent/sessions/` on Linux). Session files are created with UUID filenames and restrictive permissions (0600).

## ACP Server Deployment

Deploy an Agent Communication Protocol server for remote agent access:

### Start the Server

```bash
# Set your API key for authentication
export RVAGENT_API_KEY="your-secret-key"

# Start the ACP server
rvagent-acp
```

### Client Interaction

```bash
# Health check
curl http://localhost:8080/health

# Create a session
curl -X POST http://localhost:8080/sessions \
  -H "Authorization: Bearer $RVAGENT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"cwd": "/home/user/project"}'

# Send a prompt
curl -X POST http://localhost:8080/prompt \
  -H "Authorization: Bearer $RVAGENT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "content": [{"type": "text", "text": "List the files in src/"}]
  }'
```

### Server Configuration

The ACP server includes these security defaults:

- API key authentication via `Authorization: Bearer` header
- Rate limiting: 60 requests/minute (configurable)
- Request body size limit: 1MB (configurable)
- TLS enforcement for non-localhost connections
- CORS headers via `tower-http`

## WASM Deployment

Build rvAgent for browser or Node.js execution:

```bash
# Build for web
wasm-pack build crates/rvAgent/rvagent-wasm --target web

# Build for Node.js
wasm-pack build crates/rvAgent/rvagent-wasm --target nodejs
```

The WASM build uses `StateBackend` (in-memory) since filesystem and shell execution are unavailable in browser environments. All file operations work against the in-memory store.

## Next Steps

- Read the [Architecture Documentation](architecture.md) for the full crate dependency graph and agent lifecycle
- Review the [Security Documentation](security.md) for threat model details and all 13 security controls
- Consult the [API Reference](api-reference.md) for complete type and trait documentation
- Check the ADR series (ADR-093 through ADR-103) in `/docs/adr/` for design rationale
