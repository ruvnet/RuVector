# rvAgent

**AI Agent Framework -- Rust-native, secure-by-default**

rvAgent is a modular AI agent framework built in Rust, providing a batteries-included harness for building coding agents, autonomous assistants, and multi-agent orchestration systems. It features a typed middleware pipeline, pluggable backends, parallel tool execution, and first-class security controls.

## Architecture

rvAgent is organized as 8 crates within the RuVector workspace:

```
rvAgent/
  rvagent-core        Core types, config, model resolution, prompt builder
  rvagent-backends     Backend protocol trait + 5 implementations
  rvagent-middleware   Middleware trait + 11 middleware implementations
  rvagent-tools        Tool trait + 8 built-in tools (enum dispatch)
  rvagent-subagents    SubAgent spec, compilation, orchestration
  rvagent-cli          Terminal coding agent (ratatui TUI)
  rvagent-acp          Agent Communication Protocol server (axum)
  rvagent-wasm         WASM bindings for browser/Node.js
```

### Crate Dependency Graph

```
rvagent-cli -----> rvagent-core
    |                  |
    |              rvagent-middleware
    |                  |         \
    |              rvagent-tools  rvagent-subagents
    |                  |
    |              rvagent-backends
    |
rvagent-acp -----> rvagent-core
rvagent-wasm ----> rvagent-core
```

## Crates

| Crate | Description |
|---|---|
| `rvagent-core` | Typed `AgentState` (Arc-wrapped, O(1) clone), `Message` enum, `RvAgentConfig`, `SystemPromptBuilder`, `ChatModel` trait, `RvAgentError` |
| `rvagent-backends` | `Backend` and `SandboxBackend` async traits. Implementations: `StateBackend` (in-memory), `FilesystemBackend` (local disk + ripgrep), `LocalShellBackend` (shell exec), `CompositeBackend` (path-prefix routing), `BaseSandbox` (remote sandboxes) |
| `rvagent-middleware` | `Middleware` trait with hooks: `before_agent`, `wrap_model_call`, `modify_request`, `tools`, `state_keys`. 11 built-in middlewares (see below) |
| `rvagent-tools` | `Tool` trait with enum dispatch for 8 built-in tools: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`, `write_todos` |
| `rvagent-subagents` | `SubAgentSpec` declarative definitions, `CompiledSubAgent` compilation, state isolation (`EXCLUDED_STATE_KEYS`), parallel execution via `task` tool |
| `rvagent-cli` | Terminal agent with `ratatui` TUI, `clap` argument parsing, session management, MCP integration, headless mode |
| `rvagent-acp` | Agent Communication Protocol server built on `axum` with API key auth, rate limiting, request body limits, session management |
| `rvagent-wasm` | WASM bindings via `wasm-bindgen` for browser and Node.js deployment using `StateBackend` |

## Quick Start

Add the core crate to your `Cargo.toml`:

```toml
[dependencies]
rvagent-core = { path = "crates/rvAgent/rvagent-core" }
rvagent-backends = { path = "crates/rvAgent/rvagent-backends" }
rvagent-tools = { path = "crates/rvAgent/rvagent-tools" }
```

### Basic Usage

```rust
use rvagent_core::{
    config::RvAgentConfig,
    messages::Message,
    state::AgentState,
};

// Create a configuration with defaults (virtual_mode=true, env sanitization enabled)
let config = RvAgentConfig::default();

// Build agent state
let mut state = AgentState::with_system_message("You are a helpful assistant.");
state.push_message(Message::human("Hello, what files are in this directory?"));

// State cloning is O(1) thanks to Arc-wrapped fields
let snapshot = state.clone();
assert_eq!(state.message_count(), snapshot.message_count());
```

## Security Highlights

rvAgent is secure-by-default with 13 security controls:

- **`virtual_mode=true` by default** -- filesystem operations run in a virtual sandbox preventing path traversal and symlink attacks
- **Environment sanitization** -- sensitive env vars matching `SECRET`, `KEY`, `TOKEN`, `PASSWORD`, `CREDENTIAL`, `AWS_*`, `AZURE_*`, `GCP_*`, `DATABASE_URL`, `PRIVATE` are stripped before child process execution
- **Witness chains** -- every tool call is logged with SHAKE-256 argument hashes for audit trails
- **ASCII-only skill names** -- prevents Unicode confusable/homoglyph attacks on skill identifiers
- **Tool result sanitization** -- tool outputs are wrapped in delimited blocks to defend against indirect prompt injection
- **Unicode security** -- detection and stripping of BiDi controls, zero-width characters, and script confusable homoglyphs
- **SubAgent result validation** -- max response length (100KB default), control character stripping
- **Atomic path resolution** -- post-open `/proc/self/fd` verification prevents TOCTOU races
- **Grep literal mode** -- defaults to fixed-string matching to prevent ReDoS
- **ACP server hardening** -- API key auth, rate limiting (60 req/min), request body limits (1MB), TLS enforcement

## Performance Highlights

- **Typed `AgentState`** with `Arc`-wrapped fields -- O(1) clone on subagent spawn (vs O(n) deep copy), 5-20x middleware pipeline speedup
- **Parallel tool execution** -- multiple tool calls in a single LLM response execute concurrently via `tokio::task::JoinSet`
- **Enum dispatch** for 8 built-in tools -- eliminates vtable indirection and `async_trait` boxing on the hot path
- **`SystemPromptBuilder`** -- defers concatenation of 4+ prompt segments into a single `String::with_capacity` allocation
- **Optimized line formatting** -- `format_content_with_line_numbers` pre-calculates output size, single allocation
- **Arena allocators** -- leverages `ruvector_core::arena::Arena` for scratch allocations in grep/glob result accumulation
- **In-process search** -- uses `grep-regex`/`grep-searcher` library crates instead of subprocess `rg`
- **`parking_lot::RwLock`** -- faster reader-writer locks for concurrent backend access

## Configuration

```rust
use rvagent_core::config::{RvAgentConfig, SecurityPolicy, ResourceBudget, BackendConfig};

let config = RvAgentConfig {
    model: "anthropic:claude-sonnet-4-20250514".into(),
    name: Some("my-agent".into()),
    instructions: "You are a code reviewer.".into(),
    backend: BackendConfig {
        backend_type: "local_shell".into(),
        cwd: Some("/home/user/project".into()),
        ..Default::default()
    },
    security_policy: SecurityPolicy {
        virtual_mode: true,
        command_allowlist: vec!["cargo".into(), "npm".into()],
        ..Default::default()
    },
    resource_budget: Some(ResourceBudget {
        max_time_secs: 300,
        max_tokens: 200_000,
        max_cost_microdollars: 5_000_000, // $5
        max_tool_calls: 500,
        max_external_writes: 100,
    }),
    ..Default::default()
};
```

## CLI Usage

The `rvagent` binary provides a terminal coding agent:

```bash
# Interactive TUI session
rvagent

# Single prompt (non-interactive)
rvagent run "Fix the failing test in src/lib.rs"

# Specify model and working directory
rvagent -m openai:gpt-4o -d /path/to/project

# Resume a previous session
rvagent --resume <session-id>

# Non-interactive with prompt flag
rvagent -p "What does this codebase do?"

# Session management
rvagent session list
rvagent session delete <session-id>
```

## ACP Server Usage

The `rvagent-acp` binary runs an Agent Communication Protocol server:

```bash
# Start ACP server (default port 8080)
rvagent-acp

# Interact via HTTP
curl -X POST http://localhost:8080/prompt \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <api-key>" \
  -d '{"content": [{"type": "text", "text": "List the files in src/"}]}'

# Health check
curl http://localhost:8080/health

# Create a session
curl -X POST http://localhost:8080/sessions \
  -H "Authorization: Bearer <api-key>" \
  -d '{"cwd": "/home/user/project"}'
```

## WASM Usage

Build the WASM package for browser deployment:

```bash
wasm-pack build crates/rvAgent/rvagent-wasm --target web
```

The WASM build uses `StateBackend` (in-memory) since filesystem access is unavailable in the browser.

```javascript
import init, { create_agent, send_prompt } from './rvagent_wasm.js';

await init();
const agent = create_agent({
  model: "anthropic:claude-sonnet-4-20250514",
  instructions: "You are a helpful assistant."
});
const response = await send_prompt(agent, "Hello!");
```

## Building and Testing

```bash
# Build all rvAgent crates
cargo build -p rvagent-core -p rvagent-backends -p rvagent-middleware \
  -p rvagent-tools -p rvagent-subagents -p rvagent-cli -p rvagent-acp

# Run all tests
cargo test -p rvagent-core -p rvagent-backends -p rvagent-middleware \
  -p rvagent-tools -p rvagent-subagents -p rvagent-cli -p rvagent-acp

# Run benchmarks
cargo bench -p rvagent-core
cargo bench -p rvagent-backends
cargo bench -p rvagent-tools
cargo bench -p rvagent-middleware
```

## License

MIT OR Apache-2.0
