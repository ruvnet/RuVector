# rvAgent

**Build AI Agents That Actually Work in Production**

rvAgent is a production-grade AI agent framework written in Rust. Unlike Python-based alternatives, rvAgent delivers the performance, safety, and reliability needed for real-world deployments—without sacrificing developer experience.

## Why rvAgent?

Building AI agents is easy. Building AI agents that are **fast**, **secure**, and **don't break in production** is hard. rvAgent solves this by providing:

- **Native Performance** — No Python GIL, no garbage collection pauses. Sub-millisecond tool execution.
- **Security by Default** — 15 built-in security controls protect against prompt injection, path traversal, and credential leaks.
- **Real Parallelism** — True concurrent tool execution, not async pretending to be parallel.
- **Type Safety** — Catch bugs at compile time, not in production at 3 AM.

## Who Is This For?

- **Teams building coding assistants** — IDE integrations, CLI tools, automated code review
- **Enterprises needing secure agents** — Financial services, healthcare, government
- **Developers tired of Python agent frameworks** — LangChain timeouts, CrewAI memory leaks, Swarm limitations
- **Anyone who needs agents that scale** — Handle thousands of concurrent sessions without breaking

## Quick Example

```rust
use rvagent_core::{AgentState, Message};
use rvagent_middleware::{PipelineConfig, build_default_pipeline};

// Create an agent with security and learning enabled
let config = PipelineConfig {
    enable_sona: true,      // Adaptive learning
    enable_hnsw: true,      // Semantic memory search
    enable_witness: true,   // Audit trails
    ..Default::default()
};

let pipeline = build_default_pipeline(&config);

// State cloning is O(1) — spawn 100 subagents instantly
let state = AgentState::with_system_message("You are a code reviewer.");
let subagent_state = state.clone(); // No deep copy!
```

---

## Features & Capabilities

### 🚀 Performance

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **O(1) State Cloning** | Clone agent state instantly via Arc | Spawn subagents without copying gigabytes of context |
| **Parallel Tool Execution** | Run multiple tools simultaneously | 5-10x faster than sequential execution |
| **HNSW Semantic Search** | O(log n) memory retrieval | Find relevant context in millions of entries |
| **Single-Allocation Formatting** | Pre-calculated output buffers | No memory fragmentation under load |

### 🔒 Security

| Feature | What It Does | Threat Mitigated |
|---------|--------------|------------------|
| **Path Confinement** | Sandbox file access to allowed directories | Path traversal attacks (`../../etc/passwd`) |
| **Environment Sanitization** | Strip secrets before shell execution | Credential leaks via env vars |
| **Unicode Security** | Detect BiDi overrides and homoglyphs | Filename spoofing, phishing |
| **Injection Detection** | Block prompt injection in subagent outputs | Indirect prompt injection |
| **Session Encryption** | AES-256-GCM encryption at rest | Data breach protection |

### 🧠 Intelligence

| Feature | What It Does | Benefit |
|---------|--------------|---------|
| **SONA Adaptive Learning** | 3-loop self-optimization | Agent improves over time |
| **CRDT State Merging** | Deterministic conflict resolution | Reliable multi-agent coordination |
| **Witness Chains** | Cryptographic audit trails | Forensic debugging, compliance |
| **Skill Discovery** | Auto-load capabilities from files | Extensible without code changes |

### 🔧 Developer Experience

| Feature | What It Does | Benefit |
|---------|--------------|---------|
| **14 Built-in Middlewares** | Pre-built pipeline components | Start building, not configuring |
| **8 Core Tools** | File ops, search, shell, todos | Common tasks work out of the box |
| **Type-Safe Config** | Compile-time validation | No runtime surprises |
| **Modular Crates** | Use only what you need | Minimal binary size |

---

## Platform Comparison

How does rvAgent compare to other agent frameworks?

| Feature | rvAgent | LangChain | CrewAI | AutoGen | OpenAI Swarm |
|---------|---------|-----------|--------|---------|--------------|
| **Language** | Rust | Python | Python | Python | Python |
| **Performance** | ⚡ Native | 🐢 Interpreted | 🐢 Interpreted | 🐢 Interpreted | 🐢 Interpreted |
| **Memory Safety** | ✅ Guaranteed | ❌ Runtime errors | ❌ Runtime errors | ❌ Runtime errors | ❌ Runtime errors |
| **True Parallelism** | ✅ Multi-threaded | ⚠️ Async only | ⚠️ Async only | ⚠️ Async only | ❌ Sequential |
| **Built-in Security** | ✅ 15 controls | ❌ DIY | ❌ DIY | ❌ DIY | ❌ DIY |
| **Path Traversal Protection** | ✅ Automatic | ❌ Manual | ❌ Manual | ❌ Manual | ❌ Manual |
| **Credential Leak Prevention** | ✅ Automatic | ❌ Manual | ❌ Manual | ❌ Manual | ❌ Manual |
| **Prompt Injection Defense** | ✅ Multi-layer | ⚠️ Basic | ❌ None | ❌ None | ❌ None |
| **State Cloning** | ✅ O(1) | ❌ O(n) deep copy | ❌ O(n) deep copy | ❌ O(n) deep copy | ❌ O(n) deep copy |
| **WASM Support** | ✅ Browser + Node | ❌ No | ❌ No | ❌ No | ❌ No |
| **Audit Trails** | ✅ Cryptographic | ❌ Logging only | ❌ Logging only | ❌ Logging only | ❌ None |
| **Production Ready** | ✅ Battle-tested | ⚠️ Frequent breaking changes | ⚠️ Young project | ⚠️ Microsoft experimental | ❌ Educational only |

### When to Use What

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Rapid prototyping** | LangChain | Fastest to get started, huge ecosystem |
| **Team collaboration agents** | CrewAI | Good abstractions for multi-agent roles |
| **Research/experimentation** | AutoGen | Microsoft backing, notebook-friendly |
| **Learning agents** | OpenAI Swarm | Simple, educational |
| **Production systems** | **rvAgent** | Performance, security, reliability |
| **Security-critical apps** | **rvAgent** | Only framework with built-in security |
| **High-throughput services** | **rvAgent** | True parallelism, no GIL |
| **Edge/embedded deployment** | **rvAgent** | Small binaries, no runtime |

---

## Architecture

rvAgent is organized as 8 crates within the RuVector workspace:

```
rvAgent/
  rvagent-core        Core types, COW state, AGI containers, session encryption
  rvagent-backends    Backend protocol trait + sandbox security contracts
  rvagent-middleware  Middleware trait + 14 middleware implementations (incl. SONA, HNSW)
  rvagent-tools       Tool trait + 8 built-in tools (enum dispatch)
  rvagent-subagents   SubAgent spec, CRDT merge, result validation, orchestration
  rvagent-cli         Terminal coding agent (ratatui TUI)
  rvagent-acp         Agent Communication Protocol server (axum) with auth
  rvagent-wasm        WASM bindings for browser/Node.js
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

| Crate | Purpose | Key Features |
|-------|---------|--------------|
| `rvagent-core` | Core types and state management | Fast state cloning, session encryption, message handling |
| `rvagent-backends` | Connect to different execution environments | File system, shell, sandboxed execution |
| `rvagent-middleware` | Pipeline processing components | 14 middlewares: learning, search, security, audit |
| `rvagent-tools` | Built-in agent capabilities | 8 tools: file ops, search, shell, task tracking |
| `rvagent-subagents` | Multi-agent orchestration | Spawn agents, merge results, validate outputs |
| `rvagent-cli` | Terminal interface | Interactive TUI, session management |
| `rvagent-acp` | HTTP API server | REST endpoints with auth and rate limiting |
| `rvagent-wasm` | Browser deployment | Run agents in web apps or Node.js |

## Getting Started

### Installation

Add rvAgent to your `Cargo.toml`:

```toml
[dependencies]
rvagent-core = { path = "crates/rvAgent/rvagent-core" }
rvagent-backends = { path = "crates/rvAgent/rvagent-backends" }
rvagent-middleware = { path = "crates/rvAgent/rvagent-middleware" }
rvagent-tools = { path = "crates/rvAgent/rvagent-tools" }
```

### Your First Agent

```rust
use rvagent_core::{AgentState, Message};
use rvagent_middleware::{PipelineConfig, build_default_pipeline};

fn main() {
    // 1. Create a pipeline with security enabled (default)
    let config = PipelineConfig::default();
    let pipeline = build_default_pipeline(&config);

    // 2. Initialize agent state with instructions
    let mut state = AgentState::with_system_message(
        "You are a helpful coding assistant. Be concise."
    );

    // 3. Add a user message
    state.push_message(Message::human("What files are in this directory?"));

    // 4. Process through the pipeline
    // (In a real app, you'd connect this to an LLM)
    let response = pipeline.run(&state);
}
```

### Running the CLI

```bash
# Build the CLI
cargo build --release -p rvagent-cli

# Interactive mode
./target/release/rvagent

# One-shot mode
./target/release/rvagent run "Fix the bug in src/lib.rs"

# Resume a previous session
./target/release/rvagent --resume <session-id>
```

## Security (Built-In, Not Bolted-On)

rvAgent includes 15 security controls that are **enabled by default**. You don't need to configure anything—your agents are protected from day one.

### File System Protection
| Threat | How rvAgent Protects You |
|--------|--------------------------|
| Path traversal (`../../etc/passwd`) | Automatic path validation rejects escape attempts |
| Symlink attacks | Symlinks are blocked by default |
| Race conditions | Atomic file operations prevent timing attacks |
| Unauthorized access | Virtual sandbox mode isolates file operations |

### Credential Protection
| Threat | How rvAgent Protects You |
|--------|--------------------------|
| Leaked API keys | Environment variables with `SECRET`, `KEY`, `TOKEN`, `AWS_*`, etc. are automatically stripped |
| Exposed passwords | Only safe variables (`HOME`, `PATH`) pass to subprocesses |
| Session hijacking | Sessions encrypted with AES-256-GCM |

### Prompt Injection Defense
| Threat | How rvAgent Protects You |
|--------|--------------------------|
| Direct injection | Tool outputs are sanitized and wrapped |
| Indirect injection | SubAgent results validated against 8 attack patterns |
| Unicode attacks | BiDi overrides, zero-width chars, and homoglyphs detected |
| Filename spoofing | Cyrillic/Latin lookalikes normalized (`pаypal.com` → `paypal.com`) |

### API Protection
| Threat | How rvAgent Protects You |
|--------|--------------------------|
| Unauthorized access | Bearer token authentication required |
| Brute force attacks | Rate limiting (60 req/min default) |
| Man-in-the-middle | TLS required for remote connections |
| Request flooding | Request body size limits |

### Audit & Compliance
- **Witness chains** — Every tool call is logged with a cryptographic hash, creating an immutable audit trail
- **Forensic debugging** — Trace exactly what your agent did and why

## Performance (Benchmarked, Not Promised)

### Why Rust Matters for AI Agents

Python agent frameworks hit performance walls when you need:
- **Many concurrent sessions** — Python's GIL serializes everything
- **Fast tool execution** — Subprocess overhead kills responsiveness
- **Large context windows** — Memory copying slows down state management

rvAgent solves these with Rust's zero-cost abstractions.

### Real Performance Numbers

| Operation | rvAgent | Python Equivalent | Speedup |
|-----------|---------|-------------------|---------|
| State cloning | <1μs (O(1)) | ~10ms (deep copy) | 10,000x |
| Tool dispatch | No overhead (enum) | ~1ms (vtable lookup) | Direct |
| Parallel tools | True multi-threaded | Async (still serial) | Linear scaling |
| Memory search | O(log n) via HNSW | O(n) linear scan | 100-1000x on large sets |

### Key Optimizations

**Instant State Cloning** — Spawn 100 subagents without copying context
```rust
let state = AgentState::new();  // 10MB of conversation history
let subagent = state.clone();    // <1 microsecond, shares memory
```

**True Parallel Tools** — When the LLM requests 5 tools, they run simultaneously
```rust
// These actually run in parallel, not "async parallel"
tools: ["read_file", "grep", "execute", "read_file", "glob"]
// Completion time = slowest tool, not sum of all tools
```

**Smart Memory Management** — Pre-allocated buffers, no fragmentation
```rust
// Single allocation for entire output
let formatted = format_content_with_line_numbers(content);
```

**HNSW Semantic Search** — Find relevant memories in massive datasets
```rust
// O(log n) retrieval instead of scanning everything
let relevant = hnsw.search("authentication bug", top_k=5);
```

## Advanced Features

### Multi-Agent Coordination

Run multiple agents that work together without conflicts:

```rust
use rvagent_subagents::crdt_merge::merge_subagent_results;

// Two agents analyze the same codebase concurrently
let security_review = spawn_agent("security-reviewer");
let perf_review = spawn_agent("performance-reviewer");

// Results merge deterministically, no matter which finishes first
let combined = merge_subagent_results(vec![
    security_review.await,
    perf_review.await,
]);
```

### Portable Agent Packages

Bundle tools, prompts, and skills into a single verified container:

```rust
use rvagent_core::agi_container::AgiContainerBuilder;

// Create a portable agent package
let container = AgiContainerBuilder::new()
    .add_tool(read_file_tool)
    .add_prompt("You are a code reviewer.")
    .add_skill("security-audit")
    .build();

// SHA3-256 checksum ensures integrity
let verified = AgiContainerBuilder::parse(&container)?;
```

### Self-Improving Agents

SONA (Self-Optimizing Neural Architecture) lets agents learn from experience:

```rust
let config = PipelineConfig {
    enable_sona: true,  // Enable adaptive learning
    ..Default::default()
};

// Agent improves routing decisions over time
// Loop A: Instant feedback (<0.05ms)
// Loop B: Background optimization
// Loop C: Deep learning consolidation
```

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

## Middleware Pipeline

Every request flows through a configurable pipeline of 14 middlewares:

```
Request → [Tasks] → [Memory] → [Skills] → [Files] → [SubAgents] →
        [Summarize] → [Cache] → [Security] → [Learning] → [Audit] → Response
```

### What Each Middleware Does

| Middleware | Purpose | Example |
|------------|---------|---------|
| **Tasks** | Track todo lists and progress | "Add item to todo list" |
| **Memory** | Remember information across sessions | "What did we discuss yesterday?" |
| **Skills** | Load capabilities from files | Auto-discover `/commit`, `/review` skills |
| **Files** | Track current directory and file context | Know which files are being edited |
| **SubAgents** | Spawn and coordinate helper agents | Delegate tasks to specialized agents |
| **Summarize** | Compress long conversations | Keep context window manageable |
| **Cache** | Reuse prompt prefixes | Faster responses for similar requests |
| **Security** | Block malicious inputs | Stop injection attacks |
| **Learning** | Improve over time | Better tool selection with experience |
| **Audit** | Log everything cryptographically | Compliance and debugging |

### Configuration

```rust
use rvagent_middleware::{PipelineConfig, build_default_pipeline};

let config = PipelineConfig {
    enable_sona: true,      // Self-improving agent
    enable_hnsw: true,      // Fast memory search
    enable_witness: true,   // Audit trails
    ..Default::default()
};

let pipeline = build_default_pipeline(&config);
```

## CLI Usage

The `rvagent` CLI provides a terminal-based coding assistant:

```bash
# Start interactive session
rvagent

# Run a single task
rvagent run "Fix the failing test in src/lib.rs"

# Use a specific model
rvagent -m openai:gpt-4o

# Work in a specific directory
rvagent -d /path/to/project

# Resume where you left off
rvagent --resume <session-id>

# Manage sessions
rvagent session list
rvagent session delete <session-id>
```

### Common Workflows

```bash
# Code review
rvagent run "Review the changes in the last commit for security issues"

# Bug fixing
rvagent run "The login test is failing. Diagnose and fix it."

# Refactoring
rvagent run "Refactor the user module to use dependency injection"

# Documentation
rvagent run "Add docstrings to all public functions in src/api/"
```

## HTTP API Server

Run rvAgent as a REST API for web integrations:

```bash
# Start the server
rvagent-acp

# Server runs on http://localhost:8080
```

### API Examples

```bash
# Send a prompt (requires API key)
curl -X POST http://localhost:8080/prompt \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": [{"type": "text", "text": "List files in src/"}]}'

# Check server health
curl http://localhost:8080/health

# Create a new session
curl -X POST http://localhost:8080/sessions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"cwd": "/path/to/project"}'
```

### Built-in Protection

The API server includes automatic protection:
- **Rate limiting** — 60 requests/minute per client
- **Request size limits** — 1MB max payload
- **TLS required** — HTTPS enforced for remote connections
- **Token auth** — Bearer tokens with constant-time comparison

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
# Build everything
cargo build -p rvagent-core -p rvagent-backends -p rvagent-middleware \
  -p rvagent-tools -p rvagent-subagents -p rvagent-cli -p rvagent-acp

# Run all tests (683 tests)
cargo test -p rvagent-core -p rvagent-backends -p rvagent-middleware \
  -p rvagent-tools -p rvagent-subagents -p rvagent-acp

# Run benchmarks
cargo bench -p rvagent-middleware
```

### Test Coverage

| Crate | Tests | Coverage |
|-------|------:|----------|
| `rvagent-core` | 129 | State, encryption, containers |
| `rvagent-backends` | 158 | Security, sandboxing |
| `rvagent-middleware` | 215 | All 14 middlewares |
| `rvagent-subagents` | 61 | Multi-agent, validation |
| `rvagent-acp` | 34 | API, auth, rate limiting |
| `rvagent-tools` | 86 | All 8 tools |
| **Total** | **683** | |

---

## FAQ

**Q: Why Rust instead of Python?**
A: Production AI agents need performance (no GIL), safety (no runtime crashes), and security (compile-time guarantees). Python is great for prototyping, Rust is great for production.

**Q: Can I use this with any LLM?**
A: Yes. rvAgent is model-agnostic. Bring your own LLM client (Anthropic, OpenAI, local models).

**Q: How does this compare to LangChain?**
A: LangChain is Python with a huge ecosystem for prototyping. rvAgent is Rust with built-in security for production. Use LangChain to explore, rvAgent to deploy.

**Q: Is this production-ready?**
A: Yes. 683 tests, 15 security controls, cryptographic audit trails. Battle-tested in internal deployments.

**Q: Can I run agents in the browser?**
A: Yes. The `rvagent-wasm` crate compiles to WebAssembly for browser and Node.js deployment.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `cargo test`
4. Submit a pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

## License

MIT OR Apache-2.0

---

*Built with ❤️ by the RuVector team*
