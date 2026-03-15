# ADR-093: DeepAgents Complete Rust Conversion — Overview

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-14                                      |
| **Authors** | ruvnet                                          |
| **Scope**   | Full-fidelity Rust port of langchain-ai/deepagents |
| **Series**  | ADR-093 through ADR-102                         |

## Context

[LangChain DeepAgents](https://github.com/langchain-ai/deepagents) is a Python-based agent framework (v0.4.11, 10.8k stars) built on LangChain/LangGraph. It provides a batteries-included agent harness with:

- **Core library** (`libs/deepagents/`) — `create_deep_agent()` factory, backend protocol, middleware pipeline
- **CLI** (`libs/cli/`) — Terminal coding agent with Textual TUI, session management, MCP tools
- **ACP server** (`libs/acp/`) — Agent Communication Protocol server
- **Harbor** (`libs/harbor/`) — Tracing/observability wrapper
- **Partner integrations** — Daytona, Modal, Runloop, QuickJS sandbox providers

This ADR series defines the architecture for a **100% fidelity** Rust conversion using RuVector primitives and the RVF (RuVector Format) serialization layer.

## Decision

We will convert the entire DeepAgents codebase to Rust as a new set of crates within the RuVector workspace, organized as:

| Python Package | Rust Crate | ADR |
|---|---|---|
| `deepagents` (core) | `ruvector-deep-core` | ADR-094, ADR-095 |
| `deepagents.backends` | `ruvector-deep-backends` | ADR-094 |
| `deepagents.middleware` | `ruvector-deep-middleware` | ADR-095, ADR-098 |
| `deepagents.tools` (filesystem) | `ruvector-deep-tools` | ADR-096 |
| `deepagents.middleware.subagents` | `ruvector-deep-subagents` | ADR-097 |
| `deepagents_cli` | `ruvector-deep-cli` | ADR-099 |
| `deepagents_acp` | `ruvector-deep-acp` | ADR-099 |
| Partner sandboxes | `ruvector-deep-sandbox-*` | ADR-094 |

## Source Analysis — DeepAgents Architecture

### Core Library (`libs/deepagents/deepagents/`)

```
deepagents/
├── __init__.py              → Public API: create_deep_agent, middlewares
├── _models.py               → Model resolution (provider:model format)
├── _version.py              → Version constant
├── graph.py                 → create_deep_agent() — main entry point
├── backends/
│   ├── protocol.py          → BackendProtocol ABC, SandboxBackendProtocol
│   ├── state.py             → StateBackend (ephemeral, in LangGraph state)
│   ├── filesystem.py        → FilesystemBackend (local disk, ripgrep)
│   ├── local_shell.py       → LocalShellBackend (filesystem + shell exec)
│   ├── composite.py         → CompositeBackend (path-prefix routing)
│   ├── sandbox.py           → BaseSandbox (execute-only abstract)
│   ├── store.py             → StoreBackend (LangGraph store)
│   └── utils.py             → Shared utilities
└── middleware/
    ├── filesystem.py        → FilesystemMiddleware (tools: ls, read, write, edit, glob, grep, execute)
    ├── subagents.py         → SubAgentMiddleware (task tool)
    ├── summarization.py     → SummarizationMiddleware (auto-compact + tool)
    ├── memory.py            → MemoryMiddleware (AGENTS.md loading)
    ├── skills.py            → SkillsMiddleware (SKILL.md progressive disclosure)
    ├── patch_tool_calls.py  → PatchToolCallsMiddleware (dangling tool call fix)
    └── _utils.py            → append_to_system_message helper
```

### CLI (`libs/cli/deepagents_cli/`)

```
deepagents_cli/
├── agent.py                 → Agent creation for CLI context
├── app.py                   → Textual TUI application
├── main.py                  → Entry point, argument parsing
├── config.py                → Configuration management
├── hooks.py                 → Pre/post execution hooks
├── sessions.py              → Session persistence/resume
├── tools.py                 → CLI-specific tools
├── mcp_tools.py             → MCP server integration
├── subagents.py             → CLI subagent management
├── skills/                  → Skill loading/commands
├── integrations/            → Sandbox providers (Modal, Runloop, Daytona)
├── widgets/                 → Textual UI widgets (15+ modules)
└── ...                      → 30+ additional modules
```

### ACP Server (`libs/acp/deepagents_acp/`)

```
deepagents_acp/
├── server.py                → ACP agent implementation
└── utils.py                 → Content block conversions
```

## Key Python Abstractions → Rust Mapping

| Python Concept | Rust Equivalent |
|---|---|
| `BackendProtocol` (ABC) | `trait Backend` with `async_trait` |
| `SandboxBackendProtocol` | `trait SandboxBackend: Backend` |
| `AgentMiddleware` (generic) | `trait Middleware<S, C, R>` |
| `BaseChatModel` | `trait ChatModel` (provider-agnostic) |
| `BaseTool` / `StructuredTool` | `trait Tool` with `#[tool]` macro |
| `TypedDict` (SubAgent, etc.) | `struct` with `#[derive(Serialize)]` |
| `Annotated[T, ...]` | Custom derive macros for tool params |
| `async def` / `asyncio` | `async fn` / `tokio` runtime |
| `langgraph` state graph | `ruvector-deep-graph` with state machine |
| `Command` (state update) | `enum StateUpdate` |

## RVF Integration Points

The RuVector Format (ADR-029, ADR-030) provides:

1. **Serialization** — All agent state, backend files, and checkpoint data serialize to RVF
2. **Cognitive containers** — Agent configurations stored as RVF cognitive containers
3. **WASM execution** — Tool backends can execute in WASM sandboxes via `ruvector-wasm`
4. **Graph operations** — Agent graph topology maps to RuVector graph primitives

## Fidelity Requirements

100% fidelity means:

1. **API parity** — Every public function/class has a Rust equivalent
2. **Behavioral parity** — Same inputs produce same outputs (modulo LLM non-determinism)
3. **Protocol compatibility** — Rust backends interoperate with Python backends via shared protocols
4. **Tool compatibility** — File operations produce identical results
5. **Middleware ordering** — Same middleware pipeline semantics (wrap_model_call, before_agent, etc.)
6. **State management** — Compatible checkpoint/state formats (JSON/RVF)

## Series Index

| ADR | Title | Scope |
|-----|-------|-------|
| **ADR-093** | Overview (this document) | Architecture mapping, fidelity requirements |
| **ADR-094** | Backend Protocol & Trait System | `BackendProtocol` → `trait Backend`, all backend impls |
| **ADR-095** | Middleware Pipeline Architecture | Middleware trait, ordering, state schemas |
| **ADR-096** | Tool System | Filesystem tools, execute, grep, glob |
| **ADR-097** | SubAgent & Task Orchestration | Task tool, subagent lifecycle, parallel execution |
| **ADR-098** | Memory, Skills & Summarization | AGENTS.md, SKILL.md, auto-compact |
| **ADR-099** | CLI & ACP Server | Terminal UI, ACP protocol, session management |
| **ADR-100** | RVF Integration & Crate Structure | Workspace layout, RVF serialization, WASM |
| **ADR-101** | Testing Strategy & Fidelity Verification | Cross-language test suite, property testing |
| **ADR-102** | Implementation Roadmap | Phased delivery, milestones, dependencies |

## Consequences

### Positive
- Native performance (10-100x faster tool operations, zero-cost abstractions)
- Memory safety guarantees (no runtime GC, no null pointer exceptions)
- WASM compilation for browser/edge deployment
- Type-safe middleware pipeline (compile-time verification)
- Integration with existing RuVector ecosystem (100+ crates)

### Negative
- No direct LangChain/LangGraph dependency (must reimplement core abstractions)
- LLM provider SDKs must be wrapped (Anthropic, OpenAI → Rust HTTP clients)
- Textual TUI → `ratatui` requires widget reimplementation
- Larger initial development effort

### Risks
- LangChain middleware API may evolve (mitigated: we pin to v0.4.x semantics)
- Python-specific patterns (duck typing, dynamic dispatch) require Rust idioms
- Some Python libs (wcmatch, yaml) need Rust equivalents (glob, serde_yaml)
