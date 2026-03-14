# ADR-102: Implementation Roadmap & Phasing

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-14                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |

## Context

The DeepAgents Rust conversion spans 9 new crates, 60+ module ports, and 80+ test file equivalents. This ADR defines the implementation phases with clear milestones and dependency ordering.

## Decision

### Phase 1: Foundation (Weeks 1-3)

**Goal:** Core types, backend protocol, and state backend working.

#### Deliverables

| Crate | Modules | Tests | Status |
|---|---|---|---|
| `ruvector-deep-backends` | `protocol.rs`, `utils.rs`, `state.rs` | 15 unit tests | Foundation |
| `ruvector-deep-core` | `messages.rs`, `config.rs` | 5 unit tests | Foundation |

#### Milestone: StateBackend passes all Python-equivalent tests

```bash
cargo test -p ruvector-deep-backends
# All StateBackend operations: ls_info, read, write, edit, grep_raw, glob_info
# All utility functions: format_content_with_line_numbers, perform_string_replacement
```

### Phase 2: Backends (Weeks 3-5)

**Goal:** All 5 backend implementations complete.

#### Deliverables

| Crate | Modules | Tests |
|---|---|---|
| `ruvector-deep-backends` | `filesystem.rs`, `local_shell.rs`, `composite.rs`, `sandbox.rs`, `store.rs` | 30 unit tests |

#### Milestone: All backends pass fidelity tests

```bash
cargo test -p ruvector-deep-backends -- --test-threads=1
# FilesystemBackend: real filesystem operations with virtual_mode
# LocalShellBackend: execute with timeout, stderr prefixing
# CompositeBackend: path routing, result remapping
```

### Phase 3: Tools (Weeks 5-7)

**Goal:** All 8 tool implementations with identical behavior.

#### Deliverables

| Crate | Modules | Tests |
|---|---|---|
| `ruvector-deep-tools` | `lib.rs`, `ls.rs`, `read_file.rs`, `write_file.rs`, `edit_file.rs`, `glob.rs`, `grep.rs`, `execute.rs`, `write_todos.rs` | 25 unit tests |

#### Milestone: Tool golden-file tests pass

```bash
cargo test -p ruvector-deep-tools
# Each tool produces character-identical output to Python
# Image file detection, line number formatting, error messages
```

### Phase 4: Middleware (Weeks 7-10)

**Goal:** Complete middleware pipeline with all 9 middleware implementations.

#### Deliverables

| Crate | Modules | Tests |
|---|---|---|
| `ruvector-deep-middleware` | `lib.rs`, `todolist.rs`, `filesystem.rs`, `memory.rs`, `skills.rs`, `summarization.rs`, `prompt_caching.rs`, `patch_tool_calls.rs`, `hitl.rs`, `utils.rs` | 30 unit tests |
| `ruvector-deep-subagents` | `lib.rs`, `builder.rs`, `prompts.rs` | 15 unit tests |

#### Dependencies: Phase 2 (backends) + Phase 3 (tools)

#### Milestone: Middleware pipeline integration test passes

```bash
cargo test -p ruvector-deep-middleware -- integration
# Middleware ordering: Todo → Memory → Skills → Filesystem → SubAgent → Summarization → PromptCaching → Patch
# State isolation between parent and subagents
# System prompt injection from all middleware
```

### Phase 5: LLM Providers (Weeks 8-10, parallel with Phase 4)

**Goal:** Anthropic and OpenAI client implementations.

#### Deliverables

| Crate | Modules | Tests |
|---|---|---|
| `ruvector-deep-providers` | `lib.rs`, `anthropic.rs`, `openai.rs`, `init_chat_model.rs` | 10 unit tests |

#### Milestone: Model resolution matches Python behavior

```bash
cargo test -p ruvector-deep-providers
# "provider:model" parsing, model_matches_spec, get_model_identifier
# Anthropic streaming, OpenAI Responses API support
```

### Phase 6: Core Agent Factory (Weeks 10-12)

**Goal:** `create_deep_agent()` fully functional with all middleware.

#### Deliverables

| Crate | Modules | Tests |
|---|---|---|
| `ruvector-deep-core` | `lib.rs`, `graph.rs`, `models.rs` | 20 integration tests |

#### Dependencies: All previous phases

#### Milestone: End-to-end agent invocation with mock LLM

```bash
cargo test -p ruvector-deep-core -- e2e
# create_deep_agent() with all configurations
# Subagent spawning and result collection
# Session checkpointing and resume
```

### Phase 7: CLI (Weeks 12-16)

**Goal:** Full terminal application with ratatui TUI.

#### Deliverables

| Crate | Modules | Tests |
|---|---|---|
| `ruvector-deep-cli` | 30+ modules (see ADR-099) | 40 tests |

#### Sub-phases

1. **Week 12-13:** Core CLI (main, config, agent creation, non-interactive mode)
2. **Week 13-14:** TUI widgets (chat, messages, approval, diff)
3. **Week 14-15:** Sessions, hooks, skills, MCP integration
4. **Week 15-16:** Sandbox integrations (Modal, Runloop, Daytona)

#### Milestone: CLI passes all argument/headless tests

```bash
cargo test -p ruvector-deep-cli
cargo run -p ruvector-deep-cli -- --headless "What is 2+2?"
```

### Phase 8: ACP Server (Weeks 14-16, parallel with Phase 7)

**Goal:** ACP server implementation with axum.

#### Deliverables

| Crate | Modules | Tests |
|---|---|---|
| `ruvector-deep-acp` | `server.rs`, `utils.rs` | 10 tests |

#### Milestone: ACP protocol compliance

### Phase 9: RVF & WASM (Weeks 16-18)

**Goal:** RVF integration and WASM compilation.

#### Deliverables

| Crate | Modules | Tests |
|---|---|---|
| `ruvector-deep-wasm` | `lib.rs`, `state_backend.rs`, `agent.rs` | 10 tests |

#### Milestone: Agent runs in browser via WASM

```bash
wasm-pack build crates/ruvector-deep-wasm --target web
```

### Phase 10: Fidelity Verification (Weeks 18-20)

**Goal:** Cross-language test suite verifying 100% behavioral parity.

#### Activities

1. Run Python test suite → capture golden outputs
2. Run Rust test suite → compare against golden outputs
3. Property-based testing for edge cases
4. Performance benchmarking (Rust vs Python)
5. Documentation and API reference generation

### Dependency Graph

```
Phase 1 (Foundation) ─┐
                      ├─ Phase 2 (Backends) ─┐
                      │                       ├─ Phase 3 (Tools) ─┐
                      │                       │                    ├─ Phase 4 (Middleware) ─┐
Phase 5 (Providers) ──┘ (parallel)            │                    │                        │
                                              │                    │                        ├─ Phase 6 (Core)
                                              │                    │                        │
                                              │                    │                        ├─ Phase 7 (CLI)
                                              │                    │                        ├─ Phase 8 (ACP)
                                              │                    │                        └─ Phase 9 (WASM)
                                              │                    │
                                              └────────────────────┘
Phase 10 (Verification) — after all phases
```

### Lines of Code Estimate

| Crate | Estimated LoC (Rust) | Python Equivalent LoC |
|---|---|---|
| `ruvector-deep-backends` | ~3,500 | ~2,800 (5 files) |
| `ruvector-deep-tools` | ~1,500 | ~1,200 (tools in filesystem.py) |
| `ruvector-deep-middleware` | ~3,000 | ~2,500 (6 middleware files) |
| `ruvector-deep-subagents` | ~1,200 | ~700 (subagents.py) |
| `ruvector-deep-core` | ~1,000 | ~300 (graph.py + init) |
| `ruvector-deep-providers` | ~1,500 | ~100 (_models.py — rest is LangChain) |
| `ruvector-deep-cli` | ~8,000 | ~6,000 (30+ modules) |
| `ruvector-deep-acp` | ~800 | ~500 (2 files) |
| `ruvector-deep-wasm` | ~500 | N/A |
| **Tests** | ~5,000 | ~4,000 |
| **Total** | **~26,000** | **~18,100** |

### Risk Mitigation

| Risk | Mitigation |
|---|---|
| LangChain API changes | Pin to v0.4.x semantics, abstract behind traits |
| LLM provider SDK differences | Thin HTTP wrappers, not full SDK ports |
| Textual → ratatui gap | Focus on headless mode first, TUI second |
| WASM binary size | Feature flags, tree-shaking, wasm-opt |
| Sandbox provider API instability | Feature-gated, optional crate dependencies |

## Consequences

- 20-week implementation timeline with clear milestones
- Each phase produces independently testable crates
- Parallel work possible in Phases 5/7/8
- ~26,000 lines of Rust code for 100% fidelity conversion
- WASM deployment as a bonus capability not in Python original
