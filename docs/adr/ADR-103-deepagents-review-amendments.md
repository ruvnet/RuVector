# ADR-103: Review Amendments — Performance, RVF Integration & Security Hardening

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-14                                      |
| **Authors** | ruvnet (via review team)                        |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |
| **Amends**  | ADR-094, ADR-095, ADR-096, ADR-097, ADR-098, ADR-099, ADR-100, ADR-101 |

## Context

Three independent review agents analyzed ADR-093 through ADR-102:

1. **Performance Review** — 25 findings, 7 P0 critical
2. **RVF Capability Review** — 17 untapped integration points, 10 gap areas
3. **Security Audit** — 30 findings (5 Critical, 7 High, 6 Medium, 4 Low)

This ADR captures all actionable amendments organized by priority.

---

## Decision

### Part A: Performance Amendments

#### A1. Replace `HashMap<String, serde_json::Value>` with Typed AgentState [P0]

**Amends:** ADR-095 §AgentState type

The JSON intermediate representation imposes a "JSON tax" on every middleware interaction — 3 clone+deserialize cycles per model call, full deep-clone on subagent spawn.

**Before (ADR-095):**
```rust
pub type AgentState = HashMap<String, serde_json::Value>;
```

**After:**
```rust
pub struct AgentState {
    pub messages: Arc<Vec<Message>>,
    pub todos: Arc<Vec<TodoItem>>,
    pub files: Arc<HashMap<String, FileData>>,
    pub memory_contents: Option<Arc<HashMap<String, String>>>,
    pub skills_metadata: Option<Arc<Vec<SkillMetadata>>>,
    extensions: HashMap<String, Box<dyn Any + Send + Sync>>,
}
```

**Impact:** 5-20x middleware pipeline speedup, 10-50x subagent spawn speedup (Arc clone = O(1) vs deep clone = O(n)).

#### A2. Parallel Tool Execution [P0]

**Amends:** ADR-095 §Agent Graph Loop, ADR-097 §SubAgent invocation

When an LLM response contains multiple tool_calls, execute them concurrently:

```rust
async fn execute_tool_calls(calls: &[ToolCall], runtime: &ToolRuntime) -> Vec<ToolResult> {
    let mut set = tokio::task::JoinSet::new();
    for tc in calls {
        let tool = resolve_tool(&tc.name);
        let args = tc.args.clone();
        let rt = runtime.clone();
        set.spawn(async move { tool.ainvoke(args, &rt).await });
    }
    let mut results = Vec::with_capacity(calls.len());
    while let Some(result) = set.join_next().await {
        results.push(result.unwrap());
    }
    results
}
```

**Impact:** 2-5x speedup for multi-tool LLM responses (very common in coding agents).

#### A3. Prevent Blocking I/O in Async Context [P0]

**Amends:** ADR-094 §Backend async methods

- All subprocess invocations MUST use `tokio::process::Command`, not `std::process::Command`
- FilesystemBackend operations MUST use `tokio::task::spawn_blocking` for synchronous filesystem I/O
- Backend structs MUST use `Arc<Inner>` pattern for cheap cloning into spawn_blocking closures:

```rust
pub struct FilesystemBackend {
    inner: Arc<FilesystemBackendInner>,
}
```

**Impact:** Prevents thread pool starvation under concurrent tool execution.

#### A4. Use `grep-regex`/`grep-searcher` Instead of Subprocess ripgrep [P1]

**Amends:** ADR-094 §FilesystemBackend grep

Use ripgrep's library crates (`grep-regex`, `grep-searcher`) for in-process search instead of shelling out to `rg`:

```rust
use grep_regex::RegexMatcher;
use grep_searcher::Searcher;
```

**Impact:** Eliminates 1-5ms subprocess overhead per grep call.

#### A5. SystemPromptBuilder for Deferred Concatenation [P1]

**Amends:** ADR-095 §append_to_system_message, ADR-098 §Memory/Skills middleware

Replace 4 sequential string concatenations per model call with a builder that concatenates once:

```rust
struct SystemPromptBuilder {
    segments: SmallVec<[Cow<'static, str>; 8]>,
}
impl SystemPromptBuilder {
    fn append(&mut self, text: impl Into<Cow<'static, str>>);
    fn build(&self) -> String; // Single allocation, pre-calculated capacity
}
```

**Impact:** Reduces 4 O(n) string copies to 1 O(n) build, saving ~20-80μs per model call.

#### A6. Enum Dispatch for Built-in Tools [P1]

**Amends:** ADR-096 §Tool trait

Use enum dispatch for the 8 built-in tools, trait objects only for user-defined:

```rust
pub enum BuiltinTool { Ls, ReadFile, WriteFile, EditFile, Glob, Grep, Execute, WriteTodos, Task }
pub enum AnyTool { Builtin(BuiltinTool), Dynamic(Box<dyn Tool>) }
```

**Impact:** Eliminates vtable indirection and async_trait boxing for hot path tools.

#### A7. Optimized format_content_with_line_numbers [P1]

**Amends:** ADR-096 §Line number formatting

Pre-calculate total size, write directly to a single `String::with_capacity`:

```rust
pub fn format_content_with_line_numbers(lines: &[&str], start_line: usize) -> String {
    let total_est: usize = lines.iter().map(|l| l.len().min(2000) + 8).sum();
    let mut out = String::with_capacity(total_est);
    for (i, line) in lines.iter().enumerate() {
        if i > 0 { out.push('\n'); }
        write!(out, "{:>6}\t{}", start_line + i, &line[..line.len().min(2000)]).unwrap();
    }
    out
}
```

**Impact:** Eliminates 2000 intermediate String allocations per file read.

#### A8. Arena Allocators from ruvector-core [P1]

**Amends:** ADR-096, ADR-100 §Crate dependencies

Import `ruvector_core::arena::Arena` for scratch allocations in hot paths (line formatting, grep result accumulation, glob result building). The arena infrastructure already exists in the workspace.

#### A9. Criterion Benchmarks [P0]

**Amends:** ADR-101 §Testing Strategy

Add mandatory performance benchmarks:

```toml
# In each deep-* crate's Cargo.toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "tool_latency"
harness = false
```

Required benchmarks:
1. Tool execution latency (read_file, grep, glob, edit_file)
2. Middleware pipeline throughput (full 9-middleware chain, target <1ms)
3. State serialization round-trip (10, 100, 1000 messages)
4. Subagent spawn overhead (Arc-shared vs deep-clone)
5. Session checkpoint/resume (JSON vs rkyv vs bincode)
6. Concurrent tool execution (4 parallel greps vs sequential)
7. format_content_with_line_numbers (100, 1000, 10000 lines)
8. CompositeBackend routing (1, 5, 10, 20 routes)

---

### Part B: RVF Integration Amendments

#### B1. Concrete AGI Container Building [HIGH]

**Amends:** ADR-100 §RVF Integration Points

Replace aspirational `CognitiveLayer` references with real RVF types:

| DeepAgents Concept | RVF Segment/Tag | Integration |
|---|---|---|
| Tool registry | `AGI_TAG_TOOL_REGISTRY` (0x0105) | Serialize tool schemas into container |
| Skill library | `AGI_TAG_SKILL_LIBRARY` (0x0109) | Package skills for offline/WASM use |
| Agent prompts | `AGI_TAG_AGENT_PROMPTS` (0x0106) | Externalize prompts from source code |
| Middleware config | `SegmentType::Profile` (0x0B) | Store pipeline configuration |
| Agent orchestration | `AGI_TAG_ORCHESTRATOR` (0x0108) | Subagent topology definition |

#### B2. COW-Backed StateBackend [HIGH]

**Amends:** ADR-094 §StateBackend

Replace `Arc<RwLock<HashMap>>` with `rvf-runtime::CowEngine` for:
- O(1) state snapshots (vs full clone)
- Efficient subagent forking via COW child branches
- Automatic witness events on every mutation

```rust
pub struct CowStateBackend {
    engine: CowEngine,
    branch_id: u32,
}
impl CowStateBackend {
    pub fn fork_for_subagent(&self) -> Self {
        Self { engine: self.engine.fork_child(), branch_id: self.branch_id + 1 }
    }
}
```

#### B3. Witness Chain Middleware [HIGH]

**Amends:** ADR-095 §Middleware Pipeline, ADR-100 §RVF Integration

Add `WitnessMiddleware` to the default pipeline after `PatchToolCalls`:

```rust
pub struct WitnessMiddleware {
    builder: Arc<Mutex<WitnessBuilder>>,
}
impl Middleware for WitnessMiddleware {
    fn wrap_model_call(&self, request: ModelRequest, handler: ...) -> ModelResponse {
        let response = handler(request);
        for tool_call in &response.tool_calls {
            self.builder.lock().unwrap().add_tool_call_entry(ToolCallEntry {
                tool_name: tool_call.name.clone(),
                arguments_hash: shake256(&serde_json::to_vec(&tool_call.args).unwrap()),
                ..Default::default()
            });
        }
        response
    }
}
```

Pipeline order becomes: Todo → Memory → Skills → Filesystem → SubAgent → Summarization → PromptCaching → PatchToolCalls → **Witness** → HITL

#### B4. Resource Budget Enforcement [HIGH]

**Amends:** ADR-094 §Backend trait, ADR-097 §SubAgent

Use `rvf-types::agi_container::ResourceBudget` to enforce limits:

```rust
pub struct ResourceBudget {
    pub max_time_secs: u32,
    pub max_tokens: u64,
    pub max_cost_microdollars: u64,
    pub max_tool_calls: u32,
    pub max_external_writes: u32,
}
```

Check budgets before each tool call. Enforce `AuthorityLevel` (ReadOnly, WriteMemory, ExecuteTools, WriteExternal) on backends.

#### B5. SONA Adaptive Middleware [MEDIUM]

**Amends:** ADR-095 §Middleware Pipeline, ADR-100 §Existing Crate Integration

Add optional `SonaMiddleware` leveraging the three learning loops:

- **Loop A (Instant):** Record trajectories in `wrap_model_call` via lock-free `TrajectoryBuffer`
- **Loop B (Background):** Hourly `ReasoningBank` pattern extraction via background tokio task
- **Loop C (Deep):** Session-end consolidation with `EwcPlusPlus` for cross-session memory

#### B6. HNSW Semantic Skill/Memory Retrieval [MEDIUM]

**Amends:** ADR-098 §SkillsMiddleware, §MemoryMiddleware

Replace linear skill scanning with `ruvector-hyperbolic-hnsw` index:
- Index skill descriptions at startup
- Retrieve top-k relevant skills per query instead of injecting all
- **Impact:** 50-80% reduction in system prompt size with many skills

#### B7. CRDT State Merging for Parallel SubAgents [MEDIUM]

**Amends:** ADR-097 §SubAgent orchestration

Use `ruvector-replication::LastWriteWins` for deterministic merge of parallel subagent results:

```rust
use ruvector_replication::{VectorClock, LastWriteWins};
fn merge_subagent_results(parent: &AgentState, results: Vec<AgentState>) -> AgentState { ... }
```

---

### Part C: Security Amendments

#### C1. Atomic Path Resolution with Post-Open Verification [CRITICAL — SEC-001]

**Amends:** ADR-094 §FilesystemBackend

```rust
fn resolve_and_open(&self, path: &str, flags: i32) -> Result<File, FileOperationError> {
    let resolved = self.resolve_path(path)?;
    let file = OpenOptions::new()
        .custom_flags(libc::O_NOFOLLOW)
        .open(&resolved)?;
    // Post-open verification via /proc/self/fd/N
    let real_path = std::fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd()))?;
    if !real_path.starts_with(&self.cwd) {
        return Err(FileOperationError::InvalidPath);
    }
    Ok(file)
}
```

Change default to `virtual_mode=true` (SEC-002). Add `--no-follow` to ripgrep invocations (SEC-004).

#### C2. Shell Execution Hardening [CRITICAL — SEC-005]

**Amends:** ADR-094 §LocalShellBackend

Mandatory additions to `execute()`:
1. **Witness chain logging** for every command execution
2. **Optional command allowlist** via `CommandAllowlist` config
3. **Environment sanitization** — strip `SECRET`, `KEY`, `TOKEN`, `PASSWORD`, `CREDENTIAL`, `AWS_*`, `AZURE_*`, `GCP_*`, `DATABASE_URL`, `PRIVATE` patterns
4. **`env_clear()` + explicit safe env** — never inherit full parent environment

```rust
const SENSITIVE_ENV_PATTERNS: &[&str] = &[
    "SECRET", "KEY", "TOKEN", "PASSWORD", "CREDENTIAL",
    "AWS_", "AZURE_", "GCP_", "DATABASE_URL", "PRIVATE",
];
```

#### C3. Tool Result Sanitization [CRITICAL — SEC-009]

**Amends:** ADR-095 §Middleware Pipeline

Add `ToolResultSanitizerMiddleware` that wraps all tool results in clearly delimited blocks:

```rust
msg.with_content(format!(
    "<tool_output tool=\"{}\" id=\"{}\">\n{}\n</tool_output>",
    msg.tool_name(), msg.tool_call_id(), msg.content()
))
```

This is defense-in-depth against indirect prompt injection via file contents, grep results, or command output.

#### C4. AGENTS.md / SKILL.md Trust Verification [CRITICAL — SEC-010]

**Amends:** ADR-098 §MemoryMiddleware, §SkillsMiddleware

1. Hash verification against a signed manifest for trusted sources
2. Add `SecurityPolicy` field to `DeepAgentConfig` controlling untrusted file loading
3. Content size limits: YAML frontmatter max 4KB, skill file max 1MB (down from 10MB)
4. YAML bomb protection: explicit recursion depth and anchor expansion limits

#### C5. Sandbox Path Restriction Contract [CRITICAL — SEC-023]

**Amends:** ADR-094 §BaseSandbox

The `BaseSandbox` trait MUST specify that concrete implementations provide filesystem isolation. Add to the trait:

```rust
pub trait SandboxBackend: Backend + Send + Sync {
    fn execute(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse;
    fn id(&self) -> &str;
    /// Implementations MUST confine filesystem access to the sandbox root.
    fn sandbox_root(&self) -> &Path;
}
```

#### C6. ACP Server Authentication [HIGH — SEC-017]

**Amends:** ADR-099 §ACP Server

Mandatory axum middleware layers:
- **API key authentication** via `Authorization: Bearer` header
- **Rate limiting** (configurable, default 60 req/min)
- **Request body size limit** (default 1MB)
- **TLS enforcement** for non-localhost connections

#### C7. Unicode Security Module [HIGH — SEC-016]

**Amends:** ADR-099 §unicode_security.rs

The Rust port MUST implement full parity with Python's `unicode_security.py`:
- BiDi directional formatting controls (U+202A-U+202E, U+2066-U+2069)
- Zero-width characters (U+200B-U+200F, U+2060, U+FEFF)
- Script confusable detection (Cyrillic, Greek, Armenian homoglyphs)
- Punycode domain decoding and mixed-script URL detection

#### C8. SubAgent Result Validation [HIGH — SEC-011]

**Amends:** ADR-097 §SubAgent result handling

Add `SubAgentResultValidator`:
- Maximum response length (configurable, default 100KB)
- Strip control characters and known prompt injection patterns
- Rate-limit subagent tool calls to detect runaway behavior

#### C9. Session Encryption at Rest [MEDIUM — SEC-014, SEC-015]

**Amends:** ADR-099 §Sessions

- Session checkpoints encrypted via AES-256-GCM
- Conversation history offload uses unpredictable filenames (UUID) with 0600 permissions
- PII stripping before persistence (using patterns from `mcp-brain`)

#### C10. Skill Name ASCII Restriction [MEDIUM — SEC-022]

**Amends:** ADR-098 §validate_skill_name

Replace `c.is_alphabetic()` with `c.is_ascii_lowercase()` to prevent Unicode confusable attacks:

```rust
if (c.is_ascii_lowercase()) || c.is_ascii_digit() { continue; }
```

#### C11. CompositeBackend Path Re-Validation [MEDIUM — SEC-003]

**Amends:** ADR-094 §CompositeBackend

After prefix stripping, re-validate the resulting path against traversal:

```rust
fn route_path(&self, path: &str) -> Result<(BackendRef, String), FileOperationError> {
    let (backend, stripped, _prefix) = self.select_backend(path);
    if stripped.contains("..") || stripped.starts_with('~') {
        return Err(FileOperationError::InvalidPath);
    }
    Ok((backend, stripped))
}
```

#### C12. Tool Call ID Validation [MEDIUM — SEC-012]

**Amends:** ADR-098 §PatchToolCallsMiddleware

Validate tool call IDs: max 128 chars, ASCII alphanumeric + hyphens + underscores only.

#### C13. Grep Literal Mode Enforcement [MEDIUM — SEC-021]

**Amends:** ADR-094, ADR-096

The Python implementation uses `rg -F` (fixed-string/literal mode). The Rust port MUST default to literal mode. If regex mode is needed, use `regex` crate's built-in backtracking limits.

---

### Part D: Amended Phase Timeline

The original 20-week timeline (ADR-102) is amended to include security and integration work:

| Phase | Original | Amendment |
|---|---|---|
| 1 (Foundation) | Weeks 1-3 | Add: Typed AgentState, Arc patterns, arena integration |
| 2 (Backends) | Weeks 3-5 | Add: Atomic path resolution (C1), env sanitization (C2), literal grep (C13) |
| 3 (Tools) | Weeks 5-7 | Add: Enum dispatch (A6), grep-searcher lib (A4), line format optimization (A7) |
| 4 (Middleware) | Weeks 7-10 | Add: SystemPromptBuilder (A5), WitnessMiddleware (B3), ToolResultSanitizer (C3), trust verification (C4) |
| 5 (Providers) | Weeks 8-10 | Unchanged |
| 6 (Core) | Weeks 10-12 | Add: Parallel tool execution (A2), resource budgets (B4), subagent result validation (C8) |
| 7 (CLI) | Weeks 12-16 | Add: Unicode security (C7), session encryption (C9) |
| 8 (ACP) | Weeks 14-16 | Add: Authentication middleware (C6), TLS enforcement |
| 9 (WASM/RVF) | Weeks 16-18 | Add: Concrete AGI container building (B1), COW state (B2) |
| 10 (Verification) | Weeks 18-20 | Add: Criterion benchmarks (A9), security regression tests |
| **11 (Adaptive)** | **Weeks 20-22 (NEW)** | **SONA integration (B5), HNSW skills (B6), CRDT merge (B7)** |

Total: **22 weeks** (was 20).

---

## Consequences

### Performance
- Typed AgentState eliminates JSON tax (5-20x middleware speedup)
- Parallel tool execution (2-5x multi-tool speedup)
- Arena allocators and optimized hot paths reduce allocation pressure
- Criterion benchmarks prevent performance regressions

### Capability
- 7 concrete RVF integrations (AGI containers, COW state, witness chains, resource budgets, SONA, HNSW, CRDTs)
- Agent decisions become auditable via witness chains
- Adaptive learning via SONA enables agents that improve over sessions
- Semantic skill retrieval reduces prompt bloat

### Security
- 5 Critical, 7 High, 6 Medium findings addressed
- Defense-in-depth: atomic path resolution, env sanitization, tool result sanitization, trust verification
- ACP server hardened with auth, rate limiting, TLS
- Full Unicode security parity with Python source
- Session data encrypted at rest

### Timeline
- 2 additional weeks for Phase 11 (adaptive capabilities)
- Security hardening integrated into existing phases (no additional delay)
- Performance optimizations integrated into existing phases

---

## Appendix: Full Finding Cross-Reference

### Security Findings → Amendments

| Finding | Severity | Amendment | Phase |
|---|---|---|---|
| SEC-001 TOCTOU symlink race | Critical | C1 | 2 |
| SEC-005 Shell injection | Critical | C2 | 2 |
| SEC-009 Tool result prompt injection | Critical | C3 | 4 |
| SEC-010 AGENTS.md hijack | Critical | C4 | 4 |
| SEC-023 Sandbox path escape | Critical | C5 | 2 |
| SEC-002 virtual_mode default | High | C1 | 2 |
| SEC-004 Grep symlink following | High | C1 | 2 |
| SEC-006 Template injection | High | C2 | 2 |
| SEC-008 Env credential leak | High | C2 | 2 |
| SEC-011 SubAgent manipulation | High | C8 | 6 |
| SEC-015 History data exposure | High | C9 | 7 |
| SEC-016 Missing unicode security | High | C7 | 7 |
| SEC-017 ACP no authentication | High | C6 | 8 |
| SEC-020 YAML bomb | High | C4 | 4 |
| SEC-003 CompositeBackend traversal | Medium | C11 | 2 |
| SEC-007 Heredoc delimiter escape | Medium | C2 | 2 |
| SEC-012 Tool call ID injection | Medium | C12 | 4 |
| SEC-013 State type confusion | Medium | A1 | 1 |
| SEC-014 Unencrypted sessions | Medium | C9 | 7 |
| SEC-018 Missing TLS pinning | Medium | C6 | 8 |
| SEC-019 Sandbox credentials | Medium | C2 | 2 |
| SEC-021 ReDoS in grep | Medium | C13 | 2 |
| SEC-022 Unicode skill names | Medium | C10 | 4 |

### Performance Findings → Amendments

| Finding | Priority | Amendment | Phase |
|---|---|---|---|
| JSON AgentState tax | P0 | A1 | 1 |
| No parallel tool exec | P0 | A2 | 6 |
| Blocking I/O in async | P0 | A3 | 2 |
| No benchmarks | P0 | A9 | 10 |
| Arena allocators unused | P0 | A8 | 3 |
| Middleware pipeline overhead | P0 | A5 | 4 |
| String concatenation | P1 | A5 | 4 |
| Line formatting allocs | P1 | A7 | 3 |
| Trait object dispatch | P1 | A6 | 3 |
| Subprocess ripgrep | P1 | A4 | 2 |
| HNSW for skills | P1 | B6 | 11 |

### RVF Capability Findings → Amendments

| Gap | Severity | Amendment | Phase |
|---|---|---|---|
| No decision provenance | Critical | B3 | 4 |
| No resource governance | High | B4 | 6 |
| No adaptive learning | High | B5 | 11 |
| Linear skill scanning | Medium | B6 | 11 |
| Naive state cloning | Medium | B2 | 9 |
| No distributed agents | Medium | B7 | 11 |
| Aspirational RVF refs | Low | B1 | 9 |
