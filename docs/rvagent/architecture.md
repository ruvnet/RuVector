# rvAgent Architecture

This document describes the internal architecture of the rvAgent crate family, covering the crate dependency graph, agent lifecycle, middleware pipeline, backend protocol hierarchy, security model, and performance characteristics.

## Crate Dependency Graph

```
rvagent-cli
|-- rvagent-core
|   |-- rvagent-middleware
|   |   |-- rvagent-tools
|   |   |   |-- rvagent-backends
|   |   |   |-- rvagent-core
|   |   |-- rvagent-subagents
|   |   |   |-- rvagent-core
|   |   |   |-- rvagent-backends
|   |   |   |-- rvagent-middleware (traits only)
|   |   |   |-- rvagent-tools
|   |   |-- rvagent-backends
|   |   |-- rvagent-core
|   |-- rvagent-backends
|-- rvagent-subagents
|
rvagent-acp
|-- rvagent-core
|-- rvagent-backends
|-- rvagent-middleware
|-- rvagent-tools
|-- rvagent-subagents
|
rvagent-wasm
|-- (standalone, no workspace deps except serde/wasm-bindgen)
```

Dependencies flow strictly downward: `cli/acp` -> `core` -> `middleware` -> `tools`/`subagents` -> `backends`. There are no circular dependencies.

## Agent Lifecycle

An rvAgent invocation follows this lifecycle:

```
1. INIT
   |-- Parse RvAgentConfig (model, backend, security, middleware)
   |-- Resolve model via resolve_model("provider:model")
   |-- Construct backend (StateBackend, FilesystemBackend, LocalShellBackend, etc.)
   |-- Build middleware pipeline (ordered list of Middleware trait objects)
   |-- Compile subagent specs into CompiledSubAgent instances
   |
2. AGENT LOOP (repeats until no tool calls remain)
   |
   |-- 2a. before_agent
   |   |-- Each middleware's before_agent() runs in pipeline order
   |   |-- State updates accumulated (memory loading, skill discovery, etc.)
   |
   |-- 2b. Model Call
   |   |-- SystemPromptBuilder assembles system message from all middleware
   |   |-- wrap_model_call chain executes (outermost wraps innermost)
   |   |-- modify_request transforms applied
   |   |-- ChatModel.complete() or ChatModel.stream() invoked
   |   |-- Response: AiMessage with optional tool_calls
   |
   |-- 2c. Tool Dispatch
   |   |-- If no tool_calls: return response to user
   |   |-- Resolve each tool_call to a Tool implementation
   |   |-- Execute concurrently via tokio::task::JoinSet (ADR-103 A2)
   |   |-- Collect ToolResult for each call
   |   |-- Append ToolMessage to state.messages
   |   |-- Loop back to 2b
   |
3. RESPONSE
   |-- Final AiMessage returned to caller
   |-- State checkpointed for session resume (if session management active)
```

## Middleware Pipeline

The middleware pipeline executes in a fixed order. Each middleware can:

- Inject state via `before_agent()` (runs once per invocation)
- Wrap model calls via `wrap_model_call()` (runs on every LLM call)
- Transform requests via `modify_request()`
- Provide additional tools via `tools()`
- Declare state keys it manages via `state_keys()`

### Default Pipeline Order

```
 1. TodoListMiddleware          write_todos tool, task tracking state
 2. MemoryMiddleware            AGENTS.md loading into system prompt
 3. SkillsMiddleware            SKILL.md progressive disclosure
 4. FilesystemMiddleware        ls, read_file, write_file, edit_file, glob, grep, execute
 5. SubAgentMiddleware          task tool for subagent spawning
 6. SummarizationMiddleware     auto-compact when token budget exceeded
 7. PromptCachingMiddleware     cache control block injection (Anthropic)
 8. PatchToolCallsMiddleware    repair dangling tool calls
 9. WitnessMiddleware           SHAKE-256 tool call audit logging
10. ToolResultSanitizerMiddleware   delimited output blocks (anti-injection)
11. HumanInTheLoopMiddleware    interrupt on specified tools (optional)
```

User-defined middleware is inserted between PatchToolCallsMiddleware and WitnessMiddleware.

### Middleware Hook Execution

```
before_agent:      sequential, pipeline order (1 -> 2 -> ... -> 11)
wrap_model_call:   nested (11 wraps 10 wraps ... wraps 1 wraps base_handler)
modify_request:    sequential, pipeline order
tools:             collected from all middleware, merged into tool registry
```

## Backend Protocol Hierarchy

```
trait Backend (async_trait, Send + Sync)
|-- ls_info(path) -> Vec<FileInfo>
|-- read_file(path, offset, limit) -> Result<String, FileOperationError>
|-- write_file(path, content) -> WriteResult
|-- edit_file(path, old, new, replace_all) -> EditResult
|-- glob_info(pattern, path) -> Vec<FileInfo>
|-- grep(pattern, path, include) -> Result<Vec<GrepMatch>, String>
|-- download_files(paths) -> Vec<FileDownloadResponse>
|-- upload_files(files) -> Vec<FileUploadResponse>

trait SandboxBackend: Backend
|-- execute(command, timeout) -> ExecuteResponse
|-- id() -> &str
|-- sandbox_root() -> &Path
```

### Implementations

| Backend | Storage | Shell | Use Case |
|---|---|---|---|
| `StateBackend` | In-memory `HashMap` | No | WASM, testing, ephemeral |
| `FilesystemBackend` | Local disk | No | Read-only file access |
| `LocalShellBackend` | Local disk (extends `FilesystemBackend`) | Yes | Full coding agent |
| `CompositeBackend` | Routes to sub-backends by path prefix | Depends | Multi-workspace projects |
| `BaseSandbox` (trait) | Remote sandbox | Yes | Modal, Runloop, Daytona |

### Path Resolution

All backends enforce path safety:

1. `contains_traversal()` rejects `..` components
2. `is_safe_path_component()` rejects `.`, `..`, null bytes
3. `FilesystemBackend` uses `virtual_mode` (default: true) to confine paths within `cwd`
4. `CompositeBackend` re-validates paths after prefix stripping
5. `SandboxBackend` implementations must confine access to `sandbox_root()`

## Security Model

### Trust Boundaries

```
                    +----------------------------+
                    |  LLM Provider (external)   |
                    +----------------------------+
                              |  API calls
                    +----------------------------+
                    |       rvAgent Core          |
                    |  (middleware pipeline)      |
                    +----------------------------+
                    /           |            \
              +--------+  +----------+  +---------+
              | Memory |  |  Tools   |  | SubAgent|
              | Skills |  | (sandbox)|  | (isolated)
              +--------+  +----------+  +---------+
                                |
                    +----------------------------+
                    |   Backend (filesystem/     |
                    |   shell / sandbox)         |
                    +----------------------------+
```

### Threat Model Summary

| Threat | Control | ADR Reference |
|---|---|---|
| Path traversal / symlink race | Atomic resolve + post-open verification, `virtual_mode=true` | ADR-103 C1 |
| Shell injection | Environment sanitization, optional command allowlist | ADR-103 C2 |
| Indirect prompt injection via tool output | Tool result sanitizer middleware wraps outputs in delimited blocks | ADR-103 C3 |
| AGENTS.md / SKILL.md hijacking | Hash verification, size limits, YAML bomb protection | ADR-103 C4 |
| Sandbox path escape | `SandboxBackend.sandbox_root()` contract | ADR-103 C5 |
| ACP unauthenticated access | API key auth, rate limiting, body size limits, TLS | ADR-103 C6 |
| Unicode confusable attacks | BiDi/zero-width detection, mixed-script URL checking, ASCII skill names | ADR-103 C7, C10 |
| Subagent manipulation | Response length limits, control char stripping, rate limiting | ADR-103 C8 |
| Session data exposure | AES-256-GCM encryption at rest, UUID filenames, 0600 permissions | ADR-103 C9 |
| ReDoS in grep | Literal mode by default (`-F` flag equivalent) | ADR-103 C13 |
| Credential leakage via env | `SENSITIVE_ENV_PATTERNS` stripped before child process spawn | ADR-103 C2 |
| State type confusion | Typed `AgentState` struct replaces `HashMap<String, Value>` | ADR-103 A1 |
| Tool call ID injection | Max 128 chars, ASCII alphanumeric + hyphens + underscores | ADR-103 C12 |

## Performance Characteristics

### State Operations

| Operation | Complexity | Notes |
|---|---|---|
| `AgentState::clone()` | O(1) | Arc reference count increment |
| `AgentState::push_message()` | O(n) amortized | Copy-on-write via `Arc::make_mut` |
| `AgentState::merge_subagent()` | O(m) | m = child state size |
| Subagent spawn (state prep) | O(k) | k = number of non-excluded state keys |

### Tool Execution

| Aspect | Design |
|---|---|
| Built-in tool dispatch | Enum dispatch (no vtable) via `BuiltinTool` enum |
| User-defined tool dispatch | `Box<dyn Tool>` trait object |
| Parallel execution | `tokio::task::JoinSet` for concurrent tool calls |
| Grep | In-process via `grep-regex`/`grep-searcher` (no subprocess) |
| Line formatting | Single allocation with pre-calculated capacity |

### Middleware Pipeline

| Aspect | Design |
|---|---|
| `before_agent` overhead | O(n) where n = number of middleware |
| `wrap_model_call` overhead | O(n) nested function calls |
| System prompt construction | `SystemPromptBuilder` with `SmallVec<[Cow<'static, str>; 8]>`, single final allocation |
| State serialization | Typed struct avoids JSON parse/serialize overhead |

### Benchmarks

Each crate includes Criterion benchmarks:

- `rvagent-core`: `state_bench` -- state cloning, message operations, serialization
- `rvagent-backends`: `backend_bench` -- read/write/grep/glob latency per backend
- `rvagent-tools`: `tool_bench` -- tool invocation latency
- `rvagent-middleware`: `middleware_bench` -- full pipeline throughput (target: <1ms for 11-middleware chain)
