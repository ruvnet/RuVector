# ADR-101: Testing Strategy & Fidelity Verification

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-14                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |

## Context

DeepAgents has extensive test coverage:

- **Unit tests** — 80+ test files across `libs/deepagents/tests/unit_tests/`
- **Integration tests** — Cross-module tests in `tests/integration_tests/`
- **Eval tests** — LLM-powered behavioral tests in `tests/evals/`
- **CLI tests** — 40+ test files in `libs/cli/tests/`
- **Smoke tests** — System prompt validation

100% fidelity requires that the Rust implementation passes equivalent tests producing identical results.

## Decision

### Test Categories

#### 1. Unit Tests (Port from Python)

Each Python unit test file maps to a Rust test module:

| Python Test | Rust Test | Tests |
|---|---|---|
| `test_protocol.py` | `backends/protocol_test.rs` | FileInfo, GrepMatch, WriteResult, EditResult structs |
| `test_state_backend.py` | `backends/state_test.rs` | StateBackend CRUD, ls, grep, glob |
| `test_filesystem_backend.py` | `backends/filesystem_test.rs` | FilesystemBackend with real files |
| `test_filesystem_backend_async.py` | `backends/filesystem_async_test.rs` | Async variants |
| `test_local_shell_backend.py` | `backends/local_shell_test.rs` | Execute, timeout, output truncation |
| `test_composite_backend.py` | `backends/composite_test.rs` | Path routing, result remapping |
| `test_sandbox_backend.py` | `backends/sandbox_test.rs` | BaseSandbox command templates |
| `test_state_backend_async.py` | `backends/state_async_test.rs` | Async StateBackend |
| `test_store_backend.py` | `backends/store_test.rs` | StoreBackend persistence |
| `test_utils.py` | `backends/utils_test.rs` | format_content_with_line_numbers, perform_string_replacement |
| `test_file_system_tools.py` | `tools/filesystem_test.rs` | ls, read, write, edit, glob, grep tools |
| `test_file_system_tools_async.py` | `tools/filesystem_async_test.rs` | Async tool variants |
| `test_local_shell.py` | `tools/execute_test.rs` | Execute tool behavior |
| `test_middleware.py` | `middleware/pipeline_test.rs` | Middleware ordering, state injection |
| `test_middleware_async.py` | `middleware/pipeline_async_test.rs` | Async middleware |
| `test_subagents.py` | `subagents/task_test.rs` | Task tool, state isolation |
| `test_memory_middleware.py` | `middleware/memory_test.rs` | AGENTS.md loading |
| `test_skills_middleware.py` | `middleware/skills_test.rs` | SKILL.md parsing, validation |
| `test_summarization_middleware.py` | `middleware/summarization_test.rs` | Auto-compact trigger |
| `test_compact_tool.py` | `middleware/compact_tool_test.rs` | compact_conversation tool |
| `test_tool_schemas.py` | `tools/schema_test.rs` | Tool parameter schemas |
| `test_models.py` | `core/models_test.rs` | resolve_model, model_matches_spec |
| `test_end_to_end.py` | `core/e2e_test.rs` | Full agent invocation |
| `test_version.py` | `core/version_test.rs` | Version constant |

#### 2. Cross-Language Fidelity Tests

Golden-file tests that verify Rust output matches Python output exactly:

```rust
#[cfg(test)]
mod fidelity_tests {
    /// Test that format_content_with_line_numbers produces identical output.
    #[test]
    fn test_line_number_formatting_matches_python() {
        let lines = vec!["hello", "world", ""];
        let result = format_content_with_line_numbers(&lines, 1);
        // Must match Python's exact output character-for-character
        assert_eq!(result, "     1\thello\n     2\tworld\n     3\t");
    }

    /// Test that grep_raw produces identical GrepMatch structs.
    #[test]
    fn test_grep_matches_python_format() {
        let backend = FilesystemBackend::new(tmp_dir, false);
        // Write test file, grep, compare with Python golden output
    }

    /// Test that edit with replace_all=false rejects multiple occurrences.
    #[test]
    fn test_edit_uniqueness_check() {
        let backend = StateBackend::new(state_with_file("a\na\n"));
        let result = backend.edit("/test.txt", "a", "b", false);
        assert!(result.error.is_some());
        // Error message must match Python's exact wording
    }

    /// Test that CompositeBackend routes identically.
    #[test]
    fn test_composite_routing_matches_python() {
        // Same path inputs → same backend selection → same path stripping
    }
}
```

#### 3. Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    /// Any valid path resolves consistently between backends.
    #[test]
    fn path_resolution_consistent(path in "[a-z/]+") {
        let fs = FilesystemBackend::new(tmp, true);
        let resolved = fs.resolve_path(&path);
        // Verify: no path traversal, within root, deterministic
    }

    /// String replacement is idempotent for unique matches.
    #[test]
    fn edit_idempotent_unique(
        content in ".*",
        old in ".+",
        new in ".*",
    ) {
        // If old appears exactly once, edit succeeds
        // If old appears 0 or 2+ times, edit fails with correct error
    }

    /// Skill name validation matches spec exactly.
    #[test]
    fn skill_name_validation(name in "[a-z0-9-]{0,100}") {
        // validate_skill_name produces same result as Python
    }
}
```

#### 4. Integration Tests

```rust
// tests/integration/

/// Full agent creation and invocation with mock LLM.
#[tokio::test]
async fn test_create_deep_agent_with_defaults() {
    let agent = create_deep_agent(DeepAgentConfig {
        model: MockChatModel::new(),
        ..Default::default()
    });

    let result = agent.invoke(AgentState::from_messages(vec![
        HumanMessage::new("Hello"),
    ])).await;

    assert!(result.messages.last().unwrap().is_ai());
}

/// SubAgent task tool with parallel invocation.
#[tokio::test]
async fn test_parallel_subagent_invocation() {
    // Verify that two task tool calls execute concurrently
    // and both return results to the parent agent
}

/// Middleware pipeline ordering.
#[tokio::test]
async fn test_middleware_execution_order() {
    // Verify: TodoList → Memory → Skills → Filesystem →
    //         SubAgent → Summarization → PromptCaching → PatchToolCalls
}

/// Session persistence and resume.
#[tokio::test]
async fn test_session_round_trip() {
    // Create session → checkpoint → resume → verify state
}
```

#### 5. CLI Tests

```rust
// tests/cli/

/// CLI argument parsing matches Python's argparse behavior.
#[test]
fn test_cli_args() {
    let cli = Cli::try_parse_from(["deep", "--model", "openai:gpt-5", "-a", "myagent"]).unwrap();
    assert_eq!(cli.model.unwrap(), "openai:gpt-5");
    assert_eq!(cli.agent.unwrap(), "myagent");
}

/// Non-interactive mode produces same output format.
#[tokio::test]
async fn test_headless_mode() {
    // Run with --headless, verify JSON output matches Python
}
```

### Test Infrastructure

```rust
/// Mock ChatModel for deterministic testing.
/// Python: tests/unit_tests/chat_model.py
pub struct MockChatModel {
    responses: Vec<AIMessage>,
    call_count: AtomicUsize,
}

impl ChatModel for MockChatModel {
    fn invoke(&self, messages: &[Message]) -> AIMessage {
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        self.responses[idx].clone()
    }
}

/// Temporary directory helper for filesystem tests.
pub struct TempBackend {
    dir: tempfile::TempDir,
    backend: FilesystemBackend,
}

impl TempBackend {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let backend = FilesystemBackend::new(dir.path(), false);
        Self { dir, backend }
    }
}
```

### Coverage Targets

| Category | Target | Method |
|---|---|---|
| Backend protocol | 100% | Unit tests per method |
| Tool implementations | 100% | Golden-file fidelity tests |
| Middleware pipeline | 100% | Integration + ordering tests |
| State isolation | 100% | Property tests |
| Skill validation | 100% | Exhaustive + property tests |
| CLI args | 100% | Clap derive tests |
| Session persistence | 100% | Round-trip serialization |
| Error messages | 100% | Exact string matching |

## Consequences

- 80+ Python test files ported to Rust with identical assertions
- Golden-file tests guarantee character-for-character output fidelity
- Property-based tests catch edge cases not covered by Python suite
- MockChatModel enables deterministic agent testing without LLM calls
- CI runs both Python and Rust test suites to verify behavioral parity
