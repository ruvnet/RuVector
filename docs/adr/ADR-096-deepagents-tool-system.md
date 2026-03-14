# ADR-096: Tool System — Filesystem, Execute, Grep, Glob

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Accepted                                        |
| **Date**    | 2026-03-14                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents Rust Conversion)            |
| **Crate**   | `ruvector-deep-tools`                           |

## Context

DeepAgents' `FilesystemMiddleware` injects 7 core tools into the agent:

| Tool | Python Signature | Description |
|------|-----------------|-------------|
| `ls` | `ls(path, runtime)` | List directory contents |
| `read_file` | `read_file(file_path, offset?, limit?, runtime)` | Read file with line numbers |
| `write_file` | `write_file(file_path, content, runtime)` | Create new file |
| `edit_file` | `edit_file(file_path, old_string, new_string, replace_all?, runtime)` | String replacement |
| `glob` | `glob(pattern, path?, runtime)` | File pattern matching |
| `grep` | `grep(pattern, path?, include?, runtime)` | Literal text search |
| `execute` | `execute(command, timeout?, runtime)` | Shell command execution |

Plus a `write_todos` tool from `TodoListMiddleware`:

| Tool | Python Signature | Description |
|------|-----------------|-------------|
| `write_todos` | `write_todos(todos, runtime)` | Manage a todo list |

## Decision

### Tool Trait

```rust
// crates/ruvector-deep-tools/src/lib.rs

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Tool parameter with description (mirrors Python's Annotated[T, "description"])
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParam<T> {
    pub value: T,
    pub description: &'static str,
}

/// Runtime context passed to tool functions.
/// Python: ToolRuntime
pub struct ToolRuntime {
    pub state: AgentState,
    pub context: serde_json::Value,
    pub stream_writer: Option<Box<dyn StreamWriter>>,
    pub store: Option<Box<dyn Store>>,
    pub config: RunnableConfig,
    pub tool_call_id: Option<String>,
}

/// Result from tool execution — either content or a state update command.
/// Python: str | Command
pub enum ToolResult {
    /// Plain text result
    Text(String),
    /// State update command (used by write_file, edit_file for StateBackend)
    Command(StateUpdate),
}

/// Core tool trait.
/// Python: BaseTool / StructuredTool
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;

    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult;
    async fn ainvoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult;
}
```

### Tool Implementations

#### `ls` Tool

```rust
/// Python: FilesystemMiddleware._create_tools() → ls function
/// Lists directory contents with file metadata.
pub struct LsTool {
    backend: BackendRef,
}

impl Tool for LsTool {
    fn name(&self) -> &str { "ls" }

    fn invoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        let path: String = args["path"].as_str().unwrap_or("/").to_string();
        let backend = self.resolve_backend(runtime);
        let infos = backend.ls_info(&path);

        // Format output: Python uses specific formatting with GLOB_TIMEOUT
        let output = format_ls_output(&infos);
        ToolResult::Text(output)
    }
}
```

#### `read_file` Tool

```rust
/// Python: read_file(file_path, offset=0, limit=100, runtime)
/// DEFAULT_READ_OFFSET = 0, DEFAULT_READ_LIMIT = 100
pub struct ReadFileTool {
    backend: BackendRef,
}

impl Tool for ReadFileTool {
    fn invoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        let file_path = args["file_path"].as_str().unwrap();
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;

        let backend = self.resolve_backend(runtime);
        let content = backend.read(file_path, offset, limit);

        // Handle empty content warning
        // Python: EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
        ToolResult::Text(content)
    }
}
```

#### `write_file` Tool

```rust
/// Python: write_file(file_path, content, runtime) -> str | Command
/// Returns Command with files_update for StateBackend, text for others.
pub struct WriteFileTool {
    backend: BackendRef,
}

impl Tool for WriteFileTool {
    fn invoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        let file_path = args["file_path"].as_str().unwrap();
        let content = args["content"].as_str().unwrap();

        let backend = self.resolve_backend(runtime);
        let result = backend.write(file_path, content);

        match result.error {
            Some(err) => ToolResult::Text(err),
            None => {
                if let Some(files_update) = result.files_update {
                    // StateBackend: return Command to update LangGraph state
                    ToolResult::Command(StateUpdate::FilesUpdate(files_update))
                } else {
                    // Filesystem/Sandbox: file already written, return success
                    ToolResult::Text(format!("Successfully wrote to {}", file_path))
                }
            }
        }
    }
}
```

#### `edit_file` Tool

```rust
/// Python: edit_file(file_path, old_string, new_string, replace_all=False, runtime)
/// Exact string replacement with uniqueness check.
pub struct EditFileTool {
    backend: BackendRef,
}

impl Tool for EditFileTool {
    fn invoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        let file_path = args["file_path"].as_str().unwrap();
        let old_string = args["old_string"].as_str().unwrap();
        let new_string = args["new_string"].as_str().unwrap();
        let replace_all = args.get("replace_all").and_then(|v| v.as_bool()).unwrap_or(false);

        let backend = self.resolve_backend(runtime);
        let result = backend.edit(file_path, old_string, new_string, replace_all);

        match result.error {
            Some(err) => ToolResult::Text(err),
            None => {
                if let Some(files_update) = result.files_update {
                    ToolResult::Command(StateUpdate::FilesUpdate(files_update))
                } else {
                    let occurrences = result.occurrences.unwrap_or(1);
                    ToolResult::Text(format!(
                        "Successfully edited {} ({} occurrence{})",
                        file_path, occurrences,
                        if occurrences != 1 { "s" } else { "" }
                    ))
                }
            }
        }
    }
}
```

#### `glob` Tool

```rust
/// Python: glob(pattern, path="/", runtime) -> formatted file list
pub struct GlobTool {
    backend: BackendRef,
}

impl Tool for GlobTool {
    fn invoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        let pattern = args["pattern"].as_str().unwrap();
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("/");

        let backend = self.resolve_backend(runtime);

        // Python uses concurrent.futures with GLOB_TIMEOUT = 20.0
        let infos = backend.glob_info(pattern, path);
        let output = format_glob_output(&infos);
        ToolResult::Text(output)
    }
}
```

#### `grep` Tool

```rust
/// Python: grep(pattern, path=None, include=None, runtime) -> formatted matches
/// Note: Python param is 'include' (renamed from 'glob' for LLM clarity)
pub struct GrepTool {
    backend: BackendRef,
}

impl Tool for GrepTool {
    fn invoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        let pattern = args["pattern"].as_str().unwrap();
        let path = args.get("path").and_then(|v| v.as_str());
        let include = args.get("include").and_then(|v| v.as_str());

        let backend = self.resolve_backend(runtime);
        match backend.grep_raw(pattern, path, include) {
            Ok(matches) => {
                // Python: format_grep_matches() — path:line:text format
                let output = format_grep_output(&matches);
                ToolResult::Text(output)
            }
            Err(err) => ToolResult::Text(err),
        }
    }
}
```

#### `execute` Tool

```rust
/// Python: execute(command, timeout=None, runtime)
/// Only available when backend implements SandboxBackendProtocol.
pub struct ExecuteTool {
    backend: BackendRef,
}

impl Tool for ExecuteTool {
    fn invoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        let command = args["command"].as_str().unwrap();
        let timeout = args.get("timeout").and_then(|v| v.as_u64()).map(|t| t as u32);

        let backend = self.resolve_backend(runtime);

        // Check if backend supports execution
        if let Some(sandbox) = backend.as_sandbox() {
            let response = sandbox.execute(command, timeout);
            ToolResult::Text(response.output)
        } else {
            ToolResult::Text(
                "Error: Shell execution is not available. \
                 The current backend does not support command execution."
                .to_string()
            )
        }
    }
}
```

#### `write_todos` Tool

```rust
/// Python: TodoListMiddleware provides write_todos tool
/// Manages a structured todo list in agent state.
pub struct WriteTodosTool;

impl Tool for WriteTodosTool {
    fn name(&self) -> &str { "write_todos" }

    fn invoke(&self, args: Value, runtime: &ToolRuntime) -> ToolResult {
        let todos: Vec<TodoItem> = serde_json::from_value(args["todos"].clone()).unwrap();
        ToolResult::Command(StateUpdate::Todos(todos))
    }
}
```

### Image File Handling

```rust
/// Python: IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
/// read_file returns base64 image content blocks for image files.
const IMAGE_EXTENSIONS: &[&str] = &[".png", ".jpg", ".jpeg", ".gif", ".webp"];

fn is_image_file(path: &str) -> bool {
    IMAGE_EXTENSIONS.iter().any(|ext| path.to_lowercase().ends_with(ext))
}
```

### Output Formatting (Exact Fidelity)

```rust
/// Python: LINE_NUMBER_WIDTH = 6
/// Format: "     1\tcontent" (6-char right-aligned line number + tab + content)
const LINE_NUMBER_WIDTH: usize = 6;

pub fn format_content_with_line_numbers(lines: &[&str], start_line: usize) -> String {
    lines.iter().enumerate().map(|(i, line)| {
        let line_num = start_line + i;
        // Truncate lines longer than 2000 chars (same as Python)
        let truncated = if line.len() > 2000 { &line[..2000] } else { line };
        format!("{:>width$}\t{}", line_num, truncated, width = LINE_NUMBER_WIDTH)
    }).collect::<Vec<_>>().join("\n")
}
```

## Consequences

- All 8 tools ported with identical signatures and behavior
- Tool results match Python output format character-for-character
- StateBackend's `Command` return pattern preserved via `ToolResult::Command`
- Image file detection uses same extension set
- Line number formatting matches `cat -n` style exactly
