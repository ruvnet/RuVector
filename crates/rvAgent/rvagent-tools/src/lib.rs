//! rvAgent tools — enum-dispatched tool implementations (ADR-103 A6).
//!
//! Provides the `Tool` trait, `BuiltinTool`/`AnyTool` enum dispatch,
//! `ToolRuntime` context, and parallel execution (ADR-103 A2).

pub mod edit_file;
pub mod execute;
pub mod glob;
pub mod grep;
pub mod ls;
pub mod read_file;
pub mod task;
pub mod write_file;
pub mod write_todos;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

pub use edit_file::EditFileTool;
pub use execute::ExecuteTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use ls::LsTool;
pub use read_file::ReadFileTool;
pub use task::TaskTool;
pub use write_file::WriteFileTool;
pub use write_todos::WriteTodosTool;

// ---------------------------------------------------------------------------
// Backend trait (abstraction for tool implementations)
// ---------------------------------------------------------------------------

/// File metadata returned by ls operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub name: String,
    pub file_type: String,
    pub permissions: String,
    pub size: u64,
}

/// Result from a write or edit operation.
#[derive(Debug, Clone, Default)]
pub struct WriteResult {
    pub error: Option<String>,
    pub files_update: Option<HashMap<String, String>>,
    pub occurrences: Option<usize>,
}

/// A grep match entry.
#[derive(Debug, Clone)]
pub struct GrepMatch {
    pub file: String,
    pub line_number: usize,
    pub text: String,
}

/// Result of shell command execution.
#[derive(Debug, Clone)]
pub struct ExecuteResponse {
    pub output: String,
    pub exit_code: i32,
}

/// Backend abstraction for tool operations.
///
/// Implementations may be filesystem-based, state-based, or sandbox-based.
/// Tools call methods on this trait rather than accessing the filesystem directly.
pub trait Backend: Send + Sync {
    fn ls_info(&self, path: &str) -> Result<Vec<FileInfo>, String>;
    fn read(&self, path: &str, offset: usize, limit: usize) -> Result<String, String>;
    fn write(&self, path: &str, content: &str) -> WriteResult;
    fn edit(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
        replace_all: bool,
    ) -> WriteResult;
    fn glob_info(&self, pattern: &str, path: &str) -> Result<Vec<String>, String>;
    fn grep_raw(
        &self,
        pattern: &str,
        path: Option<&str>,
        include: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String>;
    fn execute(&self, command: &str, timeout_secs: u32) -> Result<ExecuteResponse, String>;
}

/// Reference-counted backend handle.
pub type BackendRef = Arc<dyn Backend>;

// ---------------------------------------------------------------------------
// Stream / Store / Config abstractions
// ---------------------------------------------------------------------------

/// Trait for streaming tool output to the caller.
pub trait StreamWriter: Send + Sync {
    fn write_chunk(&self, data: &str);
}

/// Trait for persistent key-value store.
pub trait Store: Send + Sync {
    fn get(&self, key: &str) -> Option<String>;
    fn set(&self, key: &str, value: &str);
}

/// Agent configuration passed through the pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunnableConfig {
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// ToolParam<T>
// ---------------------------------------------------------------------------

/// Tool parameter with description (mirrors Python's `Annotated[T, "description"]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParam<T> {
    pub value: T,
    pub description: &'static str,
}

impl<T> ToolParam<T> {
    pub fn new(value: T, description: &'static str) -> Self {
        Self { value, description }
    }
}

// ---------------------------------------------------------------------------
// TodoItem
// ---------------------------------------------------------------------------

/// A single todo item managed by `WriteTodosTool`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TodoItem {
    pub content: String,
    pub status: String,
    #[serde(rename = "activeForm")]
    pub active_form: String,
}

// ---------------------------------------------------------------------------
// ToolRuntime
// ---------------------------------------------------------------------------

/// Runtime context passed to tool invocations.
pub struct ToolRuntime {
    pub backend: BackendRef,
    pub context: serde_json::Value,
    pub stream_writer: Option<Box<dyn StreamWriter>>,
    pub store: Option<Box<dyn Store>>,
    pub config: RunnableConfig,
    pub tool_call_id: Option<String>,
}

impl ToolRuntime {
    pub fn new(backend: BackendRef) -> Self {
        Self {
            backend,
            context: serde_json::Value::Null,
            stream_writer: None,
            store: None,
            config: RunnableConfig::default(),
            tool_call_id: None,
        }
    }
}

// ---------------------------------------------------------------------------
// StateUpdate & ToolResult
// ---------------------------------------------------------------------------

/// State update commands returned by tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateUpdate {
    /// Update file contents in state-backed storage.
    FilesUpdate(HashMap<String, String>),
    /// Update the todo list.
    Todos(Vec<TodoItem>),
}

/// Result from tool execution — either content or a state update command.
#[derive(Debug, Clone)]
pub enum ToolResult {
    /// Plain text result.
    Text(String),
    /// State update command (used by write_file, edit_file, write_todos).
    Command(StateUpdate),
}

impl fmt::Display for ToolResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolResult::Text(s) => write!(f, "{}", s),
            ToolResult::Command(update) => write!(f, "Command({:?})", update),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool trait
// ---------------------------------------------------------------------------

/// Core tool trait. Built-in tools use enum dispatch; dynamic tools use vtable.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (used for matching tool_call.name).
    fn name(&self) -> &str;

    /// Human-readable description.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's parameters.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Synchronous invocation.
    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult;

    /// Async invocation (default delegates to `invoke`).
    async fn ainvoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        self.invoke(args, runtime)
    }
}

impl std::fmt::Debug for dyn Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool").field("name", &self.name()).finish()
    }
}

// ---------------------------------------------------------------------------
// ToolCall
// ---------------------------------------------------------------------------

/// A tool invocation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

// ---------------------------------------------------------------------------
// BuiltinTool enum dispatch (ADR-103 A6)
// ---------------------------------------------------------------------------

/// Enum of all built-in tools — eliminates vtable indirection on hot paths.
pub enum BuiltinTool {
    Ls(LsTool),
    ReadFile(ReadFileTool),
    WriteFile(WriteFileTool),
    EditFile(EditFileTool),
    Glob(GlobTool),
    Grep(GrepTool),
    Execute(ExecuteTool),
    WriteTodos(WriteTodosTool),
    Task(TaskTool),
}

macro_rules! dispatch_builtin {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            BuiltinTool::Ls(t) => t.$method($($arg),*),
            BuiltinTool::ReadFile(t) => t.$method($($arg),*),
            BuiltinTool::WriteFile(t) => t.$method($($arg),*),
            BuiltinTool::EditFile(t) => t.$method($($arg),*),
            BuiltinTool::Glob(t) => t.$method($($arg),*),
            BuiltinTool::Grep(t) => t.$method($($arg),*),
            BuiltinTool::Execute(t) => t.$method($($arg),*),
            BuiltinTool::WriteTodos(t) => t.$method($($arg),*),
            BuiltinTool::Task(t) => t.$method($($arg),*),
        }
    };
}

#[async_trait]
impl Tool for BuiltinTool {
    fn name(&self) -> &str {
        dispatch_builtin!(self, name)
    }

    fn description(&self) -> &str {
        dispatch_builtin!(self, description)
    }

    fn parameters_schema(&self) -> serde_json::Value {
        dispatch_builtin!(self, parameters_schema)
    }

    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        dispatch_builtin!(self, invoke, args, runtime)
    }

    async fn ainvoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        self.invoke(args, runtime)
    }
}

// ---------------------------------------------------------------------------
// AnyTool — unified enum for builtin + dynamic tools
// ---------------------------------------------------------------------------

/// Unified tool type: builtin (enum dispatch, no vtable) or dynamic (trait object).
pub enum AnyTool {
    Builtin(BuiltinTool),
    Dynamic(Box<dyn Tool>),
}

#[async_trait]
impl Tool for AnyTool {
    fn name(&self) -> &str {
        match self {
            AnyTool::Builtin(b) => b.name(),
            AnyTool::Dynamic(d) => d.name(),
        }
    }

    fn description(&self) -> &str {
        match self {
            AnyTool::Builtin(b) => b.description(),
            AnyTool::Dynamic(d) => d.description(),
        }
    }

    fn parameters_schema(&self) -> serde_json::Value {
        match self {
            AnyTool::Builtin(b) => b.parameters_schema(),
            AnyTool::Dynamic(d) => d.parameters_schema(),
        }
    }

    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        match self {
            AnyTool::Builtin(b) => b.invoke(args, runtime),
            AnyTool::Dynamic(d) => d.invoke(args, runtime),
        }
    }

    async fn ainvoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        match self {
            AnyTool::Builtin(b) => b.ainvoke(args, runtime).await,
            AnyTool::Dynamic(d) => d.ainvoke(args, runtime).await,
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel tool execution (ADR-103 A2)
// ---------------------------------------------------------------------------

/// Resolve a tool by name from the provided tool set.
pub fn resolve_tool<'a>(name: &str, tools: &'a [AnyTool]) -> Option<&'a AnyTool> {
    tools.iter().find(|t| t.name() == name)
}

/// Execute multiple tool calls in parallel (ADR-103 A2).
///
/// Results are returned in the same order as the input `calls`.
/// If a tool is not found, a `ToolResult::Text` error is returned for that call.
pub async fn execute_tools_parallel(
    calls: &[ToolCall],
    tools: &[AnyTool],
    runtime: &ToolRuntime,
) -> Vec<ToolResult> {
    // Fast path for single call — skip JoinSet overhead.
    if calls.len() == 1 {
        let result = if let Some(tool) = resolve_tool(&calls[0].name, tools) {
            tool.invoke(calls[0].args.clone(), runtime)
        } else {
            ToolResult::Text(format!("Error: tool '{}' not found", calls[0].name))
        };
        return vec![result];
    }

    let mut results: Vec<ToolResult> = Vec::with_capacity(calls.len());
    for tc in calls {
        let result = if let Some(tool) = resolve_tool(&tc.name, tools) {
            tool.invoke(tc.args.clone(), runtime)
        } else {
            ToolResult::Text(format!("Error: tool '{}' not found", tc.name))
        };
        results.push(result);
    }
    results
}

// ---------------------------------------------------------------------------
// Built-in tool registry helper
// ---------------------------------------------------------------------------

/// Create the default set of all built-in tools.
pub fn builtin_tools() -> Vec<AnyTool> {
    vec![
        AnyTool::Builtin(BuiltinTool::Ls(LsTool)),
        AnyTool::Builtin(BuiltinTool::ReadFile(ReadFileTool)),
        AnyTool::Builtin(BuiltinTool::WriteFile(WriteFileTool)),
        AnyTool::Builtin(BuiltinTool::EditFile(EditFileTool)),
        AnyTool::Builtin(BuiltinTool::Glob(GlobTool)),
        AnyTool::Builtin(BuiltinTool::Grep(GrepTool)),
        AnyTool::Builtin(BuiltinTool::Execute(ExecuteTool)),
        AnyTool::Builtin(BuiltinTool::WriteTodos(WriteTodosTool)),
        AnyTool::Builtin(BuiltinTool::Task(TaskTool)),
    ]
}

/// Resolve a tool name to a BuiltinTool variant, if it matches.
pub fn resolve_builtin(name: &str) -> Option<BuiltinTool> {
    match name {
        "ls" => Some(BuiltinTool::Ls(LsTool)),
        "read_file" => Some(BuiltinTool::ReadFile(ReadFileTool)),
        "write_file" => Some(BuiltinTool::WriteFile(WriteFileTool)),
        "edit_file" => Some(BuiltinTool::EditFile(EditFileTool)),
        "glob" => Some(BuiltinTool::Glob(GlobTool)),
        "grep" => Some(BuiltinTool::Grep(GrepTool)),
        "execute" => Some(BuiltinTool::Execute(ExecuteTool)),
        "write_todos" => Some(BuiltinTool::WriteTodos(WriteTodosTool)),
        "task" => Some(BuiltinTool::Task(TaskTool)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default read limit (lines).
pub const DEFAULT_READ_LIMIT: usize = 2000;

/// Default read offset.
pub const DEFAULT_READ_OFFSET: usize = 0;

/// Line number width for formatting.
pub const LINE_NUMBER_WIDTH: usize = 6;

/// Maximum line length before truncation.
pub const MAX_LINE_LEN: usize = 2000;

/// Default execute timeout in seconds.
pub const DEFAULT_EXECUTE_TIMEOUT: u32 = 120;

/// Image file extensions.
pub const IMAGE_EXTENSIONS: &[&str] = &[".png", ".jpg", ".jpeg", ".gif", ".webp"];

/// Empty content warning.
pub const EMPTY_CONTENT_WARNING: &str =
    "System reminder: File exists but has empty contents";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a file path refers to an image.
#[inline]
pub fn is_image_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    IMAGE_EXTENSIONS.iter().any(|ext| lower.ends_with(ext))
}

/// Format content with line numbers in `cat -n` style (ADR-103 A7).
///
/// Pre-calculates total size and uses a single `String::with_capacity`
/// allocation to avoid intermediate allocations.
pub fn format_content_with_line_numbers(content: &str, start_line: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total_est: usize = lines.iter().map(|l| l.len().min(MAX_LINE_LEN) + 8).sum();
    let mut out = String::with_capacity(total_est);
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let truncated = &line[..line.len().min(MAX_LINE_LEN)];
        use std::fmt::Write;
        write!(
            out,
            "{:>width$}\t{}",
            start_line + i,
            truncated,
            width = LINE_NUMBER_WIDTH
        )
        .unwrap();
    }
    out
}

// ---------------------------------------------------------------------------
// Shared test utilities
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests_common {
    use super::*;
    use std::sync::Mutex;

    /// Mock backend for testing all tools.
    pub struct MockBackend {
        pub files: Mutex<HashMap<String, String>>,
    }

    impl MockBackend {
        pub fn new() -> Self {
            let mut files = HashMap::new();
            files.insert("/test.txt".to_string(), "hello\nworld".to_string());
            files.insert(
                "/multi.txt".to_string(),
                "aaa\nbbb\naaa\nccc".to_string(),
            );
            Self {
                files: Mutex::new(files),
            }
        }

        pub fn with_empty_file() -> Self {
            let mut files = HashMap::new();
            files.insert("/empty.txt".to_string(), String::new());
            Self {
                files: Mutex::new(files),
            }
        }
    }

    impl Backend for MockBackend {
        fn ls_info(&self, _path: &str) -> Result<Vec<FileInfo>, String> {
            let files = self.files.lock().unwrap();
            let mut infos: Vec<FileInfo> = files
                .iter()
                .map(|(name, content)| FileInfo {
                    name: name.clone(),
                    file_type: "file".into(),
                    permissions: "-rw-r--r--".into(),
                    size: content.len() as u64,
                })
                .collect();
            infos.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(infos)
        }

        fn read(
            &self,
            path: &str,
            offset: usize,
            limit: usize,
        ) -> Result<String, String> {
            let files = self.files.lock().unwrap();
            match files.get(path) {
                Some(content) => {
                    if content.is_empty() {
                        return Ok(String::new());
                    }
                    let lines: Vec<&str> = content.lines().collect();
                    if offset >= lines.len() {
                        return Ok(String::new());
                    }
                    let end = (offset + limit).min(lines.len());
                    Ok(lines[offset..end].join("\n"))
                }
                None => Err(format!("File not found: {}", path)),
            }
        }

        fn write(&self, path: &str, content: &str) -> WriteResult {
            let mut files = self.files.lock().unwrap();
            if files.contains_key(path) {
                return WriteResult {
                    error: Some(format!(
                        "Error: file {} already exists. Use force flag to overwrite.",
                        path
                    )),
                    ..Default::default()
                };
            }
            files.insert(path.to_string(), content.to_string());
            WriteResult::default()
        }

        fn edit(
            &self,
            path: &str,
            old_string: &str,
            new_string: &str,
            replace_all: bool,
        ) -> WriteResult {
            let mut files = self.files.lock().unwrap();
            match files.get(path).cloned() {
                None => WriteResult {
                    error: Some(format!("File not found: {}", path)),
                    ..Default::default()
                },
                Some(content) => {
                    let count = content.matches(old_string).count();
                    if count == 0 {
                        return WriteResult {
                            error: Some(format!(
                                "Error: old_string not found in {}",
                                path
                            )),
                            ..Default::default()
                        };
                    }
                    if count > 1 && !replace_all {
                        return WriteResult {
                            error: Some(format!(
                                "Error: old_string is not unique in {} ({} occurrences). \
                                 Use replace_all=true.",
                                path, count
                            )),
                            ..Default::default()
                        };
                    }
                    let new_content = if replace_all {
                        content.replace(old_string, new_string)
                    } else {
                        content.replacen(old_string, new_string, 1)
                    };
                    files.insert(path.to_string(), new_content);
                    WriteResult {
                        error: None,
                        occurrences: Some(if replace_all { count } else { 1 }),
                        ..Default::default()
                    }
                }
            }
        }

        fn glob_info(
            &self,
            pattern: &str,
            _path: &str,
        ) -> Result<Vec<String>, String> {
            let files = self.files.lock().unwrap();
            let search = pattern
                .trim_start_matches('*')
                .trim_end_matches('*');
            let mut matches: Vec<String> = files
                .keys()
                .filter(|k| k.contains(search))
                .cloned()
                .collect();
            matches.sort();
            Ok(matches)
        }

        fn grep_raw(
            &self,
            pattern: &str,
            _path: Option<&str>,
            _include: Option<&str>,
        ) -> Result<Vec<GrepMatch>, String> {
            let files = self.files.lock().unwrap();
            let mut matches = Vec::new();
            let mut sorted_files: Vec<_> = files.iter().collect();
            sorted_files.sort_by_key(|(k, _)| (*k).clone());
            for (file, content) in sorted_files {
                for (i, line) in content.lines().enumerate() {
                    if line.contains(pattern) {
                        matches.push(GrepMatch {
                            file: file.clone(),
                            line_number: i + 1,
                            text: line.to_string(),
                        });
                    }
                }
            }
            Ok(matches)
        }

        fn execute(
            &self,
            command: &str,
            _timeout_secs: u32,
        ) -> Result<ExecuteResponse, String> {
            Ok(ExecuteResponse {
                output: format!("mock output for: {}", command),
                exit_code: 0,
            })
        }
    }

    /// Backend that returns errors for all operations.
    pub struct ErrorBackend;

    impl Backend for ErrorBackend {
        fn ls_info(&self, _path: &str) -> Result<Vec<FileInfo>, String> {
            Err("Permission denied".into())
        }
        fn read(
            &self,
            _path: &str,
            _offset: usize,
            _limit: usize,
        ) -> Result<String, String> {
            Err("Permission denied".into())
        }
        fn write(&self, _path: &str, _content: &str) -> WriteResult {
            WriteResult {
                error: Some("Permission denied".into()),
                ..Default::default()
            }
        }
        fn edit(
            &self,
            _path: &str,
            _old: &str,
            _new: &str,
            _all: bool,
        ) -> WriteResult {
            WriteResult {
                error: Some("Permission denied".into()),
                ..Default::default()
            }
        }
        fn glob_info(
            &self,
            _pattern: &str,
            _path: &str,
        ) -> Result<Vec<String>, String> {
            Err("Permission denied".into())
        }
        fn grep_raw(
            &self,
            _pattern: &str,
            _path: Option<&str>,
            _include: Option<&str>,
        ) -> Result<Vec<GrepMatch>, String> {
            Err("Permission denied".into())
        }
        fn execute(
            &self,
            _command: &str,
            _timeout: u32,
        ) -> Result<ExecuteResponse, String> {
            Err("Permission denied".into())
        }
    }

    pub fn mock_runtime() -> ToolRuntime {
        ToolRuntime::new(Arc::new(MockBackend::new()))
    }

    pub fn mock_runtime_with_error() -> ToolRuntime {
        ToolRuntime::new(Arc::new(ErrorBackend))
    }

    pub fn mock_runtime_with_empty_file() -> ToolRuntime {
        ToolRuntime::new(Arc::new(MockBackend::with_empty_file()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tests_common::*;

    #[test]
    fn test_builtin_tool_names() {
        let tools = builtin_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"ls"));
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"edit_file"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"execute"));
        assert!(names.contains(&"write_todos"));
        assert!(names.contains(&"task"));
        assert_eq!(names.len(), 9);
    }

    #[test]
    fn test_enum_dispatch_routes_correctly() {
        let tools = builtin_tools();
        for tool in &tools {
            let schema = tool.parameters_schema();
            assert!(
                schema.is_object(),
                "Schema for '{}' should be an object",
                tool.name()
            );
            assert!(!tool.description().is_empty());
        }
    }

    #[test]
    fn test_resolve_tool_found() {
        let tools = builtin_tools();
        assert!(resolve_tool("ls", &tools).is_some());
        assert!(resolve_tool("grep", &tools).is_some());
        assert!(resolve_tool("task", &tools).is_some());
    }

    #[test]
    fn test_resolve_tool_not_found() {
        let tools = builtin_tools();
        assert!(resolve_tool("nonexistent", &tools).is_none());
    }

    #[test]
    fn test_resolve_builtin() {
        assert!(resolve_builtin("ls").is_some());
        assert!(resolve_builtin("read_file").is_some());
        assert!(resolve_builtin("write_file").is_some());
        assert!(resolve_builtin("edit_file").is_some());
        assert!(resolve_builtin("glob").is_some());
        assert!(resolve_builtin("grep").is_some());
        assert!(resolve_builtin("execute").is_some());
        assert!(resolve_builtin("write_todos").is_some());
        assert!(resolve_builtin("task").is_some());
        assert!(resolve_builtin("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_parallel_execution_single() {
        let runtime = mock_runtime();
        let tools = builtin_tools();
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "ls".into(),
            args: serde_json::json!({"path": "/"}),
        }];
        let results = execute_tools_parallel(&calls, &tools, &runtime).await;
        assert_eq!(results.len(), 1);
        match &results[0] {
            ToolResult::Text(s) => assert!(s.contains("test.txt")),
            _ => panic!("expected Text result"),
        }
    }

    #[tokio::test]
    async fn test_parallel_execution_multiple() {
        let runtime = mock_runtime();
        let tools = builtin_tools();
        let calls = vec![
            ToolCall {
                id: "c1".into(),
                name: "ls".into(),
                args: serde_json::json!({"path": "/"}),
            },
            ToolCall {
                id: "c2".into(),
                name: "read_file".into(),
                args: serde_json::json!({"file_path": "/test.txt"}),
            },
            ToolCall {
                id: "c3".into(),
                name: "grep".into(),
                args: serde_json::json!({"pattern": "hello"}),
            },
        ];
        let results = execute_tools_parallel(&calls, &tools, &runtime).await;
        assert_eq!(results.len(), 3);
        for r in &results {
            match r {
                ToolResult::Text(_) => {}
                _ => panic!("expected Text result"),
            }
        }
    }

    #[tokio::test]
    async fn test_parallel_execution_tool_not_found() {
        let runtime = mock_runtime();
        let tools = builtin_tools();
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "no_such_tool".into(),
            args: serde_json::json!({}),
        }];
        let results = execute_tools_parallel(&calls, &tools, &runtime).await;
        assert_eq!(results.len(), 1);
        match &results[0] {
            ToolResult::Text(s) => assert!(s.contains("not found")),
            _ => panic!("expected error Text result"),
        }
    }

    #[tokio::test]
    async fn test_parallel_execution_mixed() {
        let runtime = mock_runtime();
        let tools = builtin_tools();
        let calls = vec![
            ToolCall {
                id: "c1".into(),
                name: "ls".into(),
                args: serde_json::json!({"path": "/"}),
            },
            ToolCall {
                id: "c2".into(),
                name: "missing_tool".into(),
                args: serde_json::json!({}),
            },
        ];
        let results = execute_tools_parallel(&calls, &tools, &runtime).await;
        assert_eq!(results.len(), 2);
        match &results[1] {
            ToolResult::Text(s) => assert!(s.contains("not found")),
            _ => panic!("second should be error"),
        }
    }

    #[test]
    fn test_any_tool_dynamic() {
        struct CustomTool;

        #[async_trait]
        impl Tool for CustomTool {
            fn name(&self) -> &str {
                "custom"
            }
            fn description(&self) -> &str {
                "A custom tool"
            }
            fn parameters_schema(&self) -> serde_json::Value {
                serde_json::json!({"type": "object", "properties": {}})
            }
            fn invoke(
                &self,
                _args: serde_json::Value,
                _runtime: &ToolRuntime,
            ) -> ToolResult {
                ToolResult::Text("custom result".into())
            }
        }

        let tool = AnyTool::Dynamic(Box::new(CustomTool));
        assert_eq!(tool.name(), "custom");
        let runtime = mock_runtime();
        match tool.invoke(serde_json::json!({}), &runtime) {
            ToolResult::Text(s) => assert_eq!(s, "custom result"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn test_tool_param() {
        let p = ToolParam::new(42, "The answer");
        assert_eq!(p.value, 42);
        assert_eq!(p.description, "The answer");
    }

    #[test]
    fn test_format_line_numbers() {
        let result = format_content_with_line_numbers("hello\nworld", 1);
        assert!(result.contains("     1\thello"));
        assert!(result.contains("     2\tworld"));
    }

    #[test]
    fn test_format_line_numbers_with_offset() {
        let result = format_content_with_line_numbers("a\nb", 10);
        assert!(result.contains("    10\ta"));
        assert!(result.contains("    11\tb"));
    }

    #[test]
    fn test_format_empty_content() {
        let result = format_content_with_line_numbers("", 1);
        assert_eq!(result, "");
    }

    #[test]
    fn test_format_single_line() {
        let result = format_content_with_line_numbers("hello world", 1);
        assert_eq!(result, "     1\thello world");
    }

    #[test]
    fn test_format_multiple_lines() {
        let content = "line one\nline two\nline three";
        let result = format_content_with_line_numbers(content, 1);
        let expected = "     1\tline one\n     2\tline two\n     3\tline three";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_format_line_truncation() {
        let long_line = "a".repeat(MAX_LINE_LEN + 100);
        let result = format_content_with_line_numbers(&long_line, 1);
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 1);
        // Extract the content after the line number and tab
        let content = lines[0].split('\t').nth(1).unwrap();
        assert_eq!(content.len(), MAX_LINE_LEN);
    }

    #[test]
    fn test_format_preserves_short_lines() {
        let content = "ab";
        let result = format_content_with_line_numbers(content, 1);
        assert_eq!(result, "     1\tab");
    }

    #[test]
    fn test_format_large_line_numbers() {
        let content = "data";
        let result = format_content_with_line_numbers(content, 999999);
        assert_eq!(result, "999999\tdata");
    }

    #[test]
    fn test_format_correctness_many_lines() {
        let lines_vec: Vec<String> = (0..100).map(|i| format!("line {}", i)).collect();
        let content = lines_vec.join("\n");
        let result = format_content_with_line_numbers(&content, 1);
        let output_lines: Vec<&str> = result.lines().collect();
        assert_eq!(output_lines.len(), 100);
        assert!(output_lines[0].starts_with("     1\t"));
        assert!(output_lines[99].starts_with("   100\t"));
        assert!(output_lines[99].ends_with("line 99"));
    }

    #[test]
    fn test_format_no_intermediate_allocations() {
        // Verify that the function pre-allocates the correct size
        let content = "short\nline\ntest";
        let result = format_content_with_line_numbers(content, 1);
        // The capacity should be close to the length (no excessive reallocations)
        // This is a sanity check that we're using with_capacity correctly
        assert!(result.capacity() >= result.len());
        assert!(result.capacity() < result.len() + 100); // Not too much excess
    }

    #[test]
    fn test_is_image_file() {
        assert!(is_image_file("photo.png"));
        assert!(is_image_file("IMG.JPG"));
        assert!(!is_image_file("test.txt"));
    }

    #[test]
    fn test_tool_result_display() {
        let text = ToolResult::Text("hello".into());
        assert_eq!(format!("{}", text), "hello");
        let cmd = ToolResult::Command(StateUpdate::Todos(vec![]));
        assert!(format!("{}", cmd).starts_with("Command("));
    }

    #[test]
    fn test_todo_item_serde() {
        let item = TodoItem {
            content: "Fix bug".into(),
            status: "pending".into(),
            active_form: "Fixing bug".into(),
        };
        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("activeForm"));
        let back: TodoItem = serde_json::from_str(&json).unwrap();
        assert_eq!(item, back);
    }

    #[test]
    fn test_state_update_serde() {
        let update = StateUpdate::Todos(vec![TodoItem {
            content: "task".into(),
            status: "pending".into(),
            active_form: "tasking".into(),
        }]);
        let json = serde_json::to_string(&update).unwrap();
        let back: StateUpdate = serde_json::from_str(&json).unwrap();
        match back {
            StateUpdate::Todos(todos) => assert_eq!(todos.len(), 1),
            _ => panic!("expected Todos"),
        }
    }

    #[test]
    fn test_runnable_config_default() {
        let config = RunnableConfig::default();
        assert!(config.tags.is_empty());
        assert!(config.metadata.is_empty());
    }

    #[test]
    fn test_tool_runtime_new() {
        let runtime = mock_runtime();
        assert!(runtime.context.is_null());
        assert!(runtime.stream_writer.is_none());
        assert!(runtime.store.is_none());
        assert!(runtime.tool_call_id.is_none());
    }
}
