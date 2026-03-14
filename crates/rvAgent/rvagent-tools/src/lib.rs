//! rvAgent tools — ls, read, write, edit, glob, grep, execute, todos, task.
//!
//! Provides the `Tool` async trait, `BuiltinTool` enum dispatch (ADR-103 A6),
//! `AnyTool` wrapper, and `ToolRuntime` context.

pub mod edit_file;
pub mod execute;
pub mod glob_tool;
pub mod grep;
pub mod ls;
pub mod read_file;
pub mod write_file;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Tool trait (ADR-096)
// ---------------------------------------------------------------------------

/// Result from tool execution — either plain text or a state update command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolResult {
    /// Plain text result.
    Text(String),
    /// State update command (files, todos, etc.).
    Command(StateUpdate),
}

/// State update returned by tools that modify agent state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateUpdate {
    FilesUpdate(HashMap<String, serde_json::Value>),
    Todos(Vec<serde_json::Value>),
}

/// Runtime context passed to tool functions.
#[derive(Debug, Clone)]
pub struct ToolRuntime {
    pub context: serde_json::Value,
    pub tool_call_id: Option<String>,
    pub cwd: Option<String>,
}

impl ToolRuntime {
    pub fn new() -> Self {
        Self {
            context: serde_json::Value::Null,
            tool_call_id: None,
            cwd: None,
        }
    }

    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }
}

impl Default for ToolRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// Core tool trait (ADR-096).
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;

    /// Synchronous invocation.
    fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult;

    /// Async invocation (defaults to sync).
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
// Enum dispatch for built-in tools (ADR-103 A6)
// ---------------------------------------------------------------------------

/// Built-in tool variants — enum dispatch eliminates vtable indirection.
#[derive(Debug, Clone)]
pub enum BuiltinTool {
    Ls,
    ReadFile,
    WriteFile,
    EditFile,
    Glob,
    Grep,
    Execute,
    WriteTodos,
    Task,
}

impl BuiltinTool {
    /// Get the tool name as a static string.
    pub fn tool_name(&self) -> &'static str {
        match self {
            Self::Ls => "ls",
            Self::ReadFile => "read_file",
            Self::WriteFile => "write_file",
            Self::EditFile => "edit_file",
            Self::Glob => "glob",
            Self::Grep => "grep",
            Self::Execute => "execute",
            Self::WriteTodos => "write_todos",
            Self::Task => "task",
        }
    }

    /// Invoke the builtin tool with the given args.
    pub fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        match self {
            Self::Ls => ls::invoke(args, runtime),
            Self::ReadFile => read_file::invoke(args, runtime),
            Self::WriteFile => write_file::invoke(args, runtime),
            Self::EditFile => edit_file::invoke(args, runtime),
            Self::Glob => glob_tool::invoke(args, runtime),
            Self::Grep => grep::invoke(args, runtime),
            Self::Execute => execute::invoke(args, runtime),
            Self::WriteTodos => {
                ToolResult::Text("write_todos: stub".into())
            }
            Self::Task => {
                ToolResult::Text("task: stub".into())
            }
        }
    }

    /// Async invocation.
    pub async fn ainvoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        match self {
            Self::Execute => execute::ainvoke(args, runtime).await,
            _ => self.invoke(args, runtime),
        }
    }
}

/// Wrapper that unifies built-in (enum dispatch) and dynamic (trait object) tools.
#[derive(Debug)]
pub enum AnyTool {
    Builtin(BuiltinTool),
    Dynamic(Box<dyn Tool>),
}

impl AnyTool {
    pub fn tool_name(&self) -> &str {
        match self {
            Self::Builtin(b) => b.tool_name(),
            Self::Dynamic(d) => d.name(),
        }
    }

    pub fn invoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        match self {
            Self::Builtin(b) => b.invoke(args, runtime),
            Self::Dynamic(d) => d.invoke(args, runtime),
        }
    }

    pub async fn ainvoke(&self, args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
        match self {
            Self::Builtin(b) => b.ainvoke(args, runtime).await,
            Self::Dynamic(d) => d.ainvoke(args, runtime).await,
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel tool execution (ADR-103 A2)
// ---------------------------------------------------------------------------

/// Execute multiple tool calls concurrently using tokio::JoinSet.
pub async fn execute_tool_calls_parallel(
    tools: &[AnyTool],
    calls: Vec<(usize, serde_json::Value)>,
    runtime: &ToolRuntime,
) -> Vec<(usize, ToolResult)> {
    use tokio::task::JoinSet;

    let runtime = Arc::new(runtime.clone());
    let mut set = JoinSet::new();

    for (idx, args) in calls {
        let rt = runtime.clone();
        let builtin = match &tools[idx] {
            AnyTool::Builtin(b) => Some(b.clone()),
            AnyTool::Dynamic(_) => None,
        };

        if let Some(b) = builtin {
            set.spawn(async move {
                let result = b.ainvoke(args, &rt).await;
                (idx, result)
            });
        } else {
            let result = tools[idx].invoke(args.clone(), &runtime);
            set.spawn(async move { (idx, result) });
        }
    }

    let mut results = Vec::new();
    while let Some(res) = set.join_next().await {
        if let Ok(r) = res {
            results.push(r);
        }
    }
    results.sort_by_key(|(idx, _)| *idx);
    results
}

// ---------------------------------------------------------------------------
// Tool resolution
// ---------------------------------------------------------------------------

/// Resolve a tool name to a BuiltinTool variant, if it matches.
pub fn resolve_builtin(name: &str) -> Option<BuiltinTool> {
    match name {
        "ls" => Some(BuiltinTool::Ls),
        "read_file" => Some(BuiltinTool::ReadFile),
        "write_file" => Some(BuiltinTool::WriteFile),
        "edit_file" => Some(BuiltinTool::EditFile),
        "glob" => Some(BuiltinTool::Glob),
        "grep" => Some(BuiltinTool::Grep),
        "execute" => Some(BuiltinTool::Execute),
        "write_todos" => Some(BuiltinTool::WriteTodos),
        "task" => Some(BuiltinTool::Task),
        _ => None,
    }
}
