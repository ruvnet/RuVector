//! FilesystemMiddleware — registers file operation tools (ls, read_file, write_file,
//! edit_file, glob, grep, execute).

use async_trait::async_trait;
use serde_json;

use crate::{AgentState, AgentStateUpdate, Middleware, RunnableConfig, Runtime, Tool};

/// Middleware that provides file operation tools.
///
/// - `before_agent`: registers the filesystem backend with runtime
/// - `tools()`: returns ls, read_file, write_file, edit_file, glob, grep, execute tools
pub struct FilesystemMiddleware {
    /// Working directory root for file operations.
    cwd: Option<String>,
}

impl FilesystemMiddleware {
    pub fn new() -> Self {
        Self { cwd: None }
    }

    pub fn with_cwd(cwd: impl Into<String>) -> Self {
        Self {
            cwd: Some(cwd.into()),
        }
    }
}

impl Default for FilesystemMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for FilesystemMiddleware {
    fn name(&self) -> &str {
        "filesystem"
    }

    fn before_agent(
        &self,
        _state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        if let Some(cwd) = &self.cwd {
            let mut update = AgentStateUpdate::default();
            update.extensions.insert(
                "filesystem_cwd".into(),
                serde_json::Value::String(cwd.clone()),
            );
            Some(update)
        } else {
            None
        }
    }

    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![
            Box::new(LsTool),
            Box::new(ReadFileTool),
            Box::new(WriteFileTool),
            Box::new(EditFileTool),
            Box::new(GlobTool),
            Box::new(GrepTool),
            Box::new(ExecuteTool),
        ]
    }
}

// ---------------------------------------------------------------------------
// Tool implementations (stubs — actual I/O delegated to backend at runtime)
// ---------------------------------------------------------------------------

macro_rules! fs_tool {
    ($name:ident, $tool_name:expr, $desc:expr, $schema:expr) => {
        struct $name;
        impl Tool for $name {
            fn name(&self) -> &str {
                $tool_name
            }
            fn description(&self) -> &str {
                $desc
            }
            fn parameters_schema(&self) -> serde_json::Value {
                $schema
            }
            fn invoke(&self, _args: serde_json::Value) -> Result<String, String> {
                Err("filesystem tool must be invoked through the agent runtime".into())
            }
        }
    };
}

fs_tool!(
    LsTool,
    "ls",
    "List files and directories at a given path.",
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Directory path to list" }
        },
        "required": ["path"]
    })
);

fs_tool!(
    ReadFileTool,
    "read_file",
    "Read the contents of a file. Supports offset and limit for large files.",
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "File path to read" },
            "offset": { "type": "integer", "description": "Line offset (0-based)" },
            "limit": { "type": "integer", "description": "Maximum lines to read" }
        },
        "required": ["path"]
    })
);

fs_tool!(
    WriteFileTool,
    "write_file",
    "Write content to a file, creating it if necessary.",
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "File path to write" },
            "content": { "type": "string", "description": "Content to write" }
        },
        "required": ["path", "content"]
    })
);

fs_tool!(
    EditFileTool,
    "edit_file",
    "Edit a file by replacing old_string with new_string.",
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "File path to edit" },
            "old_string": { "type": "string", "description": "Text to find and replace" },
            "new_string": { "type": "string", "description": "Replacement text" }
        },
        "required": ["path", "old_string", "new_string"]
    })
);

fs_tool!(
    GlobTool,
    "glob",
    "Find files matching a glob pattern.",
    serde_json::json!({
        "type": "object",
        "properties": {
            "pattern": { "type": "string", "description": "Glob pattern (e.g. **/*.rs)" },
            "path": { "type": "string", "description": "Base directory to search" }
        },
        "required": ["pattern"]
    })
);

fs_tool!(
    GrepTool,
    "grep",
    "Search file contents using a pattern. Uses literal mode by default (SEC-021).",
    serde_json::json!({
        "type": "object",
        "properties": {
            "pattern": { "type": "string", "description": "Search pattern" },
            "path": { "type": "string", "description": "Directory or file to search" },
            "literal": { "type": "boolean", "description": "Use literal (fixed-string) mode (default: true)" }
        },
        "required": ["pattern"]
    })
);

fs_tool!(
    ExecuteTool,
    "execute",
    "Execute a shell command. Subject to command allowlist and environment sanitization (SEC-005).",
    serde_json::json!({
        "type": "object",
        "properties": {
            "command": { "type": "string", "description": "Shell command to execute" },
            "timeout": { "type": "integer", "description": "Timeout in seconds" }
        },
        "required": ["command"]
    })
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_middleware_name() {
        let mw = FilesystemMiddleware::new();
        assert_eq!(mw.name(), "filesystem");
    }

    #[test]
    fn test_tools_count() {
        let mw = FilesystemMiddleware::new();
        let tools = mw.tools();
        assert_eq!(tools.len(), 7);
    }

    #[test]
    fn test_tool_names() {
        let mw = FilesystemMiddleware::new();
        let tools = mw.tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"ls"));
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"edit_file"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"execute"));
    }

    #[test]
    fn test_before_agent_no_cwd() {
        let mw = FilesystemMiddleware::new();
        let state = AgentState::default();
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        assert!(mw.before_agent(&state, &runtime, &config).is_none());
    }

    #[test]
    fn test_before_agent_with_cwd() {
        let mw = FilesystemMiddleware::with_cwd("/tmp/test");
        let state = AgentState::default();
        let runtime = Runtime::new();
        let config = RunnableConfig::default();
        let update = mw.before_agent(&state, &runtime, &config);
        assert!(update.is_some());
        let ext = &update.unwrap().extensions;
        assert_eq!(
            ext.get("filesystem_cwd").and_then(|v| v.as_str()),
            Some("/tmp/test")
        );
    }

    #[test]
    fn test_tools_return_error_without_runtime() {
        let mw = FilesystemMiddleware::new();
        for tool in mw.tools() {
            let result = tool.invoke(serde_json::json!({}));
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_tool_schemas_are_objects() {
        let mw = FilesystemMiddleware::new();
        for tool in mw.tools() {
            let schema = tool.parameters_schema();
            assert_eq!(schema["type"], "object");
        }
    }
}
