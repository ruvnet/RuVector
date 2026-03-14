//! `write_file` tool — creates or overwrites files.

use crate::{ToolResult, ToolRuntime};
use std::fs;
use std::path::Path;

/// Standalone write_file invocation using filesystem directly.
pub fn invoke_standalone(args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
    let file_path = match args.get("file_path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolResult::Text("Error: file_path is required".into()),
    };
    let content = match args.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return ToolResult::Text("Error: content is required".into()),
    };

    let base = runtime.cwd.as_deref().unwrap_or(".");
    let full_path = if Path::new(file_path).is_absolute() {
        Path::new(file_path).to_path_buf()
    } else {
        Path::new(base).join(file_path)
    };

    if let Some(parent) = full_path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return ToolResult::Text(format!("Error creating directory: {}", e));
        }
    }

    match fs::write(&full_path, content) {
        Ok(()) => ToolResult::Text(format!("Successfully wrote to {}", full_path.display())),
        Err(e) => ToolResult::Text(format!("Error writing {}: {}", full_path.display(), e)),
    }
}
