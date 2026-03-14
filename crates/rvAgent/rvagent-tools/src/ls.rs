//! `ls` tool — lists directory contents with file metadata.

use crate::{ToolResult, ToolRuntime};
use std::fs;
use std::path::Path;

/// Standalone ls invocation using filesystem directly.
pub fn invoke_standalone(args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
    let path_str = args
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    let base = runtime.cwd.as_deref().unwrap_or(".");
    let full_path = if Path::new(path_str).is_absolute() {
        Path::new(path_str).to_path_buf()
    } else {
        Path::new(base).join(path_str)
    };

    match fs::read_dir(&full_path) {
        Ok(entries) => {
            let mut lines: Vec<String> = Vec::new();
            for entry in entries.flatten() {
                let meta = entry.metadata();
                let is_dir = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);
                let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
                let name = entry.file_name().to_string_lossy().to_string();
                let suffix = if is_dir { "/" } else { "" };
                lines.push(format!("{}{}\t{}", name, suffix, size));
            }
            lines.sort();
            ToolResult::Text(lines.join("\n"))
        }
        Err(e) => ToolResult::Text(format!("Error listing {}: {}", full_path.display(), e)),
    }
}
