//! `glob` tool — file pattern matching (ADR-096).

use crate::{ToolResult, ToolRuntime};
use std::path::Path;

/// Synchronous glob invocation.
pub fn invoke(args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
    let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolResult::Text("Error: pattern is required".into()),
    };

    let base_path = args
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(".");

    let base = runtime.cwd.as_deref().unwrap_or(".");
    let search_path = Path::new(base).join(base_path);
    let full_pattern = search_path.join(pattern);

    match glob::glob(full_pattern.to_str().unwrap_or("")) {
        Ok(paths) => {
            let mut matches: Vec<String> = Vec::new();
            for entry in paths.flatten() {
                matches.push(entry.display().to_string());
            }
            matches.sort();
            if matches.is_empty() {
                ToolResult::Text("No files matched the pattern.".into())
            } else {
                ToolResult::Text(matches.join("\n"))
            }
        }
        Err(e) => ToolResult::Text(format!("Error in glob pattern: {}", e)),
    }
}
