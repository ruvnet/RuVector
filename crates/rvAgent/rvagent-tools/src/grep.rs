//! `grep` tool — literal text search (NOT regex, per ADR-103 C13).

use crate::{ToolResult, ToolRuntime};
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Standalone grep invocation using filesystem directly.
/// Uses literal (fixed-string) matching per ADR-103 C13.
pub fn invoke_standalone(args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
    let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolResult::Text("Error: pattern is required".into()),
    };

    let search_path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
    let include = args.get("include").and_then(|v| v.as_str());

    let base = runtime.cwd.as_deref().unwrap_or(".");
    let full_path = if Path::new(search_path).is_absolute() {
        Path::new(search_path).to_path_buf()
    } else {
        Path::new(base).join(search_path)
    };

    let mut results: Vec<String> = Vec::new();

    for entry in WalkDir::new(&full_path)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();

        if let Some(inc) = include {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if let Ok(glob) = glob::Pattern::new(inc) {
                if !glob.matches(name) {
                    continue;
                }
            }
        }

        if let Ok(content) = fs::read_to_string(path) {
            for (line_num, line) in content.lines().enumerate() {
                if line.contains(pattern) {
                    results.push(format!(
                        "{}:{}:{}",
                        path.display(),
                        line_num + 1,
                        line
                    ));
                }
            }
        }
    }

    if results.is_empty() {
        ToolResult::Text("No results found.".into())
    } else {
        ToolResult::Text(results.join("\n"))
    }
}
