//! `edit_file` tool — exact string replacement in files.

use crate::{ToolResult, ToolRuntime};
use std::fs;
use std::path::Path;

/// Standalone edit_file invocation using filesystem directly.
pub fn invoke_standalone(args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
    let file_path = match args.get("file_path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolResult::Text("Error: file_path is required".into()),
    };
    let old_string = match args.get("old_string").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ToolResult::Text("Error: old_string is required".into()),
    };
    let new_string = match args.get("new_string").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return ToolResult::Text("Error: new_string is required".into()),
    };
    let replace_all = args
        .get("replace_all")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let base = runtime.cwd.as_deref().unwrap_or(".");
    let full_path = if Path::new(file_path).is_absolute() {
        Path::new(file_path).to_path_buf()
    } else {
        Path::new(base).join(file_path)
    };

    let content = match fs::read_to_string(&full_path) {
        Ok(c) => c,
        Err(e) => return ToolResult::Text(format!("Error reading {}: {}", full_path.display(), e)),
    };

    let count = content.matches(old_string).count();

    if count == 0 {
        return ToolResult::Text(format!(
            "Error: old_string not found in {}",
            full_path.display()
        ));
    }

    if !replace_all && count > 1 {
        return ToolResult::Text(format!(
            "Error: old_string found {} times in {}. Use replace_all=true or provide a more unique string.",
            count,
            full_path.display()
        ));
    }

    let new_content = if replace_all {
        content.replace(old_string, new_string)
    } else {
        content.replacen(old_string, new_string, 1)
    };

    match fs::write(&full_path, &new_content) {
        Ok(()) => {
            let occurrences = if replace_all { count } else { 1 };
            ToolResult::Text(format!(
                "Successfully edited {} ({} occurrence{})",
                full_path.display(),
                occurrences,
                if occurrences != 1 { "s" } else { "" }
            ))
        }
        Err(e) => ToolResult::Text(format!("Error writing {}: {}", full_path.display(), e)),
    }
}
