//! `read_file` tool — reads file content with line numbers.

use crate::{format_content_with_line_numbers, is_image_file, ToolResult, ToolRuntime,
            DEFAULT_READ_LIMIT, DEFAULT_READ_OFFSET, EMPTY_CONTENT_WARNING};
use std::fs;
use std::path::Path;

/// Standalone read_file invocation using filesystem directly.
pub fn invoke_standalone(args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
    let file_path = match args.get("file_path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolResult::Text("Error: file_path is required".into()),
    };

    if is_image_file(file_path) {
        return ToolResult::Text(format!(
            "[Image file: {}. Image content cannot be displayed as text.]",
            file_path
        ));
    }

    let offset = args
        .get("offset")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_READ_OFFSET as u64) as usize;
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_READ_LIMIT);

    let base = runtime.cwd.as_deref().unwrap_or(".");
    let full_path = if Path::new(file_path).is_absolute() {
        Path::new(file_path).to_path_buf()
    } else {
        Path::new(base).join(file_path)
    };

    match fs::read_to_string(&full_path) {
        Ok(content) => {
            if content.is_empty() {
                return ToolResult::Text(EMPTY_CONTENT_WARNING.to_string());
            }

            let all_lines: Vec<&str> = content.lines().collect();
            let total = all_lines.len();
            let start = offset.min(total);
            let end = (start + limit).min(total);
            let slice = &all_lines[start..end];

            let rejoined = slice.join("\n");
            let formatted = format_content_with_line_numbers(&rejoined, start + 1);

            let mut result = formatted;
            if end < total {
                result.push_str(&format!(
                    "\n\n... ({} more lines not shown)",
                    total - end
                ));
            }
            ToolResult::Text(result)
        }
        Err(e) => ToolResult::Text(format!("Error reading {}: {}", full_path.display(), e)),
    }
}
