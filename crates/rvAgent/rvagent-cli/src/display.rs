//! Output formatting for the rvAgent CLI.
//!
//! Provides terminal-friendly rendering of agent messages, including:
//! - Markdown rendering hints
//! - Syntax highlighting markers
//! - Tool call result formatting
//! - Error display with suggestions

use rvagent_core::messages::{Message, ToolCall};

// ---------------------------------------------------------------------------
// Message display
// ---------------------------------------------------------------------------

/// Print an assistant message to stdout (non-interactive mode).
pub fn print_assistant_message(msg: &Message) {
    match msg {
        Message::Ai(ai) => {
            if !ai.content.is_empty() {
                println!();
                print_markdown(&ai.content);
            }
            for tc in &ai.tool_calls {
                print_tool_call(tc);
            }
        }
        Message::Tool(tool) => {
            print_tool_result(&tool.tool_call_id, &tool.content);
        }
        other => {
            println!("{}", other.content());
        }
    }
}

// ---------------------------------------------------------------------------
// Markdown rendering (terminal-friendly subset)
// ---------------------------------------------------------------------------

/// Render a markdown string to the terminal with basic formatting.
///
/// This is a lightweight renderer that handles:
/// - Code blocks (``` fenced)
/// - Inline code (`backticks`)
/// - Headers (# prefix)
/// - Bold (**text**)
/// - Bullet lists (- items)
pub fn print_markdown(text: &str) {
    let mut in_code_block = false;
    let mut code_lang = String::new();

    for line in text.lines() {
        if line.starts_with("```") {
            if in_code_block {
                // End of code block.
                in_code_block = false;
                code_lang.clear();
                println!("  {}", "---");
            } else {
                // Start of code block.
                in_code_block = true;
                code_lang = line.trim_start_matches('`').trim().to_string();
                if code_lang.is_empty() {
                    println!("  [code]");
                } else {
                    println!("  [{}]", code_lang);
                }
            }
            continue;
        }

        if in_code_block {
            // Inside a code block — print with indent and syntax hint marker.
            println!("  | {}", line);
            continue;
        }

        // Headers.
        if line.starts_with("### ") {
            println!("\n=== {} ===\n", &line[4..]);
        } else if line.starts_with("## ") {
            println!("\n== {} ==\n", &line[3..]);
        } else if line.starts_with("# ") {
            println!("\n= {} =\n", &line[2..]);
        } else if line.starts_with("- ") || line.starts_with("* ") {
            // Bullet list items.
            println!("  * {}", &line[2..]);
        } else if line.starts_with("> ") {
            // Block quotes.
            println!("  | {}", &line[2..]);
        } else {
            println!("{}", line);
        }
    }

    if in_code_block {
        // Unterminated code block — close it.
        println!("  ---");
    }
}

// ---------------------------------------------------------------------------
// Tool call display
// ---------------------------------------------------------------------------

/// Print a tool call invocation.
pub fn print_tool_call(tc: &ToolCall) {
    println!();
    println!("[tool] {} (id: {})", tc.name, tc.id);

    // Print arguments if they're an object.
    if let Some(obj) = tc.args.as_object() {
        for (key, value) in obj {
            let display_val = format_arg_value(value);
            println!("  {}: {}", key, display_val);
        }
    }
}

/// Print a tool execution result.
pub fn print_tool_result(tool_call_id: &str, content: &str) {
    println!("[result:{}]", tool_call_id);

    // Truncate very long results for display.
    let max_display = 2000;
    if content.len() > max_display {
        println!(
            "{}... ({} chars truncated)",
            &content[..max_display],
            content.len() - max_display
        );
    } else {
        println!("{}", content);
    }
}

/// Format a tool argument value for display.
fn format_arg_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => {
            if s.len() > 200 {
                format!("\"{}...\" ({} chars)", &s[..200], s.len())
            } else {
                format!("\"{}\"", s)
            }
        }
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        other => {
            let s = serde_json::to_string(other).unwrap_or_default();
            if s.len() > 200 {
                format!("{}... ({} chars)", &s[..200], s.len())
            } else {
                s
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Error display
// ---------------------------------------------------------------------------

/// Display an error with contextual suggestions.
pub fn print_error(error: &anyhow::Error) {
    eprintln!();
    eprintln!("[error] {}", error);

    // Walk the error chain for context.
    let mut source = error.source();
    while let Some(cause) = source {
        eprintln!("  caused by: {}", cause);
        source = std::error::Error::source(cause);
    }

    // Provide suggestions based on common error patterns.
    let msg = format!("{}", error);
    if msg.contains("API key") || msg.contains("ANTHROPIC_API_KEY") {
        eprintln!();
        eprintln!("  hint: Set your API key with:");
        eprintln!("    export ANTHROPIC_API_KEY=sk-...");
    } else if msg.contains("session not found") {
        eprintln!();
        eprintln!("  hint: List available sessions with:");
        eprintln!("    rvagent session list");
    } else if msg.contains("permission denied") {
        eprintln!();
        eprintln!("  hint: Check file permissions in the working directory.");
    }
}

/// Format a syntax-highlighted code snippet label for terminal display.
///
/// Returns a label like `[rust]`, `[python]`, etc. based on the language identifier.
pub fn syntax_label(lang: &str) -> String {
    match lang.to_lowercase().as_str() {
        "rs" | "rust" => "[rust]".to_string(),
        "py" | "python" => "[python]".to_string(),
        "js" | "javascript" => "[javascript]".to_string(),
        "ts" | "typescript" => "[typescript]".to_string(),
        "sh" | "bash" | "shell" => "[shell]".to_string(),
        "json" => "[json]".to_string(),
        "toml" => "[toml]".to_string(),
        "yaml" | "yml" => "[yaml]".to_string(),
        "" => "[code]".to_string(),
        other => format!("[{}]", other),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax_label() {
        assert_eq!(syntax_label("rust"), "[rust]");
        assert_eq!(syntax_label("rs"), "[rust]");
        assert_eq!(syntax_label("py"), "[python]");
        assert_eq!(syntax_label(""), "[code]");
        assert_eq!(syntax_label("go"), "[go]");
    }

    #[test]
    fn test_format_arg_value_string() {
        let val = serde_json::json!("hello");
        assert_eq!(format_arg_value(&val), "\"hello\"");
    }

    #[test]
    fn test_format_arg_value_long_string() {
        let long = "x".repeat(300);
        let val = serde_json::json!(long);
        let result = format_arg_value(&val);
        assert!(result.contains("300 chars"));
        assert!(result.len() < 300);
    }

    #[test]
    fn test_format_arg_value_null() {
        let val = serde_json::Value::Null;
        assert_eq!(format_arg_value(&val), "null");
    }

    #[test]
    fn test_format_arg_value_bool() {
        assert_eq!(format_arg_value(&serde_json::json!(true)), "true");
    }

    #[test]
    fn test_format_arg_value_number() {
        assert_eq!(format_arg_value(&serde_json::json!(42)), "42");
    }
}
