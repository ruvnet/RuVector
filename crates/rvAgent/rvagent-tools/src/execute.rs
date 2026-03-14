//! `execute` tool — shell command execution (ADR-096, ADR-103 C2).

use crate::{ToolResult, ToolRuntime};
use std::time::Duration;

/// Default timeout in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Standalone synchronous execute invocation.
pub fn invoke_standalone(args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
    let command = match args.get("command").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return ToolResult::Text("Error: command is required".into()),
    };

    let cwd = runtime.cwd.as_deref().unwrap_or(".");

    let result = std::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", std::env::var("HOME").unwrap_or_default())
        .output();

    match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let exit_code = output.status.code().unwrap_or(-1);

            let mut text = String::new();
            if !stdout.is_empty() {
                text.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str("STDERR:\n");
                text.push_str(&stderr);
            }
            if exit_code != 0 {
                text.push_str(&format!("\n\nExit code: {}", exit_code));
            }

            ToolResult::Text(text)
        }
        Err(e) => ToolResult::Text(format!("Error executing command: {}", e)),
    }
}

/// Async execute invocation using tokio (ADR-103 A3).
pub async fn ainvoke_standalone(args: serde_json::Value, runtime: &ToolRuntime) -> ToolResult {
    let command = match args.get("command").and_then(|v| v.as_str()) {
        Some(c) => c.to_string(),
        None => return ToolResult::Text("Error: command is required".into()),
    };

    let timeout_secs = args
        .get("timeout")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_TIMEOUT_SECS);

    let cwd = runtime.cwd.as_deref().unwrap_or(".").to_string();

    let result = tokio::time::timeout(
        Duration::from_secs(timeout_secs),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&command)
            .current_dir(&cwd)
            .env_clear()
            .env("PATH", std::env::var("PATH").unwrap_or_default())
            .env("HOME", std::env::var("HOME").unwrap_or_default())
            .output(),
    )
    .await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let exit_code = output.status.code().unwrap_or(-1);

            let mut text = String::new();
            if !stdout.is_empty() {
                text.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str("STDERR:\n");
                text.push_str(&stderr);
            }
            if exit_code != 0 {
                text.push_str(&format!("\n\nExit code: {}", exit_code));
            }

            ToolResult::Text(text)
        }
        Ok(Err(e)) => ToolResult::Text(format!("Error executing command: {}", e)),
        Err(_) => ToolResult::Text(format!(
            "Error: command timed out after {} seconds",
            timeout_secs
        )),
    }
}
