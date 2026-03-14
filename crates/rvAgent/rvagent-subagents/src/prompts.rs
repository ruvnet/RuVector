//! Prompt constants for subagent orchestration.
//!
//! These constants define the task tool description, system prompts, and
//! handoff message format used when spawning and managing subagents.

/// Description for the `task` tool that appears in the tool registry.
///
/// The `{available_agents}` placeholder is replaced at runtime with the
/// list of compiled subagent names and descriptions.
pub const TASK_TOOL_DESCRIPTION: &str = "\
Launch a new agent that has access to the same tools as you. \
When you are searching for a keyword or file and are not confident \
that you will find the right match in the first few tries, use the \
task tool to perform the search for you.

When you use the task tool, you should provide a detailed natural \
language description of what you want the agent to do, including \
any relevant context from the conversation so far.

The available subagent types are:
{available_agents}

IMPORTANT: Each invocation of the task tool creates a NEW agent \
with no memory of previous invocations. Do not reference previous \
task results — instead, include all necessary context in the \
description.

You should use subagent_type to select the most appropriate agent \
for the task. If unsure, use \"general-purpose\".";

/// System prompt appended to the parent agent's system message when
/// the subagent middleware is active.
///
/// Instructs the model on when and how to use the `task` tool.
pub const TASK_SYSTEM_PROMPT: &str = "\
You have access to a `task` tool that lets you spawn subagents. \
Use it when:
- You need to search for files or content and want thorough results
- The task can be parallelized (e.g., searching multiple directories)
- You want to delegate a self-contained subtask
- The subtask requires a different set of tools or capabilities

Each subagent runs in isolation: it cannot see your conversation \
history, todos, or structured responses. You must pass all relevant \
context in the task description.

When spawning multiple tasks, you can invoke the task tool multiple \
times in a single response — they will execute concurrently.";

/// Format template for handoff messages between parent and subagent.
///
/// Placeholders:
/// - `{agent_name}`: name of the subagent being invoked
/// - `{description}`: the task description passed to the subagent
/// - `{result}`: the subagent's final response (used in result messages)
pub const HANDOFF_FORMAT: &str = "\
[SubAgent Handoff — {agent_name}]
Task: {description}
---
{result}";

/// Format a handoff message for spawning a subagent.
pub fn format_handoff_spawn(agent_name: &str, description: &str) -> String {
    HANDOFF_FORMAT
        .replace("{agent_name}", agent_name)
        .replace("{description}", description)
        .replace("{result}", "(pending)")
}

/// Format a handoff message with a completed result.
pub fn format_handoff_result(agent_name: &str, description: &str, result: &str) -> String {
    HANDOFF_FORMAT
        .replace("{agent_name}", agent_name)
        .replace("{description}", description)
        .replace("{result}", result)
}

/// Build the task tool description with concrete agent list.
pub fn build_task_tool_description(agents: &[(String, String)]) -> String {
    let agents_desc = agents
        .iter()
        .map(|(name, desc)| format!("- {}: {}", name, desc))
        .collect::<Vec<_>>()
        .join("\n");

    TASK_TOOL_DESCRIPTION.replace("{available_agents}", &agents_desc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_tool_description_has_placeholder() {
        assert!(TASK_TOOL_DESCRIPTION.contains("{available_agents}"));
    }

    #[test]
    fn test_build_task_tool_description() {
        let agents = vec![
            ("coder".to_string(), "Writes code".to_string()),
            ("researcher".to_string(), "Searches docs".to_string()),
        ];
        let desc = build_task_tool_description(&agents);
        assert!(desc.contains("- coder: Writes code"));
        assert!(desc.contains("- researcher: Searches docs"));
        assert!(!desc.contains("{available_agents}"));
    }

    #[test]
    fn test_format_handoff_spawn() {
        let msg = format_handoff_spawn("coder", "Fix the bug in main.rs");
        assert!(msg.contains("[SubAgent Handoff — coder]"));
        assert!(msg.contains("Fix the bug in main.rs"));
        assert!(msg.contains("(pending)"));
    }

    #[test]
    fn test_format_handoff_result() {
        let msg = format_handoff_result("coder", "Fix the bug", "Bug fixed in line 42.");
        assert!(msg.contains("coder"));
        assert!(msg.contains("Bug fixed in line 42."));
        assert!(!msg.contains("(pending)"));
    }

    #[test]
    fn test_task_system_prompt_nonempty() {
        assert!(!TASK_SYSTEM_PROMPT.is_empty());
        assert!(TASK_SYSTEM_PROMPT.contains("task"));
    }
}
