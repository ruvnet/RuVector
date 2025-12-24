//! Agent Executor Module
//!
//! Provides the runtime for executing agent tasks with tool invocation and state management.

use crate::spec::{AgentSpec, OutputParser, ToolSpec};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Stdio;
use tokio::process::Command;
use tracing::{debug, info, warn};

/// Executes agent tasks with full audit logging
pub struct AgentExecutor {
    /// The agent specification being executed
    spec: AgentSpec,

    /// Execution state
    state: ExecutionState,

    /// Audit log of all actions
    audit_log: Vec<AuditEntry>,
}

impl AgentExecutor {
    /// Create a new executor for the given agent spec
    pub fn new(spec: AgentSpec) -> Self {
        Self {
            spec,
            state: ExecutionState::new(),
            audit_log: Vec::new(),
        }
    }

    /// Execute a task
    pub async fn execute_task(&mut self, task: &str) -> Result<ExecutionResult, ExecutionError> {
        info!("Agent '{}' executing task: {}", self.spec.name, task);

        self.log_audit(AuditEntry::new(
            "task_start",
            task.to_string(),
            serde_json::json!({"agent": self.spec.name}),
        ));

        // Create task context
        let context = TaskContext {
            task: task.to_string(),
            agent_name: self.spec.name.clone(),
            agent_role: format!("{:?}", self.spec.role),
            workspace: std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            state: self.state.clone(),
        };

        // Execute with context
        let result = self.execute_with_context(context).await?;

        self.log_audit(AuditEntry::new(
            "task_complete",
            task.to_string(),
            serde_json::json!({"success": result.success}),
        ));

        Ok(result)
    }

    /// Execute with a specific context
    async fn execute_with_context(
        &mut self,
        mut context: TaskContext,
    ) -> Result<ExecutionResult, ExecutionError> {
        let mut outputs = Vec::new();
        let mut errors = Vec::new();

        // Clone the tools to avoid borrow checker issues
        let tools = self.spec.tools.clone();

        // Parse the task and determine which tools to use
        // This is a simplified version - a real implementation would use LLM reasoning
        for tool in &tools {
            debug!("Considering tool: {}", tool.name);

            // Simple heuristic: if task mentions the tool or its purpose
            if context.task.to_lowercase().contains(&tool.name.to_lowercase()) {
                match self.invoke_tool(tool, &context).await {
                    Ok(output) => {
                        info!("Tool '{}' executed successfully", tool.name);
                        outputs.push(output.clone());
                        context.state.add_output(tool.name.clone(), output);
                    }
                    Err(e) => {
                        warn!("Tool '{}' failed: {}", tool.name, e);
                        errors.push(format!("Tool '{}' error: {}", tool.name, e));
                    }
                }
            }
        }

        // Update internal state
        self.state = context.state.clone();

        Ok(ExecutionResult {
            success: errors.is_empty(),
            outputs,
            errors,
            state: context.state,
        })
    }

    /// Invoke a specific tool
    async fn invoke_tool(
        &mut self,
        tool: &ToolSpec,
        context: &TaskContext,
    ) -> Result<String, ExecutionError> {
        // Prepare variables for template expansion
        let mut vars = HashMap::new();
        vars.insert("input".to_string(), context.task.clone());
        vars.insert("workspace".to_string(), context.workspace.clone());
        vars.insert("agent".to_string(), context.agent_name.clone());

        // Expand argument template
        let args = tool
            .expand_args(&vars)
            .map_err(|e| ExecutionError::ToolError(format!("Template expansion failed: {}", e)))?;

        self.log_audit(AuditEntry::new(
            "tool_invoke",
            tool.name.clone(),
            serde_json::json!({
                "command": &tool.command,
                "args": &args,
            }),
        ));

        // Execute the command
        let output = self.run_command(&tool.command, &args, tool).await?;

        // Parse the output
        let parsed = self.parse_output(&output, &tool.output_parser)?;

        self.log_audit(AuditEntry::new(
            "tool_complete",
            tool.name.clone(),
            serde_json::json!({"output_length": parsed.len()}),
        ));

        Ok(parsed)
    }

    /// Run a shell command
    async fn run_command(
        &self,
        command: &str,
        args: &str,
        tool: &ToolSpec,
    ) -> Result<String, ExecutionError> {
        let mut cmd = Command::new(command);

        // Split args (simplified - real implementation would handle quoting)
        for arg in args.split_whitespace() {
            cmd.arg(arg);
        }

        // Set working directory if specified
        if let Some(ref wd) = tool.working_dir {
            cmd.current_dir(wd);
        }

        // Set environment variables
        for (key, value) in &tool.env_vars {
            cmd.env(key, value);
        }

        // Capture output
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let child = cmd
            .spawn()
            .map_err(|e| ExecutionError::ToolError(format!("Failed to spawn command: {}", e)))?;

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| ExecutionError::ToolError(format!("Command failed: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ExecutionError::ToolError(format!(
                "Command failed with status {}: {}",
                output.status, stderr
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Parse tool output based on parser type
    fn parse_output(
        &self,
        output: &str,
        parser: &OutputParser,
    ) -> Result<String, ExecutionError> {
        match parser {
            OutputParser::Raw => Ok(output.to_string()),

            OutputParser::Json => {
                // Validate JSON
                serde_json::from_str::<serde_json::Value>(output)
                    .map_err(|e| ExecutionError::ParseError(format!("Invalid JSON: {}", e)))?;
                Ok(output.to_string())
            }

            OutputParser::Lines => {
                let lines: Vec<&str> = output.lines().collect();
                Ok(format!("{} lines of output", lines.len()))
            }

            OutputParser::Regex { pattern, .. } => {
                let re = regex::Regex::new(pattern)
                    .map_err(|e| ExecutionError::ParseError(format!("Invalid regex: {}", e)))?;

                if re.is_match(output) {
                    Ok(output.to_string())
                } else {
                    Err(ExecutionError::ParseError("No regex match".into()))
                }
            }

            OutputParser::Custom(name) => {
                warn!("Custom parser '{}' not implemented, using raw", name);
                Ok(output.to_string())
            }
        }
    }

    /// Get the audit log
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    /// Log an audit entry
    fn log_audit(&mut self, entry: AuditEntry) {
        debug!("Audit: {} - {}", entry.action, entry.description);
        self.audit_log.push(entry);
    }

    /// Get the current execution state
    pub fn state(&self) -> &ExecutionState {
        &self.state
    }

    /// Export audit log to JSON
    pub fn export_audit_log(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.audit_log)
    }
}

/// Context passed to task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    /// The task description
    pub task: String,

    /// Agent name
    pub agent_name: String,

    /// Agent role
    pub agent_role: String,

    /// Working directory
    pub workspace: String,

    /// Execution state
    pub state: ExecutionState,
}

/// Execution state maintained between tool invocations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    /// Outputs from previous tool invocations
    outputs: HashMap<String, String>,

    /// Custom state variables
    variables: HashMap<String, String>,
}

impl ExecutionState {
    /// Create new empty state
    pub fn new() -> Self {
        Self {
            outputs: HashMap::new(),
            variables: HashMap::new(),
        }
    }

    /// Add a tool output
    pub fn add_output(&mut self, tool: String, output: String) {
        self.outputs.insert(tool, output);
    }

    /// Get a tool's output
    pub fn get_output(&self, tool: &str) -> Option<&String> {
        self.outputs.get(tool)
    }

    /// Set a variable
    pub fn set_variable(&mut self, key: String, value: String) {
        self.variables.insert(key, value);
    }

    /// Get a variable
    pub fn get_variable(&self, key: &str) -> Option<&String> {
        self.variables.get(key)
    }
}

impl Default for ExecutionState {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Whether execution was successful
    pub success: bool,

    /// Outputs from tools
    pub outputs: Vec<String>,

    /// Any errors encountered
    pub errors: Vec<String>,

    /// Final state
    pub state: ExecutionState,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Timestamp
    #[serde(with = "chrono::serde::ts_milliseconds")]
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Action type
    pub action: String,

    /// Description
    pub description: String,

    /// Additional data
    pub data: serde_json::Value,
}

impl AuditEntry {
    /// Create a new audit entry
    pub fn new(action: impl Into<String>, description: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            action: action.into(),
            description: description.into(),
            data,
        }
    }
}

/// Execution errors
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("Tool error: {0}")]
    ToolError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("State error: {0}")]
    StateError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{tool, AgentBuilder};
    use crate::spec::AgentRole;

    #[tokio::test]
    async fn test_executor_creation() {
        let spec = AgentBuilder::new("Test Agent")
            .role(AgentRole::Tester)
            .instruction("Run tests")
            .build()
            .unwrap();

        let executor = AgentExecutor::new(spec);
        assert_eq!(executor.spec.name, "Test Agent");
    }

    #[tokio::test]
    async fn test_execution_state() {
        let mut state = ExecutionState::new();
        state.add_output("tool1".into(), "output1".into());
        state.set_variable("var1".into(), "value1".into());

        assert_eq!(state.get_output("tool1"), Some(&"output1".to_string()));
        assert_eq!(state.get_variable("var1"), Some(&"value1".to_string()));
    }

    #[test]
    fn test_audit_log() {
        let entry = AuditEntry::new(
            "test_action",
            "Test description",
            serde_json::json!({"key": "value"}),
        );

        assert_eq!(entry.action, "test_action");
        assert_eq!(entry.description, "Test description");
    }
}
