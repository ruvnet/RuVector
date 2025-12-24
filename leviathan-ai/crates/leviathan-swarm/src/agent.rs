//! Agent specification and state management

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Agent state
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentState {
    /// Agent is idle and ready for work
    Idle,
    /// Agent is currently executing a task
    Running,
    /// Agent is blocked waiting for dependencies
    Blocked,
    /// Agent has completed its task
    Complete,
    /// Agent has failed
    Failed,
}

/// Agent specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    /// Unique agent identifier
    pub id: Uuid,
    /// Agent name
    pub name: String,
    /// Agent capabilities (e.g., "rust", "python", "testing")
    pub capabilities: Vec<String>,
    /// Tools available to the agent (as command strings)
    pub tools: Vec<String>,
    /// Current agent state
    pub state: AgentState,
    /// Agent metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
}

impl AgentSpec {
    /// Create a new agent specification
    pub fn new(name: impl Into<String>, capabilities: Vec<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            capabilities,
            tools: Vec::new(),
            state: AgentState::Idle,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a tool to the agent
    pub fn with_tool(mut self, tool: impl Into<String>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Add multiple tools to the agent
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Add metadata to the agent
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Update agent state
    pub fn set_state(&mut self, state: AgentState) {
        self.state = state;
        self.updated_at = Utc::now();
    }

    /// Check if agent has a specific capability
    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|c| c == capability)
    }

    /// Check if agent has a specific tool
    pub fn has_tool(&self, tool: &str) -> bool {
        self.tools.iter().any(|t| t == tool)
    }
}

/// Result of agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Agent ID
    pub agent_id: Uuid,
    /// Task ID
    pub task_id: Uuid,
    /// Execution output
    pub output: String,
    /// Execution success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution metrics
    pub metrics: AgentMetrics,
    /// Audit trail of operations
    pub audit_trail: Vec<AuditEntry>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Agent execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Execution duration in milliseconds
    pub duration_ms: u64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// CPU time in milliseconds
    pub cpu_time_ms: u64,
    /// Number of operations performed
    pub operations_count: usize,
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self {
            duration_ms: 0,
            memory_bytes: 0,
            cpu_time_ms: 0,
            operations_count: 0,
        }
    }
}

/// Audit trail entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Timestamp of the operation
    pub timestamp: DateTime<Utc>,
    /// Operation description
    pub operation: String,
    /// Additional context
    pub context: HashMap<String, String>,
}

impl AuditEntry {
    /// Create a new audit entry
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            operation: operation.into(),
            context: HashMap::new(),
        }
    }

    /// Add context to the audit entry
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_spec_creation() {
        let agent = AgentSpec::new("test-agent", vec!["rust".to_string(), "testing".to_string()])
            .with_tool("cargo")
            .with_tool("rustfmt")
            .with_metadata("version", "1.0.0");

        assert_eq!(agent.name, "test-agent");
        assert_eq!(agent.capabilities.len(), 2);
        assert_eq!(agent.tools.len(), 2);
        assert_eq!(agent.state, AgentState::Idle);
        assert!(agent.has_capability("rust"));
        assert!(agent.has_tool("cargo"));
    }

    #[test]
    fn test_agent_state_transitions() {
        let mut agent = AgentSpec::new("test-agent", vec!["rust".to_string()]);

        assert_eq!(agent.state, AgentState::Idle);

        agent.set_state(AgentState::Running);
        assert_eq!(agent.state, AgentState::Running);

        agent.set_state(AgentState::Complete);
        assert_eq!(agent.state, AgentState::Complete);
    }

    #[test]
    fn test_audit_entry() {
        let entry = AuditEntry::new("task_started")
            .with_context("task_id", "123")
            .with_context("agent_id", "456");

        assert_eq!(entry.operation, "task_started");
        assert_eq!(entry.context.len(), 2);
        assert_eq!(entry.context.get("task_id"), Some(&"123".to_string()));
    }
}
