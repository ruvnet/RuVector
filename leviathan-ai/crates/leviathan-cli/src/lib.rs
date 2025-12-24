//! Leviathan CLI Framework
//!
//! Provides CLI framework types, action sequences, and TUI components
//! for the Leviathan AI system.

pub mod action;
pub mod config;
pub mod tui;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// CLI framework types and utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliContext {
    /// Configuration loaded from leviathan.toml
    pub config: config::LeviathanConfig,

    /// Working directory
    pub work_dir: PathBuf,

    /// Session ID for tracking
    pub session_id: Uuid,

    /// Verbose output flag
    pub verbose: bool,
}

impl CliContext {
    /// Create a new CLI context
    pub fn new(config_path: Option<PathBuf>, verbose: bool) -> Result<Self> {
        let config = if let Some(path) = config_path {
            config::LeviathanConfig::load(&path)?
        } else {
            config::LeviathanConfig::default()
        };

        let work_dir = std::env::current_dir()?;
        let session_id = Uuid::new_v4();

        Ok(Self {
            config,
            work_dir,
            session_id,
            verbose,
        })
    }

    /// Get the audit log path
    pub fn audit_log_path(&self) -> PathBuf {
        self.config.audit.log_path.clone()
    }

    /// Get the data directory
    pub fn data_dir(&self) -> PathBuf {
        self.config.data_dir.clone()
    }
}

/// CLI command result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub success: bool,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

impl CommandResult {
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: None,
        }
    }

    pub fn success_with_data(message: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            success: true,
            message: message.into(),
            data: Some(data),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            data: None,
        }
    }
}

/// Agent status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub id: String,
    pub name: String,
    pub status: String,
    pub task: Option<String>,
    pub uptime_secs: u64,
    pub actions_completed: usize,
}

/// Swarm status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStatus {
    pub active_agents: usize,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub topology: String,
}

/// DAG node for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    pub id: String,
    pub label: String,
    pub dependencies: Vec<String>,
    pub status: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_result() {
        let result = CommandResult::success("Test successful");
        assert!(result.success);
        assert_eq!(result.message, "Test successful");
        assert!(result.data.is_none());
    }

    #[test]
    fn test_cli_context_creation() {
        let ctx = CliContext::new(None, false).unwrap();
        assert!(!ctx.verbose);
        assert!(ctx.work_dir.exists());
    }
}
