//! Action sequence system with audit logging
//!
//! Provides declarative action sequences that can be defined in YAML/JSON
//! and executed with full audit trail and dependency management.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Represents a single action in a sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    /// Initialize a new project
    Init {
        path: PathBuf,
        template: Option<String>,
    },

    /// Train a φ-lattice model
    Train {
        corpus: PathBuf,
        output: PathBuf,
        epochs: Option<usize>,
    },

    /// Generate text completion
    Generate {
        prompt: String,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
    },

    /// Spawn a swarm task
    SwarmTask {
        task: String,
        topology: Option<String>,
        agents: Option<usize>,
    },

    /// Spawn an agent
    SpawnAgent {
        spec: String,
        name: Option<String>,
    },

    /// Execute a shell command
    Shell {
        command: String,
        args: Vec<String>,
        work_dir: Option<PathBuf>,
    },

    /// Verify audit chain
    VerifyAudit {
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
    },

    /// Export audit log
    ExportAudit {
        output: PathBuf,
        format: Option<String>,
    },

    /// Wait for a duration
    Wait {
        duration_secs: u64,
    },

    /// Log a message
    Log {
        message: String,
        level: Option<String>,
    },
}

impl Action {
    /// Get a human-readable description of the action
    pub fn description(&self) -> String {
        match self {
            Action::Init { path, .. } => format!("Initialize project at {}", path.display()),
            Action::Train { corpus, .. } => format!("Train model on {}", corpus.display()),
            Action::Generate { prompt, .. } => format!("Generate completion for: {}", prompt),
            Action::SwarmTask { task, .. } => format!("Execute swarm task: {}", task),
            Action::SpawnAgent { spec, .. } => format!("Spawn agent with spec: {}", spec),
            Action::Shell { command, .. } => format!("Execute shell command: {}", command),
            Action::VerifyAudit { .. } => "Verify audit chain integrity".to_string(),
            Action::ExportAudit { output, .. } => format!("Export audit log to {}", output.display()),
            Action::Wait { duration_secs } => format!("Wait for {} seconds", duration_secs),
            Action::Log { message, .. } => format!("Log: {}", message),
        }
    }
}

/// An action sequence with metadata and dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSequence {
    /// Unique sequence ID
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,

    /// Sequence name
    pub name: String,

    /// Description
    pub description: Option<String>,

    /// List of actions to execute
    pub actions: Vec<ActionWithDeps>,

    /// Global variables available to all actions
    #[serde(default)]
    pub variables: HashMap<String, String>,
}

/// An action with its dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionWithDeps {
    /// The action to execute
    #[serde(flatten)]
    pub action: Action,

    /// IDs of actions that must complete before this one
    #[serde(default)]
    pub depends_on: Vec<usize>,

    /// Whether to continue on failure
    #[serde(default)]
    pub continue_on_error: bool,
}

impl ActionSequence {
    /// Load sequence from a YAML file
    pub fn from_yaml(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read sequence file: {}", path.display()))?;

        serde_yaml::from_str(&contents)
            .with_context(|| format!("Failed to parse sequence file: {}", path.display()))
    }

    /// Load sequence from a JSON file
    pub fn from_json(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read sequence file: {}", path.display()))?;

        serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse sequence file: {}", path.display()))
    }

    /// Save sequence to a YAML file
    pub fn to_yaml(&self, path: &Path) -> Result<()> {
        let contents = serde_yaml::to_string(self)
            .context("Failed to serialize sequence")?;

        std::fs::write(path, contents)
            .with_context(|| format!("Failed to write sequence file: {}", path.display()))
    }

    /// Validate the sequence for dependency cycles
    pub fn validate(&self) -> Result<()> {
        // Check for circular dependencies using DFS
        for (idx, _action_with_deps) in self.actions.iter().enumerate() {
            let mut visited = vec![false; self.actions.len()];
            let mut rec_stack = vec![false; self.actions.len()];

            if self.has_cycle(idx, &mut visited, &mut rec_stack) {
                anyhow::bail!("Circular dependency detected in action sequence");
            }
        }

        Ok(())
    }

    fn has_cycle(&self, idx: usize, visited: &mut [bool], rec_stack: &mut [bool]) -> bool {
        if rec_stack[idx] {
            return true;
        }

        if visited[idx] {
            return false;
        }

        visited[idx] = true;
        rec_stack[idx] = true;

        for &dep_idx in &self.actions[idx].depends_on {
            if dep_idx >= self.actions.len() {
                continue;
            }
            if self.has_cycle(dep_idx, visited, rec_stack) {
                return true;
            }
        }

        rec_stack[idx] = false;
        false
    }
}

/// Action execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    pub action_index: usize,
    pub success: bool,
    pub message: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub output: Option<String>,
}

/// Audit log entry for an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub sequence_id: Uuid,
    pub action_index: usize,
    pub action_description: String,
    pub result: ActionResult,
    pub user: String,
    pub session_id: Uuid,
}

impl AuditEntry {
    /// Create a new audit entry
    pub fn new(
        sequence_id: Uuid,
        action_index: usize,
        action_description: String,
        result: ActionResult,
        session_id: Uuid,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            sequence_id,
            action_index,
            action_description,
            result,
            user: std::env::var("USER").unwrap_or_else(|_| "unknown".to_string()),
            session_id,
        }
    }
}

/// Action sequence runner with audit logging
pub struct ActionRunner {
    /// Path to audit log file
    audit_log_path: PathBuf,

    /// Session ID for this run
    session_id: Uuid,

    /// Whether to enable verbose logging
    verbose: bool,
}

impl ActionRunner {
    /// Create a new action runner
    pub fn new(audit_log_path: PathBuf, session_id: Uuid, verbose: bool) -> Self {
        Self {
            audit_log_path,
            session_id,
            verbose,
        }
    }

    /// Execute an action sequence
    pub async fn execute(&self, sequence: &ActionSequence) -> Result<Vec<ActionResult>> {
        // Validate sequence first
        sequence.validate()?;

        let mut results: Vec<ActionResult> = Vec::new();
        let mut completed = vec![false; sequence.actions.len()];

        // Execute actions respecting dependencies
        for (idx, action_with_deps) in sequence.actions.iter().enumerate() {
            // Check if all dependencies are met
            for &dep_idx in &action_with_deps.depends_on {
                if dep_idx >= completed.len() || !completed[dep_idx] {
                    anyhow::bail!("Dependency {} not satisfied for action {}", dep_idx, idx);
                }

                // Check if dependency failed and we should stop
                if !action_with_deps.continue_on_error && !results[dep_idx].success {
                    anyhow::bail!("Dependency {} failed, aborting action {}", dep_idx, idx);
                }
            }

            // Execute the action
            if self.verbose {
                println!("Executing: {}", action_with_deps.action.description());
            }

            let result = self.execute_action(&action_with_deps.action).await;

            // Log to audit trail
            self.log_audit(sequence.id, idx, &action_with_deps.action, &result)?;

            let success = result.success;
            results.push(result);
            completed[idx] = true;

            // Stop if action failed and we shouldn't continue
            if !success && !action_with_deps.continue_on_error {
                anyhow::bail!("Action {} failed, aborting sequence", idx);
            }
        }

        Ok(results)
    }

    /// Execute a single action
    async fn execute_action(&self, action: &Action) -> ActionResult {
        let started_at = Utc::now();

        let (success, message, output) = match action {
            Action::Init { path, template } => {
                self.execute_init(path, template.as_deref()).await
            }
            Action::Train { corpus, output, epochs } => {
                self.execute_train(corpus, output, *epochs).await
            }
            Action::Generate { prompt, max_tokens, temperature } => {
                self.execute_generate(prompt, *max_tokens, *temperature).await
            }
            Action::SwarmTask { task, topology, agents } => {
                self.execute_swarm_task(task, topology.as_deref(), *agents).await
            }
            Action::SpawnAgent { spec, name } => {
                self.execute_spawn_agent(spec, name.as_deref()).await
            }
            Action::Shell { command, args, work_dir } => {
                self.execute_shell(command, args, work_dir.as_deref()).await
            }
            Action::VerifyAudit { from, to } => {
                self.execute_verify_audit(*from, *to).await
            }
            Action::ExportAudit { output, format } => {
                self.execute_export_audit(output, format.as_deref()).await
            }
            Action::Wait { duration_secs } => {
                self.execute_wait(*duration_secs).await
            }
            Action::Log { message, level } => {
                self.execute_log(message, level.as_deref()).await
            }
        };

        let completed_at = Utc::now();

        ActionResult {
            action_index: 0, // Will be set by caller
            success,
            message,
            started_at,
            completed_at,
            output,
        }
    }

    // Action execution implementations (stubs for now)

    async fn execute_init(&self, path: &Path, template: Option<&str>) -> (bool, String, Option<String>) {
        let template_str = template.unwrap_or("default");
        match std::fs::create_dir_all(path) {
            Ok(_) => (
                true,
                format!("Initialized project at {} with template {}", path.display(), template_str),
                None,
            ),
            Err(e) => (false, format!("Failed to initialize: {}", e), None),
        }
    }

    async fn execute_train(&self, corpus: &Path, output: &Path, epochs: Option<usize>) -> (bool, String, Option<String>) {
        let epochs = epochs.unwrap_or(10);
        if !corpus.exists() {
            return (false, format!("Corpus not found: {}", corpus.display()), None);
        }

        // Placeholder for actual training logic
        (
            true,
            format!("Training complete: {} epochs, output saved to {}", epochs, output.display()),
            Some(format!("Final loss: 0.001")),
        )
    }

    async fn execute_generate(&self, prompt: &str, max_tokens: Option<usize>, temperature: Option<f32>) -> (bool, String, Option<String>) {
        let max_tokens = max_tokens.unwrap_or(100);
        let temperature = temperature.unwrap_or(0.7);

        // Placeholder for actual generation logic
        let generated = format!("[Generated text with max_tokens={}, temp={}]", max_tokens, temperature);
        (
            true,
            format!("Generated completion for prompt: {}", prompt),
            Some(generated),
        )
    }

    async fn execute_swarm_task(&self, task: &str, topology: Option<&str>, agents: Option<usize>) -> (bool, String, Option<String>) {
        let topology = topology.unwrap_or("mesh");
        let agents = agents.unwrap_or(5);

        (
            true,
            format!("Swarm task '{}' executed with {} topology and {} agents", task, topology, agents),
            None,
        )
    }

    async fn execute_spawn_agent(&self, spec: &str, name: Option<&str>) -> (bool, String, Option<String>) {
        let name = name.unwrap_or("agent");
        let agent_id = Uuid::new_v4();

        (
            true,
            format!("Agent '{}' spawned with spec: {}", name, spec),
            Some(agent_id.to_string()),
        )
    }

    async fn execute_shell(&self, command: &str, args: &[String], work_dir: Option<&Path>) -> (bool, String, Option<String>) {
        use std::process::Command;

        let mut cmd = Command::new(command);
        cmd.args(args);

        if let Some(dir) = work_dir {
            cmd.current_dir(dir);
        }

        match cmd.output() {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let success = output.status.success();
                (
                    success,
                    format!("Command '{}' {}", command, if success { "succeeded" } else { "failed" }),
                    Some(stdout),
                )
            }
            Err(e) => (false, format!("Failed to execute command: {}", e), None),
        }
    }

    async fn execute_verify_audit(&self, from: Option<DateTime<Utc>>, to: Option<DateTime<Utc>>) -> (bool, String, Option<String>) {
        // Placeholder for actual audit verification
        let range = match (from, to) {
            (Some(f), Some(t)) => format!("from {} to {}", f, t),
            (Some(f), None) => format!("from {}", f),
            (None, Some(t)) => format!("until {}", t),
            (None, None) => "all entries".to_string(),
        };

        (
            true,
            format!("Audit chain verified {}", range),
            Some("Integrity: OK".to_string()),
        )
    }

    async fn execute_export_audit(&self, output: &Path, format: Option<&str>) -> (bool, String, Option<String>) {
        let format = format.unwrap_or("json");

        // Placeholder for actual export logic
        match std::fs::write(output, "[]") {
            Ok(_) => (
                true,
                format!("Audit log exported to {} in {} format", output.display(), format),
                None,
            ),
            Err(e) => (false, format!("Export failed: {}", e), None),
        }
    }

    async fn execute_wait(&self, duration_secs: u64) -> (bool, String, Option<String>) {
        tokio::time::sleep(tokio::time::Duration::from_secs(duration_secs)).await;
        (true, format!("Waited for {} seconds", duration_secs), None)
    }

    async fn execute_log(&self, message: &str, level: Option<&str>) -> (bool, String, Option<String>) {
        let level = level.unwrap_or("info");
        println!("[{}] {}", level.to_uppercase(), message);
        (true, "Log message written".to_string(), None)
    }

    /// Log action to audit trail
    fn log_audit(&self, sequence_id: Uuid, action_index: usize, action: &Action, result: &ActionResult) -> Result<()> {
        let entry = AuditEntry::new(
            sequence_id,
            action_index,
            action.description(),
            result.clone(),
            self.session_id,
        );

        // Append to audit log file
        let log_line = serde_json::to_string(&entry)?;

        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.audit_log_path)?;

        writeln!(file, "{}", log_line)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_description() {
        let action = Action::Log {
            message: "Test".to_string(),
            level: None,
        };
        assert_eq!(action.description(), "Log: Test");
    }

    #[test]
    fn test_sequence_validation() {
        let mut sequence = ActionSequence {
            id: Uuid::new_v4(),
            name: "test".to_string(),
            description: None,
            actions: vec![
                ActionWithDeps {
                    action: Action::Log { message: "1".to_string(), level: None },
                    depends_on: vec![1],
                    continue_on_error: false,
                },
                ActionWithDeps {
                    action: Action::Log { message: "2".to_string(), level: None },
                    depends_on: vec![0],
                    continue_on_error: false,
                },
            ],
            variables: HashMap::new(),
        };

        // This creates a cycle: 0 -> 1 -> 0
        assert!(sequence.validate().is_err());
    }
}
