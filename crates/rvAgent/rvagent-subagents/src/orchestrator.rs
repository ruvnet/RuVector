//! SubAgent orchestrator — spawn and parallel execution (ADR-097, ADR-103 A2).

use crate::{
    prepare_subagent_state, AgentState, CompiledSubAgent, SubAgentResult, SubAgentResultValidator,
    ValidationConfig, ValidationError,
};
use std::time::Instant;

/// Orchestrates subagent execution, including parallel spawning and result validation.
pub struct SubAgentOrchestrator {
    compiled: Vec<CompiledSubAgent>,
    validator: SubAgentResultValidator,
}

impl SubAgentOrchestrator {
    /// Create a new orchestrator from compiled subagents with default validation.
    pub fn new(compiled: Vec<CompiledSubAgent>) -> Self {
        Self::new_with_validation(compiled, ValidationConfig::default())
    }

    /// Create a new orchestrator with custom validation configuration.
    pub fn new_with_validation(
        compiled: Vec<CompiledSubAgent>,
        validation_config: ValidationConfig,
    ) -> Self {
        Self {
            compiled,
            validator: SubAgentResultValidator::new(validation_config),
        }
    }

    /// Find a compiled subagent by name.
    pub fn find(&self, name: &str) -> Option<&CompiledSubAgent> {
        self.compiled.iter().find(|c| c.spec.name == name)
    }

    /// Return the number of compiled subagents.
    pub fn len(&self) -> usize {
        self.compiled.len()
    }

    /// Spawn a single subagent (mock/stub — returns a validated result).
    ///
    /// Returns `None` if the subagent is not found or validation fails.
    pub fn spawn_sync(
        &self,
        name: &str,
        parent_state: &AgentState,
        task_description: &str,
    ) -> Result<SubAgentResult, SpawnError> {
        let _compiled = self.find(name).ok_or(SpawnError::SubAgentNotFound {
            name: name.to_string(),
        })?;
        let _child_state = prepare_subagent_state(parent_state, task_description);

        let start = Instant::now();

        // In a real implementation, this would run the agent graph.
        // For now, return a stub result.
        let result_message = format!("SubAgent '{}' completed task: {}", name, task_description);

        // Validate the result content (C8: SubAgent Result Validation)
        let validated_message = self
            .validator
            .validate(&result_message)
            .map_err(SpawnError::ValidationFailed)?;

        // Validate tool call count
        let tool_calls_count = 0; // Stub value
        self.validator
            .validate_tool_calls(tool_calls_count)
            .map_err(SpawnError::ValidationFailed)?;

        Ok(SubAgentResult {
            agent_name: name.to_string(),
            result_message: validated_message,
            tool_calls_count,
            duration: start.elapsed(),
        })
    }
}

/// Errors that can occur during subagent spawning.
#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    #[error("SubAgent not found: {name}")]
    SubAgentNotFound { name: String },

    #[error("Validation failed: {0}")]
    ValidationFailed(#[from] ValidationError),

    #[error("Execution failed: {reason}")]
    ExecutionFailed { reason: String },
}

/// Spawn multiple subagents in parallel (async).
///
/// Returns results for successful spawns and logs errors for failures.
pub async fn spawn_parallel(
    orchestrator: &SubAgentOrchestrator,
    tasks: Vec<(&str, &AgentState, &str)>,
) -> Vec<Result<SubAgentResult, SpawnError>> {
    // In a real implementation, this would use tokio::JoinSet.
    // For now, execute sequentially and collect all results (Ok or Err).
    let mut results = Vec::new();
    for (name, state, desc) in tasks {
        let result = orchestrator.spawn_sync(name, state, desc);
        results.push(result);
    }
    results
}
