//! SubAgent orchestrator — spawn and parallel execution (ADR-097, ADR-103 A2).

use crate::{
    AgentState, CompiledSubAgent, SubAgentResult,
    prepare_subagent_state,
};
use std::time::Instant;

/// Orchestrates subagent execution, including parallel spawning.
pub struct SubAgentOrchestrator {
    compiled: Vec<CompiledSubAgent>,
}

impl SubAgentOrchestrator {
    /// Create a new orchestrator from compiled subagents.
    pub fn new(compiled: Vec<CompiledSubAgent>) -> Self {
        Self { compiled }
    }

    /// Find a compiled subagent by name.
    pub fn find(&self, name: &str) -> Option<&CompiledSubAgent> {
        self.compiled.iter().find(|c| c.spec.name == name)
    }

    /// Return the number of compiled subagents.
    pub fn len(&self) -> usize {
        self.compiled.len()
    }

    /// Spawn a single subagent (mock/stub — returns a result based on the spec).
    pub fn spawn_sync(
        &self,
        name: &str,
        parent_state: &AgentState,
        task_description: &str,
    ) -> Option<SubAgentResult> {
        let _compiled = self.find(name)?;
        let _child_state = prepare_subagent_state(parent_state, task_description);

        let start = Instant::now();

        // In a real implementation, this would run the agent graph.
        // For now, return a stub result.
        let result_message = format!(
            "SubAgent '{}' completed task: {}",
            name, task_description
        );

        Some(SubAgentResult {
            agent_name: name.to_string(),
            result_message,
            tool_calls_count: 0,
            duration: start.elapsed(),
        })
    }
}

/// Spawn multiple subagents in parallel (async).
pub async fn spawn_parallel(
    orchestrator: &SubAgentOrchestrator,
    tasks: Vec<(&str, &AgentState, &str)>,
) -> Vec<SubAgentResult> {
    // In a real implementation, this would use tokio::JoinSet.
    // For now, execute sequentially.
    let mut results = Vec::new();
    for (name, state, desc) in tasks {
        if let Some(result) = orchestrator.spawn_sync(name, state, desc) {
            results.push(result);
        }
    }
    results
}
