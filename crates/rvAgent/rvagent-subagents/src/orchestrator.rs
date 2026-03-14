//! Subagent orchestration — spawn, parallel execution, and result merging.
//!
//! The `SubAgentOrchestrator` manages the lifecycle of subagent invocations,
//! enforcing state isolation per ADR-097 and supporting concurrent execution
//! via `tokio::task::JoinSet`.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use crate::{
    prepare_subagent_state, extract_result_message, merge_subagent_state,
    AgentState, CompiledSubAgent, SubAgentResult, SubAgentSpec,
    EXCLUDED_STATE_KEYS,
};

/// Orchestrates subagent spawning, execution, and result collection.
///
/// Enforces state isolation: parent messages, todos, and completion data
/// never leak into subagent context, and subagent-specific keys never
/// overwrite parent state.
#[derive(Debug, Clone)]
pub struct SubAgentOrchestrator {
    /// Maximum wall-clock time for a single subagent invocation.
    pub timeout: Duration,

    /// Maximum number of tool calls a subagent may make.
    pub max_tool_calls: usize,

    /// Maximum number of concurrent subagents.
    pub max_concurrent: usize,
}

impl Default for SubAgentOrchestrator {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            max_tool_calls: 100,
            max_concurrent: 8,
        }
    }
}

impl SubAgentOrchestrator {
    /// Create a new orchestrator with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an orchestrator with custom limits.
    pub fn with_limits(timeout: Duration, max_tool_calls: usize, max_concurrent: usize) -> Self {
        Self {
            timeout,
            max_tool_calls,
            max_concurrent,
        }
    }

    /// Spawn a single subagent with the given input.
    ///
    /// The parent state is filtered through `EXCLUDED_STATE_KEYS` to ensure
    /// isolation. The subagent receives only a single human message with the
    /// task description.
    ///
    /// Returns the subagent result, or an error if execution fails or times out.
    pub async fn spawn(
        &self,
        compiled: &CompiledSubAgent,
        input: &str,
        parent_state: &AgentState,
    ) -> Result<SubAgentResult, SubAgentError> {
        let agent_name = compiled.spec.name.clone();
        info!(agent = %agent_name, "Spawning subagent");

        let subagent_state = prepare_subagent_state(parent_state, input);

        let start = Instant::now();

        // Simulate subagent execution.
        // In production, this would invoke the compiled graph's runnable.
        let result_state = self.execute_subagent(compiled, subagent_state).await?;

        let duration = start.elapsed();

        let result_message = extract_result_message(&result_state)
            .unwrap_or_else(|| "(no response)".to_string());

        let tool_calls_count = result_state
            .get("tool_calls_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        if tool_calls_count > self.max_tool_calls {
            warn!(
                agent = %agent_name,
                calls = tool_calls_count,
                max = self.max_tool_calls,
                "Subagent exceeded tool call limit"
            );
            return Err(SubAgentError::ToolCallLimitExceeded {
                agent: agent_name,
                count: tool_calls_count,
                limit: self.max_tool_calls,
            });
        }

        debug!(
            agent = %agent_name,
            duration_ms = duration.as_millis(),
            tool_calls = tool_calls_count,
            "Subagent completed"
        );

        Ok(SubAgentResult {
            agent_name,
            result_message,
            tool_calls_count,
            duration,
        })
    }

    /// Spawn multiple subagents concurrently and collect all results.
    ///
    /// Uses `tokio::task::JoinSet` for concurrent execution. Each subagent
    /// runs in isolation with its own filtered state. Results are collected
    /// in completion order.
    ///
    /// Respects `max_concurrent` — if more specs than the limit are provided,
    /// they are batched.
    pub async fn spawn_parallel(
        &self,
        compiled_agents: &[CompiledSubAgent],
        inputs: &[String],
        parent_state: &AgentState,
    ) -> Vec<Result<SubAgentResult, SubAgentError>> {
        assert_eq!(
            compiled_agents.len(),
            inputs.len(),
            "compiled_agents and inputs must have the same length"
        );

        if compiled_agents.is_empty() {
            return Vec::new();
        }

        info!(count = compiled_agents.len(), "Spawning parallel subagents");

        let mut all_results = Vec::with_capacity(compiled_agents.len());

        // Process in batches of max_concurrent
        for chunk_start in (0..compiled_agents.len()).step_by(self.max_concurrent) {
            let chunk_end = (chunk_start + self.max_concurrent).min(compiled_agents.len());
            let mut join_set = JoinSet::new();

            for i in chunk_start..chunk_end {
                let agent_name = compiled_agents[i].spec.name.clone();
                let input = inputs[i].clone();
                let subagent_state = prepare_subagent_state(parent_state, &input);
                let timeout = self.timeout;
                let max_tool_calls = self.max_tool_calls;
                let graph = compiled_agents[i].graph.clone();
                let middleware = compiled_agents[i].middleware_pipeline.clone();

                join_set.spawn(async move {
                    let start = Instant::now();

                    // Simulate execution with timeout
                    let exec_result = tokio::time::timeout(timeout, async {
                        // In production, this would run the compiled graph.
                        // Here we return the state as-is to simulate completion.
                        Ok::<AgentState, SubAgentError>(subagent_state)
                    })
                    .await;

                    let duration = start.elapsed();

                    match exec_result {
                        Ok(Ok(result_state)) => {
                            let result_message = extract_result_message(&result_state)
                                .unwrap_or_else(|| "(no response)".to_string());
                            let tool_calls_count = result_state
                                .get("tool_calls_count")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as usize;

                            if tool_calls_count > max_tool_calls {
                                Err(SubAgentError::ToolCallLimitExceeded {
                                    agent: agent_name,
                                    count: tool_calls_count,
                                    limit: max_tool_calls,
                                })
                            } else {
                                Ok(SubAgentResult {
                                    agent_name,
                                    result_message,
                                    tool_calls_count,
                                    duration,
                                })
                            }
                        }
                        Ok(Err(e)) => Err(e),
                        Err(_) => Err(SubAgentError::Timeout {
                            agent: agent_name,
                            duration: timeout,
                        }),
                    }
                });
            }

            // Collect results from this batch
            while let Some(result) = join_set.join_next().await {
                match result {
                    Ok(r) => all_results.push(r),
                    Err(e) => all_results.push(Err(SubAgentError::JoinError(e.to_string()))),
                }
            }
        }

        all_results
    }

    /// Merge results from one or more subagents back into the parent state.
    ///
    /// Only non-excluded keys are merged. If multiple subagents modify the
    /// same key, the last writer wins (future work: CRDT merge per ADR-103 B7).
    pub fn merge_results(
        &self,
        parent_state: &mut AgentState,
        result_states: &[AgentState],
    ) {
        for state in result_states {
            merge_subagent_state(parent_state, state);
        }
    }

    /// Execute a compiled subagent against a prepared state.
    ///
    /// In production, this invokes the agent graph's runnable. The current
    /// implementation returns the input state augmented with an AI response
    /// message for testing purposes.
    async fn execute_subagent(
        &self,
        compiled: &CompiledSubAgent,
        mut state: AgentState,
    ) -> Result<AgentState, SubAgentError> {
        // Simulate the agent producing a response
        let messages = state
            .entry("messages".to_string())
            .or_insert_with(|| serde_json::json!([]));

        if let Some(arr) = messages.as_array_mut() {
            arr.push(serde_json::json!({
                "type": "ai",
                "content": format!("[{}] Task completed.", compiled.spec.name)
            }));
        }

        Ok(state)
    }
}

/// Errors that can occur during subagent orchestration.
#[derive(Debug, thiserror::Error)]
pub enum SubAgentError {
    /// Subagent exceeded the maximum allowed tool calls.
    #[error("subagent '{agent}' exceeded tool call limit: {count}/{limit}")]
    ToolCallLimitExceeded {
        agent: String,
        count: usize,
        limit: usize,
    },

    /// Subagent execution timed out.
    #[error("subagent '{agent}' timed out after {duration:?}")]
    Timeout {
        agent: String,
        duration: Duration,
    },

    /// Task join error (panic in spawned task).
    #[error("join error: {0}")]
    JoinError(String),

    /// Graph execution error.
    #[error("execution error: {0}")]
    Execution(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompiledSubAgent, SubAgentSpec};

    fn mock_compiled(name: &str) -> CompiledSubAgent {
        CompiledSubAgent {
            spec: SubAgentSpec::new(name, "Test agent"),
            graph: vec!["start".into(), format!("agent:{}", name), "end".into()],
            middleware_pipeline: vec!["prompt_caching".into()],
            backend: "read_only".into(),
        }
    }

    fn empty_parent_state() -> AgentState {
        let mut state = AgentState::new();
        state.insert("messages".into(), serde_json::json!([
            {"type": "human", "content": "parent message"}
        ]));
        state.insert("custom_data".into(), serde_json::json!("shared"));
        state
    }

    #[tokio::test]
    async fn test_spawn_single() {
        let orch = SubAgentOrchestrator::new();
        let compiled = mock_compiled("tester");
        let parent = empty_parent_state();

        let result = orch.spawn(&compiled, "Do the test", &parent).await.unwrap();
        assert_eq!(result.agent_name, "tester");
        assert!(!result.result_message.is_empty());
        assert!(result.duration.as_nanos() > 0);
    }

    #[tokio::test]
    async fn test_state_isolation_parent_messages_not_leaked() {
        let orch = SubAgentOrchestrator::new();
        let compiled = mock_compiled("isolated");
        let parent = empty_parent_state();

        let result = orch.spawn(&compiled, "Check isolation", &parent).await.unwrap();

        // The result should NOT contain the parent's "parent message"
        assert!(!result.result_message.contains("parent message"));
    }

    #[tokio::test]
    async fn test_spawn_parallel_collects_all() {
        let orch = SubAgentOrchestrator::new();
        let agents = vec![
            mock_compiled("agent-a"),
            mock_compiled("agent-b"),
            mock_compiled("agent-c"),
        ];
        let inputs = vec![
            "Task A".to_string(),
            "Task B".to_string(),
            "Task C".to_string(),
        ];
        let parent = empty_parent_state();

        let results = orch.spawn_parallel(&agents, &inputs, &parent).await;

        assert_eq!(results.len(), 3);
        let names: Vec<String> = results
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|r| r.agent_name.clone())
            .collect();
        assert!(names.contains(&"agent-a".to_string()));
        assert!(names.contains(&"agent-b".to_string()));
        assert!(names.contains(&"agent-c".to_string()));
    }

    #[tokio::test]
    async fn test_spawn_parallel_empty() {
        let orch = SubAgentOrchestrator::new();
        let results = orch
            .spawn_parallel(&[], &[], &AgentState::new())
            .await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_merge_results_excludes_messages() {
        let orch = SubAgentOrchestrator::new();
        let mut parent = AgentState::new();
        parent.insert("messages".into(), serde_json::json!([{"type": "human", "content": "hi"}]));

        let mut child = AgentState::new();
        child.insert("messages".into(), serde_json::json!([{"type": "ai", "content": "bye"}]));
        child.insert("findings".into(), serde_json::json!("important"));

        orch.merge_results(&mut parent, &[child]);

        // Parent messages should be untouched
        let msgs = parent.get("messages").unwrap().as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"], "hi");

        // Non-excluded keys should be merged
        assert_eq!(parent.get("findings").unwrap(), &serde_json::json!("important"));
    }

    #[tokio::test]
    async fn test_tool_call_limit_exceeded() {
        let orch = SubAgentOrchestrator::with_limits(Duration::from_secs(10), 2, 4);

        let compiled = mock_compiled("runaway");
        let mut parent = AgentState::new();
        parent.insert("messages".into(), serde_json::json!([]));
        parent.insert("tool_calls_count".into(), serde_json::json!(5));

        let result = orch.spawn(&compiled, "Go wild", &parent).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("exceeded tool call limit"));
    }

    #[tokio::test]
    async fn test_max_concurrent_batching() {
        let orch = SubAgentOrchestrator::with_limits(Duration::from_secs(60), 100, 2);
        let agents: Vec<_> = (0..5).map(|i| mock_compiled(&format!("a{}", i))).collect();
        let inputs: Vec<_> = (0..5).map(|i| format!("Task {}", i)).collect();
        let parent = empty_parent_state();

        let results = orch.spawn_parallel(&agents, &inputs, &parent).await;
        // All 5 should complete even with max_concurrent=2
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.is_ok()));
    }
}
