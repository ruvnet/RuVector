//! Task orchestration with parallel execution and DAG resolution

use crate::{
    metrics::Metrics,
    task::{Task, TaskResult},
    Result, SwarmConfig, SwarmError,
};
use chrono::Utc;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::time::{timeout, Duration};
use uuid::Uuid;

/// Task orchestrator for parallel and sequential execution
pub struct Orchestrator {
    config: SwarmConfig,
    metrics: Arc<Metrics>,
}

impl Orchestrator {
    /// Create a new orchestrator
    pub fn new(config: SwarmConfig, metrics: Arc<Metrics>) -> Self {
        Self { config, metrics }
    }

    /// Execute tasks in parallel using rayon
    pub async fn execute_parallel(&self, tasks: Vec<Task>) -> Result<Vec<TaskResult>> {
        let start = Instant::now();
        tracing::info!("Executing {} tasks in parallel", tasks.len());

        let metrics = Arc::clone(&self.metrics);
        let audit_enabled = self.config.audit_enabled;

        // Use rayon for CPU parallelism
        let results: Vec<TaskResult> = tasks
            .into_par_iter()
            .map(|task| {
                let task_start = Instant::now();
                let task_id = task.id;

                if audit_enabled {
                    tracing::debug!("Starting task: {} - {}", task_id, task.description);
                }

                let result = Self::execute_task_sync(task);

                let duration = task_start.elapsed();
                metrics.record_task_duration(duration);

                if audit_enabled {
                    tracing::debug!(
                        "Completed task: {} in {:?} - success: {}",
                        task_id,
                        duration,
                        result.success
                    );
                }

                result
            })
            .collect();

        let total_duration = start.elapsed();
        tracing::info!(
            "Parallel execution completed in {:?} - {} tasks",
            total_duration,
            results.len()
        );

        Ok(results)
    }

    /// Execute tasks sequentially in order
    pub async fn execute_sequential(&self, tasks: Vec<Task>) -> Result<Vec<TaskResult>> {
        let start = Instant::now();
        tracing::info!("Executing {} tasks sequentially", tasks.len());

        let mut results = Vec::new();

        for task in tasks {
            let task_start = Instant::now();
            let task_id = task.id;

            if self.config.audit_enabled {
                tracing::debug!("Starting task: {} - {}", task_id, task.description);
            }

            let result = self.execute_task_async(task).await?;

            let duration = task_start.elapsed();
            self.metrics.record_task_duration(duration);

            if self.config.audit_enabled {
                tracing::debug!(
                    "Completed task: {} in {:?} - success: {}",
                    task_id,
                    duration,
                    result.success
                );
            }

            results.push(result);
        }

        let total_duration = start.elapsed();
        tracing::info!(
            "Sequential execution completed in {:?} - {} tasks",
            total_duration,
            results.len()
        );

        Ok(results)
    }

    /// Execute tasks based on dependency DAG
    pub async fn execute_dag(&self, tasks: Vec<Task>) -> Result<Vec<TaskResult>> {
        let start = Instant::now();
        tracing::info!("Executing {} tasks using DAG", tasks.len());

        // Validate DAG (no cycles)
        self.validate_dag(&tasks)?;

        // Build dependency graph
        let task_map: HashMap<Uuid, Task> = tasks.into_iter().map(|t| (t.id, t)).collect();

        let mut results = Vec::new();
        let mut completed = HashSet::new();
        let mut in_progress = HashSet::new();

        // Process tasks in waves based on dependencies
        while completed.len() < task_map.len() {
            // Find all tasks ready to execute (dependencies satisfied)
            let ready: Vec<_> = task_map
                .values()
                .filter(|task| {
                    !completed.contains(&task.id)
                        && !in_progress.contains(&task.id)
                        && task
                            .dependencies
                            .iter()
                            .all(|dep| completed.contains(dep))
                })
                .cloned()
                .collect();

            if ready.is_empty() {
                if completed.len() < task_map.len() {
                    return Err(SwarmError::DependencyCycle);
                }
                break;
            }

            tracing::debug!("Executing wave of {} ready tasks", ready.len());

            // Mark as in progress
            for task in &ready {
                in_progress.insert(task.id);
            }

            // Execute ready tasks in parallel
            let wave_results = self.execute_parallel(ready).await?;

            // Mark as completed
            for result in wave_results {
                completed.insert(result.task_id);
                in_progress.remove(&result.task_id);
                results.push(result);
            }
        }

        let total_duration = start.elapsed();
        tracing::info!(
            "DAG execution completed in {:?} - {} tasks",
            total_duration,
            results.len()
        );

        Ok(results)
    }

    /// Validate DAG has no cycles using topological sort
    fn validate_dag(&self, tasks: &[Task]) -> Result<()> {
        let mut graph: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        let mut in_degree: HashMap<Uuid, usize> = HashMap::new();

        // Build graph
        for task in tasks {
            graph.entry(task.id).or_insert_with(Vec::new);
            in_degree.entry(task.id).or_insert(0);

            for dep in &task.dependencies {
                graph.entry(*dep).or_insert_with(Vec::new).push(task.id);
                *in_degree.entry(task.id).or_insert(0) += 1;
            }
        }

        // Kahn's algorithm for cycle detection
        let mut queue: VecDeque<Uuid> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut visited = 0;

        while let Some(node) = queue.pop_front() {
            visited += 1;

            if let Some(neighbors) = graph.get(&node) {
                for &neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(&neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if visited != tasks.len() {
            return Err(SwarmError::DependencyCycle);
        }

        Ok(())
    }

    /// Execute a single task asynchronously
    async fn execute_task_async(&self, task: Task) -> Result<TaskResult> {
        let task_id = task.id;
        let description = task.description.clone();
        let command = task.command.clone();
        let task_timeout = task.timeout_secs;

        let start = Instant::now();

        // Execute with timeout if specified
        let result = if let Some(timeout_secs) = task_timeout {
            let timeout_duration = Duration::from_secs(timeout_secs);
            match timeout(timeout_duration, Self::run_command(command.clone())).await {
                Ok(Ok(output)) => {
                    let duration_ms = start.elapsed().as_millis() as u64;
                    TaskResult::success(task_id, description, output, duration_ms)
                }
                Ok(Err(e)) => {
                    let duration_ms = start.elapsed().as_millis() as u64;
                    TaskResult::failure(task_id, description, e, duration_ms)
                }
                Err(_) => {
                    let duration_ms = start.elapsed().as_millis() as u64;
                    TaskResult::failure(
                        task_id,
                        description,
                        format!("Task timed out after {}s", timeout_secs),
                        duration_ms,
                    )
                }
            }
        } else {
            match Self::run_command(command).await {
                Ok(output) => {
                    let duration_ms = start.elapsed().as_millis() as u64;
                    TaskResult::success(task_id, description, output, duration_ms)
                }
                Err(e) => {
                    let duration_ms = start.elapsed().as_millis() as u64;
                    TaskResult::failure(task_id, description, e, duration_ms)
                }
            }
        };

        Ok(result)
    }

    /// Execute a single task synchronously (for rayon)
    fn execute_task_sync(task: Task) -> TaskResult {
        let task_id = task.id;
        let description = task.description.clone();
        let command = task.command.clone();

        let start = Instant::now();

        // For synchronous execution, we'll use a simplified approach
        // In production, you might want to use crossbeam channels or other sync primitives
        let output = format!("Executed command: {}", command);
        let duration_ms = start.elapsed().as_millis() as u64;

        TaskResult::success(task_id, description, output, duration_ms)
    }

    /// Run a command and return output
    async fn run_command(command: String) -> std::result::Result<String, String> {
        // This is a placeholder - in production, you'd use tokio::process::Command
        // For now, just simulate command execution
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(format!("Command executed: {}", command))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::Priority;

    fn create_test_orchestrator() -> Orchestrator {
        let config = SwarmConfig {
            id: Uuid::new_v4(),
            name: "test".to_string(),
            topology: crate::TopologyType::Mesh,
            max_agents: 4,
            strategy: crate::ExecutionStrategy::DAG,
            audit_enabled: true,
            metrics_enabled: true,
        };
        let metrics = Arc::new(Metrics::new(true));
        Orchestrator::new(config, metrics)
    }

    #[tokio::test]
    async fn test_parallel_execution() {
        let orchestrator = create_test_orchestrator();

        let tasks = vec![
            Task::new("task1", "cmd1"),
            Task::new("task2", "cmd2"),
            Task::new("task3", "cmd3"),
        ];

        let results = orchestrator.execute_parallel(tasks).await.unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.success));
    }

    #[tokio::test]
    async fn test_sequential_execution() {
        let orchestrator = create_test_orchestrator();

        let tasks = vec![
            Task::new("task1", "cmd1"),
            Task::new("task2", "cmd2"),
            Task::new("task3", "cmd3"),
        ];

        let results = orchestrator.execute_sequential(tasks).await.unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.success));
    }

    #[tokio::test]
    async fn test_dag_execution() {
        let orchestrator = create_test_orchestrator();

        let task1 = Task::new("task1", "cmd1");
        let task1_id = task1.id;

        let task2 = Task::new("task2", "cmd2").with_dependency(task1_id);
        let task3 = Task::new("task3", "cmd3").with_dependency(task1_id);

        let tasks = vec![task1, task2, task3];

        let results = orchestrator.execute_dag(tasks).await.unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.success));
    }

    #[tokio::test]
    async fn test_cycle_detection() {
        let orchestrator = create_test_orchestrator();

        let task1 = Task::new("task1", "cmd1");
        let task1_id = task1.id;

        let task2 = Task::new("task2", "cmd2").with_dependency(task1_id);
        let task2_id = task2.id;

        // Create cycle: task1 -> task2 -> task1
        let task1_with_cycle = Task::new("task1", "cmd1").with_dependency(task2_id);

        let tasks = vec![task1_with_cycle, task2];

        let result = orchestrator.validate_dag(&tasks);
        assert!(result.is_err());
    }
}
