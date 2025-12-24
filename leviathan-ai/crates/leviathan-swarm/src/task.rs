//! Task management and queuing

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use uuid::Uuid;

/// Task priority
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Task specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier
    pub id: Uuid,
    /// Task description
    pub description: String,
    /// Task dependencies (must complete before this task)
    pub dependencies: Vec<Uuid>,
    /// Command to execute
    pub command: String,
    /// Task priority
    pub priority: Priority,
    /// Timeout in seconds
    pub timeout_secs: Option<u64>,
    /// Retry count on failure
    pub retry_count: u32,
    /// Task metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl Task {
    /// Create a new task
    pub fn new(description: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            description: description.into(),
            dependencies: Vec::new(),
            command: command.into(),
            priority: Priority::Normal,
            timeout_secs: None,
            retry_count: 0,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// Add a dependency to the task
    pub fn with_dependency(mut self, task_id: Uuid) -> Self {
        self.dependencies.push(task_id);
        self
    }

    /// Set task priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set task timeout
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = Some(timeout_secs);
        self
    }

    /// Set retry count
    pub fn with_retry(mut self, retry_count: u32) -> Self {
        self.retry_count = retry_count;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if task has unmet dependencies
    pub fn has_dependencies(&self, completed: &HashSet<Uuid>) -> bool {
        !self.dependencies.iter().all(|dep| completed.contains(dep))
    }
}

/// Wrapper for priority queue ordering
#[derive(Debug, Clone)]
struct PriorityTask {
    task: Task,
    priority: Priority,
}

impl PartialEq for PriorityTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PriorityTask {}

impl PartialOrd for PriorityTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

/// Priority-based task queue
pub struct TaskQueue {
    queue: BinaryHeap<PriorityTask>,
    task_map: HashMap<Uuid, Task>,
}

impl TaskQueue {
    /// Create a new task queue
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            task_map: HashMap::new(),
        }
    }

    /// Add a task to the queue
    pub fn enqueue(&mut self, task: Task) {
        let priority = task.priority;
        let task_id = task.id;
        self.task_map.insert(task_id, task.clone());
        self.queue.push(PriorityTask { task, priority });
    }

    /// Remove and return the highest priority task
    pub fn dequeue(&mut self) -> Option<Task> {
        self.queue.pop().map(|pt| {
            self.task_map.remove(&pt.task.id);
            pt.task
        })
    }

    /// Peek at the highest priority task without removing it
    pub fn peek(&self) -> Option<&Task> {
        self.queue.peek().map(|pt| &pt.task)
    }

    /// Get a task by ID
    pub fn get(&self, task_id: &Uuid) -> Option<&Task> {
        self.task_map.get(task_id)
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Get all tasks ready to execute (no unmet dependencies)
    pub fn get_ready_tasks(&self, completed: &HashSet<Uuid>) -> Vec<Task> {
        self.task_map
            .values()
            .filter(|task| !task.has_dependencies(completed))
            .cloned()
            .collect()
    }
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task ID
    pub task_id: Uuid,
    /// Task description
    pub description: String,
    /// Success status
    pub success: bool,
    /// Output from task execution
    pub output: String,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution duration in milliseconds
    pub duration_ms: u64,
    /// Start timestamp
    pub started_at: DateTime<Utc>,
    /// Completion timestamp
    pub completed_at: DateTime<Utc>,
    /// Additional outputs/artifacts
    pub outputs: HashMap<String, String>,
}

impl TaskResult {
    /// Create a successful task result
    pub fn success(task_id: Uuid, description: String, output: String, duration_ms: u64) -> Self {
        let now = Utc::now();
        let started_at = now - chrono::Duration::milliseconds(duration_ms as i64);

        Self {
            task_id,
            description,
            success: true,
            output,
            error: None,
            duration_ms,
            started_at,
            completed_at: now,
            outputs: HashMap::new(),
        }
    }

    /// Create a failed task result
    pub fn failure(
        task_id: Uuid,
        description: String,
        error: String,
        duration_ms: u64,
    ) -> Self {
        let now = Utc::now();
        let started_at = now - chrono::Duration::milliseconds(duration_ms as i64);

        Self {
            task_id,
            description,
            success: false,
            output: String::new(),
            error: Some(error),
            duration_ms,
            started_at,
            completed_at: now,
            outputs: HashMap::new(),
        }
    }

    /// Add an output artifact
    pub fn with_output(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.outputs.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_creation() {
        let task = Task::new("test task", "echo hello")
            .with_priority(Priority::High)
            .with_timeout(60)
            .with_retry(3)
            .with_metadata("key", "value");

        assert_eq!(task.description, "test task");
        assert_eq!(task.command, "echo hello");
        assert_eq!(task.priority, Priority::High);
        assert_eq!(task.timeout_secs, Some(60));
        assert_eq!(task.retry_count, 3);
    }

    #[test]
    fn test_task_dependencies() {
        let task1_id = Uuid::new_v4();
        let task2_id = Uuid::new_v4();

        let task = Task::new("dependent task", "echo test")
            .with_dependency(task1_id)
            .with_dependency(task2_id);

        let mut completed = HashSet::new();
        assert!(task.has_dependencies(&completed));

        completed.insert(task1_id);
        assert!(task.has_dependencies(&completed));

        completed.insert(task2_id);
        assert!(!task.has_dependencies(&completed));
    }

    #[test]
    fn test_task_queue_priority() {
        let mut queue = TaskQueue::new();

        let low = Task::new("low", "cmd").with_priority(Priority::Low);
        let high = Task::new("high", "cmd").with_priority(Priority::High);
        let normal = Task::new("normal", "cmd").with_priority(Priority::Normal);

        queue.enqueue(low);
        queue.enqueue(normal.clone());
        queue.enqueue(high.clone());

        // Should dequeue in priority order: High -> Normal -> Low
        let first = queue.dequeue().unwrap();
        assert_eq!(first.description, "high");

        let second = queue.dequeue().unwrap();
        assert_eq!(second.description, "normal");

        let third = queue.dequeue().unwrap();
        assert_eq!(third.description, "low");
    }

    #[test]
    fn test_task_result() {
        let task_id = Uuid::new_v4();
        let result = TaskResult::success(task_id, "test".to_string(), "output".to_string(), 100)
            .with_output("key", "value");

        assert!(result.success);
        assert_eq!(result.duration_ms, 100);
        assert_eq!(result.outputs.get("key"), Some(&"value".to_string()));
    }
}
