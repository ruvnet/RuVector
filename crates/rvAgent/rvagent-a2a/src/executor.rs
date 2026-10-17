//! Executor abstraction (ADR-159 — "Executor abstraction: TaskRunner").
//!
//! `TaskRunner` is the trait the A2A server delegates actual execution to.
//! `Executor` is the location-transparent wrapper: `Local` runs via a
//! `TaskRunner` implementation in-process, `Remote` delegates to a peer
//! over HTTP via `crate::client::A2aClient`.
//!
//! For M1 we ship `InMemoryRunner`, a zero-dependency runner that echoes
//! the input message back as an artifact and walks the full submitted →
//! working → completed transition synchronously.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;

use crate::{
    error::A2aError,
    identity::AgentID,
    types::{AgentCard, Task, TaskSpec},
};

// ---------------------------------------------------------------------------
// TaskRunner trait.
// ---------------------------------------------------------------------------

/// The agent-implementation side of A2A. Anything that can run a [`TaskSpec`]
/// to completion — a local LLM, a WASM plugin, a remote service adapter —
/// implements this trait.
// `async_trait` adds a bare `#[must_use]` to methods whose boxed-future return type is
// already `#[must_use]`; clippy 1.99 flags that as `double_must_use`.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait TaskRunner: Send + Sync {
    async fn run(&self, spec: TaskSpec) -> Result<Task, A2aError>;
    async fn cancel(&self, task_id: &str) -> Result<(), A2aError>;
}

// ---------------------------------------------------------------------------
// Peer + Executor.
// ---------------------------------------------------------------------------

/// Remote peer handle. The `client` is shared so several `Peer`s can reuse
/// the same underlying HTTP connection pool.
#[derive(Clone)]
pub struct Peer {
    pub id: AgentID,
    pub card: AgentCard,
    pub base_url: url::Url,
    pub client: Arc<crate::client::A2aClient>,
}

impl std::fmt::Debug for Peer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Peer")
            .field("id", &self.id)
            .field("base_url", &self.base_url)
            .finish()
    }
}

/// Location-transparent dispatch. `Local` runs via a trait object;
/// `Remote` forwards over the wire. `Remote` is boxed so the two variants
/// have comparable size and the enum doesn't trip clippy's
/// `large-enum-variant` lint.
#[derive(Clone)]
pub enum Executor {
    Local(Arc<dyn TaskRunner>),
    Remote(Box<Peer>),
}

impl Executor {
    #[tracing::instrument(level = "debug", skip(self, spec))]
    pub async fn run(&self, spec: TaskSpec) -> Result<Task, A2aError> {
        match self {
            Executor::Local(runner) => runner.run(spec).await,
            Executor::Remote(peer) => {
                // TODO(client.rs owner): confirm `A2aClient::send_task`
                // signature is `(base_url: &str, spec: TaskSpec) -> Result<Task, A2aError>`.
                peer.client.send_task(peer.base_url.as_str(), spec).await
            }
        }
    }

    #[tracing::instrument(level = "debug", skip(self))]
    pub async fn cancel(&self, id: &str) -> Result<(), A2aError> {
        match self {
            Executor::Local(runner) => runner.cancel(id).await,
            Executor::Remote(peer) => peer.client.cancel_task(peer.base_url.as_str(), id).await,
        }
    }

    pub fn id_label(&self) -> String {
        match self {
            Executor::Local(_) => "local".into(),
            Executor::Remote(peer) => format!("remote:{}", peer.id),
        }
    }
}

// ---------------------------------------------------------------------------
// M1 default runner — echo.
// ---------------------------------------------------------------------------

/// Minimal runnable default. Takes the input message, wraps it as a single
/// artifact, and transitions the task submitted → working → completed.
/// Useful for smoke-testing the server wiring before a real runner lands.
#[derive(Default, Clone)]
pub struct InMemoryRunner;

impl InMemoryRunner {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl TaskRunner for InMemoryRunner {
    #[tracing::instrument(level = "debug", skip(self, spec))]
    async fn run(&self, spec: TaskSpec) -> Result<Task, A2aError> {
        use crate::types::{Artifact, Message, Part, Role, Task, TaskStatus};

        // Minimum two history entries: the caller's input, then an agent
        // response echoing it. This satisfies the r2 state-machine
        // contract (`Submitted → … → Completed`) in a synchronous runner.
        let input_parts = spec.message.parts.clone();
        let agent_reply = Message {
            role: Role::Agent,
            parts: vec![Part::Text {
                text: "echo".into(),
            }],
            metadata: serde_json::Value::Null,
        };
        let artifact = Artifact {
            name: Some("echo".into()),
            description: Some("InMemoryRunner echo of the input message".into()),
            parts: input_parts,
            index: 0,
            append: false,
            last_chunk: true,
            metadata: serde_json::Value::Null,
        };
        Ok(Task {
            id: spec.id,
            session_id: None,
            status: TaskStatus {
                state: crate::types::TaskState::Completed,
                timestamp: Utc::now(),
                message: None,
            },
            history: vec![spec.message, agent_reply],
            artifacts: vec![artifact],
            metadata: spec.metadata,
        })
    }

    async fn cancel(&self, _task_id: &str) -> Result<(), A2aError> {
        // Synchronous runner: nothing to cancel; treat as a successful noop.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::TaskContext;
    use crate::types::{Message, Part, Role, TaskState};

    fn spec() -> TaskSpec {
        TaskSpec {
            id: "t-test".into(),
            skill: "echo".into(),
            message: Message {
                role: Role::User,
                parts: vec![Part::Text {
                    text: "hello".into(),
                }],
                metadata: serde_json::Value::Null,
            },
            policy: None,
            context: TaskContext::new_root(AgentID("0".repeat(64))),
            metadata: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn inmemory_runner_completes_and_echoes() {
        let runner = InMemoryRunner::new();
        let task = runner.run(spec()).await.expect("run");
        assert_eq!(task.status.state, TaskState::Completed);
        assert!(!task.artifacts.is_empty());
    }

    #[tokio::test]
    async fn local_executor_dispatches_to_runner() {
        let exec = Executor::Local(Arc::new(InMemoryRunner::new()));
        let task = exec.run(spec()).await.expect("run");
        assert_eq!(task.status.state, TaskState::Completed);
    }

    #[tokio::test]
    async fn local_executor_cancel_is_noop_ok() {
        let exec = Executor::Local(Arc::new(InMemoryRunner::new()));
        assert!(exec.cancel("task-does-not-exist").await.is_ok());
    }

    #[test]
    fn id_label_local() {
        let exec = Executor::Local(Arc::new(InMemoryRunner::new()));
        assert_eq!(exec.id_label(), "local");
    }
}
