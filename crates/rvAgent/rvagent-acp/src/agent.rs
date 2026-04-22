//! ACP agent — wraps `RvAgentConfig` and manages sessions.
//!
//! Provides thread-safe session CRUD and prompt execution.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use async_trait::async_trait;

use rvagent_core::config::RvAgentConfig;
use rvagent_core::error::Result as CoreResult;
use rvagent_core::graph::{AgentGraph, GraphConfig, ToolExecutor};
use rvagent_core::messages::{Message, ToolCall};
use rvagent_core::models::ChatModel;
use rvagent_core::state::AgentState;

use crate::types::{
    ContentBlock, CreateSessionRequest, PromptResponse, ResponseMessage, SessionInfo,
};

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// An active agent session holding conversation history.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Session {
    pub info: SessionInfo,
    pub messages: Vec<Message>,
    pub cwd: Option<String>,
}

impl Session {
    fn new(cwd: Option<String>) -> Self {
        Self {
            info: SessionInfo {
                id: Uuid::new_v4().to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                message_count: 0,
            },
            messages: Vec::new(),
            cwd,
        }
    }
}

// ---------------------------------------------------------------------------
// Placeholder tool executor
// ---------------------------------------------------------------------------

/// Placeholder tool executor for the ACP server.
///
/// Returns the tool name as the result, allowing end-to-end testing of the
/// AgentGraph pipeline without requiring real tool implementations.
struct AcpToolExecutor;

#[async_trait]
impl ToolExecutor for AcpToolExecutor {
    async fn execute(&self, call: &ToolCall, _state: &AgentState) -> CoreResult<String> {
        Ok(format!("[ACP] tool '{}' executed", call.name))
    }
}

// ---------------------------------------------------------------------------
// Stub chat model (no API key required)
// ---------------------------------------------------------------------------

/// A stub `ChatModel` for the ACP server that echoes user messages
/// without calling any external API.
///
/// This allows the ACP server to function end-to-end for testing and
/// local development without requiring an LLM provider API key.
struct StubModel;

#[async_trait]
impl ChatModel for StubModel {
    async fn complete(&self, messages: &[Message]) -> CoreResult<Message> {
        // Find the last human message and produce an intelligent echo.
        let user_text = messages
            .iter()
            .rev()
            .find_map(|m| match m {
                Message::Human(h) => Some(h.content.as_str()),
                _ => None,
            })
            .unwrap_or("(no user message)");

        let response = format!(
            "I received your message ({} chars). Processing complete.",
            user_text.len()
        );
        Ok(Message::ai(response))
    }

    async fn stream(&self, messages: &[Message]) -> CoreResult<Vec<Message>> {
        let msg = self.complete(messages).await?;
        Ok(vec![msg])
    }
}

// ---------------------------------------------------------------------------
// AcpAgent
// ---------------------------------------------------------------------------

/// The core ACP agent that manages sessions and handles prompts.
///
/// Thread-safe via `Arc<RwLock<…>>` on the session map.
pub struct AcpAgent {
    #[allow(dead_code)]
    config: RvAgentConfig,
    sessions: Arc<RwLock<HashMap<String, Session>>>,
}

impl AcpAgent {
    /// Create a new `AcpAgent` with the given configuration.
    pub fn new(config: RvAgentConfig) -> Self {
        Self {
            config,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new session, returning its info.
    pub async fn create_session(&self, req: &CreateSessionRequest) -> SessionInfo {
        let session = Session::new(req.cwd.clone());
        let info = session.info.clone();
        let mut sessions = self.sessions.write().await;
        sessions.insert(info.id.clone(), session);
        info
    }

    /// List all active sessions.
    pub async fn list_sessions(&self) -> Vec<SessionInfo> {
        let sessions = self.sessions.read().await;
        sessions.values().map(|s| s.info.clone()).collect()
    }

    /// Get a single session by ID.
    pub async fn get_session(&self, id: &str) -> Option<SessionInfo> {
        let sessions = self.sessions.read().await;
        sessions.get(id).map(|s| s.info.clone())
    }

    /// Delete a session by ID. Returns `true` if it existed.
    pub async fn delete_session(&self, id: &str) -> bool {
        let mut sessions = self.sessions.write().await;
        sessions.remove(id).is_some()
    }

    /// Execute a prompt against a session.
    ///
    /// If `session_id` is `None`, a new session is created automatically.
    /// Returns the agent's response along with the session ID used.
    pub async fn prompt(
        &self,
        session_id: Option<&str>,
        content: Vec<ContentBlock>,
    ) -> Result<PromptResponse, String> {
        // Resolve or create session.
        let sid = match session_id {
            Some(id) => {
                let sessions = self.sessions.read().await;
                if !sessions.contains_key(id) {
                    return Err(format!("session not found: {}", id));
                }
                id.to_string()
            }
            None => {
                let info = self.create_session(&CreateSessionRequest::default()).await;
                info.id
            }
        };

        // Convert content blocks into a user message.
        let user_text = content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        let user_msg = Message::human(&user_text);

        // Run the prompt through an AgentGraph with a stub model.
        //
        // In production, the model would be resolved from `self.config`
        // and real tools/middleware would be wired in. The stub model
        // allows the server to run without an API key.
        let graph_config = GraphConfig {
            max_iterations: 10,
            parallel_tools: false,
        };
        let graph = AgentGraph::with_config(StubModel, AcpToolExecutor, graph_config);

        let mut agent_state = AgentState::new();
        agent_state.push_message(user_msg.clone());

        let final_state = graph
            .run(agent_state)
            .await
            .map_err(|e| format!("agent graph error: {}", e))?;

        // Extract the last AI message as the response.
        let response_text = final_state
            .messages
            .iter()
            .rev()
            .find_map(|m| match m {
                Message::Ai(ai) => Some(ai.content.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "No response generated.".to_string());
        let ai_msg = Message::ai(&response_text);

        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(&sid) {
                session.messages.push(user_msg);
                session.messages.push(ai_msg);
                session.info.message_count = session.messages.len();
            }
        }

        Ok(PromptResponse {
            session_id: sid,
            messages: vec![ResponseMessage {
                role: "assistant".into(),
                content: vec![ContentBlock::Text {
                    text: response_text,
                }],
            }],
        })
    }

    /// Access the underlying agent configuration.
    #[allow(dead_code)]
    pub fn config(&self) -> &RvAgentConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_agent() -> AcpAgent {
        AcpAgent::new(RvAgentConfig::default())
    }

    #[tokio::test]
    async fn test_create_session() {
        let agent = default_agent();
        let info = agent.create_session(&CreateSessionRequest::default()).await;
        assert!(!info.id.is_empty());
        assert_eq!(info.message_count, 0);
    }

    #[tokio::test]
    async fn test_list_sessions() {
        let agent = default_agent();
        assert!(agent.list_sessions().await.is_empty());

        agent.create_session(&CreateSessionRequest::default()).await;
        agent.create_session(&CreateSessionRequest::default()).await;

        assert_eq!(agent.list_sessions().await.len(), 2);
    }

    #[tokio::test]
    async fn test_get_session() {
        let agent = default_agent();
        let info = agent.create_session(&CreateSessionRequest::default()).await;

        let fetched = agent.get_session(&info.id).await;
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().id, info.id);

        assert!(agent.get_session("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_delete_session() {
        let agent = default_agent();
        let info = agent.create_session(&CreateSessionRequest::default()).await;

        assert!(agent.delete_session(&info.id).await);
        assert!(!agent.delete_session(&info.id).await);
        assert!(agent.get_session(&info.id).await.is_none());
    }

    #[tokio::test]
    async fn test_prompt_creates_session_if_missing() {
        let agent = default_agent();
        let resp = agent
            .prompt(
                None,
                vec![ContentBlock::Text {
                    text: "hello".into(),
                }],
            )
            .await
            .unwrap();

        assert!(!resp.session_id.is_empty());
        assert_eq!(resp.messages.len(), 1);
        assert_eq!(resp.messages[0].role, "assistant");
    }

    #[tokio::test]
    async fn test_prompt_existing_session() {
        let agent = default_agent();
        let info = agent.create_session(&CreateSessionRequest::default()).await;

        let resp = agent
            .prompt(
                Some(&info.id),
                vec![ContentBlock::Text {
                    text: "test".into(),
                }],
            )
            .await
            .unwrap();

        assert_eq!(resp.session_id, info.id);

        // Session should now have 2 messages (user + assistant).
        let updated = agent.get_session(&info.id).await.unwrap();
        assert_eq!(updated.message_count, 2);
    }

    #[tokio::test]
    async fn test_prompt_nonexistent_session_returns_error() {
        let agent = default_agent();
        let result = agent
            .prompt(
                Some("bad_id"),
                vec![ContentBlock::Text { text: "hi".into() }],
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_prompt_response_valid_content() {
        let agent = default_agent();
        let resp = agent
            .prompt(
                None,
                vec![
                    ContentBlock::Text { text: "one".into() },
                    ContentBlock::Text { text: "two".into() },
                ],
            )
            .await
            .unwrap();

        // Response should contain a non-empty text block from the agent.
        let text = match &resp.messages[0].content[0] {
            ContentBlock::Text { text } => text.as_str(),
            _ => panic!("expected text block"),
        };
        assert!(!text.is_empty());
    }
}
