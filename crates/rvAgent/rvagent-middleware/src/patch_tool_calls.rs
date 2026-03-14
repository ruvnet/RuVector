//! PatchToolCalls middleware stub.
use async_trait::async_trait;
use crate::{Middleware, AgentState, AgentStateUpdate, Runtime, RunnableConfig};

pub struct PatchToolCallsMiddleware;
impl PatchToolCallsMiddleware {
    pub fn new() -> Self { Self }
}
#[async_trait]
impl Middleware for PatchToolCallsMiddleware {
    fn name(&self) -> &str { "patch_tool_calls" }

    fn before_agent(
        &self,
        state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        if state.messages.is_empty() {
            return None;
        }

        let mut patched = Vec::new();
        for (i, msg) in state.messages.iter().enumerate() {
            patched.push(msg.clone());

            if msg.role == crate::Role::Assistant && !msg.tool_calls.is_empty() {
                for tc in &msg.tool_calls {
                    let has_response = state.messages[i..].iter().any(|m| {
                        m.role == crate::Role::Tool
                            && m.tool_call_id.as_deref() == Some(&tc.id)
                    });
                    if !has_response {
                        patched.push(crate::Message::tool(
                            format!(
                                "Tool call {} with id {} was cancelled",
                                tc.name, tc.id
                            ),
                            &tc.id,
                            &tc.name,
                        ));
                    }
                }
            }
        }

        let mut update = AgentStateUpdate::default();
        update.messages = Some(patched);
        Some(update)
    }
}
