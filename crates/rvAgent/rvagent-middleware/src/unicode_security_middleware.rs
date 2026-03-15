//! Unicode Security Middleware for rvAgent.
//!
//! Automatically checks tool inputs and outputs for Unicode-based security threats.

use crate::unicode_security::{UnicodeIssue, UnicodeSecurityChecker, UnicodeSecurityConfig};
use crate::{AgentState, AgentStateUpdate, Message, Middleware, Role, Runtime, RunnableConfig};
use async_trait::async_trait;
use tracing::{debug, warn};

/// Middleware that applies Unicode security checks to all tool calls and results.
///
/// ## Security Model
///
/// - **Tool inputs**: Checked before execution, dangerous characters are sanitized
/// - **Tool outputs**: Checked after execution, warnings logged for dangerous content
/// - **User messages**: Optionally checked (configurable)
/// - **System messages**: Never modified (trusted)
///
/// ## Configuration
///
/// - `strict()`: Block all dangerous Unicode (recommended for production)
/// - `permissive()`: Only block BiDi and zero-width (for development)
/// - `check_user_input`: Whether to check user messages (default: false)
/// - `sanitize_inputs`: Whether to sanitize tool inputs (default: true)
/// - `sanitize_outputs`: Whether to sanitize tool outputs (default: false, log only)
pub struct UnicodeSecurityMiddleware {
    checker: UnicodeSecurityChecker,
    check_user_input: bool,
    sanitize_inputs: bool,
    sanitize_outputs: bool,
}

impl UnicodeSecurityMiddleware {
    /// Create a new middleware with strict security.
    pub fn strict() -> Self {
        Self {
            checker: UnicodeSecurityChecker::strict(),
            check_user_input: false,
            sanitize_inputs: true,
            sanitize_outputs: false,
        }
    }

    /// Create a new middleware with custom configuration.
    pub fn new(config: UnicodeSecurityConfig) -> Self {
        Self {
            checker: UnicodeSecurityChecker::new(config),
            check_user_input: false,
            sanitize_inputs: true,
            sanitize_outputs: false,
        }
    }

    /// Enable checking user input messages.
    pub fn with_user_input_check(mut self, enabled: bool) -> Self {
        self.check_user_input = enabled;
        self
    }

    /// Enable sanitizing tool inputs (removes dangerous characters).
    pub fn with_input_sanitization(mut self, enabled: bool) -> Self {
        self.sanitize_inputs = enabled;
        self
    }

    /// Enable sanitizing tool outputs (removes dangerous characters).
    pub fn with_output_sanitization(mut self, enabled: bool) -> Self {
        self.sanitize_outputs = enabled;
        self
    }

    /// Check a message for Unicode security issues.
    fn check_message(&self, msg: &Message) -> Vec<UnicodeIssue> {
        self.checker.check(&msg.content)
    }

    /// Log detected issues.
    fn log_issues(&self, issues: &[UnicodeIssue], context: &str) {
        if !issues.is_empty() {
            warn!(
                "Unicode security issues detected in {}: {} issues",
                context,
                issues.len()
            );
            for issue in issues {
                warn!("  - {}", issue);
            }
        }
    }
}

#[async_trait]
impl Middleware for UnicodeSecurityMiddleware {
    fn name(&self) -> &str {
        "unicode_security"
    }

    async fn abefore_agent(
        &self,
        state: &AgentState,
        _runtime: &Runtime,
        _config: &RunnableConfig,
    ) -> Option<AgentStateUpdate> {
        let mut modified = false;
        let mut new_messages = Vec::new();

        for msg in &state.messages {
            let mut msg_copy = msg.clone();

            match msg.role {
                Role::User if self.check_user_input => {
                    let issues = self.check_message(msg);
                    if !issues.is_empty() {
                        self.log_issues(&issues, "user message");

                        // Sanitize if configured
                        if self.sanitize_inputs {
                            msg_copy.content = self.checker.sanitize(&msg.content);
                            modified = true;
                            debug!("Sanitized user message");
                        }
                    }
                }
                Role::Tool => {
                    let issues = self.check_message(msg);
                    if !issues.is_empty() {
                        self.log_issues(
                            &issues,
                            &format!("tool result: {}", msg.tool_name.as_deref().unwrap_or("unknown")),
                        );

                        // Sanitize if configured
                        if self.sanitize_outputs {
                            msg_copy.content = self.checker.sanitize(&msg.content);
                            modified = true;
                            debug!("Sanitized tool output");
                        }
                    }
                }
                _ => {
                    // Don't modify system or assistant messages
                }
            }

            new_messages.push(msg_copy);

            // Check tool call arguments (in assistant messages)
            if msg.role == Role::Assistant {
                for tool_call in &msg.tool_calls {
                    if let Some(args_str) = tool_call.args.as_str() {
                        let issues = self.checker.check(args_str);
                        if !issues.is_empty() {
                            self.log_issues(&issues, &format!("tool call: {}", tool_call.name));
                        }
                    } else if let Some(obj) = tool_call.args.as_object() {
                        // Check each string field in the arguments
                        for (key, value) in obj {
                            if let Some(s) = value.as_str() {
                                let issues = self.checker.check(s);
                                if !issues.is_empty() {
                                    self.log_issues(
                                        &issues,
                                        &format!("tool call {}.{}", tool_call.name, key),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        if modified {
            Some(AgentStateUpdate {
                messages: Some(new_messages),
                todos: None,
                extensions: Default::default(),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, ToolCall};

    #[tokio::test]
    async fn test_strict_middleware() {
        let mw = UnicodeSecurityMiddleware::strict();
        assert_eq!(mw.name(), "unicode_security");
    }

    #[tokio::test]
    async fn test_detect_bidi_in_tool_result() {
        let mw = UnicodeSecurityMiddleware::strict();

        let state = AgentState {
            messages: vec![Message::tool(
                "evil\u{202E}txt.exe", // BiDi override
                "tc-1",
                "filesystem",
            )],
            todos: vec![],
            extensions: Default::default(),
        };

        let runtime = Runtime::new();
        let config = RunnableConfig::default();

        // Should detect but not modify (sanitize_outputs = false by default)
        let update = mw.abefore_agent(&state, &runtime, &config).await;
        assert!(update.is_none());

        // Enable sanitization
        let mw2 = UnicodeSecurityMiddleware::strict().with_output_sanitization(true);
        let update2 = mw2.abefore_agent(&state, &runtime, &config).await;
        assert!(update2.is_some());

        let new_msgs = update2.unwrap().messages.unwrap();
        assert_eq!(new_msgs[0].content, "eviltxt.exe"); // BiDi stripped
    }

    #[tokio::test]
    async fn test_detect_zero_width_in_user_message() {
        let mw = UnicodeSecurityMiddleware::strict()
            .with_user_input_check(true)
            .with_input_sanitization(true);

        let state = AgentState {
            messages: vec![Message::user("Hello\u{200B}world")],
            todos: vec![],
            extensions: Default::default(),
        };

        let runtime = Runtime::new();
        let config = RunnableConfig::default();

        let update = mw.abefore_agent(&state, &runtime, &config).await;
        assert!(update.is_some());

        let new_msgs = update.unwrap().messages.unwrap();
        assert_eq!(new_msgs[0].content, "Helloworld");
    }

    #[tokio::test]
    async fn test_check_tool_call_arguments() {
        let mw = UnicodeSecurityMiddleware::strict();

        let state = AgentState {
            messages: vec![{
                let mut msg = Message::assistant("");
                msg.tool_calls = vec![ToolCall {
                    id: "tc-1".to_string(),
                    name: "write_file".to_string(),
                    args: serde_json::json!({
                        "path": "test.txt",
                        "content": "evil\u{202E}txt.exe"
                    }),
                }];
                msg
            }],
            todos: vec![],
            extensions: Default::default(),
        };

        let runtime = Runtime::new();
        let config = RunnableConfig::default();

        // Should detect (logs warning) but not modify
        let update = mw.abefore_agent(&state, &runtime, &config).await;
        assert!(update.is_none());
    }

    #[tokio::test]
    async fn test_confusable_detection() {
        // Without output sanitization, should only log warnings
        let mw = UnicodeSecurityMiddleware::strict().with_output_sanitization(false);

        let state = AgentState {
            messages: vec![Message::tool("pаypal.com", "tc-1", "browser")], // Cyrillic 'а'
            todos: vec![],
            extensions: Default::default(),
        };

        let runtime = Runtime::new();
        let config = RunnableConfig::default();

        let update = mw.abefore_agent(&state, &runtime, &config).await;
        // Should detect confusable and log, but not modify (sanitize_outputs = false)
        assert!(update.is_none());
    }

    #[tokio::test]
    async fn test_safe_content_unmodified() {
        let mw = UnicodeSecurityMiddleware::strict()
            .with_user_input_check(true)
            .with_input_sanitization(true);

        let state = AgentState {
            messages: vec![Message::user("Hello world"), Message::tool("OK", "tc-1", "test")],
            todos: vec![],
            extensions: Default::default(),
        };

        let runtime = Runtime::new();
        let config = RunnableConfig::default();

        let update = mw.abefore_agent(&state, &runtime, &config).await;
        assert!(update.is_none()); // No modification needed
    }

    #[tokio::test]
    async fn test_system_messages_never_modified() {
        let mw = UnicodeSecurityMiddleware::strict()
            .with_user_input_check(true)
            .with_input_sanitization(true);

        let state = AgentState {
            messages: vec![Message::system("System\u{202E}message")],
            todos: vec![],
            extensions: Default::default(),
        };

        let runtime = Runtime::new();
        let config = RunnableConfig::default();

        let update = mw.abefore_agent(&state, &runtime, &config).await;
        assert!(update.is_none()); // System messages are never modified
    }

    #[tokio::test]
    async fn test_permissive_config() {
        let mw =
            UnicodeSecurityMiddleware::new(UnicodeSecurityConfig::permissive())
                .with_output_sanitization(true);

        let state = AgentState {
            messages: vec![
                Message::tool("pаypal.com", "tc-1", "test"), // Confusable (should pass)
                Message::tool("evil\u{202E}txt.exe", "tc-2", "test"), // BiDi (should be caught)
            ],
            todos: vec![],
            extensions: Default::default(),
        };

        let runtime = Runtime::new();
        let config = RunnableConfig::default();

        let update = mw.abefore_agent(&state, &runtime, &config).await;
        assert!(update.is_some());

        let new_msgs = update.unwrap().messages.unwrap();
        // First message unchanged (confusables not checked in permissive mode)
        assert_eq!(new_msgs[0].content, "pаypal.com");
        // Second message sanitized (BiDi always checked)
        assert_eq!(new_msgs[1].content, "eviltxt.exe");
    }

    #[tokio::test]
    async fn test_multiple_messages() {
        let mw = UnicodeSecurityMiddleware::strict()
            .with_user_input_check(true)
            .with_input_sanitization(true)
            .with_output_sanitization(true);

        let state = AgentState {
            messages: vec![
                Message::user("Hello\u{200B}world"),
                Message::assistant("Response"),
                Message::tool("evil\u{202E}txt.exe", "tc-1", "filesystem"),
            ],
            todos: vec![],
            extensions: Default::default(),
        };

        let runtime = Runtime::new();
        let config = RunnableConfig::default();

        let update = mw.abefore_agent(&state, &runtime, &config).await;
        assert!(update.is_some());

        let new_msgs = update.unwrap().messages.unwrap();
        assert_eq!(new_msgs.len(), 3);
        assert_eq!(new_msgs[0].content, "Helloworld"); // User message sanitized
        assert_eq!(new_msgs[1].content, "Response"); // Assistant unchanged
        assert_eq!(new_msgs[2].content, "eviltxt.exe"); // Tool result sanitized
    }
}
