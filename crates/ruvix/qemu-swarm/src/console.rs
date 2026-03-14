//! Console I/O multiplexing for QEMU swarm.
//!
//! Aggregates and routes console output from multiple QEMU nodes.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tracing::{debug, error, info, warn};

use crate::error::{SwarmError, SwarmResult};
use crate::node::NodeId;

/// A message from a node's console.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsoleMessage {
    /// Source node ID.
    pub node_id: NodeId,

    /// Node index.
    pub node_index: usize,

    /// Timestamp when received.
    pub timestamp: DateTime<Utc>,

    /// Message content.
    pub content: String,

    /// Message severity (if parseable).
    pub severity: MessageSeverity,

    /// Message category (if parseable).
    pub category: Option<String>,
}

impl ConsoleMessage {
    /// Create a new console message.
    pub fn new(node_id: NodeId, node_index: usize, content: String) -> Self {
        let severity = Self::detect_severity(&content);
        let category = Self::detect_category(&content);

        Self {
            node_id,
            node_index,
            timestamp: Utc::now(),
            content,
            severity,
            category,
        }
    }

    /// Detect message severity from content.
    fn detect_severity(content: &str) -> MessageSeverity {
        let lower = content.to_lowercase();
        if lower.contains("panic") || lower.contains("fatal") || lower.contains("crash") {
            MessageSeverity::Panic
        } else if lower.contains("error") || lower.contains("fail") {
            MessageSeverity::Error
        } else if lower.contains("warn") {
            MessageSeverity::Warning
        } else if lower.contains("debug") || lower.contains("trace") {
            MessageSeverity::Debug
        } else {
            MessageSeverity::Info
        }
    }

    /// Detect message category from content.
    fn detect_category(content: &str) -> Option<String> {
        // Try to extract category from common log formats like [CATEGORY] or <CATEGORY>
        if let Some(start) = content.find('[') {
            if let Some(end) = content[start..].find(']') {
                return Some(content[start + 1..start + end].to_string());
            }
        }
        if let Some(start) = content.find('<') {
            if let Some(end) = content[start..].find('>') {
                return Some(content[start + 1..start + end].to_string());
            }
        }
        None
    }

    /// Format for display.
    pub fn format(&self, show_timestamp: bool) -> String {
        let node_label = format!("[N{}]", self.node_index);
        let severity_label = match self.severity {
            MessageSeverity::Panic => "[PANIC]",
            MessageSeverity::Error => "[ERROR]",
            MessageSeverity::Warning => "[WARN]",
            MessageSeverity::Debug => "[DEBUG]",
            MessageSeverity::Info => "",
        };

        if show_timestamp {
            format!(
                "{} {} {} {}",
                self.timestamp.format("%H:%M:%S%.3f"),
                node_label,
                severity_label,
                self.content
            )
        } else {
            format!("{} {} {}", node_label, severity_label, self.content)
        }
    }
}

/// Message severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageSeverity {
    /// Debug-level message.
    Debug,
    /// Informational message.
    Info,
    /// Warning message.
    Warning,
    /// Error message.
    Error,
    /// Panic or crash.
    Panic,
}

impl Default for MessageSeverity {
    fn default() -> Self {
        Self::Info
    }
}

/// Filter for console messages.
#[derive(Debug, Clone, Default)]
pub struct ConsoleFilter {
    /// Minimum severity to include.
    pub min_severity: Option<MessageSeverity>,

    /// Only include messages from specific nodes.
    pub node_filter: Option<Vec<usize>>,

    /// Only include messages containing this text.
    pub text_filter: Option<String>,

    /// Only include messages with specific categories.
    pub category_filter: Option<Vec<String>>,

    /// Exclude messages matching this pattern.
    pub exclude_pattern: Option<String>,
}

impl ConsoleFilter {
    /// Create a new filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by minimum severity.
    pub fn min_severity(mut self, severity: MessageSeverity) -> Self {
        self.min_severity = Some(severity);
        self
    }

    /// Filter by nodes.
    pub fn nodes(mut self, nodes: Vec<usize>) -> Self {
        self.node_filter = Some(nodes);
        self
    }

    /// Filter by text content.
    pub fn containing(mut self, text: impl Into<String>) -> Self {
        self.text_filter = Some(text.into());
        self
    }

    /// Filter by category.
    pub fn categories(mut self, categories: Vec<String>) -> Self {
        self.category_filter = Some(categories);
        self
    }

    /// Exclude messages matching pattern.
    pub fn exclude(mut self, pattern: impl Into<String>) -> Self {
        self.exclude_pattern = Some(pattern.into());
        self
    }

    /// Check if a message passes the filter.
    pub fn matches(&self, msg: &ConsoleMessage) -> bool {
        // Check severity
        if let Some(ref min) = self.min_severity {
            let msg_level = match msg.severity {
                MessageSeverity::Debug => 0,
                MessageSeverity::Info => 1,
                MessageSeverity::Warning => 2,
                MessageSeverity::Error => 3,
                MessageSeverity::Panic => 4,
            };
            let min_level = match min {
                MessageSeverity::Debug => 0,
                MessageSeverity::Info => 1,
                MessageSeverity::Warning => 2,
                MessageSeverity::Error => 3,
                MessageSeverity::Panic => 4,
            };
            if msg_level < min_level {
                return false;
            }
        }

        // Check node filter
        if let Some(ref nodes) = self.node_filter {
            if !nodes.contains(&msg.node_index) {
                return false;
            }
        }

        // Check text filter
        if let Some(ref text) = self.text_filter {
            if !msg.content.contains(text) {
                return false;
            }
        }

        // Check category filter
        if let Some(ref categories) = self.category_filter {
            if let Some(ref cat) = msg.category {
                if !categories.contains(cat) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check exclude pattern
        if let Some(ref pattern) = self.exclude_pattern {
            if msg.content.contains(pattern) {
                return false;
            }
        }

        true
    }
}

/// Console I/O aggregator for the swarm.
pub struct ConsoleIO {
    /// Message receivers from all nodes.
    receivers: HashMap<NodeId, Receiver<ConsoleMessage>>,

    /// Message senders for writing to nodes.
    senders: HashMap<NodeId, Sender<Vec<u8>>>,

    /// Aggregated message buffer.
    buffer: Arc<RwLock<Vec<ConsoleMessage>>>,

    /// Maximum buffer size.
    max_buffer_size: usize,

    /// Active filter.
    filter: Arc<RwLock<ConsoleFilter>>,

    /// Statistics.
    stats: Arc<RwLock<ConsoleStats>>,
}

/// Console statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsoleStats {
    /// Total messages received.
    pub total_messages: u64,

    /// Messages per node.
    pub messages_per_node: HashMap<usize, u64>,

    /// Errors detected.
    pub errors_detected: u64,

    /// Panics detected.
    pub panics_detected: u64,

    /// Start time.
    pub start_time: Option<DateTime<Utc>>,
}

impl ConsoleIO {
    /// Create a new console I/O aggregator.
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            receivers: HashMap::new(),
            senders: HashMap::new(),
            buffer: Arc::new(RwLock::new(Vec::with_capacity(max_buffer_size))),
            max_buffer_size,
            filter: Arc::new(RwLock::new(ConsoleFilter::default())),
            stats: Arc::new(RwLock::new(ConsoleStats::default())),
        }
    }

    /// Register a node's console socket.
    pub async fn register_node(
        &mut self,
        node_id: NodeId,
        node_index: usize,
        socket_path: PathBuf,
    ) -> SwarmResult<()> {
        let (msg_tx, msg_rx) = bounded::<ConsoleMessage>(1024);
        let (write_tx, write_rx) = bounded::<Vec<u8>>(256);

        self.receivers.insert(node_id, msg_rx);
        self.senders.insert(node_id, write_tx);

        let buffer = Arc::clone(&self.buffer);
        let max_buffer_size = self.max_buffer_size;
        let stats = Arc::clone(&self.stats);
        let filter = Arc::clone(&self.filter);

        // Spawn async task to read from socket
        tokio::spawn(async move {
            // Wait for socket
            for _ in 0..50 {
                if socket_path.exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }

            match UnixStream::connect(&socket_path).await {
                Ok(stream) => {
                    let (reader, mut writer) = stream.into_split();
                    let mut reader = BufReader::new(reader);
                    let mut line = String::new();

                    // Initialize stats
                    {
                        let mut s = stats.write();
                        if s.start_time.is_none() {
                            s.start_time = Some(Utc::now());
                        }
                    }

                    loop {
                        tokio::select! {
                            result = reader.read_line(&mut line) => {
                                match result {
                                    Ok(0) => break, // EOF
                                    Ok(_) => {
                                        let content = line.trim().to_string();
                                        if !content.is_empty() {
                                            let msg = ConsoleMessage::new(
                                                node_id,
                                                node_index,
                                                content,
                                            );

                                            // Update stats
                                            {
                                                let mut s = stats.write();
                                                s.total_messages += 1;
                                                *s.messages_per_node.entry(node_index).or_insert(0) += 1;
                                                if msg.severity == MessageSeverity::Error {
                                                    s.errors_detected += 1;
                                                }
                                                if msg.severity == MessageSeverity::Panic {
                                                    s.panics_detected += 1;
                                                }
                                            }

                                            // Apply filter and add to buffer
                                            let passes_filter = filter.read().matches(&msg);
                                            if passes_filter {
                                                let mut buf = buffer.write();
                                                buf.push(msg.clone());
                                                if buf.len() > max_buffer_size {
                                                    buf.remove(0);
                                                }
                                            }

                                            // Send to channel
                                            let _ = msg_tx.try_send(msg);
                                        }
                                        line.clear();
                                    }
                                    Err(e) => {
                                        error!(node = %node_id, "Console read error: {}", e);
                                        break;
                                    }
                                }
                            }
                            // Handle write requests
                            _ = async {
                                if let Ok(data) = write_rx.recv() {
                                    let _ = writer.write_all(&data).await;
                                }
                            } => {}
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        node = %node_id,
                        path = %socket_path.display(),
                        "Failed to connect to console: {}",
                        e
                    );
                }
            }
        });

        info!(node = %node_id, "Console registered");
        Ok(())
    }

    /// Write to a node's console.
    pub fn write_to_node(&self, node_id: NodeId, data: &[u8]) -> SwarmResult<()> {
        if let Some(sender) = self.senders.get(&node_id) {
            sender
                .try_send(data.to_vec())
                .map_err(|_| SwarmError::ConsoleIO("Write channel full".to_string()))?;
            Ok(())
        } else {
            Err(SwarmError::NodeNotFound(node_id.to_string()))
        }
    }

    /// Broadcast to all nodes.
    pub fn broadcast(&self, data: &[u8]) -> SwarmResult<()> {
        for (node_id, sender) in &self.senders {
            if sender.try_send(data.to_vec()).is_err() {
                warn!(node = %node_id, "Failed to broadcast to node");
            }
        }
        Ok(())
    }

    /// Set the active filter.
    pub fn set_filter(&self, filter: ConsoleFilter) {
        *self.filter.write() = filter;
    }

    /// Get recent messages.
    pub fn recent_messages(&self, limit: usize) -> Vec<ConsoleMessage> {
        let buffer = self.buffer.read();
        let start = buffer.len().saturating_sub(limit);
        buffer[start..].to_vec()
    }

    /// Get messages matching a filter.
    pub fn filtered_messages(&self, filter: &ConsoleFilter, limit: usize) -> Vec<ConsoleMessage> {
        let buffer = self.buffer.read();
        buffer
            .iter()
            .rev()
            .filter(|msg| filter.matches(msg))
            .take(limit)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Wait for a specific message pattern.
    pub async fn wait_for_pattern(
        &self,
        pattern: &str,
        timeout: Duration,
    ) -> SwarmResult<ConsoleMessage> {
        let start = Instant::now();
        let check_interval = Duration::from_millis(100);

        loop {
            if start.elapsed() > timeout {
                return Err(SwarmError::timeout(
                    "wait_for_pattern",
                    format!("Pattern '{}' not found", pattern),
                ));
            }

            {
                let buffer = self.buffer.read();
                for msg in buffer.iter().rev() {
                    if msg.content.contains(pattern) {
                        return Ok(msg.clone());
                    }
                }
            }

            tokio::time::sleep(check_interval).await;
        }
    }

    /// Wait for messages from all nodes.
    pub async fn wait_for_all_nodes(
        &self,
        node_count: usize,
        pattern: &str,
        timeout: Duration,
    ) -> SwarmResult<()> {
        let start = Instant::now();
        let check_interval = Duration::from_millis(100);
        let mut seen_nodes = std::collections::HashSet::new();

        loop {
            if start.elapsed() > timeout {
                return Err(SwarmError::timeout(
                    "wait_for_all_nodes",
                    format!(
                        "Only {} of {} nodes matched pattern '{}'",
                        seen_nodes.len(),
                        node_count,
                        pattern
                    ),
                ));
            }

            {
                let buffer = self.buffer.read();
                for msg in buffer.iter() {
                    if msg.content.contains(pattern) {
                        seen_nodes.insert(msg.node_index);
                    }
                }
            }

            if seen_nodes.len() >= node_count {
                return Ok(());
            }

            tokio::time::sleep(check_interval).await;
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> ConsoleStats {
        self.stats.read().clone()
    }

    /// Clear the message buffer.
    pub fn clear_buffer(&self) {
        self.buffer.write().clear();
    }

    /// Export messages to JSON.
    pub fn export_json(&self) -> SwarmResult<String> {
        let buffer = self.buffer.read();
        serde_json::to_string_pretty(&*buffer).map_err(SwarmError::Json)
    }

    /// Format messages for display.
    pub fn format_output(&self, limit: usize, show_timestamp: bool) -> String {
        let messages = self.recent_messages(limit);
        messages
            .iter()
            .map(|m| m.format(show_timestamp))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_severity_detection() {
        let msg = ConsoleMessage::new(NodeId::new(), 0, "PANIC: kernel crash".to_string());
        assert_eq!(msg.severity, MessageSeverity::Panic);

        let msg = ConsoleMessage::new(NodeId::new(), 0, "Error: failed to load".to_string());
        assert_eq!(msg.severity, MessageSeverity::Error);

        let msg = ConsoleMessage::new(NodeId::new(), 0, "Warning: low memory".to_string());
        assert_eq!(msg.severity, MessageSeverity::Warning);
    }

    #[test]
    fn test_category_detection() {
        let msg = ConsoleMessage::new(NodeId::new(), 0, "[KERNEL] initialized".to_string());
        assert_eq!(msg.category, Some("KERNEL".to_string()));

        let msg = ConsoleMessage::new(NodeId::new(), 0, "<DRIVER> loaded".to_string());
        assert_eq!(msg.category, Some("DRIVER".to_string()));
    }

    #[test]
    fn test_filter() {
        let filter = ConsoleFilter::new()
            .min_severity(MessageSeverity::Warning)
            .nodes(vec![0, 1]);

        let msg1 = ConsoleMessage::new(NodeId::new(), 0, "Error: test".to_string());
        assert!(filter.matches(&msg1));

        let msg2 = ConsoleMessage::new(NodeId::new(), 2, "Error: test".to_string());
        assert!(!filter.matches(&msg2)); // Wrong node

        let msg3 = ConsoleMessage::new(NodeId::new(), 0, "Info: test".to_string());
        assert!(!filter.matches(&msg3)); // Severity too low
    }
}
