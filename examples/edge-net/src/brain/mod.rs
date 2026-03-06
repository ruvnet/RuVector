//! Brain Integration Module
//!
//! Bridges edge-net distributed compute with pi brain shared intelligence.
//! Edge nodes interact with the brain through the relay WebSocket connection.
//!
//! Since this runs in WASM, actual WebSocket calls go through JavaScript interop.
//! The BrainBridge prepares request JSON that the JavaScript host sends through
//! the relay WebSocket. Each method returns a JSON string representing the
//! request to be sent.
//!
//! ## rUv Rewards
//!
//! Brain operations earn rUv (Resource Utility Vouchers):
//! - `search`: 0.5 rUv per query
//! - `share`: 5.0 rUv per knowledge contribution
//! - `vote`: 0.2 rUv per quality vote
//! - `lora_pull`: 0.1 rUv per LoRA weight pull
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ruvector_edge_net::brain::BrainBridge;
//!
//! let bridge = BrainBridge::new("ws://localhost:8080/relay", "node-abc123");
//!
//! // Get a search request to send via JS WebSocket
//! let request_json = bridge.brain_search("authentication patterns", 10);
//! // JS host sends request_json through relay WebSocket
//! ```

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use serde_json::json;

/// Get current timestamp in milliseconds (portable across wasm/native)
fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0.0
    }
}

/// rUv reward amounts for brain operations
mod rewards {
    pub const SEARCH: f64 = 0.5;
    pub const SHARE: f64 = 5.0;
    pub const VOTE: f64 = 0.2;
    pub const LORA_PULL: f64 = 0.1;
    pub const STATUS: f64 = 0.0;
    pub const LIST: f64 = 0.1;
}

/// Brain operation types that edge nodes can perform
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BrainOperation {
    /// Search shared brain knowledge
    Search { query: String, limit: usize },
    /// Share knowledge with the brain
    Share {
        title: String,
        content: String,
        category: String,
        tags: Vec<String>,
    },
    /// Vote on brain knowledge quality
    Vote {
        memory_id: String,
        /// "up" or "down"
        direction: String,
    },
    /// Get brain system status
    Status,
    /// List brain memories by category
    List {
        category: Option<String>,
        limit: usize,
    },
    /// Get latest LoRA consensus weights
    LoraLatest,
}

impl BrainOperation {
    /// Get the rUv reward for this operation type
    pub fn ruv_reward(&self) -> f64 {
        match self {
            BrainOperation::Search { .. } => rewards::SEARCH,
            BrainOperation::Share { .. } => rewards::SHARE,
            BrainOperation::Vote { .. } => rewards::VOTE,
            BrainOperation::Status => rewards::STATUS,
            BrainOperation::List { .. } => rewards::LIST,
            BrainOperation::LoraLatest => rewards::LORA_PULL,
        }
    }

    /// Get the operation name as a string
    pub fn name(&self) -> &'static str {
        match self {
            BrainOperation::Search { .. } => "search",
            BrainOperation::Share { .. } => "share",
            BrainOperation::Vote { .. } => "vote",
            BrainOperation::Status => "status",
            BrainOperation::List { .. } => "list",
            BrainOperation::LoraLatest => "lora_latest",
        }
    }
}

/// Result from a brain operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainResult {
    /// Whether the operation succeeded
    pub success: bool,
    /// Result data (operation-specific)
    pub data: serde_json::Value,
    /// rUv earned from this operation
    pub ruv_earned: f64,
    /// Operation type name
    pub operation: String,
}

/// Record of a completed brain operation for tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
struct BrainOpRecord {
    /// Operation type
    operation: String,
    /// rUv earned
    ruv_earned: f64,
    /// Timestamp (ms since epoch)
    timestamp: f64,
    /// Whether the operation succeeded
    success: bool,
}

/// Brain integration client for edge-net nodes
///
/// Prepares brain operation requests as JSON for the JavaScript host
/// to send through the relay WebSocket. Tracks rUv earned and
/// operation history locally.
#[wasm_bindgen]
pub struct BrainBridge {
    /// Relay WebSocket URL
    relay_url: String,
    /// Pi-Key identity for attribution
    node_identity: String,
    /// Accumulated rUv from brain operations
    brain_ruv_earned: f64,
    /// Total operations performed
    operations_count: u64,
    /// Quality score from brain interactions (0.0 - 1.0)
    brain_reputation: f32,
    /// Operation history for audit
    history: Vec<BrainOpRecord>,
}

#[wasm_bindgen]
impl BrainBridge {
    /// Create a new BrainBridge for connecting to the shared brain
    ///
    /// # Arguments
    /// * `relay_url` - WebSocket URL of the relay server
    /// * `node_identity` - Pi-Key identity hex string for attribution
    #[wasm_bindgen(constructor)]
    pub fn new(relay_url: &str, node_identity: &str) -> BrainBridge {
        BrainBridge {
            relay_url: relay_url.to_string(),
            node_identity: node_identity.to_string(),
            brain_ruv_earned: 0.0,
            operations_count: 0,
            brain_reputation: 0.5, // Start neutral
            history: Vec::new(),
        }
    }

    /// Search the shared brain knowledge
    ///
    /// Returns a JSON request string for the JS host to send via WebSocket.
    /// Earns 0.5 rUv per search.
    #[wasm_bindgen(js_name = brainSearch)]
    pub fn brain_search(&mut self, query: &str, limit: usize) -> String {
        let op = BrainOperation::Search {
            query: query.to_string(),
            limit: limit.min(100), // Cap limit
        };
        self.prepare_request(op)
    }

    /// Share knowledge with the brain
    ///
    /// Returns a JSON request string for the JS host to send via WebSocket.
    /// Earns 5.0 rUv per share (highest reward for contributing knowledge).
    ///
    /// # Arguments
    /// * `title` - Title of the knowledge being shared
    /// * `content` - The knowledge content
    /// * `category` - Category (e.g., "pattern", "optimization", "security")
    /// * `tags_json` - JSON array of tags (e.g., `["rust", "wasm"]`)
    #[wasm_bindgen(js_name = brainShare)]
    pub fn brain_share(
        &mut self,
        title: &str,
        content: &str,
        category: &str,
        tags_json: &str,
    ) -> String {
        let tags: Vec<String> = serde_json::from_str(tags_json).unwrap_or_default();

        let op = BrainOperation::Share {
            title: title.to_string(),
            content: content.to_string(),
            category: category.to_string(),
            tags,
        };
        self.prepare_request(op)
    }

    /// Vote on brain knowledge quality
    ///
    /// Returns a JSON request string for the JS host to send via WebSocket.
    /// Earns 0.2 rUv per vote.
    ///
    /// # Arguments
    /// * `memory_id` - ID of the memory to vote on
    /// * `direction` - "up" or "down"
    #[wasm_bindgen(js_name = brainVote)]
    pub fn brain_vote(&mut self, memory_id: &str, direction: &str) -> String {
        // Validate direction
        let direction = match direction {
            "up" | "down" => direction.to_string(),
            _ => "up".to_string(), // Default to upvote
        };

        let op = BrainOperation::Vote {
            memory_id: memory_id.to_string(),
            direction,
        };
        self.prepare_request(op)
    }

    /// Get brain system status
    ///
    /// Returns a JSON request string for the JS host to send via WebSocket.
    /// No rUv reward for status checks.
    #[wasm_bindgen(js_name = brainStatus)]
    pub fn brain_status(&mut self) -> String {
        self.prepare_request(BrainOperation::Status)
    }

    /// List brain memories by category
    ///
    /// Returns a JSON request string for the JS host to send via WebSocket.
    /// Earns 0.1 rUv per list query.
    ///
    /// # Arguments
    /// * `category` - Category to filter by (empty string for all)
    /// * `limit` - Maximum number of results
    #[wasm_bindgen(js_name = brainList)]
    pub fn brain_list(&mut self, category: &str, limit: usize) -> String {
        let category = if category.is_empty() {
            None
        } else {
            Some(category.to_string())
        };

        let op = BrainOperation::List {
            category,
            limit: limit.min(100),
        };
        self.prepare_request(op)
    }

    /// Get latest LoRA consensus weights
    ///
    /// Returns a JSON request string for the JS host to send via WebSocket.
    /// Earns 0.1 rUv per pull.
    #[wasm_bindgen(js_name = brainLoraLatest)]
    pub fn brain_lora_latest(&mut self) -> String {
        self.prepare_request(BrainOperation::LoraLatest)
    }

    /// Get total rUv earned from brain operations
    #[wasm_bindgen(js_name = getBrainRuv)]
    pub fn get_brain_ruv(&self) -> f64 {
        self.brain_ruv_earned
    }

    /// Get total brain operation count
    #[wasm_bindgen(js_name = getBrainOpsCount)]
    pub fn get_brain_ops_count(&self) -> u64 {
        self.operations_count
    }

    /// Get brain reputation score (0.0 - 1.0)
    #[wasm_bindgen(js_name = getBrainReputation)]
    pub fn get_brain_reputation(&self) -> f32 {
        self.brain_reputation
    }

    /// Get relay URL
    #[wasm_bindgen(js_name = getRelayUrl)]
    pub fn get_relay_url(&self) -> String {
        self.relay_url.clone()
    }

    /// Get node identity
    #[wasm_bindgen(js_name = getNodeIdentity)]
    pub fn get_node_identity(&self) -> String {
        self.node_identity.clone()
    }

    /// Record an operation result (called by JS after WebSocket response)
    ///
    /// Updates rUv earned, operation count, and reputation based on
    /// the result of a brain operation.
    ///
    /// # Arguments
    /// * `result_json` - JSON string with the BrainResult from the relay
    #[wasm_bindgen(js_name = recordResult)]
    pub fn record_result(&mut self, result_json: &str) -> bool {
        let result: BrainResult = match serde_json::from_str(result_json) {
            Ok(r) => r,
            Err(_) => return false,
        };

        let timestamp = now_ms();

        if result.success {
            self.brain_ruv_earned += result.ruv_earned;
            self.operations_count += 1;

            // Improve reputation on success (weighted moving average)
            self.brain_reputation = self.brain_reputation * 0.95 + 0.05;
        } else {
            // Slight reputation decrease on failure
            self.brain_reputation = self.brain_reputation * 0.98;
        }

        // Clamp reputation
        self.brain_reputation = self.brain_reputation.clamp(0.0, 1.0);

        self.history.push(BrainOpRecord {
            operation: result.operation,
            ruv_earned: result.ruv_earned,
            timestamp,
            success: result.success,
        });

        // Keep history bounded (last 1000 operations)
        if self.history.len() > 1000 {
            self.history.drain(0..self.history.len() - 1000);
        }

        true
    }

    /// Get operation history as JSON
    #[wasm_bindgen(js_name = getHistory)]
    pub fn get_history(&self, limit: usize) -> String {
        let limit = limit.min(self.history.len());
        let recent: Vec<&BrainOpRecord> = self.history.iter().rev().take(limit).collect();
        serde_json::to_string(&recent).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get brain bridge statistics as JSON
    #[wasm_bindgen(js_name = getStats)]
    pub fn get_stats(&self) -> String {
        let stats = json!({
            "relay_url": self.relay_url,
            "node_identity": self.node_identity,
            "ruv_earned": self.brain_ruv_earned,
            "operations_count": self.operations_count,
            "brain_reputation": self.brain_reputation,
            "history_size": self.history.len(),
        });
        stats.to_string()
    }
}

impl BrainBridge {
    /// Prepare a brain operation request as JSON for the relay WebSocket
    ///
    /// The returned JSON has the format:
    /// ```json
    /// {
    ///   "type": "brain_request",
    ///   "relay_url": "ws://...",
    ///   "node_identity": "...",
    ///   "operation": { ... },
    ///   "ruv_reward": 0.5,
    ///   "request_id": "..."
    /// }
    /// ```
    fn prepare_request(&mut self, op: BrainOperation) -> String {
        let ruv_reward = op.ruv_reward();
        let op_name = op.name().to_string();

        let request_id = format!(
            "brain-{}-{}",
            self.operations_count,
            now_ms() as u64
        );

        let request = json!({
            "type": "brain_request",
            "relay_url": self.relay_url,
            "node_identity": self.node_identity,
            "operation": op,
            "operation_name": op_name,
            "ruv_reward": ruv_reward,
            "request_id": request_id,
        });

        request.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_bridge_creation() {
        let bridge = BrainBridge::new("ws://localhost:8080/relay", "node-abc123");
        assert_eq!(bridge.get_relay_url(), "ws://localhost:8080/relay");
        assert_eq!(bridge.get_node_identity(), "node-abc123");
        assert_eq!(bridge.get_brain_ruv(), 0.0);
        assert_eq!(bridge.get_brain_ops_count(), 0);
        assert!((bridge.get_brain_reputation() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_brain_operation_rewards() {
        assert!((BrainOperation::Search {
            query: "test".into(),
            limit: 10,
        }.ruv_reward() - 0.5).abs() < f64::EPSILON);

        assert!((BrainOperation::Share {
            title: "t".into(),
            content: "c".into(),
            category: "cat".into(),
            tags: vec![],
        }.ruv_reward() - 5.0).abs() < f64::EPSILON);

        assert!((BrainOperation::Vote {
            memory_id: "m".into(),
            direction: "up".into(),
        }.ruv_reward() - 0.2).abs() < f64::EPSILON);

        assert!((BrainOperation::LoraLatest.ruv_reward() - 0.1).abs() < f64::EPSILON);
        assert!((BrainOperation::Status.ruv_reward()).abs() < f64::EPSILON);
        assert!((BrainOperation::List {
            category: None,
            limit: 10,
        }.ruv_reward() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_brain_operation_names() {
        assert_eq!(BrainOperation::Search { query: "q".into(), limit: 5 }.name(), "search");
        assert_eq!(BrainOperation::Share {
            title: "t".into(), content: "c".into(),
            category: "cat".into(), tags: vec![],
        }.name(), "share");
        assert_eq!(BrainOperation::Vote {
            memory_id: "m".into(), direction: "up".into(),
        }.name(), "vote");
        assert_eq!(BrainOperation::Status.name(), "status");
        assert_eq!(BrainOperation::List { category: None, limit: 5 }.name(), "list");
        assert_eq!(BrainOperation::LoraLatest.name(), "lora_latest");
    }

    #[test]
    fn test_brain_search_request() {
        let mut bridge = BrainBridge::new("ws://localhost:8080/relay", "node-abc");
        let request = bridge.brain_search("auth patterns", 10);

        let parsed: serde_json::Value = serde_json::from_str(&request).unwrap();
        assert_eq!(parsed["type"], "brain_request");
        assert_eq!(parsed["relay_url"], "ws://localhost:8080/relay");
        assert_eq!(parsed["node_identity"], "node-abc");
        assert_eq!(parsed["operation_name"], "search");
        assert_eq!(parsed["ruv_reward"], 0.5);
    }

    #[test]
    fn test_brain_share_request() {
        let mut bridge = BrainBridge::new("ws://relay:8080", "node-xyz");
        let request = bridge.brain_share(
            "Auth Pattern",
            "JWT with refresh tokens",
            "security",
            r#"["jwt", "auth"]"#,
        );

        let parsed: serde_json::Value = serde_json::from_str(&request).unwrap();
        assert_eq!(parsed["type"], "brain_request");
        assert_eq!(parsed["operation_name"], "share");
        assert_eq!(parsed["ruv_reward"], 5.0);

        let op = &parsed["operation"];
        assert_eq!(op["Share"]["title"], "Auth Pattern");
        assert_eq!(op["Share"]["content"], "JWT with refresh tokens");
        assert_eq!(op["Share"]["category"], "security");
    }

    #[test]
    fn test_brain_vote_validates_direction() {
        let mut bridge = BrainBridge::new("ws://relay:8080", "node-1");

        // Valid directions
        let request = bridge.brain_vote("mem-1", "up");
        let parsed: serde_json::Value = serde_json::from_str(&request).unwrap();
        assert_eq!(parsed["operation"]["Vote"]["direction"], "up");

        let request = bridge.brain_vote("mem-2", "down");
        let parsed: serde_json::Value = serde_json::from_str(&request).unwrap();
        assert_eq!(parsed["operation"]["Vote"]["direction"], "down");

        // Invalid direction defaults to "up"
        let request = bridge.brain_vote("mem-3", "sideways");
        let parsed: serde_json::Value = serde_json::from_str(&request).unwrap();
        assert_eq!(parsed["operation"]["Vote"]["direction"], "up");
    }

    #[test]
    fn test_brain_list_empty_category() {
        let mut bridge = BrainBridge::new("ws://relay:8080", "node-1");

        let request = bridge.brain_list("", 20);
        let parsed: serde_json::Value = serde_json::from_str(&request).unwrap();
        assert_eq!(parsed["operation"]["List"]["category"], serde_json::Value::Null);

        let request = bridge.brain_list("security", 20);
        let parsed: serde_json::Value = serde_json::from_str(&request).unwrap();
        assert_eq!(parsed["operation"]["List"]["category"], "security");
    }

    #[test]
    fn test_brain_search_caps_limit() {
        let mut bridge = BrainBridge::new("ws://relay:8080", "node-1");
        let request = bridge.brain_search("test", 500);
        let parsed: serde_json::Value = serde_json::from_str(&request).unwrap();
        assert_eq!(parsed["operation"]["Search"]["limit"], 100);
    }

    #[test]
    fn test_brain_stats() {
        let bridge = BrainBridge::new("ws://relay:8080", "node-1");
        let stats = bridge.get_stats();
        let parsed: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(parsed["ruv_earned"], 0.0);
        assert_eq!(parsed["operations_count"], 0);
    }
}
