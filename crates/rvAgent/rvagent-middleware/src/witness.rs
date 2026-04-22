//! WitnessMiddleware — logs each tool call to a witness chain (ADR-103 B3).
//!
//! Records tool_name, arguments_hash (SHA3-256), timestamp.
//! Thread-safe via `Arc<Mutex<WitnessBuilder>>`.
//!
//! ## ADR-106 Integration (Phase 4 — Unified Witness Format)
//!
//! When `rvf_witness` is enabled in [`RvfBridgeConfig`], the witness builder
//! produces RVF wire-format witness bundles compatible with
//! `rvf-types::witness::WitnessHeader` and `ToolCallEntry`. This enables
//! deterministic replay and audit across both the rvAgent framework and the
//! RuVix kernel's witness log.

use async_trait::async_trait;
use chrono::Utc;
use sha3::{Digest, Sha3_256};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rvagent_core::rvf_bridge::{
    GovernanceMode, PolicyCheck, RvfToolCallEntry, RvfWitnessHeader, TaskOutcome,
    WITNESS_HEADER_SIZE, WIT_HAS_TRACE,
};

use crate::{Middleware, ModelHandler, ModelRequest, ModelResponse};

/// A single entry in the witness chain.
#[derive(Debug, Clone)]
pub struct WitnessEntry {
    /// Name of the tool that was called.
    pub tool_name: String,
    /// SHA3-256 hash of the serialized arguments.
    pub arguments_hash: String,
    /// ISO 8601 timestamp of the call.
    pub timestamp: String,
    /// Sequential index in the witness chain.
    pub sequence: u64,
    /// Wall-clock latency in milliseconds (ADR-106).
    pub latency_ms: Option<u32>,
    /// Policy check result (ADR-106).
    pub policy_check: PolicyCheck,
}

/// Builder that accumulates witness entries in a thread-safe manner.
#[derive(Debug)]
pub struct WitnessBuilder {
    entries: Vec<WitnessEntry>,
    next_sequence: u64,
    /// Task ID for RVF witness header (UUID bytes).
    task_id: [u8; 16],
    /// Governance mode for RVF witness header.
    governance_mode: GovernanceMode,
    /// Total cost in microdollars accumulated across entries.
    total_cost_microdollars: u32,
    /// Total tokens accumulated across entries.
    total_tokens: u32,
    /// Whether to produce RVF wire-format bundles.
    rvf_enabled: bool,
    /// Start instant for latency tracking.
    start_time: Option<Instant>,
}

impl WitnessBuilder {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_sequence: 0,
            task_id: [0u8; 16],
            governance_mode: GovernanceMode::Approved,
            total_cost_microdollars: 0,
            total_tokens: 0,
            rvf_enabled: false,
            start_time: None,
        }
    }

    /// Create a builder with RVF wire-format support enabled.
    pub fn with_rvf(task_id: [u8; 16], governance_mode: GovernanceMode) -> Self {
        Self {
            entries: Vec::new(),
            next_sequence: 0,
            task_id,
            governance_mode,
            total_cost_microdollars: 0,
            total_tokens: 0,
            rvf_enabled: true,
            start_time: Some(Instant::now()),
        }
    }

    /// Add a tool call entry to the witness chain.
    pub fn add_tool_call_entry(&mut self, tool_name: &str, args: &serde_json::Value) {
        let arguments_hash = compute_arguments_hash(args);
        let entry = WitnessEntry {
            tool_name: tool_name.to_string(),
            arguments_hash,
            timestamp: Utc::now().to_rfc3339(),
            sequence: self.next_sequence,
            latency_ms: None,
            policy_check: PolicyCheck::Allowed,
        };
        self.next_sequence += 1;
        self.entries.push(entry);
    }

    /// Add a tool call entry with RVF-compatible metadata.
    pub fn add_rvf_tool_call(
        &mut self,
        tool_name: &str,
        args: &serde_json::Value,
        latency_ms: u32,
        policy_check: PolicyCheck,
        cost_microdollars: u32,
        tokens: u32,
    ) {
        let arguments_hash = compute_arguments_hash(args);
        let entry = WitnessEntry {
            tool_name: tool_name.to_string(),
            arguments_hash,
            timestamp: Utc::now().to_rfc3339(),
            sequence: self.next_sequence,
            latency_ms: Some(latency_ms),
            policy_check,
        };
        self.next_sequence += 1;
        self.total_cost_microdollars += cost_microdollars;
        self.total_tokens += tokens;
        self.entries.push(entry);
    }

    /// Get all entries in the witness chain.
    pub fn entries(&self) -> &[WitnessEntry] {
        &self.entries
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether RVF wire-format output is enabled.
    pub fn is_rvf_enabled(&self) -> bool {
        self.rvf_enabled
    }

    /// Build an RVF wire-format witness header from the accumulated entries.
    ///
    /// Returns `None` if RVF mode is not enabled.
    pub fn build_rvf_header(&self, outcome: TaskOutcome) -> Option<RvfWitnessHeader> {
        if !self.rvf_enabled {
            return None;
        }

        let total_latency_ms = self
            .start_time
            .map(|s| s.elapsed().as_millis() as u32)
            .unwrap_or(0);

        Some(RvfWitnessHeader {
            version: 1,
            flags: WIT_HAS_TRACE,
            task_id: self.task_id,
            policy_hash: {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                self.governance_mode.hash(&mut hasher);
                let h = hasher.finish();
                h.to_le_bytes()
            },
            created_ns: Utc::now().timestamp_nanos_opt().unwrap_or(0) as u64,
            outcome,
            governance_mode: self.governance_mode,
            tool_call_count: self.entries.len() as u16,
            total_cost_microdollars: self.total_cost_microdollars,
            total_latency_ms,
            total_tokens: self.total_tokens,
            retry_count: 0,
            section_count: 1, // trace section only
            total_bundle_size: WITNESS_HEADER_SIZE as u32,
        })
    }

    /// Convert entries to RVF tool call entries.
    pub fn to_rvf_entries(&self) -> Vec<RvfToolCallEntry> {
        self.entries
            .iter()
            .map(|e| {
                let args_hash = truncate_hash_to_8(&e.arguments_hash);
                RvfToolCallEntry {
                    action: e.tool_name.clone(),
                    args_hash,
                    result_hash: [0u8; 8], // Result hash not tracked in current implementation
                    latency_ms: e.latency_ms.unwrap_or(0),
                    cost_microdollars: 0,
                    tokens: 0,
                    policy_check: e.policy_check,
                }
            })
            .collect()
    }
}

impl Default for WitnessBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Truncate a hex hash string to 8 bytes — zero-allocation version.
///
/// Decodes up to 16 hex characters (8 bytes) directly into the output
/// array without intermediate `Vec` allocation.
#[inline]
fn truncate_hash_to_8(hex: &str) -> [u8; 8] {
    let mut result = [0u8; 8];
    let hex_bytes = hex.as_bytes();
    let pairs = (hex_bytes.len() / 2).min(8);
    for i in 0..pairs {
        let hi = hex_nibble(hex_bytes[i * 2]);
        let lo = hex_nibble(hex_bytes[i * 2 + 1]);
        result[i] = (hi << 4) | lo;
    }
    result
}

/// Convert a single ASCII hex character to its 4-bit value.
#[inline(always)]
const fn hex_nibble(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

/// Hex encoding lookup table — avoids per-byte `write!` formatting overhead.
const HEX_LUT: &[u8; 16] = b"0123456789abcdef";

/// Compute SHA3-256 hash of tool call arguments.
///
/// Uses a lookup-table hex encoder to avoid 32 `write!` calls per hash
/// (each of which invokes the formatting machinery).
pub fn compute_arguments_hash(args: &serde_json::Value) -> String {
    let serialized = serde_json::to_vec(args).unwrap_or_default();
    let mut hasher = Sha3_256::new();
    hasher.update(&serialized);
    let result = hasher.finalize();
    // Fast hex encode via LUT — no formatting overhead
    let mut hex = Vec::with_capacity(64);
    for &b in result.iter() {
        hex.push(HEX_LUT[(b >> 4) as usize]);
        hex.push(HEX_LUT[(b & 0x0f) as usize]);
    }
    // SAFETY: HEX_LUT only contains ASCII bytes
    unsafe { String::from_utf8_unchecked(hex) }
}

/// Middleware that records tool calls to a witness chain for auditability.
///
/// After the model handler returns, each tool call in the response is logged
/// with its name, argument hash, and timestamp.
pub struct WitnessMiddleware {
    builder: Arc<Mutex<WitnessBuilder>>,
}

impl WitnessMiddleware {
    pub fn new() -> Self {
        Self {
            builder: Arc::new(Mutex::new(WitnessBuilder::new())),
        }
    }

    pub fn with_builder(builder: Arc<Mutex<WitnessBuilder>>) -> Self {
        Self { builder }
    }

    /// Get a reference to the witness builder for inspection.
    pub fn builder(&self) -> &Arc<Mutex<WitnessBuilder>> {
        &self.builder
    }
}

impl Default for WitnessMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for WitnessMiddleware {
    fn name(&self) -> &str {
        "witness"
    }

    fn wrap_model_call(&self, request: ModelRequest, handler: &dyn ModelHandler) -> ModelResponse {
        let response = handler.call(request);

        // Log each tool call to the witness chain
        if !response.tool_calls.is_empty() {
            let mut builder = self.builder.lock().unwrap();
            for tc in &response.tool_calls {
                builder.add_tool_call_entry(&tc.name, &tc.args);
            }
        }

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, ToolCall};

    struct ToolCallHandler {
        tool_calls: Vec<ToolCall>,
    }
    impl ModelHandler for ToolCallHandler {
        fn call(&self, _request: ModelRequest) -> ModelResponse {
            let mut response = ModelResponse::text("done");
            response.tool_calls = self.tool_calls.clone();
            response
        }
    }

    #[test]
    fn test_middleware_name() {
        let mw = WitnessMiddleware::new();
        assert_eq!(mw.name(), "witness");
    }

    #[test]
    fn test_compute_arguments_hash() {
        let args = serde_json::json!({"path": "test.txt"});
        let hash = compute_arguments_hash(&args);
        assert_eq!(hash.len(), 64); // SHA3-256 = 32 bytes = 64 hex
                                    // Deterministic
        assert_eq!(hash, compute_arguments_hash(&args));
        // Different args -> different hash
        let other = serde_json::json!({"path": "other.txt"});
        assert_ne!(hash, compute_arguments_hash(&other));
    }

    #[test]
    fn test_witness_builder() {
        let mut builder = WitnessBuilder::new();
        assert!(builder.is_empty());

        builder.add_tool_call_entry("read_file", &serde_json::json!({"path": "a.txt"}));
        assert_eq!(builder.len(), 1);
        assert_eq!(builder.entries()[0].tool_name, "read_file");
        assert_eq!(builder.entries()[0].sequence, 0);

        builder.add_tool_call_entry("write_file", &serde_json::json!({}));
        assert_eq!(builder.len(), 2);
        assert_eq!(builder.entries()[1].sequence, 1);
    }

    #[test]
    fn test_wrap_model_call_records_tool_calls() {
        let mw = WitnessMiddleware::new();
        let handler = ToolCallHandler {
            tool_calls: vec![
                ToolCall {
                    id: "call-1".into(),
                    name: "read_file".into(),
                    args: serde_json::json!({"path": "test.txt"}),
                },
                ToolCall {
                    id: "call-2".into(),
                    name: "execute".into(),
                    args: serde_json::json!({"command": "ls"}),
                },
            ],
        };

        let request = ModelRequest::new(vec![Message::user("test")]);
        let _response = mw.wrap_model_call(request, &handler);

        let builder = mw.builder().lock().unwrap();
        assert_eq!(builder.len(), 2);
        assert_eq!(builder.entries()[0].tool_name, "read_file");
        assert_eq!(builder.entries()[1].tool_name, "execute");
    }

    #[test]
    fn test_wrap_model_call_no_tool_calls() {
        let mw = WitnessMiddleware::new();
        let handler = ToolCallHandler { tool_calls: vec![] };

        let request = ModelRequest::new(vec![]);
        let _response = mw.wrap_model_call(request, &handler);

        let builder = mw.builder().lock().unwrap();
        assert!(builder.is_empty());
    }

    #[test]
    fn test_thread_safety() {
        let builder = Arc::new(Mutex::new(WitnessBuilder::new()));
        let mw1 = WitnessMiddleware::with_builder(builder.clone());
        let mw2 = WitnessMiddleware::with_builder(builder.clone());

        let handler1 = ToolCallHandler {
            tool_calls: vec![ToolCall {
                id: "c1".into(),
                name: "tool_a".into(),
                args: serde_json::json!({}),
            }],
        };
        let handler2 = ToolCallHandler {
            tool_calls: vec![ToolCall {
                id: "c2".into(),
                name: "tool_b".into(),
                args: serde_json::json!({}),
            }],
        };

        let req1 = ModelRequest::new(vec![]);
        let req2 = ModelRequest::new(vec![]);
        mw1.wrap_model_call(req1, &handler1);
        mw2.wrap_model_call(req2, &handler2);

        let builder = builder.lock().unwrap();
        assert_eq!(builder.len(), 2);
    }

    #[test]
    fn test_witness_entry_has_timestamp() {
        let mut builder = WitnessBuilder::new();
        builder.add_tool_call_entry("test", &serde_json::json!({}));
        let entry = &builder.entries()[0];
        assert!(!entry.timestamp.is_empty());
        // Should be valid ISO 8601
        assert!(entry.timestamp.contains('T'));
    }

    // --- ADR-106 RVF witness tests ---

    #[test]
    fn test_rvf_builder_disabled_by_default() {
        let builder = WitnessBuilder::new();
        assert!(!builder.is_rvf_enabled());
        assert!(builder.build_rvf_header(TaskOutcome::Solved).is_none());
    }

    #[test]
    fn test_rvf_builder_enabled() {
        let task_id = [0x42u8; 16];
        let builder = WitnessBuilder::with_rvf(task_id, GovernanceMode::Approved);
        assert!(builder.is_rvf_enabled());
    }

    #[test]
    fn test_rvf_tool_call_entry() {
        let task_id = [0x42u8; 16];
        let mut builder = WitnessBuilder::with_rvf(task_id, GovernanceMode::Autonomous);
        builder.add_rvf_tool_call(
            "read_file",
            &serde_json::json!({"path": "test.txt"}),
            150,
            PolicyCheck::Allowed,
            500,
            200,
        );
        builder.add_rvf_tool_call(
            "execute",
            &serde_json::json!({"command": "ls"}),
            300,
            PolicyCheck::Confirmed,
            1000,
            400,
        );

        assert_eq!(builder.len(), 2);
        assert_eq!(builder.entries()[0].latency_ms, Some(150));
        assert_eq!(builder.entries()[1].policy_check, PolicyCheck::Confirmed);
    }

    #[test]
    fn test_rvf_header_generation() {
        let task_id = [0x42u8; 16];
        let mut builder = WitnessBuilder::with_rvf(task_id, GovernanceMode::Approved);
        builder.add_rvf_tool_call(
            "test_tool",
            &serde_json::json!({}),
            100,
            PolicyCheck::Allowed,
            500,
            200,
        );

        let header = builder.build_rvf_header(TaskOutcome::Solved).unwrap();
        assert_eq!(header.version, 1);
        assert_eq!(header.task_id, task_id);
        assert_eq!(header.outcome, TaskOutcome::Solved);
        assert_eq!(header.governance_mode, GovernanceMode::Approved);
        assert_eq!(header.tool_call_count, 1);
        assert_eq!(header.total_cost_microdollars, 500);
        assert_eq!(header.total_tokens, 200);
        assert!(header.flags & WIT_HAS_TRACE != 0);
    }

    #[test]
    fn test_rvf_header_wire_format() {
        let task_id = [0x42u8; 16];
        let mut builder = WitnessBuilder::with_rvf(task_id, GovernanceMode::Restricted);
        builder.add_rvf_tool_call(
            "tool",
            &serde_json::json!({}),
            50,
            PolicyCheck::Allowed,
            100,
            50,
        );

        let header = builder.build_rvf_header(TaskOutcome::Failed).unwrap();
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), WITNESS_HEADER_SIZE);

        // Verify magic bytes
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, rvagent_core::rvf_bridge::WITNESS_MAGIC);

        // Roundtrip
        let decoded = RvfWitnessHeader::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.task_id, task_id);
        assert_eq!(decoded.outcome, TaskOutcome::Failed);
        assert_eq!(decoded.governance_mode, GovernanceMode::Restricted);
    }

    #[test]
    fn test_to_rvf_entries() {
        let task_id = [0x42u8; 16];
        let mut builder = WitnessBuilder::with_rvf(task_id, GovernanceMode::Approved);
        builder.add_rvf_tool_call(
            "read_file",
            &serde_json::json!({"path": "test.txt"}),
            150,
            PolicyCheck::Allowed,
            500,
            200,
        );

        let rvf_entries = builder.to_rvf_entries();
        assert_eq!(rvf_entries.len(), 1);
        assert_eq!(rvf_entries[0].action, "read_file");
        assert_eq!(rvf_entries[0].latency_ms, 150);
        assert_eq!(rvf_entries[0].policy_check, PolicyCheck::Allowed);
    }

    #[test]
    fn test_truncate_hash() {
        let hash = compute_arguments_hash(&serde_json::json!({"key": "value"}));
        let truncated = truncate_hash_to_8(&hash);
        // Should produce 8 non-zero bytes from a valid hash
        assert!(truncated.iter().any(|b| *b != 0));
    }
}
