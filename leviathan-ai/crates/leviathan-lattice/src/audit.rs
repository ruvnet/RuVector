//! Audit logging for regulatory compliance
//!
//! Every operation is logged with cryptographic verification for:
//! - FFIEC compliance
//! - BCBS 239 data lineage
//! - SR 11-7 model risk management
//! - Full reproducibility

use super::TrainResult;
use serde::{Deserialize, Serialize};
use blake3::Hash;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String};

/// Cryptographically verifiable audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Monotonic sequence number
    pub seq: u64,
    /// Unix timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Event type and data
    pub event: AuditEvent,
    /// BLAKE3 hash of previous entry (chain integrity)
    pub prev_hash: [u8; 32],
    /// BLAKE3 hash of this entry
    pub hash: [u8; 32],
}

/// Auditable events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEvent {
    /// System initialization
    Init { config: String },
    /// Training completed
    Train(TrainResult),
    /// Prediction made
    Predict {
        prefix_len: usize,
        result: Option<u32>,
    },
    /// Generation completed
    Generate {
        prompt_hash: [u8; 32],
        tokens_generated: usize,
        output_hash: [u8; 32],
    },
    /// Configuration change
    ConfigChange {
        key: String,
        old_value: String,
        new_value: String,
    },
    /// Checkpoint created
    Checkpoint { id: u64 },
    /// Custom event for extensibility
    Custom { event_type: String, data: String },
}

/// Immutable, append-only audit log with hash chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    entries: Vec<AuditEntry>,
    current_hash: [u8; 32],
}

impl AuditLog {
    pub fn new() -> Self {
        let genesis_hash = blake3::hash(b"LEVIATHAN_AI_GENESIS_v1").as_bytes().clone();
        Self {
            entries: Vec::new(),
            current_hash: genesis_hash,
        }
    }

    /// Record a new audit event
    pub fn record(&mut self, event: AuditEvent) {
        let seq = self.entries.len() as u64;
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let prev_hash = self.current_hash;

        // Compute hash of this entry
        let entry_data = bincode::serialize(&(&seq, &timestamp_ns, &event, &prev_hash))
            .unwrap_or_default();
        let hash = blake3::hash(&entry_data).as_bytes().clone();

        let entry = AuditEntry {
            seq,
            timestamp_ns,
            event,
            prev_hash,
            hash,
        };

        self.current_hash = hash;
        self.entries.push(entry);
    }

    /// Verify integrity of the entire audit chain
    pub fn verify_integrity(&self) -> Result<(), AuditVerifyError> {
        let mut expected_prev = blake3::hash(b"LEVIATHAN_AI_GENESIS_v1").as_bytes().clone();

        for (idx, entry) in self.entries.iter().enumerate() {
            // Verify sequence number
            if entry.seq != idx as u64 {
                return Err(AuditVerifyError::SequenceMismatch {
                    expected: idx as u64,
                    found: entry.seq,
                });
            }

            // Verify prev_hash chain
            if entry.prev_hash != expected_prev {
                return Err(AuditVerifyError::ChainBroken { at_index: idx });
            }

            // Recompute and verify hash
            let entry_data = bincode::serialize(&(
                &entry.seq,
                &entry.timestamp_ns,
                &entry.event,
                &entry.prev_hash,
            ))
            .unwrap_or_default();
            let computed_hash = blake3::hash(&entry_data).as_bytes().clone();

            if entry.hash != computed_hash {
                return Err(AuditVerifyError::HashMismatch { at_index: idx });
            }

            expected_prev = entry.hash;
        }

        Ok(())
    }

    /// Get all entries
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// Get entry count
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Export to JSON for external audit
    pub fn export_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Get current chain head hash
    pub fn head_hash(&self) -> [u8; 32] {
        self.current_hash
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum AuditVerifyError {
    SequenceMismatch { expected: u64, found: u64 },
    ChainBroken { at_index: usize },
    HashMismatch { at_index: usize },
}

impl std::fmt::Display for AuditVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SequenceMismatch { expected, found } => {
                write!(f, "Sequence mismatch: expected {expected}, found {found}")
            }
            Self::ChainBroken { at_index } => {
                write!(f, "Hash chain broken at index {at_index}")
            }
            Self::HashMismatch { at_index } => {
                write!(f, "Hash mismatch at index {at_index}")
            }
        }
    }
}

impl std::error::Error for AuditVerifyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_integrity() {
        let mut log = AuditLog::new();

        log.record(AuditEvent::Init {
            config: "default".to_string(),
        });
        log.record(AuditEvent::Train(TrainResult {
            vocab_size: 100,
            sequence_count: 10,
            total_tokens: 500,
            duration_ns: 1000,
        }));

        assert!(log.verify_integrity().is_ok());
    }

    #[test]
    fn test_audit_tamper_detection() {
        let mut log = AuditLog::new();

        log.record(AuditEvent::Init {
            config: "default".to_string(),
        });

        // Tamper with the log
        if let Some(entry) = log.entries.get_mut(0) {
            entry.timestamp_ns = 999;
        }

        assert!(log.verify_integrity().is_err());
    }
}
