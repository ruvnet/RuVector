//! Error types for the proof engine.

use core::fmt;

/// Result type alias for proof operations.
pub type ProofResult<T> = Result<T, ProofError>;

/// Errors that can occur during proof operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofError {
    /// Proof token has expired.
    Expired {
        /// When the proof expired (ns since epoch).
        valid_until: u64,
        /// Current time (ns since epoch).
        current_time: u64,
    },

    /// Nonce has already been consumed.
    NonceReused {
        /// The reused nonce value.
        nonce: u64,
    },

    /// Mutation hash mismatch.
    HashMismatch {
        /// Expected hash prefix (first 8 bytes).
        expected_prefix: u64,
        /// Actual hash prefix (first 8 bytes).
        actual_prefix: u64,
    },

    /// Missing PROVE capability rights.
    MissingProveRights {
        /// Object ID that requires PROVE rights.
        object_id: u64,
    },

    /// Invalid proof tier for the operation.
    InvalidTier {
        /// Expected tier.
        expected: ProofTier,
        /// Actual tier.
        actual: ProofTier,
    },

    /// Merkle witness verification failed.
    MerkleVerificationFailed {
        /// Path element index where verification failed.
        failed_at_index: u8,
    },

    /// Coherence verification failed.
    CoherenceVerificationFailed {
        /// Coherence score that failed the threshold.
        score: u16,
        /// Required minimum score.
        threshold: u16,
    },

    /// Proof cache is full.
    CacheFull {
        /// Current cache size.
        size: usize,
        /// Maximum cache capacity.
        capacity: usize,
    },

    /// Internal error during proof generation.
    InternalError {
        /// Error code for debugging.
        code: u32,
    },

    /// Proof payload does not match tier requirements.
    PayloadMismatch {
        /// Expected payload type for the tier.
        tier: ProofTier,
    },

    /// Attestation serialization error.
    AttestationError,

    /// Witness log overflow.
    WitnessLogFull {
        /// Current log size.
        size: usize,
    },
}

// Import ProofTier for error variants
use ruvix_types::ProofTier;

impl fmt::Display for ProofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expired {
                valid_until,
                current_time,
            } => {
                write!(
                    f,
                    "proof expired: valid_until={valid_until}, current={current_time}"
                )
            }
            Self::NonceReused { nonce } => {
                write!(f, "nonce already consumed: {nonce:#x}")
            }
            Self::HashMismatch {
                expected_prefix,
                actual_prefix,
            } => {
                write!(
                    f,
                    "mutation hash mismatch: expected={expected_prefix:#x}, actual={actual_prefix:#x}"
                )
            }
            Self::MissingProveRights { object_id } => {
                write!(f, "missing PROVE rights on object {object_id:#x}")
            }
            Self::InvalidTier { expected, actual } => {
                write!(
                    f,
                    "invalid proof tier: expected={}, actual={}",
                    expected.as_str(),
                    actual.as_str()
                )
            }
            Self::MerkleVerificationFailed { failed_at_index } => {
                write!(f, "Merkle verification failed at index {failed_at_index}")
            }
            Self::CoherenceVerificationFailed { score, threshold } => {
                write!(f, "coherence score {score} below threshold {threshold}")
            }
            Self::CacheFull { size, capacity } => {
                write!(f, "proof cache full: {size}/{capacity}")
            }
            Self::InternalError { code } => {
                write!(f, "internal proof engine error: code={code:#x}")
            }
            Self::PayloadMismatch { tier } => {
                write!(f, "payload does not match tier {}", tier.as_str())
            }
            Self::AttestationError => {
                write!(f, "attestation serialization error")
            }
            Self::WitnessLogFull { size } => {
                write!(f, "witness log full: {size} entries")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ProofError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ProofError::Expired {
            valid_until: 1000,
            current_time: 2000,
        };
        let msg = format!("{err}");
        assert!(msg.contains("expired"));
        assert!(msg.contains("1000"));
        assert!(msg.contains("2000"));
    }

    #[test]
    fn test_nonce_reused_display() {
        let err = ProofError::NonceReused { nonce: 0xDEADBEEF };
        let msg = format!("{err}");
        assert!(msg.contains("nonce"));
        assert!(msg.contains("0xdeadbeef"));
    }

    #[test]
    fn test_hash_mismatch_display() {
        let err = ProofError::HashMismatch {
            expected_prefix: 0x1234,
            actual_prefix: 0x5678,
        };
        let msg = format!("{err}");
        assert!(msg.contains("mismatch"));
    }

    #[test]
    fn test_missing_prove_rights_display() {
        let err = ProofError::MissingProveRights { object_id: 0x1000 };
        let msg = format!("{err}");
        assert!(msg.contains("PROVE"));
        assert!(msg.contains("0x1000"));
    }

    #[test]
    fn test_invalid_tier_display() {
        let err = ProofError::InvalidTier {
            expected: ProofTier::Deep,
            actual: ProofTier::Reflex,
        };
        let msg = format!("{err}");
        assert!(msg.contains("Deep"));
        assert!(msg.contains("Reflex"));
    }
}
