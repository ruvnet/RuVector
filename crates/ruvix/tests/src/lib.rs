//! Integration test support for the RuVix Cognition Kernel.
//!
//! This crate provides integration tests that verify end-to-end behavior
//! across the RuVix kernel primitives: Task, Capability, Region, Queue,
//! Timer, and Proof systems as defined in ADR-087.
//!
//! # Test Categories
//!
//! 1. **Syscall Flow Tests**: End-to-end syscall execution paths
//! 2. **Capability-Gated Access Tests**: Access control verification
//! 3. **Proof-Gated Mutation Tests**: Rejection of mutations without proofs
//! 4. **Section 17 Acceptance Tests**: Full ADR-087 acceptance criteria
//!
//! # Usage
//!
//! This crate is intended for testing only and is not published.
//! Run tests with:
//!
//! ```bash
//! cargo test -p ruvix-integration
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

// Re-export test dependencies for integration tests
pub use ruvix_cap;
pub use ruvix_queue;
pub use ruvix_region;
pub use ruvix_types;

/// Test harness configuration for integration tests.
#[derive(Debug, Clone)]
pub struct TestHarnessConfig {
    /// Maximum number of tasks in the test environment.
    pub max_tasks: usize,
    /// Maximum capabilities per task.
    pub max_caps_per_task: usize,
    /// Region backing size for tests.
    pub region_backing_size: usize,
    /// Queue capacity for tests.
    pub queue_capacity: usize,
}

impl Default for TestHarnessConfig {
    fn default() -> Self {
        Self {
            max_tasks: 16,
            max_caps_per_task: 256,
            region_backing_size: 4096,
            queue_capacity: 64,
        }
    }
}

/// Test environment for integration tests.
///
/// Provides a controlled environment with pre-configured kernel primitives
/// for testing interactions between subsystems.
pub struct TestEnvironment {
    config: TestHarnessConfig,
}

impl TestEnvironment {
    /// Create a new test environment with default configuration.
    pub fn new() -> Self {
        Self::with_config(TestHarnessConfig::default())
    }

    /// Create a new test environment with custom configuration.
    pub fn with_config(config: TestHarnessConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &TestHarnessConfig {
        &self.config
    }
}

impl Default for TestEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

/// Attestation record for checkpoint/restart verification.
///
/// Used in Section 17 acceptance tests to verify bit-identical state
/// after checkpoint and restart operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationRecord {
    /// Hash of the region state.
    pub region_hash: u64,
    /// Number of operations performed.
    pub operation_count: u64,
    /// Checkpoint sequence number.
    pub checkpoint_seq: u32,
}

impl AttestationRecord {
    /// Create a new attestation record.
    pub fn new(region_hash: u64, operation_count: u64, checkpoint_seq: u32) -> Self {
        Self {
            region_hash,
            operation_count,
            checkpoint_seq,
        }
    }

    /// Verify that two attestation records match (bit-identical state).
    pub fn verify_identical(&self, other: &Self) -> bool {
        self.region_hash == other.region_hash
            && self.operation_count == other.operation_count
            && self.checkpoint_seq == other.checkpoint_seq
    }
}

/// Simple hash function for test verification (FNV-1a).
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harness_config_default() {
        let config = TestHarnessConfig::default();
        assert_eq!(config.max_tasks, 16);
        assert_eq!(config.max_caps_per_task, 256);
        assert_eq!(config.region_backing_size, 4096);
        assert_eq!(config.queue_capacity, 64);
    }

    #[test]
    fn test_environment_creation() {
        let env = TestEnvironment::new();
        assert_eq!(env.config().max_tasks, 16);
    }

    #[test]
    fn test_attestation_record() {
        let record1 = AttestationRecord::new(0x12345678, 100, 1);
        let record2 = AttestationRecord::new(0x12345678, 100, 1);
        let record3 = AttestationRecord::new(0x87654321, 100, 1);

        assert!(record1.verify_identical(&record2));
        assert!(!record1.verify_identical(&record3));
    }

    #[test]
    fn test_fnv1a_hash() {
        let data1 = b"hello world";
        let data2 = b"hello world";
        let data3 = b"goodbye world";

        assert_eq!(fnv1a_hash(data1), fnv1a_hash(data2));
        assert_ne!(fnv1a_hash(data1), fnv1a_hash(data3));
    }
}
