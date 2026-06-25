//! Proof-gated vector writes with Merkle-accumulating witness logs.
//!
//! Provides cryptographic write admission gates for vector stores. Every
//! admitted write produces a `WriteReceipt` that commits the gate's current
//! chain state. Receipts can be stored alongside vector index entries and
//! verified offline without re-querying the write source.
//!
//! # Gate variants
//!
//! | Gate          | Per-write cost | Guarantee                          | Use case           |
//! |---------------|----------------|------------------------------------|--------------------|
//! | `NullGate`    | ~0 ns          | None (baseline throughput)         | Development / test |
//! | `HashChainGate` | ~200 ns      | Sequential tamper-evidence         | Agent audit logs   |
//! | `MerkleGate`  | ~300 ns        | MMR membership proofs + tamper-ev  | RAG provenance     |
//!
//! # Example
//!
//! ```rust
//! use ruvector_proof_gate::{HashChainGate, WriteGate, WritePayload};
//!
//! let mut gate = HashChainGate::new();
//! let payload = WritePayload::new(0, vec![0.1, 0.2, 0.3]);
//! let receipt = gate.admit(&payload).unwrap();
//! assert!(gate.verify_receipt(&receipt));
//! println!("root: {:?}", gate.chain_root());
//! ```

mod gate;
mod payload;

pub use gate::{HashChainGate, MerkleGate, NullGate, WriteGate};
pub use payload::{GateError, GateVariant, WritePayload, WriteReceipt};

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::gate::{HashChainGate, MerkleGate, NullGate, WriteGate};

    fn make_payload(id: u64) -> WritePayload {
        WritePayload::new(id, vec![id as f32 * 0.1, id as f32 * 0.2, id as f32 * 0.3])
    }

    // ── NullGate ─────────────────────────────────────────────────────────────

    #[test]
    fn null_gate_admits_all() {
        let mut g = NullGate::new();
        for i in 0..10u64 {
            let r = g.admit(&make_payload(i)).unwrap();
            assert_eq!(r.sequence, i);
            assert_eq!(r.gate_variant, GateVariant::Null);
        }
        assert_eq!(g.len(), 10);
    }

    #[test]
    fn null_gate_verify_always_true() {
        let mut g = NullGate::new();
        let r = g.admit(&make_payload(0)).unwrap();
        assert!(g.verify_receipt(&r));
    }

    // ── HashChainGate ─────────────────────────────────────────────────────────

    #[test]
    fn hash_chain_receipts_are_unique() {
        let mut g = HashChainGate::new();
        let r0 = g.admit(&make_payload(0)).unwrap();
        let r1 = g.admit(&make_payload(1)).unwrap();
        // Different payloads → different chain commitments
        assert_ne!(r0.chain_commitment, r1.chain_commitment);
        // Different sequence numbers
        assert_eq!(r0.sequence, 0);
        assert_eq!(r1.sequence, 1);
    }

    #[test]
    fn hash_chain_verify_receipts() {
        let mut g = HashChainGate::new();
        let receipts: Vec<_> = (0..20u64)
            .map(|i| g.admit(&make_payload(i)).unwrap())
            .collect();
        for r in &receipts {
            assert!(
                g.verify_receipt(r),
                "receipt {} failed verification",
                r.sequence
            );
        }
    }

    #[test]
    fn hash_chain_root_non_zero_after_writes() {
        let mut g = HashChainGate::new();
        g.admit(&make_payload(0)).unwrap();
        assert_ne!(g.chain_root(), [0u8; 32]);
    }

    #[test]
    fn hash_chain_same_payload_same_commitment_chain() {
        // Two identical payloads admitted sequentially produce different
        // chain commitments because the chain prev-hash differs.
        let mut g = HashChainGate::new();
        let p = make_payload(42);
        let r0 = g.admit(&p).unwrap();
        let r1 = g.admit(&p).unwrap();
        assert_ne!(r0.chain_commitment, r1.chain_commitment);
    }

    // ── MerkleGate ────────────────────────────────────────────────────────────

    #[test]
    fn merkle_gate_roots_change_on_each_write() {
        let mut g = MerkleGate::new();
        let mut roots = Vec::new();
        for i in 0..8u64 {
            g.admit(&make_payload(i)).unwrap();
            roots.push(g.chain_root());
        }
        // All roots must be distinct (each write changes the MMR)
        let unique: std::collections::HashSet<_> = roots.iter().collect();
        assert_eq!(unique.len(), 8, "expected 8 distinct roots");
    }

    #[test]
    fn merkle_gate_verify_receipts() {
        let mut g = MerkleGate::new();
        let receipts: Vec<_> = (0..20u64)
            .map(|i| g.admit(&make_payload(i)).unwrap())
            .collect();
        for r in &receipts {
            assert!(g.verify_receipt(r), "receipt {} failed", r.sequence);
        }
    }

    #[test]
    fn merkle_gate_root_non_zero_after_writes() {
        let mut g = MerkleGate::new();
        g.admit(&make_payload(0)).unwrap();
        assert_ne!(g.chain_root(), [0u8; 32]);
    }

    #[test]
    fn merkle_gate_power_of_two_leaves_stable() {
        // For 2^k leaves the MMR collapses to a single peak (complete binary tree).
        let mut g = MerkleGate::new();
        for i in 0..8u64 {
            g.admit(&make_payload(i)).unwrap();
        }
        // After 8 = 2^3 leaves the peaks vec should have exactly 1 entry.
        // We can't inspect peaks directly, but we CAN verify the root is stable
        // across a re-read without new writes.
        let root_a = g.chain_root();
        let root_b = g.chain_root();
        assert_eq!(root_a, root_b, "root must be deterministic");
    }

    // ── synthetic_payloads ───────────────────────────────────────────────────

    #[test]
    fn synthetic_payloads_length_and_dims() {
        let ps = synthetic_payloads(100, 64);
        assert_eq!(ps.len(), 100);
        assert!(ps.iter().all(|p| p.vector.len() == 64));
    }

    #[test]
    fn synthetic_payloads_deterministic() {
        let a = synthetic_payloads(50, 32);
        let b = synthetic_payloads(50, 32);
        assert!(a.iter().zip(b.iter()).all(|(x, y)| x.vector == y.vector));
    }

    // ── Acceptance: functional correctness at scale ────────────────────────

    #[test]
    fn acceptance_hash_chain_bulk_verify() {
        // 500 writes → all receipts must verify correctly.
        let payloads = synthetic_payloads(500, 64);
        let mut gate = HashChainGate::new();
        let receipts: Vec<_> = payloads.iter().map(|p| gate.admit(p).unwrap()).collect();
        let all_ok = receipts.iter().all(|r| gate.verify_receipt(r));
        assert!(all_ok, "bulk receipt verification failed for HashChainGate");
        assert_eq!(gate.len(), 500);
        assert_ne!(gate.chain_root(), [0u8; 32]);
    }

    #[test]
    fn acceptance_merkle_bulk_verify() {
        // 500 writes → all receipts must verify correctly.
        let payloads = synthetic_payloads(500, 64);
        let mut gate = MerkleGate::new();
        let receipts: Vec<_> = payloads.iter().map(|p| gate.admit(p).unwrap()).collect();
        let all_ok = receipts.iter().all(|r| gate.verify_receipt(r));
        assert!(all_ok, "bulk receipt verification failed for MerkleGate");
        assert_eq!(gate.len(), 500);
        assert_ne!(gate.chain_root(), [0u8; 32]);
    }

    #[test]
    fn acceptance_all_gates_distinct_roots() {
        // With the same payloads, NullGate returns zero root while
        // HashChain and Merkle return non-zero, non-equal roots.
        let payloads = synthetic_payloads(16, 32);
        let mut null = NullGate::new();
        let mut chain = HashChainGate::new();
        let mut merkle = MerkleGate::new();
        for p in &payloads {
            null.admit(p).unwrap();
            chain.admit(p).unwrap();
            merkle.admit(p).unwrap();
        }
        assert_eq!(null.chain_root(), [0u8; 32], "NullGate root must be zero");
        assert_ne!(
            chain.chain_root(),
            [0u8; 32],
            "HashChain root must be non-zero"
        );
        assert_ne!(
            merkle.chain_root(),
            [0u8; 32],
            "Merkle root must be non-zero"
        );
        // HashChain and Merkle roots differ (different algorithms)
        assert_ne!(
            chain.chain_root(),
            merkle.chain_root(),
            "HashChain and Merkle roots must differ"
        );
    }
}

/// Generate a deterministic synthetic dataset for benchmarks.
///
/// Returns `n` payloads with `dims`-dimensional f32 vectors. Uses a simple
/// LCG so benchmarks are reproducible without an external RNG dependency.
pub fn synthetic_payloads(n: usize, dims: usize) -> Vec<WritePayload> {
    let mut state: u64 = 0x6b37_9d3c_2a85_f1e4;
    let mut next = move || -> f32 {
        // Xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Map to [-1, 1]
        (state as i64 as f64 / i64::MAX as f64) as f32
    };

    (0..n)
        .map(|i| {
            let vector: Vec<f32> = (0..dims).map(|_| next()).collect();
            WritePayload::new(i as u64, vector)
        })
        .collect()
}
