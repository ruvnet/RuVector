//! Integration tests for proof-gated mutations.
//!
//! These tests verify the kernel invariant from ADR-087:
//! "The kernel physically prevents state mutation without a valid proof token."
//!
//! Test scenarios:
//! 1. Valid proof -> mutation succeeds
//! 2. Wrong hash -> ProofRejected
//! 3. Expired proof -> ProofRejected
//! 4. Reused nonce -> ProofRejected
//! 5. Insufficient rights -> InsufficientRights
//! 6. Wrong tier -> ProofRejected (for Deep policy)

use ruvix_region::backing::StaticBacking;
use ruvix_types::{
    CapRights, Capability, GraphMutation, KernelError, ObjectType, ProofPayload, ProofTier,
    ProofToken, RegionHandle, VectorKey,
};
use ruvix_vecgraph::{
    GraphStoreBuilder, KernelGraphStore, KernelVectorStore, ProofPolicy, VectorStoreBuilder,
};

// ============================================================================
// Vector Store Proof-Gated Tests
// ============================================================================

fn create_vector_store() -> KernelVectorStore<StaticBacking<16384>> {
    let data_backing = StaticBacking::<16384>::new();
    let hnsw_backing = StaticBacking::<16384>::new();
    let witness_backing = StaticBacking::<16384>::new();

    VectorStoreBuilder::new(4, 10)  // Small capacity for tests
        .with_proof_policy(ProofPolicy::standard())
        .build(
            data_backing,
            hnsw_backing,
            witness_backing,
            RegionHandle::new(1, 0),
            RegionHandle::new(2, 0),
            RegionHandle::new(3, 0),
            1,
        )
        .unwrap()
}

fn create_full_capability() -> Capability {
    Capability::new(
        1,
        ObjectType::VectorStore,
        CapRights::READ | CapRights::WRITE | CapRights::PROVE,
        0,
        1,
    )
}

fn compute_vector_mutation_hash(key: VectorKey, data: &[f32]) -> [u8; 32] {
    let mut hash = [0u8; 32];
    let key_bytes = key.raw().to_le_bytes();
    hash[0..8].copy_from_slice(&key_bytes);
    for (i, &value) in data.iter().enumerate() {
        let bytes = value.to_le_bytes();
        let offset = (8 + (i * 4)) % 24;
        for j in 0..4 {
            hash[offset + j] ^= bytes[j];
        }
    }
    hash
}

#[test]
fn test_vector_valid_proof_succeeds() {
    let mut store = create_vector_store();
    let cap = create_full_capability();

    let key = VectorKey::new(1);
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let hash = compute_vector_mutation_hash(key, &data);

    let proof = ProofToken::new(
        hash,
        ProofTier::Standard,
        ProofPayload::Hash { hash },
        2_000_000_000, // Valid until 2 seconds
        1,
    );

    let result = store.vector_put_proved(key, &data, &proof, &cap, 1_000_000_000);

    assert!(result.is_ok());
    assert_eq!(store.len(), 1);

    // Verify we can read it back
    let (retrieved, _meta) = store.vector_get(key, &cap).unwrap();
    assert_eq!(retrieved, data);
}

#[test]
fn test_vector_wrong_hash_rejected() {
    let mut store = create_vector_store();
    let cap = create_full_capability();

    let key = VectorKey::new(1);
    let data = vec![1.0, 2.0, 3.0, 4.0];

    // Create proof with wrong hash
    let wrong_hash = [0xAB; 32];
    let proof = ProofToken::new(
        wrong_hash,
        ProofTier::Standard,
        ProofPayload::Hash { hash: wrong_hash },
        2_000_000_000,
        1,
    );

    let result = store.vector_put_proved(key, &data, &proof, &cap, 1_000_000_000);

    assert_eq!(result, Err(KernelError::ProofRejected));
    assert_eq!(store.len(), 0); // No mutation occurred
}

#[test]
fn test_vector_expired_proof_rejected() {
    let mut store = create_vector_store();
    let cap = create_full_capability();

    let key = VectorKey::new(1);
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let hash = compute_vector_mutation_hash(key, &data);

    // Create proof that expires at 500ms
    let proof = ProofToken::new(
        hash,
        ProofTier::Standard,
        ProofPayload::Hash { hash },
        500_000_000, // Expires at 500ms
        1,
    );

    // Try to use at 1 second (after expiry)
    let result = store.vector_put_proved(key, &data, &proof, &cap, 1_000_000_000);

    assert_eq!(result, Err(KernelError::ProofRejected));
    assert_eq!(store.len(), 0);
}

#[test]
fn test_vector_nonce_reuse_rejected() {
    let mut store = create_vector_store();
    let cap = create_full_capability();

    let key1 = VectorKey::new(1);
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let hash1 = compute_vector_mutation_hash(key1, &data1);

    // First mutation with nonce 42
    let proof1 = ProofToken::new(
        hash1,
        ProofTier::Standard,
        ProofPayload::Hash { hash: hash1 },
        2_000_000_000,
        42, // Nonce
    );

    store
        .vector_put_proved(key1, &data1, &proof1, &cap, 1_000_000_000)
        .unwrap();

    // Second mutation with same nonce
    let key2 = VectorKey::new(2);
    let data2 = vec![5.0, 6.0, 7.0, 8.0];
    let hash2 = compute_vector_mutation_hash(key2, &data2);

    let proof2 = ProofToken::new(
        hash2,
        ProofTier::Standard,
        ProofPayload::Hash { hash: hash2 },
        2_000_000_000,
        42, // Same nonce - should be rejected
    );

    let result = store.vector_put_proved(key2, &data2, &proof2, &cap, 1_000_000_001);

    assert_eq!(result, Err(KernelError::ProofRejected));
    assert_eq!(store.len(), 1); // Only first mutation succeeded
}

#[test]
fn test_vector_insufficient_rights_rejected() {
    let mut store = create_vector_store();

    // Capability without PROVE right
    let cap = Capability::new(
        1,
        ObjectType::VectorStore,
        CapRights::READ | CapRights::WRITE, // Missing PROVE
        0,
        1,
    );

    let key = VectorKey::new(1);
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let hash = compute_vector_mutation_hash(key, &data);

    let proof = ProofToken::new(
        hash,
        ProofTier::Standard,
        ProofPayload::Hash { hash },
        2_000_000_000,
        1,
    );

    let result = store.vector_put_proved(key, &data, &proof, &cap, 1_000_000_000);

    assert_eq!(result, Err(KernelError::InsufficientRights));
    assert_eq!(store.len(), 0);
}

#[test]
fn test_vector_read_without_prove_right_succeeds() {
    let mut store = create_vector_store();
    let full_cap = create_full_capability();

    // Insert with full capability
    let key = VectorKey::new(1);
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let hash = compute_vector_mutation_hash(key, &data);

    let proof = ProofToken::new(
        hash,
        ProofTier::Standard,
        ProofPayload::Hash { hash },
        2_000_000_000,
        1,
    );

    store
        .vector_put_proved(key, &data, &proof, &full_cap, 1_000_000_000)
        .unwrap();

    // Read with capability that only has READ
    let read_cap = Capability::new(1, ObjectType::VectorStore, CapRights::READ, 0, 1);

    let result = store.vector_get(key, &read_cap);
    assert!(result.is_ok());

    let (retrieved, _) = result.unwrap();
    assert_eq!(retrieved, data);
}

// ============================================================================
// Graph Store Proof-Gated Tests
// ============================================================================

fn create_graph_store() -> KernelGraphStore<StaticBacking<16384>> {
    let node_backing = StaticBacking::<16384>::new();
    let edge_backing = StaticBacking::<16384>::new();
    let witness_backing = StaticBacking::<16384>::new();

    GraphStoreBuilder::new(10)  // Small capacity for tests
        .with_proof_policy(ProofPolicy::standard())
        .build(
            node_backing,
            edge_backing,
            witness_backing,
            RegionHandle::new(1, 0),
            RegionHandle::new(2, 0),
            RegionHandle::new(3, 0),
            1,
        )
        .unwrap()
}

fn create_graph_capability() -> Capability {
    Capability::new(
        1,
        ObjectType::GraphStore,
        CapRights::READ | CapRights::WRITE | CapRights::PROVE,
        0,
        1,
    )
}

fn compute_graph_mutation_hash(mutation: &GraphMutation) -> [u8; 32] {
    let mut hash = [0u8; 32];
    hash[0] = mutation.kind as u8;
    hash[1..9].copy_from_slice(&mutation.node_a.to_le_bytes());
    hash[9..17].copy_from_slice(&mutation.node_b.to_le_bytes());
    hash[17..21].copy_from_slice(&mutation.weight_fp.to_le_bytes());
    hash[21..25].copy_from_slice(&mutation.partition_hint.to_le_bytes());
    hash
}

#[test]
fn test_graph_valid_proof_succeeds() {
    let mut store = create_graph_store();
    let cap = create_graph_capability();

    let mutation = GraphMutation::add_node(1);
    let hash = compute_graph_mutation_hash(&mutation);

    let proof = ProofToken::new(
        hash,
        ProofTier::Standard,
        ProofPayload::Hash { hash },
        2_000_000_000,
        1,
    );

    let result = store.graph_apply_proved(&mutation, &proof, &cap, 1_000_000_000);

    assert!(result.is_ok());
    assert_eq!(store.node_count(), 1);
}

#[test]
fn test_graph_wrong_hash_rejected() {
    let mut store = create_graph_store();
    let cap = create_graph_capability();

    let mutation = GraphMutation::add_node(1);
    let wrong_hash = [0xCD; 32];

    let proof = ProofToken::new(
        wrong_hash,
        ProofTier::Standard,
        ProofPayload::Hash { hash: wrong_hash },
        2_000_000_000,
        1,
    );

    let result = store.graph_apply_proved(&mutation, &proof, &cap, 1_000_000_000);

    assert_eq!(result, Err(KernelError::ProofRejected));
    assert_eq!(store.node_count(), 0);
}

#[test]
fn test_graph_add_edge_requires_nodes() {
    let mut store = create_graph_store();
    let cap = create_graph_capability();

    // Try to add edge without nodes
    let mutation = GraphMutation::add_edge(1, 2, 0.5);
    let hash = compute_graph_mutation_hash(&mutation);

    let proof = ProofToken::new(
        hash,
        ProofTier::Standard,
        ProofPayload::Hash { hash },
        2_000_000_000,
        1,
    );

    let result = store.graph_apply_proved(&mutation, &proof, &cap, 1_000_000_000);

    assert_eq!(result, Err(KernelError::NotFound));
}

#[test]
fn test_graph_complete_workflow() {
    let mut store = create_graph_store();
    let cap = create_graph_capability();

    let mut nonce = 1u64;

    // Add node 1
    let add1 = GraphMutation::add_node(1);
    let hash1 = compute_graph_mutation_hash(&add1);
    let proof1 = ProofToken::new(
        hash1,
        ProofTier::Standard,
        ProofPayload::Hash { hash: hash1 },
        2_000_000_000,
        nonce,
    );
    nonce += 1;
    store
        .graph_apply_proved(&add1, &proof1, &cap, 1_000_000_000)
        .unwrap();

    // Add node 2
    let add2 = GraphMutation::add_node(2);
    let hash2 = compute_graph_mutation_hash(&add2);
    let proof2 = ProofToken::new(
        hash2,
        ProofTier::Standard,
        ProofPayload::Hash { hash: hash2 },
        2_000_000_000,
        nonce,
    );
    nonce += 1;
    store
        .graph_apply_proved(&add2, &proof2, &cap, 1_000_000_001)
        .unwrap();

    // Add edge 1->2
    let edge = GraphMutation::add_edge(1, 2, 0.75);
    let hash3 = compute_graph_mutation_hash(&edge);
    let proof3 = ProofToken::new(
        hash3,
        ProofTier::Standard,
        ProofPayload::Hash { hash: hash3 },
        2_000_000_000,
        nonce,
    );
    nonce += 1;
    store
        .graph_apply_proved(&edge, &proof3, &cap, 1_000_000_002)
        .unwrap();

    assert_eq!(store.node_count(), 2);
    assert_eq!(store.edge_count(), 1);

    // Update edge weight
    let update = GraphMutation::update_edge_weight(1, 2, 0.9);
    let hash4 = compute_graph_mutation_hash(&update);
    let proof4 = ProofToken::new(
        hash4,
        ProofTier::Standard,
        ProofPayload::Hash { hash: hash4 },
        2_000_000_000,
        nonce,
    );
    nonce += 1;
    store
        .graph_apply_proved(&update, &proof4, &cap, 1_000_000_003)
        .unwrap();

    // Remove edge
    let remove_edge = GraphMutation::remove_edge(1, 2);
    let hash5 = compute_graph_mutation_hash(&remove_edge);
    let proof5 = ProofToken::new(
        hash5,
        ProofTier::Standard,
        ProofPayload::Hash { hash: hash5 },
        2_000_000_000,
        nonce,
    );
    nonce += 1;
    store
        .graph_apply_proved(&remove_edge, &proof5, &cap, 1_000_000_004)
        .unwrap();

    assert_eq!(store.edge_count(), 0);

    // Remove node
    let remove_node = GraphMutation::remove_node(1);
    let hash6 = compute_graph_mutation_hash(&remove_node);
    let proof6 = ProofToken::new(
        hash6,
        ProofTier::Standard,
        ProofPayload::Hash { hash: hash6 },
        2_000_000_000,
        nonce,
    );
    store
        .graph_apply_proved(&remove_node, &proof6, &cap, 1_000_000_005)
        .unwrap();

    assert_eq!(store.node_count(), 1);
}

#[test]
fn test_graph_witness_log_integrity() {
    let mut store = create_graph_store();
    let cap = create_graph_capability();

    // Perform 10 mutations
    for i in 0..10 {
        let mutation = GraphMutation::add_node(i);
        let hash = compute_graph_mutation_hash(&mutation);
        let proof = ProofToken::new(
            hash,
            ProofTier::Standard,
            ProofPayload::Hash { hash },
            2_000_000_000,
            i,
        );
        store
            .graph_apply_proved(&mutation, &proof, &cap, 1_000_000_000 + i)
            .unwrap();
    }

    // Verify witness log has all entries
    assert_eq!(store.witness_entry_count(), 10);
}

// ============================================================================
// Deep Proof Policy Tests
// ============================================================================

#[test]
fn test_deep_policy_requires_deep_tier() {
    let node_backing = StaticBacking::<16384>::new();
    let edge_backing = StaticBacking::<16384>::new();
    let witness_backing = StaticBacking::<16384>::new();

    let mut store = GraphStoreBuilder::new(10)  // Small capacity for tests
        .with_proof_policy(ProofPolicy::deep()) // Requires Deep tier
        .build(
            node_backing,
            edge_backing,
            witness_backing,
            RegionHandle::new(1, 0),
            RegionHandle::new(2, 0),
            RegionHandle::new(3, 0),
            1,
        )
        .unwrap();

    let cap = create_graph_capability();

    let mutation = GraphMutation::add_node(1);
    let hash = compute_graph_mutation_hash(&mutation);

    // Standard tier proof should be rejected
    let standard_proof = ProofToken::new(
        hash,
        ProofTier::Standard,
        ProofPayload::Hash { hash },
        2_000_000_000,
        1,
    );

    let result = store.graph_apply_proved(&mutation, &standard_proof, &cap, 1_000_000_000);

    assert_eq!(result, Err(KernelError::ProofRejected));

    // Deep tier proof should succeed
    let deep_proof = ProofToken::new(
        hash,
        ProofTier::Deep,
        ProofPayload::CoherenceCert {
            score_before: 10000,
            score_after: 9500,
            partition_id: 0,
            signature: [0u8; 64],
        },
        2_000_000_000,
        2,
    );

    let result = store.graph_apply_proved(&mutation, &deep_proof, &cap, 1_000_000_001);

    assert!(result.is_ok());
}

// ============================================================================
// Reflex Policy Tests (Fast Path)
// ============================================================================

#[test]
fn test_reflex_policy_accepts_all_tiers() {
    let node_backing = StaticBacking::<16384>::new();
    let edge_backing = StaticBacking::<16384>::new();
    let witness_backing = StaticBacking::<16384>::new();

    let mut store = GraphStoreBuilder::new(10)  // Small capacity for tests
        .with_proof_policy(ProofPolicy::reflex()) // Most permissive
        .build(
            node_backing,
            edge_backing,
            witness_backing,
            RegionHandle::new(1, 0),
            RegionHandle::new(2, 0),
            RegionHandle::new(3, 0),
            1,
        )
        .unwrap();

    let cap = create_graph_capability();

    // Reflex tier should work
    let mutation1 = GraphMutation::add_node(1);
    let hash1 = compute_graph_mutation_hash(&mutation1);
    let reflex_proof = ProofToken::new(
        hash1,
        ProofTier::Reflex,
        ProofPayload::Hash { hash: hash1 },
        2_000_000_000,
        1,
    );

    assert!(store
        .graph_apply_proved(&mutation1, &reflex_proof, &cap, 1_000_000_000)
        .is_ok());

    // Standard tier should also work
    let mutation2 = GraphMutation::add_node(2);
    let hash2 = compute_graph_mutation_hash(&mutation2);
    let standard_proof = ProofToken::new(
        hash2,
        ProofTier::Standard,
        ProofPayload::Hash { hash: hash2 },
        2_000_000_000,
        2,
    );

    assert!(store
        .graph_apply_proved(&mutation2, &standard_proof, &cap, 1_000_000_001)
        .is_ok());

    // Deep tier should also work
    let mutation3 = GraphMutation::add_node(3);
    let hash3 = compute_graph_mutation_hash(&mutation3);
    let deep_proof = ProofToken::new(
        hash3,
        ProofTier::Deep,
        ProofPayload::CoherenceCert {
            score_before: 10000,
            score_after: 9500,
            partition_id: 0,
            signature: [0u8; 64],
        },
        2_000_000_000,
        3,
    );

    assert!(store
        .graph_apply_proved(&mutation3, &deep_proof, &cap, 1_000_000_002)
        .is_ok());

    assert_eq!(store.node_count(), 3);
}

// ============================================================================
// Concurrent Nonce Tests
// ============================================================================

#[test]
fn test_multiple_proofs_unique_nonces() {
    let mut store = create_vector_store();
    let cap = create_full_capability();

    // Insert 10 vectors with unique nonces (matches store capacity)
    for i in 0..10 {
        let key = VectorKey::new(i);
        let data = vec![i as f32; 4];
        let hash = compute_vector_mutation_hash(key, &data);

        let proof = ProofToken::new(
            hash,
            ProofTier::Standard,
            ProofPayload::Hash { hash },
            2_000_000_000,
            i, // Unique nonce
        );

        let result = store.vector_put_proved(key, &data, &proof, &cap, 1_000_000_000 + i);
        assert!(result.is_ok(), "Mutation {} should succeed", i);
    }

    assert_eq!(store.len(), 10);
}

#[test]
fn test_coherence_metadata_tracking() {
    let mut store = create_vector_store();
    let cap = create_full_capability();

    let key = VectorKey::new(1);
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let hash = compute_vector_mutation_hash(key, &data);

    let proof = ProofToken::new(
        hash,
        ProofTier::Standard,
        ProofPayload::Hash { hash },
        2_000_000_000,
        1,
    );

    store
        .vector_put_proved(key, &data, &proof, &cap, 1_000_000_000)
        .unwrap();

    // Read and verify coherence
    let (_, coherence) = store.vector_get(key, &cap).unwrap();

    assert_eq!(coherence.coherence_score, 10000); // Initial full coherence
    assert!(coherence.mutation_epoch > 0);
}
