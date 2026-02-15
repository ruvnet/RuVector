//! Integration tests for rvf-adapter-claude-flow.
//!
//! Exercises the full adapter lifecycle: create -> ingest -> search -> audit -> close ->
//! reopen -> verify.

use rvf_adapter_claude_flow::{ClaudeFlowConfig, RvfMemoryStore};
use tempfile::TempDir;

fn make_embedding(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut x = seed;
    for _ in 0..dim {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((x >> 33) as f32) / (u32::MAX as f32) - 0.5);
    }
    v
}

#[test]
fn create_open_ingest_search() {
    let dir = TempDir::new().unwrap();
    let config = ClaudeFlowConfig::new(dir.path(), "test-agent")
        .with_dimension(4)
        .with_namespace("test-session");

    // Create store and ingest memories
    {
        let mut store = RvfMemoryStore::create(config.clone()).unwrap();

        // Ingest three memories
        let memories = [
            ("theme", "dark", Some("ui"), make_embedding(4, 1)),
            ("lang", "en", Some("profile"), make_embedding(4, 2)),
            ("project", "rvf", Some("work"), make_embedding(4, 3)),
        ];

        for (key, value, ns, ref embedding) in &memories {
            let id = store.ingest_memory(key, value, *ns, embedding).unwrap();
            assert!(id > 0);
        }

        // Search memories
        let query = make_embedding(4, 2);
        let results = store.search_memories(&query, 3);
        assert_eq!(results.len(), 3);
        // closest to seed=2
        assert_eq!(results[0].key, "lang");

        store.close().unwrap();
    }

    // Reopen and verify persistence
    {
        let store = RvfMemoryStore::open(config).unwrap();
        let status = store.status();
        assert_eq!(status.total_vectors, 3);

        store.close().unwrap();
    }
}

#[test]
fn coordination_and_learning() {
    let dir = TempDir::new().unwrap();
    let config = ClaudeFlowConfig::new(dir.path(), "swarm-agent").with_dimension(4);

    let mut store = RvfMemoryStore::create(config).unwrap();

    // Record coordination state
    store
        .coordination()
        .record_state("agent-1", "status", "active")
        .unwrap();
    store
        .coordination()
        .record_consensus_vote("leader-election", "agent-1", true)
        .unwrap();

    // Store learning pattern
    let pattern_id = store
        .learning()
        .store_pattern("convergent", "batch processing", 0.85)
        .unwrap();
    assert!(pattern_id > 0);

    // Verify coordination state
    let states = store.coordination_ref().get_all_states();
    assert_eq!(states.len(), 1);
    assert_eq!(states[0].value, "active");

    let votes = store.coordination_ref().get_votes("leader-election");
    assert_eq!(votes.len(), 1);
    assert!(votes[0].vote);

    // Verify learning pattern
    let top_patterns = store.learning_ref().get_top_patterns(5);
    assert_eq!(top_patterns.len(), 1);
    assert_eq!(top_patterns[0].pattern_type, "convergent");

    store.close().unwrap();
}

#[test]
fn audit_trail_integrity() {
    let dir = TempDir::new().unwrap();
    let config = ClaudeFlowConfig::new(dir.path(), "audit-agent").with_dimension(4);

    let mut store = RvfMemoryStore::create(config.clone()).unwrap();

    // Record various actions
    store.witness().record_action("init", &[]).unwrap();
    store.witness().record_search(5).unwrap();
    store.witness().record_session("start").unwrap();

    // Verify witness chain integrity
    assert!(store.witness_ref().verify().is_ok());

    // Get entries and check ordering
    let entries: Vec<_> = store.witness_ref().entries().collect();
    assert_eq!(entries.len(), 3);

    // Timestamps should be monotonic
    assert!(entries[0].timestamp_ns <= entries[1].timestamp_ns);
    assert!(entries[1].timestamp_ns <= entries[2].timestamp_ns);

    store.close().unwrap();

    // Reopen and verify chain persists
    let store = RvfMemoryStore::open(config).unwrap();
    assert!(store.witness_ref().verify().is_ok());

    let reopened_entries: Vec<_> = store.witness_ref().entries().collect();
    assert_eq!(reopened_entries.len(), 3);

    store.close().unwrap();
}

#[test]
fn duplicate_key_replacement() {
    let dir = TempDir::new().unwrap();
    let config = ClaudeFlowConfig::new(dir.path(), "dup-agent").with_dimension(4);

    let mut store = RvfMemoryStore::create(config).unwrap();

    let id1 = store
        .ingest_memory("pref", "light", None, &make_embedding(4, 1))
        .unwrap();

    let id2 = store
        .ingest_memory("pref", "dark", None, &make_embedding(4, 2))
        .unwrap();

    assert_ne!(id1, id2);

    // Search should return the newer value
    let results = store.search_memories(&make_embedding(4, 2), 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].value, "dark");

    store.close().unwrap();
}

#[test]
fn namespace_isolation() {
    let dir = TempDir::new().unwrap();
    let config = ClaudeFlowConfig::new(dir.path(), "ns-agent").with_dimension(4);

    let mut store = RvfMemoryStore::create(config).unwrap();

    // Ingest same key in different namespaces
    store
        .ingest_memory("key", "value1", Some("ns1"), &make_embedding(4, 1))
        .unwrap();
    store
        .ingest_memory("key", "value2", Some("ns2"), &make_embedding(4, 2))
        .unwrap();

    // Search should find both
    let results = store.search_memories(&make_embedding(4, 1), 5);
    assert_eq!(results.len(), 2);

    // Verify namespace isolation
    let ns1_values: Vec<_> = results
        .iter()
        .filter(|r| r.namespace.as_deref() == Some("ns1"))
        .collect();
    let ns2_values: Vec<_> = results
        .iter()
        .filter(|r| r.namespace.as_deref() == Some("ns2"))
        .collect();

    assert_eq!(ns1_values.len(), 1);
    assert_eq!(ns1_values[0].value, "value1");
    assert_eq!(ns2_values.len(), 1);
    assert_eq!(ns2_values[0].value, "value2");

    store.close().unwrap();
}
