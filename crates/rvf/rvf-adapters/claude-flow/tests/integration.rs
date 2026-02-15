//! Integration tests for rvf-adapter-claude-flow.  
//!  
//! Exercises the full adapter lifecycle: create -> ingest -> search -> audit -> close ->  
//! reopen -> verify. Tests both minimal memory-only and full swarm-capable modes.  
  
use std::path::PathBuf;  
use tempfile::TempDir;  
use rvf_adapter_claude_flow::{ClaudeFlowConfig, RvfMemoryStore};  
  
fn make_embedding(dim: usize, seed: u64) -> Vec<f32> {  
    let mut v = Vec::with_capacity(dim);  
    let mut x = seed;  
    for _ in 0..dim {  
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);  
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
            ("theme", "dark", "ui", make_embedding(4, 1)),  
            ("lang", "en", "profile", make_embedding(4, 2)),  
            ("project", "rvf", "work", make_embedding(4, 3)),  
        ];  
  
        for (key, value, ns, embedding) in memories {  
            let id = store.ingest_memory(key, value, Some(ns), &embedding).unwrap();  
            assert!(id > 0);  
              
            // Record witness for audit trail  
            store.witness().record_ingest(key, value, ns).unwrap();  
        }  
  
        // Search memories  
        let query = make_embedding(4, 2);  
        let results = store.search_memories(&query, 3).unwrap();  
        assert_eq!(results.len(), 3);  
        assert_eq!(results[0].key, "lang"); // closest to seed=2  
  
        store.close().unwrap();  
    }  
  
    // Reopen and verify persistence  
    {  
        let store = RvfMemoryStore::open(config).unwrap();  
        let status = store.status();  
        assert_eq!(status.total_vectors, 3);  
  
        // Search again after reopen  
        let query = make_embedding(4, 1);  
        let results = store.search_memories(&query, 2).unwrap();  
        assert_eq!(results.len(), 2);  
        assert_eq!(results[0].key, "theme");  
  
        // Verify witness trail  
        let entries: Vec<_> = store.witness().entries().take(3).collect();  
        assert_eq!(entries.len(), 3);  
  
        store.close().unwrap();  
    }  
}  
  
#[test]  
fn coordination_and_learning() {  
    let dir = TempDir::new().unwrap();  
    let config = ClaudeFlowConfig::new(dir.path(), "swarm-agent")  
        .with_dimension(4);  
  
    let mut store = RvfMemoryStore::create(config).unwrap();  
  
    // Record coordination state  
    store.coordination()  
        .record_state("agent-1", "status", "active")  
        .unwrap();  
    store.coordination()  
        .record_consensus_vote("leader-election", "agent-1", true)  
        .unwrap();  
  
    // Store learning pattern  
    let pattern_id = store.learning()  
        .store_pattern("convergent", "batch processing", 0.85)  
        .unwrap();  
    assert!(pattern_id > 0);  
  
    // Verify coordination state  
    let states = store.coordination().get_agent_states("agent-1");  
    assert_eq!(states.len(), 1);  
    assert_eq!(states[0].value, "active");  
  
    let votes = store.coordination().get_votes("leader-election");  
    assert_eq!(votes.len(), 1);  
    assert!(votes[0].vote);  
  
    // Verify learning pattern  
    let top_patterns = store.learning().get_top_patterns(5);  
    assert_eq!(top_patterns.len(), 1);  
    assert_eq!(top_patterns[0].pattern_type, "convergent");  
  
    store.close().unwrap();  
}  
  
#[test]  
fn audit_trail_integrity() {  
    let dir = TempDir::new().unwrap();  
    let config = ClaudeFlowConfig::new(dir.path(), "audit-agent")  
        .with_dimension(4);  
  
    let mut store = RvfMemoryStore::create(config).unwrap();  
  
    // Record various actions  
    store.witness().record_action("init", &[]).unwrap();  
    store.witness().record_search("query", 5).unwrap();  
    store.witness().record_session("start").unwrap();  
  
    // Verify witness chain integrity  
    assert!(store.witness().verify().is_ok());  
  
    // Get entries and check ordering  
    let entries: Vec<_> = store.witness().entries().collect();  
    assert_eq!(entries.len(), 3);  
      
    // Timestamps should be monotonic  
    assert!(entries[0].timestamp_ns <= entries[1].timestamp_ns);  
    assert!(entries[1].timestamp_ns <= entries[2].timestamp_ns);  
  
    store.close().unwrap();  
  
    // Reopen and verify chain persists  
    let store = RvfMemoryStore::open(config).unwrap();  
    assert!(store.witness().verify().is_ok());  
      
    let reopened_entries: Vec<_> = store.witness().entries().collect();  
    assert_eq!(reopened_entries.len(), 3);  
  
    store.close().unwrap();  
}  
  
#[test]  
fn duplicate_key_replacement() {  
    let dir = TempDir::new().unwrap();  
    let config = ClaudeFlowConfig::new(dir.path(), "dup-agent")  
        .with_dimension(4);  
  
    let mut store = RvfMemoryStore::create(config).unwrap();  
  
    // Ingest memory with key "pref"  
    let id1 = store  
        .ingest_memory("pref", "light", None, &make_embedding(4, 1))  
        .unwrap();  
  
    // Ingest again with same key but different value  
    let id2 = store  
        .ingest_memory("pref", "dark", None, &make_embedding(4, 2))  
        .unwrap();  
  
    assert_ne!(id1, id2);  
  
    // Search should return the newer value  
    let results = store.search_memories(&make_embedding(4, 2), 1).unwrap();  
    assert_eq!(results.len(), 1);  
    assert_eq!(results[0].value, "dark");  
  
    store.close().unwrap();  
}  
  
#[test]  
fn namespace_isolation() {  
    let dir = TempDir::new().unwrap();  
    let config = ClaudeFlowConfig::new(dir.path(), "ns-agent")  
        .with_dimension(4);  
  
    let mut store = RvfMemoryStore::create(config).unwrap();  
  
    // Ingest same key in different namespaces  
    store  
        .ingest_memory("key", "value1", Some("ns1"), &make_embedding(4, 1))  
        .unwrap();  
    store  
        .ingest_memory("key", "value2", Some("ns2"), &make_embedding(4, 2))  
        .unwrap();  
  
    // Search should find both  
    let results = store.search_memories(&make_embedding(4, 1), 5).unwrap();  
    assert_eq!(results.len(), 2);  
  
    // Verify namespace isolation  
    let ns1_values: Vec<_> = results.iter()  
        .filter(|r| r.namespace.as_deref() == Some("ns1"))  
        .collect();  
    let ns2_values: Vec<_> = results.iter()  
        .filter(|r| r.namespace.as_deref() == Some("ns2"))  
        .collect();  
      
    assert_eq!(ns1_values.len(), 1);  
    assert_eq!(ns1_values[0].value, "value1");  
    assert_eq!(ns2_values.len(), 1);  
    assert_eq!(ns2_values[0].value, "value2");  
  
    store.close().unwrap();  
}