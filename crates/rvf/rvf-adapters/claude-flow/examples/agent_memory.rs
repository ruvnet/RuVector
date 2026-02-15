//! # Agent Memory
//!
//! Category: Agentic
//!
//! **What this demonstrates:**
//! - Persistent agent memory across sessions with witness audit trails
//! - Memory ingestion with metadata (key, value, namespace)
//! - Semantic search via embedding similarity
//! - Tamper-evident audit logging via WITNESS_SEG
//! - Session recall by reopening an RVF store
//!
//! **RVF segments used:** VEC_SEG, META_SEG, WITNESS_SEG, MANIFEST_SEG
//!
//! **Run:** `cargo run --example agent_memory`

use tempfile::TempDir;

use rvf_adapter_claude_flow::{ClaudeFlowConfig, RvfMemoryStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ------------------------------------------------------------------
    // 1. Create a temporary directory and configure the memory store
    // ------------------------------------------------------------------
    let tmp_dir = TempDir::new()?;
    let config = ClaudeFlowConfig::new(tmp_dir.path(), "agent-demo")
        .with_dimension(4)
        .with_namespace("session-42");

    // ------------------------------------------------------------------
    // 2. Create the store and ingest memory entries
    // ------------------------------------------------------------------
    let mut store = RvfMemoryStore::create(config.clone())?;

    // Ingest three memories with deterministic embeddings
    let memories = [
        ("user-theme", "dark-mode", Some("ui"), vec![0.1, 0.2, 0.3, 0.4]),
        ("user-lang", "english", Some("profile"), vec![0.2, 0.3, 0.4, 0.5]),
        ("project", "rvf-adapter", Some("work"), vec![0.3, 0.4, 0.5, 0.6]),
    ];

    for (i, (key, value, ns, ref embedding)) in memories.iter().enumerate() {
        let id = store.ingest_memory(key, value, *ns, embedding)?;
        println!("Ingested memory {}: id={}, key={}, value={}", i + 1, id, key, value);
    }

    // ------------------------------------------------------------------
    // 3. Search memories by embedding similarity
    // ------------------------------------------------------------------
    let query = vec![0.15, 0.25, 0.35, 0.45];
    let results = store.search_memories(&query, 3);
    println!("\nSearch results (k=3):");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  {}. key={}, value={}, namespace={:.12}, distance={:.4}",
            i + 1,
            r.key,
            r.value,
            r.namespace.as_deref().unwrap_or(""),
            r.distance
        );
    }

    // ------------------------------------------------------------------
    // 4. Demonstrate session recall: close and reopen the store
    // ------------------------------------------------------------------
    store.close()?;
    let store_reopened = RvfMemoryStore::open(config)?;
    let status = store_reopened.status();
    println!(
        "\nReopened store: {} vectors, epoch {}",
        status.total_vectors, status.current_epoch
    );

    // Verify we can search memories after reopen
    let results_after = store_reopened.search_memories(&query, 2);
    println!("Search after reopen (k=2):");
    for r in results_after.iter() {
        println!("  - key={}, value={}", r.key, r.value);
    }

    // ------------------------------------------------------------------
    // 5. Show witness audit trail entries
    // ------------------------------------------------------------------
    let entries: Vec<_> = store_reopened.witness_ref().entries().take(3).collect();
    println!("\nWitness audit trail (first 3 entries):");
    for (i, e) in entries.iter().enumerate() {
        println!(
            "  {}. type=0x{:02x}, timestamp_ns={}",
            i + 1,
            e.witness_type,
            e.timestamp_ns
        );
    }

    store_reopened.close()?;
    Ok(())
}
