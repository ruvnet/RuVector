//! # Swarm Knowledge  
//!  
//! Category: Agentic  
//!  
//! **What this demonstrates:**  
//! - Multi-agent shared knowledge base with concurrent writes  
//! - Cross-agent semantic search across shared memories  
//! - Swarm coordination state tracking (agent status, topology)  
//! - Learning pattern sharing and effectiveness scoring  
//! - Tamper-evident audit trails for all swarm operations  
//!  
//! **RVF segments used:** VEC_SEG, META_SEG, WITNESS_SEG, SKETCH_SEG, MANIFEST_SEG  
//!  
//! **Run:** `cargo run --example swarm_knowledge`  
  
use std::path::PathBuf;  
use tempfile::TempDir;  
  
use rvf_adapter_claude_flow::{ClaudeFlowConfig, RvfMemoryStore};  
  
fn main() -> Result<(), Box<dyn std::error::Error>> {  
    // ------------------------------------------------------------------  
    // 1. Create a temporary directory and configure the swarm store  
    // ------------------------------------------------------------------  
    let tmp_dir = TempDir::new()?;  
    let config = ClaudeFlowConfig::new(tmp_dir.path(), "agent-coordinator")  
        .with_dimension(4)  
        .with_namespace("swarm-knowledge");  
  
    // ------------------------------------------------------------------  
    // 2. Create the store and simulate multiple agents sharing knowledge  
    // ------------------------------------------------------------------  
    let mut store = RvfMemoryStore::create(config)?;  
    let base_ts = 1739606400; // 2025-01-15 00:00:00 UTC  
  
    // Agents in the swarm with different knowledge contributions  
    let agents = [  
        ("agent-planner", "planning-strategies", "coordination"),  
        ("agent-researcher", "research-findings", "knowledge"),  
        ("agent-executor", "execution-patterns", "operations"),  
    ];  
  
    // Each agent shares knowledge with deterministic embeddings  
    for (i, (agent_id, key, ns)) in agents.iter().enumerate() {  
        let embedding = vec![  
            0.1 + (i as f32) * 0.1,  
            0.2 + (i as f32) * 0.1,  
            0.3 + (i as f32) * 0.1,  
            0.4 + (i as f32) * 0.1,  
        ];  
  
        let id = store.ingest_memory(key, &format!("Shared by {}", agent_id), Some(*ns), &embedding)?;  
        println!("Agent {} shared knowledge {}: id={}, key={}", agent_id, i + 1, id, key);  
  
        // Record agent state in coordination  
        store.coordination().record_state(agent_id, "status", "active")?;  
        store.coordination().record_state(agent_id, "last_shared", key)?;  
  
        // Record witness for knowledge sharing  
        store.witness().record_action("share-knowledge", &[agent_id, key])?;  
  
        // Store learning pattern for this type of knowledge  
        let pattern_type = if *ns == "coordination" { "convergent" } else { "divergent" };  
        store.learning().store_pattern(  
            pattern_type,  
            &format!("{} pattern for {}", pattern_type, key),  
            0.75 + (i as f32) * 0.05,  
        )?;  
    }  
  
    // ------------------------------------------------------------------  
    // 3. Simulate topology change in swarm coordination  
    // ------------------------------------------------------------------  
    store.coordination().record_state("swarm", "topology", "mesh")?;  
    store.coordination().record_state("swarm", "coordinator", "agent-coordinator")?;  
    store.witness().record_action("topology-change", &["mesh"])?;  
  
    // ------------------------------------------------------------------  
    // 4. Cross-agent semantic search  
    // ------------------------------------------------------------------  
    let query = vec![0.15, 0.25, 0.35, 0.45]; // Closest to agent-planner's knowledge  
    let results = store.search_memories(&query, 3)?;  
    println!("\nCross-agent knowledge search (k=3):");  
    for (i, r) in results.iter().enumerate() {  
        println!(  
            "  {}. key={}, value={}, namespace={:.12}, distance={:.4}",  
            i + 1, r.key, r.value, r.namespace.as_deref().unwrap_or(""), r.distance  
        );  
    }  
  
    // ------------------------------------------------------------------  
    // 5. Retrieve swarm coordination state  
    // ------------------------------------------------------------------  
    println!("\nSwarm coordination state:");  
    let all_states = store.coordination().get_all_states();  
    for state in &all_states {  
        if state.agent_id == "swarm" {  
            println!("  Swarm: {} = {}", state.key, state.value);  
        } else {  
            println!("  {}: {} = {}", state.agent_id, state.key, state.value);  
        }  
    }  
  
    // ------------------------------------------------------------------  
    // 6. Retrieve shared learning patterns  
    // ------------------------------------------------------------------  
    println!("\nShared learning patterns:");  
    let top_patterns = store.learning().get_top_patterns(3);  
    for (i, pattern) in top_patterns.iter().enumerate() {  
        println!(  
            "  {}. type={}, description={}, score={:.2}",  
            i + 1, pattern.pattern_type, pattern.description, pattern.score  
        );  
    }  
  
    // ------------------------------------------------------------------  
    // 7. Verify audit trail  
    // ------------------------------------------------------------------  
    let witness_entries: Vec<_> = store.witness().entries().take(5).collect();  
    println!("\nAudit trail (first 5 entries):");  
    for (i, e) in witness_entries.iter().enumerate() {  
        println!(  
            "  {}. type=0x{:02x}, timestamp_ns={}",  
            i + 1,  
            e.witness_type,  
            e.timestamp_ns  
        );  
    }  
  
    // ------------------------------------------------------------------  
    // 8. Demonstrate session persistence  
    // ------------------------------------------------------------------  
    store.close()?;  
    let config_reopen = ClaudeFlowConfig::new(tmp_dir.path(), "agent-coordinator")  
        .with_dimension(4)  
        .with_namespace("swarm-knowledge");  
    let store_reopened = RvfMemoryStore::open(config_reopen)?;  
    let status = store_reopened.status();  
    println!(  
        "\nReopened swarm store: {} vectors, epoch {}",  
        status.total_vectors, status.current_epoch  
    );  
  
    // Verify coordination state persistence  
    let swarm_states = store_reopened.coordination().get_agent_states("swarm");  
    println!("Swarm topology after reopen: {:?}", swarm_states);  
  
    store_reopened.close()?;  
    Ok(())  
}