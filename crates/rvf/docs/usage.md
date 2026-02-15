 ---
  How to Use RVF

  1. CLI - Quick Operations

  # Alias for convenience (or use cargo run -p rvf-cli --)
  alias rvf="cargo run -p rvf-cli --"

  # Create a store (dimension = your embedding size)
  rvf create mydata.rvf --dimension 384

  # Ingest vectors from JSON
  # Format: [{"id": 1, "vector": [0.1, 0.2, ...]}, ...]
  rvf ingest mydata.rvf --input data.json

  # Search nearest neighbors
  rvf query mydata.rvf --vector "0.1,0.2,0.3,..." --k 10

  # Check store info
  rvf status mydata.rvf

  # Inspect segments and lineage
  rvf inspect mydata.rvf

  # Delete vectors, then compact
  rvf delete mydata.rvf --ids "3,5"
  rvf compact mydata.rvf

  # Branch a child (COW - only stores diffs)
  rvf derive parent.rvf child.rvf

  # Verify audit trail
  rvf verify-witness mydata.rvf

  2. Rust API - Embed in Your App

  use rvf_runtime::{RvfStore, RvfOptions, QueryOptions};

  // Create store
  let mut store = RvfStore::create("vectors.rvf", RvfOptions {
      dimension: 384,
      ..Default::default()
  })?;

  // Ingest vectors (batch)
  store.ingest_batch(&[&embedding], &[1], None)?;

  // Query k-NN
  let results = store.query(&query_vec, 10, &QueryOptions::default())?;
  // results = [SearchResult { id: 1, distance: 0.01 }, ...]

  store.close()?;

  3. Claude-Flow Adapter - Agent Memory

  This is what we built. It wraps RVF for AI agent use cases:

  use rvf_adapter_claude_flow::{ClaudeFlowConfig, RvfMemoryStore};

  // Configure
  let config = ClaudeFlowConfig::new("./agent-data", "agent-1")
      .with_dimension(384)
      .with_namespace("session-42");

  // Create store
  let mut store = RvfMemoryStore::create(config.clone())?;

  // Store memories with semantic embeddings
  store.ingest_memory("user-pref", "dark-mode", Some("ui"), &embedding)?;

  // Search by similarity
  let results = store.search_memories(&query_embedding, 10);

  // Swarm coordination (multi-agent state)
  store.coordination().record_state("agent-1", "status", "active")?;
  store.coordination().record_consensus_vote("leader-election", "agent-1", true)?;

  // Learning patterns (track what works)
  store.learning().store_pattern("convergent", "batch strategy", 0.85)?;

  // Tamper-evident audit trail
  store.witness().record_action("decision", &["chose-plan-B"])?;

  // Persist and reopen later
  store.close()?;
  let store = RvfMemoryStore::open(config)?;  // memories survive restarts

  Key Concepts
  What: Store a vector
  How: ingest_batch / ingest_memory
  ────────────────────────────────────────
  What: Search by similarity
  How: query / search_memories (returns nearest neighbors by L2/cosine distance)
  ────────────────────────────────────────
  What: Attach metadata
  How: Key-value pairs per vector, filterable with AND/OR/IN/RANGE
  ────────────────────────────────────────
  What: Audit everything
  How: Witness chain - every operation gets a hash-linked 73-byte entry
  ────────────────────────────────────────
  What: Branch without copying
  How: derive creates a COW child that shares the parent's data
  ────────────────────────────────────────
  What: Persist across sessions
  How: close() then open() - everything survives restarts
  ────────────────────────────────────────
  What: Multi-agent
  How: Coordination state + consensus votes + shared knowledge base
  The file is self-contained - one .rvf file holds vectors, indexes, metadata, audit trails, and
  lineage. No external database needed.