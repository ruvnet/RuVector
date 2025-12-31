# RuVector Edge - Distributed AI Swarm Communication

Edge AI swarm communication using `ruv-swarm-transport` with RuVector intelligence synchronization and production-grade P2P security.

## Features

- **🔐 Production-Grade Security**: Ed25519/X25519 crypto, AES-256-GCM encryption
- **🌐 Multi-Transport**: WebSocket, SharedMemory, and WASM support
- **🧠 Distributed Learning**: Sync Q-learning patterns across agents
- **💾 Shared Memory**: Vector memory for collaborative RAG
- **📦 Tensor Compression**: LZ4 + quantization for efficient transfer
- **🔄 Real-time Sync**: Automatic pattern propagation
- **🎯 Agent Roles**: Coordinator, Worker, Scout, Specialist
- **🌍 GUN Integration**: Decentralized P2P database for swarm state

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       RuVector Edge                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    P2P Swarm Layer                          │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │ │
│  │  │ Identity │ │  Crypto  │ │ Envelope │ │   Registry    │  │ │
│  │  │ Ed25519  │ │ AES-GCM  │ │  Signed  │ │ Membership    │  │ │
│  │  │ X25519   │ │ Canonical│ │  Tasks   │ │ Heartbeats    │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Transport Layer                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │ │
│  │  │  WebSocket   │  │ SharedMemory │  │    WASM      │      │ │
│  │  │  (Remote)    │  │   (Local)    │  │  (Browser)   │      │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                RuVector Integration                         │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ ┌───────┐  │ │
│  │  │ Intelligence │  │   Vector    │  │  Tensor  │ │  GUN  │  │ │
│  │  │    Sync      │  │   Memory    │  │ Compress │ │ Sync  │  │ │
│  │  └─────────────┘  └─────────────┘  └──────────┘ └───────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Add to your Cargo.toml
cargo add ruv-swarm-transport

# Or build this example
cd examples/edge
cargo build --release
```

### Run Demo

```bash
# Run the demo (local swarm simulation)
cargo run --bin edge-demo

# Expected output:
# 🚀 RuVector Edge Swarm Demo
# ✅ Coordinator created: coordinator-001
# ✅ Worker created: worker-001
# ✅ Worker created: worker-002
# ✅ Worker created: worker-003
# 📚 Simulating distributed learning...
```

### Run Coordinator

```bash
# Start a coordinator
cargo run --bin edge-coordinator -- --id coord-001

# With WebSocket transport
cargo run --bin edge-coordinator -- --transport websocket --listen 0.0.0.0:8080
```

### Run Agent

```bash
# Start a worker agent
cargo run --bin edge-agent -- --role worker

# Connect to coordinator
cargo run --bin edge-agent -- --coordinator ws://localhost:8080

# As a scout
cargo run --bin edge-agent -- --role scout --id scout-001
```

## P2P Swarm (Production-Grade Security)

The `p2p` module provides enterprise-grade security for swarm coordination:

### Security Features

| Feature | Implementation |
|---------|----------------|
| **Identity Keys** | Ed25519 for signing |
| **Key Exchange** | X25519 ECDH + HKDF |
| **Encryption** | AES-256-GCM |
| **Signatures** | Canonical JSON (sorted keys) |
| **Replay Protection** | Nonces + counters + timestamps |
| **Identity Binding** | Registry-based (never trust envelope keys) |

### Create a Secure P2P Swarm

```rust
use ruvector_edge::p2p::{P2PSwarmV2, IdentityManager};
use ruvector_edge::p2p::envelope::{TaskEnvelope, TaskBudgets, TaskStatus};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create swarm coordinator
    let mut swarm = P2PSwarmV2::new(
        "agent-001",
        None,  // Generate random swarm key (or provide [u8; 32])
        vec!["executor".to_string(), "coordinator".to_string()],
    );

    // Connect to network
    swarm.connect().await?;

    // Register a peer (from signed registration)
    let peer_identity = IdentityManager::new();
    let registration = peer_identity.create_registration(
        "peer-001",
        vec!["worker".to_string()],
    );
    swarm.register_member(registration);

    // Publish encrypted message (auto-signed)
    let message = b"Hello, swarm!";
    let msg_id = swarm.publish("chat", message)?;

    // Store artifact and get CID
    let data = b"Q-table data here";
    let cid = swarm.store_artifact(data, true)?;  // compressed

    // Create signed artifact pointer
    let pointer = swarm.create_artifact_pointer(
        ruvector_edge::p2p::envelope::ArtifactType::QTable,
        &cid,
        "100x10",
    );

    Ok(())
}
```

### Task Execution with Receipts

```rust
use ruvector_edge::p2p::envelope::{TaskEnvelope, TaskBudgets, TaskStatus};

// Submit task
let task = TaskEnvelope::new(
    "task-001".to_string(),
    "local:module_cid".to_string(),
    "process".to_string(),
    "local:input_cid".to_string(),
    [0u8; 32],  // output schema hash
    TaskBudgets {
        fuel_limit: 1_000_000,
        memory_mb: 128,
        timeout_ms: 30_000,
    },
    "requester-001".to_string(),
    deadline,
    1,  // priority
);

swarm.submit_task(task.clone())?;

// Claim task for execution
let claim = swarm.claim_task("task-001")?;

// After execution, create signed receipt (full binding)
let receipt = swarm.create_receipt(
    &task,
    "local:result_cid".to_string(),
    TaskStatus::Success,
    500_000,   // fuel_used
    64,        // memory_peak_mb
    1500,      // execution_ms
    input_hash,
    output_hash,
    module_hash,
);
```

### Derive Session Keys for Direct Channels

```rust
// After both parties are registered in registry
let session_key = swarm.derive_session_key("peer-001")?;

// Use for direct WebRTC channel encryption
// (separate from swarm broadcast key)
```

## GUN Integration (Decentralized Sync)

Integrate with GUN for true P2P decentralized state:

```rust
use ruvector_edge::gun::{GunSync, GunSwarmBuilder};

// Create GUN-backed swarm
let gun_sync = GunSwarmBuilder::new("my-swarm")
    .with_public_relays()  // Use public GUN relays
    .encrypted(true)        // Enable SEA encryption
    .sync_interval(1000)    // 1 second sync
    .build("agent-001");

// Connect to GUN network
gun_sync.connect().await?;

// Publish pattern to all peers (via GUN)
gun_sync.publish_pattern(&pattern).await?;

// Announce presence
gun_sync.announce_peer().await?;

// Get all patterns from swarm
let patterns = gun_sync.get_patterns().await;

// Sync learning state to GUN
let synced = gun_sync.sync_learning_state(&learning_state).await?;

// Import patterns from GUN to local state
let imported = gun_sync.import_to_learning_state(&mut learning_state).await;
```

## Basic Swarm Agent

```rust
use ruvector_edge::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let config = SwarmConfig::default()
        .with_agent_id("my-agent")
        .with_role(AgentRole::Worker)
        .with_transport(Transport::WebSocket);

    let mut agent = SwarmAgent::new(config).await?;

    // Join swarm
    agent.join_swarm("ws://coordinator:8080").await?;

    // Learn from experience
    agent.learn("edit_ts", "typescript-developer", 0.9).await;

    // Get best action
    let actions = vec!["coder".to_string(), "reviewer".to_string()];
    if let Some((action, confidence)) = agent.get_best_action("edit_ts", &actions).await {
        println!("Best action: {} ({:.0}% confidence)", action, confidence * 100.0);
    }

    // Store vector memory
    let embedding = vec![0.1, 0.2, 0.3, 0.4];
    agent.store_memory("API authentication flow", embedding).await?;

    // Search memory
    let query = vec![0.1, 0.2, 0.3, 0.4];
    let results = agent.search_memory(&query, 5).await;

    Ok(())
}
```

## Distributed Learning Sync

```rust
use ruvector_edge::intelligence::IntelligenceSync;

// Create sync manager
let sync = IntelligenceSync::new("agent-001");

// Update patterns locally
sync.update_pattern("edit_rs", "rust-developer", 0.95).await;

// Serialize for network transfer
let data = sync.serialize_state().await?;

// Merge peer state (federated learning)
let merge_result = sync.merge_peer_state("peer-002", &peer_data).await?;
println!("Merged {} patterns from peer", merge_result.merged_patterns);

// Get aggregated stats
let stats = sync.get_swarm_stats().await;
println!("Swarm: {} agents, {} patterns", stats.total_agents, stats.total_patterns);
```

## Tensor Compression

```rust
use ruvector_edge::compression::{TensorCodec, CompressionLevel};

// Create codec with quantization
let codec = TensorCodec::with_level(CompressionLevel::Quantized8);

// Compress tensor (75% size reduction)
let tensor: Vec<f32> = vec![0.1, 0.2, 0.3, /* ... */];
let compressed = codec.compress_tensor(&tensor)?;

// Decompress
let restored = codec.decompress_tensor(&compressed)?;
```

## Transport Options

| Transport | Use Case | Latency | Throughput |
|-----------|----------|---------|------------|
| WebSocket | Remote agents, cloud | Medium | High |
| SharedMemory | Local multi-process | Ultra-low | Very High |
| WASM | Browser-based agents | Low | Medium |

## Compression Levels

| Level | Ratio | Quality | Use Case |
|-------|-------|---------|----------|
| None | 1.0x | Lossless | Debugging |
| Fast | ~2x | Lossless | Default |
| High | ~3x | Lossless | Bandwidth-limited |
| Quantized8 | ~6x | Near-lossless | Pattern sync |
| Quantized4 | ~12x | Lossy | Archive |

## Agent Roles

| Role | Responsibilities |
|------|------------------|
| **Coordinator** | Manages swarm, distributes tasks |
| **Worker** | Executes tasks, learns patterns |
| **Scout** | Explores codebase, gathers context |
| **Specialist** | Domain expert (Rust, ML, etc.) |

## Protocol Messages

```
JOIN      → Agent joining swarm
LEAVE     → Agent leaving gracefully
PING/PONG → Heartbeat
SYNC_PATTERNS → Share learning state
REQUEST_PATTERNS → Request delta from peer
SYNC_MEMORIES → Share vector memories
BROADCAST_TASK → Distribute task to swarm
TASK_RESULT → Return task result
```

## P2P Security Model

### Identity Trust Chain

```
1. Member Registration (signed with Ed25519)
   └── agent_id + ed25519_pubkey + x25519_pubkey + capabilities + joined_at
   └── signature covers ALL fields

2. Registry Verification
   └── verify registration signature
   └── check x25519_pubkey present
   └── check capabilities non-empty
   └── store in member_registry

3. Message Verification
   └── resolve sender from registry (NEVER trust envelope key)
   └── verify signature using registry key
   └── check nonce/counter/timestamp
   └── decrypt with swarm key

4. Task Receipt Binding
   └── signature covers ALL fields including:
       - module_cid, input_cid, entrypoint
       - input_hash, output_hash, module_hash
       - task_envelope_hash (full traceability)
```

### Key Derivation

```
Session Key = HKDF-SHA256(
    IKM: X25519(our_private, peer_public),
    Salt: SHA256(sorted(our_x25519_pub || peer_x25519_pub)),
    Info: "p2p-swarm-v2:{swarm_id}"
)
```

## Environment Variables

```bash
RUST_LOG=info          # Logging level
SWARM_COORDINATOR=ws://localhost:8080  # Default coordinator
SWARM_SYNC_INTERVAL=1000  # Sync interval in ms
```

## Integration with RuVector

This example integrates with the main RuVector ecosystem:

- **Learning Engine**: 9 RL algorithms for pattern learning
- **TensorCompress**: Adaptive compression based on access frequency
- **ONNX Embeddings**: Local semantic embeddings (all-MiniLM-L6-v2)
- **GNN/Attention**: Graph neural networks for code understanding

## Performance

| Metric | Value |
|--------|-------|
| Sync latency (SharedMemory) | < 1ms |
| Sync latency (WebSocket) | 5-50ms |
| Pattern merge throughput | 10K/sec |
| Compression ratio | 2-12x |
| Max agents per swarm | 1000+ |
| Ed25519 sign | ~50K ops/sec |
| AES-256-GCM encrypt | ~1 GB/sec |

## Feature Flags

```toml
[features]
default = ["websocket", "shared-memory"]
websocket = ["ruv-swarm-transport/default"]
shared-memory = []
wasm = ["ruv-swarm-transport/wasm", "wasm-bindgen", "web-sys", "js-sys"]
gun = ["dep:gundb"]
full = ["websocket", "shared-memory", "gun"]
```

## License

MIT
