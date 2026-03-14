# RVF Swarm Consensus Demo

Multi-agent swarm coordination with Byzantine fault-tolerant consensus running on the RuVix Cognition Kernel.

## Overview

This demo showcases how RuVix provides **native primitives** for multi-agent AI systems:

```
┌─────────────────────────────────────────────────────────────┐
│                     SWARM ARCHITECTURE                       │
│                                                              │
│  ┌─────────────┐       PROPOSE        ┌─────────────┐       │
│  │ Coordinator │ ─────────────────────▶ │  Validators │       │
│  │   (Leader)  │ ◀───────────────────── │   (BFT)    │       │
│  └──────┬──────┘       VOTE           └─────────────┘       │
│         │                                                    │
│         │ COMMIT (on quorum)                                 │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    WORKERS                               │ │
│  │   [W1] ─┬─ [W2] ─┬─ [W3] ─┬─ [W4]                       │ │
│  │         │        │        │                              │ │
│  │         └────────┴────────┘                              │ │
│  │              PARALLEL EXECUTION                          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              SHARED VECTOR MEMORY                        │ │
│  │        [Proof-gated] [Witness-logged] [Immutable]        │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Why RuVix for Swarms?

| Traditional OS | RuVix |
|---------------|-------|
| Process isolation via memory protection | **Capability-based** isolation with unforgeable tokens |
| IPC via pipes/sockets (byte streams) | **Typed queues** with coherence scores |
| File-based shared state | **Native vector stores** with proof-gated mutations |
| Manual consensus implementation | **Built-in BFT** with witness logging |
| No attestation | **Every mutation witnessed** and tamper-evident |

## Files

```
swarm-consensus/
├── Cargo.toml          # Rust package manifest
├── README.md           # This file
├── swarm.rvf.json      # RVF cognitive container manifest
└── src/
    └── main.rs         # Demo implementation
```

## RVF Manifest Structure

The `swarm.rvf.json` defines:

### Agents

```json
{
  "agents": [
    {
      "id": "coordinator",
      "role": "leader",
      "count": 1,
      "capabilities": ["CAP_BROADCAST", "CAP_PROPOSE"]
    },
    {
      "id": "validator",
      "role": "validator",
      "count": 3,
      "capabilities": ["CAP_VOTE", "CAP_VERIFY"]
    },
    {
      "id": "worker",
      "role": "executor",
      "count": 4,
      "capabilities": ["CAP_EXECUTE", "CAP_REPORT"]
    }
  ]
}
```

### Consensus Configuration

```json
{
  "consensus": {
    "algorithm": "pbft",
    "fault_tolerance": "byzantine",
    "quorum": "2f+1",
    "timeout_ms": 1000
  }
}
```

With `n=3` validators and `f=1` fault tolerance, the system requires `2f+1 = 3` votes for quorum.

### Memory Regions

```json
{
  "regions": [
    {
      "name": "shared-memory",
      "type": "vector_store",
      "dimensions": 128,
      "capacity": 10000,
      "policy": "append-only"
    },
    {
      "name": "consensus-log",
      "type": "witness_log",
      "policy": "immutable"
    }
  ]
}
```

## Running the Demo

### On RuVix (QEMU or Hardware)

```bash
# Build the RVF package
ruvix build --rvf swarm.rvf.json --release

# Run on QEMU
ruvix run --target qemu-aarch64 --rvf swarm.rvf

# Run on Raspberry Pi 4
ruvix flash --device /dev/disk4 --rvf swarm.rvf
ruvix monitor --port /dev/tty.usbserial
```

### Hosted Mode (Testing)

```bash
cd crates/ruvix/examples/rvf-demos/swarm-consensus
cargo run
```

## Consensus Flow

```
1. PROPOSE: Coordinator creates task distribution proposal
   └─ Proof required (Reflex tier, <10μs)
   └─ Broadcast to all validators

2. VOTE: Each validator verifies and votes
   └─ Proof required per vote
   └─ Sent to coordinator inbox

3. COMMIT: Coordinator checks quorum (2f+1)
   └─ Proof required (Standard tier)
   └─ Attestation emitted to witness log
   └─ Broadcast commit to workers

4. EXECUTE: Workers process in parallel
   └─ Read input from vector store
   └─ Write output with proof
   └─ Report completion

5. AGGREGATE: Coordinator collects results
   └─ Verify all worker proofs
   └─ Final attestation
```

## Key RuVix Primitives Used

### 1. Capability-Based Isolation

```rust
let task = kernel.spawn_task(
    CapabilityRights::QUEUE_SEND | CapabilityRights::PROOF_EMIT,
    priority,
)?;
```

Each agent has explicit capabilities. A worker cannot broadcast (no `CAP_BROADCAST`). A validator cannot execute tasks (no `CAP_EXECUTE`).

### 2. Typed Queue IPC

```rust
kernel.dispatch(Syscall::QueueSend {
    queue: broadcast,
    message: serialize_message(&msg),
    proof: Some(proof),  // Proof-gated send
})?;
```

Zero-copy message passing with optional proof requirements.

### 3. Proof-Gated Vector Mutations

```rust
let proof = kernel.create_proof(output_hash, ProofTier::Standard, task_id)?;

kernel.dispatch(Syscall::VectorPutProved {
    store: vector_store,
    key: output_key,
    data: result,
    proof,  // Required for mutation
})?;
```

No proof = no mutation. Period.

### 4. Witness Logging

```rust
kernel.dispatch(Syscall::AttestEmit {
    record_type: "consensus_commit",
    data: &commit_hash,
    proof,
})?;
```

Every consensus decision is recorded immutably for audit and replay.

## Comparison: RuVix vs Linux Swarm

| Operation | Linux (Python/asyncio) | RuVix |
|-----------|----------------------|-------|
| Agent isolation | Process + namespace | Capability tokens |
| IPC latency | ~50-100μs (socket) | ~45ns (queue) |
| Consensus | Manual Redis/etcd | Native PBFT |
| Shared state | Redis/memcached | Native vector store |
| Proof verification | Application layer | Kernel syscall |
| Audit log | Application logging | Witness log (tamper-evident) |

## Extensions

This demo can be extended for:

1. **Distributed Inference**: Shard model across workers, aggregate outputs
2. **RAG Pipelines**: Parallel document retrieval with consensus ranking
3. **Multi-Agent Reasoning**: Chain-of-thought with validator verification
4. **Federated Learning**: Gradient aggregation with Byzantine tolerance

## Related

- [ADR-087: RuVix Cognition Kernel](../../../../docs/adr/ADR-087-ruvix-cognition-kernel.md)
- [RVF Specification](../../../rvf/)
- [Claude-Flow Swarm Integration](https://github.com/ruvnet/claude-flow)
