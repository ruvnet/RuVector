# ruvix-nucleus

Integration crate for the RuVix Cognition Kernel (ADR-087).

## Overview

The `ruvix-nucleus` crate brings all RuVix subsystems together and provides:

- **Syscall dispatch table** for all 12 syscalls defined in ADR-087
- **Kernel struct** coordinating all subsystems
- **Deterministic replay** support for checkpoint/restore
- **Witness log** for attestation and auditability
- **Proof-gated mutations** with multi-tier verification

## Architecture

```
                    +-------------------+
                    |    RuVix Nucleus  |
                    |  (this crate)     |
                    +--------+----------+
                             |
         +-------------------+-------------------+
         |           |           |               |
   +-----v-----+ +---v---+ +-----v-----+ +-------v-------+
   | RegionMgr | | CapMgr| | QueueMgr  | | ProofEngine   |
   +-----------+ +-------+ +-----------+ +---------------+
         |           |           |               |
   +-----v-----+ +---v---+ +-----v-----+ +-------v-------+
   | VectorMgr | |GraphMgr| | Scheduler | | WitnessLog    |
   +-----------+ +-------+ +-----------+ +---------------+
```

## Syscalls

The kernel provides 12 syscalls as defined in ADR-087 Section 3.1:

| # | Syscall | Proof Required | Description |
|---|---------|----------------|-------------|
| 0 | `TaskSpawn` | No | Create a new task |
| 1 | `CapGrant` | No | Grant a capability to another task |
| 2 | `RegionMap` | No | Map a memory region |
| 3 | `QueueSend` | No | Send a message to a queue |
| 4 | `QueueRecv` | No | Receive a message from a queue |
| 5 | `TimerWait` | No | Wait for a timer deadline |
| 6 | `RvfMount` | Yes | Mount an RVF package |
| 7 | `AttestEmit` | Yes | Emit an attestation record |
| 8 | `VectorGet` | No | Get a vector from a kernel store |
| 9 | `VectorPutProved` | Yes | Put a vector with proof |
| 10 | `GraphApplyProved` | Yes | Apply a graph mutation with proof |
| 11 | `SensorSubscribe` | No | Subscribe to sensor events |

## Syscall Invariants (ADR-087 Section 3.2)

1. **Capability-gated**: No syscall succeeds without appropriate capability
2. **Proof-required for mutation**: `vector_put_proved`, `graph_apply_proved`, `rvf_mount`
3. **Bounded latency**: No unbounded loops in syscall path
4. **Witness-logged**: Every mutation emits a witness record
5. **No allocation in syscall path**: Pre-allocated structures only

## Proof Tiers

The proof engine supports three verification tiers:

| Tier | Latency | Use Case |
|------|---------|----------|
| **Reflex** | <10us | Real-time hotpath mutations |
| **Standard** | ~100us | Normal operations |
| **Deep** | ~1ms | Security-critical operations |

## Usage

```rust
use ruvix_nucleus::{Kernel, KernelConfig, Syscall, VectorStoreConfig, ProofTier};

// Create and boot kernel
let mut kernel = Kernel::new(KernelConfig::default());
kernel.boot(0, [0u8; 32]).expect("Boot failed");

// Create a vector store
let config = VectorStoreConfig::new(4, 100); // 4 dimensions, 100 capacity
let store = kernel.create_vector_store(config).unwrap();

// Create a proof for mutation
let mutation_hash = [1u8; 32];
let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, 1);

// Execute a proved vector put
use ruvix_nucleus::{VectorKey, SyscallResult};

let result = kernel.dispatch(Syscall::VectorPutProved {
    store,
    key: VectorKey::new(1),
    data: vec![1.0, 2.0, 3.0, 4.0],
    proof,
}).expect("VectorPutProved failed");

assert!(matches!(result, SyscallResult::VectorStored));
```

## Deterministic Replay

The kernel supports deterministic replay from checkpoint + witness log:

```rust
use ruvix_nucleus::{CheckpointConfig, Checkpoint};

// Create checkpoint
let checkpoint = kernel.checkpoint(CheckpointConfig::full()).unwrap();

// Later: restore and replay
let bytes = checkpoint.to_bytes();
let restored = Checkpoint::from_bytes(&bytes).unwrap();

// Verify state matches checkpoint
assert!(kernel.verify_checkpoint(&restored));
```

## Witness Log

Every mutation is recorded in the witness log for auditability:

```rust
let (boot_count, mount_count, mutation_count) = kernel.get_witness_counts();
println!("Boot records: {}", boot_count);
println!("Mount records: {}", mount_count);
println!("Mutation records: {}", mutation_count);

// Serialize for external verification
let log_bytes = kernel.witness_log().to_bytes();
```

## Features

| Feature | Description |
|---------|-------------|
| `std` (default) | Standard library support with heap allocation |
| `alloc` | Heap allocation without full std |
| `sha2` (default) | SHA-256 for proof verification |
| `coherence` | Enable coherence scoring |
| `stats` | Enable statistics collection |

## no_std Support

The crate supports `#![no_std]` environments with the `alloc` feature:

```toml
[dependencies]
ruvix-nucleus = { version = "0.1", default-features = false, features = ["alloc"] }
```

Without `alloc`, the crate uses fixed-size arrays with compile-time limits:
- Maximum 16 vector stores
- Maximum 16 graph stores
- Maximum 64 tasks

## Benchmarks

Run syscall latency benchmarks:

```bash
cargo bench -p ruvix-nucleus
```

Benchmarks include:
- Individual syscall latencies
- Proof tier comparison
- Vector dimension scaling
- Checkpoint operations

## Testing

Run the full test suite:

```bash
cargo test -p ruvix-nucleus
```

Test categories:
- `tests/acceptance.rs` - Full ADR-087 Section 17 acceptance test
- `tests/syscall_tests.rs` - Integration tests for all 12 syscalls
- `tests/deterministic_replay.rs` - Checkpoint and replay verification

## ADR-087 Compliance

This crate implements the RuVix Cognition Kernel as specified in ADR-087:

- Section 3.1: All 12 syscalls implemented
- Section 3.2: All 5 invariants enforced
- Section 17: Full acceptance test passing

## License

MIT OR Apache-2.0
