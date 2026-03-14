# ruvix-types

Core kernel interface types for the RuVix Cognition Kernel (ADR-087).

## Overview

This crate provides all kernel interface types for RuVix. It is designed to be `no_std` compatible with zero external dependencies, ensuring it can be used in both kernel code and RVF component code.

## The Six Kernel Primitives

RuVix has exactly six kernel primitives:

| Primitive | Purpose | Analog |
|-----------|---------|--------|
| **Task** | Unit of concurrent execution with capability set | seL4 TCB |
| **Capability** | Unforgeable typed token granting access to a resource | seL4 capability |
| **Region** | Contiguous memory with access policy | seL4 Untyped + frame |
| **Queue** | Typed ring buffer for inter-task communication | io_uring SQ/CQ |
| **Timer** | Deadline-driven scheduling primitive | POSIX timer_create |
| **Proof** | Cryptographic attestation gating state mutation | Novel (ADR-047) |

## Type Categories

### Handle Types

All kernel objects are referenced through handles:

```rust
use ruvix_types::{TaskHandle, CapHandle, RegionHandle, QueueHandle, VectorStoreHandle, GraphHandle};

let task = TaskHandle::new(1, 0);       // Task ID 1, generation 0
let cap = CapHandle::new(42, 1);        // Capability slot 42, epoch 1
let region = RegionHandle::new(0x1000); // Region at ID 0x1000
```

### Capability Rights

Fine-grained access control through bitflags:

```rust
use ruvix_types::CapRights;

let read_write = CapRights::READ | CapRights::WRITE;
let full_access = CapRights::all();
let read_only = CapRights::READ;
```

Available rights: `READ`, `WRITE`, `GRANT`, `REVOKE`, `PROVE`, `GRANT_ONCE`

### Proof Types

Cryptographic proof tokens for mutation authorization:

```rust
use ruvix_types::{ProofToken, ProofTier, ProofPayload};

// Proof tokens carry tier, nonce, timestamp, and hash
let tier = ProofTier::Reflex;  // <100ns
let tier = ProofTier::Standard; // <100us
let tier = ProofTier::Deep;     // <10ms
```

### Region Policies

Memory access policies:

```rust
use ruvix_types::RegionPolicy;

let immutable = RegionPolicy::Immutable;           // Set once, never modified
let append = RegionPolicy::AppendOnly { max_size: 4096 };  // Only append
let slab = RegionPolicy::Slab { slot_size: 64 };   // Fixed-size slots
```

## Constants

Important constants from ADR-087:

```rust
use ruvix_types::{ATTESTATION_SIZE, MAX_DELEGATION_DEPTH, REFLEX_CACHE_TTL_MS, REFLEX_CACHE_SIZE};

assert_eq!(ATTESTATION_SIZE, 82);           // ADR-047 witness size
assert_eq!(MAX_DELEGATION_DEPTH, 8);        // Section 20.2
assert_eq!(REFLEX_CACHE_TTL_MS, 100);       // 100ms cache TTL
assert_eq!(REFLEX_CACHE_SIZE, 64);          // 64 entry cache
```

## Features

- `std` (default): Enable standard library support
- `alloc`: Enable alloc crate support for heap allocation

## Design Philosophy

- **Zero dependencies**: Pure Rust, no external crates
- **`no_std` compatible**: Works in kernel and embedded contexts
- **`forbid(unsafe_code)`**: Safety guaranteed at compile time
- **Exhaustive docs**: Every public item is documented

## Integration

This crate is the foundation of the RuVix type system. All other ruvix crates depend on it:

- `ruvix-cap`: Capability management using `CapHandle`, `CapRights`
- `ruvix-region`: Memory regions using `RegionHandle`, `RegionPolicy`
- `ruvix-queue`: IPC queues using `QueueHandle`, `MsgPriority`
- `ruvix-proof`: Proof engine using `ProofToken`, `ProofTier`
- `ruvix-vecgraph`: Vector stores using `VectorKey`, `CoherenceMeta`
- `ruvix-sched`: Scheduler using `TaskHandle`, `TaskPriority`

## License

MIT OR Apache-2.0
