# RuVix Cognition Kernel - Phase A Architecture

## Executive Summary

This document specifies the architecture for Phase A (Linux-Hosted Nucleus) of the RuVix Cognition Kernel as defined in ADR-087. It provides detailed guidance for implementation agents including crate dependency graphs, module structures, type definitions, syscall interfaces, and integration strategies with existing RuVector crates.

**Target**: 18-week implementation delivering a Linux-hosted prototype with all 12 syscalls and the complete acceptance test.

---

## 1. Crate Dependency Graph

### 1.1 Dependency DAG

```
                          +------------------+
                          |   ruvix-types    |  (no_std, no deps)
                          |  Core kernel     |
                          |  type definitions|
                          +--------+---------+
                                   |
          +------------------------+------------------------+
          |                        |                        |
          v                        v                        v
+------------------+    +------------------+    +------------------+
|  ruvix-region    |    |   ruvix-queue    |    |    ruvix-cap     |
|  Memory regions  |    |  Queue-based IPC |    |  Capability mgr  |
|  (slab, append,  |    |  (io_uring style)|    |  (seL4 inspired) |
|   immutable)     |    |                  |    |                  |
+--------+---------+    +--------+---------+    +--------+---------+
         |                       |                       |
         |                       |                       |
         +-----------+-----------+-----------+-----------+
                     |
                     v
          +------------------+
          |   ruvix-proof    |
          |  Proof engine    |
          |  (token gen/     |
          |   verification)  |
          +--------+---------+
                   |
                   v
          +------------------+
          | ruvix-vecgraph   |
          | Kernel-resident  |
          | vector & graph   |
          | stores           |
          +--------+---------+
                   |
                   v
          +------------------+
          |   ruvix-sched    |
          | Coherence-aware  |
          | scheduler        |
          +--------+---------+
                   |
                   v
          +------------------+
          |   ruvix-boot     |
          | RVF boot loader  |
          | (manifest parse, |
          |  component mount)|
          +--------+---------+
                   |
                   v
          +------------------+
          |  ruvix-nucleus   |
          | Integration crate|
          | (syscall impl,   |
          |  acceptance test)|
          +------------------+
```

### 1.2 Dependency Matrix

| Crate | Dependencies | no_std | alloc | std |
|-------|-------------|--------|-------|-----|
| `ruvix-types` | none | Required | Optional | Optional |
| `ruvix-region` | `ruvix-types` | Required | Required | Optional |
| `ruvix-queue` | `ruvix-types` | Required | Required | Optional |
| `ruvix-cap` | `ruvix-types` | Required | Optional | Optional |
| `ruvix-proof` | `ruvix-types`, `ruvix-region`, `ruvix-cap` | Required | Required | Optional |
| `ruvix-vecgraph` | `ruvix-types`, `ruvix-region`, `ruvix-proof` | Required | Required | Optional |
| `ruvix-sched` | `ruvix-types`, `ruvix-cap`, `ruvix-vecgraph` | Required | Required | Optional |
| `ruvix-boot` | `ruvix-types`, `ruvix-region`, `ruvix-cap`, `ruvix-proof`, `ruvix-vecgraph` | Required | Required | std feature |
| `ruvix-nucleus` | all above | no_std feature | Required | Required for Phase A |

### 1.3 External Integration Points

| RuVix Crate | Integrates With | Integration Type |
|-------------|-----------------|------------------|
| `ruvix-types` | - | Standalone (no external deps) |
| `ruvix-region` | `memmap2` (std) | Linux mmap backend |
| `ruvix-proof` | `ruvector-verified` | ProofAttestation, ProofEnvironment |
| `ruvix-vecgraph` | `ruvector-core` | HNSW algorithm extraction |
| `ruvix-vecgraph` | `ruvector-graph-transformer` | Graph mutation substrate |
| `ruvix-boot` | `rvf` workspace | RVF manifest parsing |
| `ruvix-boot` | `rvf-crypto` | ML-DSA-65 signature verification |

---

## 2. Module Structure

### 2.1 ruvix-types

```
ruvix-types/
├── Cargo.toml
└── src/
    ├── lib.rs              # Re-exports, feature gates
    ├── task.rs             # TaskHandle, TaskPriority, TaskControlBlock
    ├── capability.rs       # Capability, CapHandle, CapRights, ObjectType
    ├── region.rs           # RegionHandle, RegionPolicy, RegionDescriptor
    ├── queue.rs            # QueueHandle, MsgPriority, QueueDescriptor
    ├── timer.rs            # TimerSpec, TimerHandle
    ├── proof.rs            # ProofToken, ProofTier, ProofAttestation, ProofPayload
    ├── vector.rs           # VectorKey, VectorStoreHandle, CoherenceMeta
    ├── graph.rs            # GraphHandle, GraphMutation, PartitionMeta
    ├── sensor.rs           # SensorDescriptor, SensorType, SubscriptionHandle
    ├── rvf.rs              # RvfComponentId, RvfMountHandle, AttestPayload
    ├── error.rs            # KernelError enum (comprehensive error types)
    ├── handle.rs           # Handle generation, ObjectId, epoch management
    └── constants.rs        # MAX_CAPABILITY_DEPTH, PROOF_CACHE_SIZE, etc.
```

**Key Design Principles**:
- Zero external dependencies
- All types `#[repr(C)]` for FFI compatibility
- All handles contain epoch for use-after-free detection
- Comprehensive `KernelError` covering all syscall failure modes

### 2.2 ruvix-region

```
ruvix-region/
├── Cargo.toml
└── src/
    ├── lib.rs              # Public API, feature gates
    ├── policy.rs           # RegionPolicy enum implementation
    ├── allocator/
    │   ├── mod.rs
    │   ├── slab.rs         # SlabAllocator for fixed-size slots
    │   ├── append.rs       # AppendOnlyAllocator with monotonic cursor
    │   └── immutable.rs    # ImmutableRegion (set once, dedupe)
    ├── backend/
    │   ├── mod.rs
    │   ├── linux.rs        # mmap backend (std feature)
    │   ├── memory.rs       # Pure memory backend (no_std)
    │   └── traits.rs       # RegionBackend trait
    ├── manager.rs          # RegionManager: create, map, unmap, reclaim
    ├── protection.rs       # Access policy enforcement
    └── dedup.rs            # Content-addressable deduplication for immutable
```

**Key Invariants**:
- No demand paging (all physically backed at map time)
- Slab allocations use generation counters per slot
- Append-only regions have monotonic write cursor
- Immutable regions may be deduplicated by content hash

### 2.3 ruvix-queue

```
ruvix-queue/
├── Cargo.toml
└── src/
    ├── lib.rs              # Public API
    ├── ring.rs             # RingBuffer: power-of-2 ring implementation
    ├── kernel_queue.rs     # KernelQueue struct (from ADR-087)
    ├── send.rs             # queue_send implementation
    ├── recv.rs             # queue_recv implementation (blocking + timeout)
    ├── descriptor.rs       # Zero-copy descriptor format
    ├── schema.rs           # WIT type validation at send time
    ├── priority.rs         # MsgPriority ordering
    └── metrics.rs          # Queue statistics, overflow counters
```

**Zero-Copy Protocol**:
1. Sender places descriptor (region_handle, offset, length) in ring
2. Kernel validates: referenced region is `Immutable` or `AppendOnly`
3. Kernel rejects descriptors pointing into `Slab` regions (TOCTOU mitigation)
4. Receiver reads directly from shared region via descriptor

### 2.4 ruvix-cap

```
ruvix-cap/
├── Cargo.toml
└── src/
    ├── lib.rs              # Public API
    ├── capability.rs       # Capability struct, CapRights bitflags
    ├── table.rs            # CapabilityTable: per-task capability storage
    ├── derivation.rs       # DerivationTree: tracks cap_grant chains
    ├── grant.rs            # cap_grant implementation with rights subsetting
    ├── revoke.rs           # Revocation with propagation through derivation tree
    ├── badge.rs            # Capability badging for demultiplexing
    ├── epoch.rs            # Epoch-based capability invalidation
    └── audit.rs            # Periodic capability audit against manifest
```

**Derivation Rules** (from ADR-087 Section 6.2):
- Task can only grant capabilities it holds
- Granted rights must be subset of held rights
- `PROVE` required for `vector_put_proved`/`graph_apply_proved`
- `GRANT` required for `cap_grant`
- Maximum delegation depth: 8 (configurable per-RVF)
- `GRANT_ONCE` right available for non-transitive delegation

### 2.5 ruvix-proof

```
ruvix-proof/
├── Cargo.toml
└── src/
    ├── lib.rs              # Public API
    ├── token.rs            # ProofToken generation and structure
    ├── tier.rs             # ProofTier: Reflex, Standard, Deep
    ├── verify.rs           # Proof verification (hash match, expiry, nonce)
    ├── attestation.rs      # ProofAttestation (82-byte compatible with ADR-047)
    ├── witness_log.rs      # Append-only witness log using ruvix-region
    ├── cache.rs            # Reflex-tier proof cache (64 entries, 100ms TTL)
    ├── composition.rs      # Regional proof composition for mincut partitions
    ├── nonce.rs            # Nonce tracking (single-use enforcement)
    └── integration/
        ├── mod.rs
        └── verified.rs     # Integration with ruvector-verified ProofEnvironment
```

**Proof Lifecycle** (from ADR-087 Section 8.2):
1. Task prepares mutation, computes mutation_hash
2. Task requests proof from Proof Engine (RVF component)
3. Proof Engine evaluates against coherence state, issues ProofToken
4. Task calls syscall with ProofToken
5. Kernel verifies: hash match, not expired, nonce unused, PROVE right held
6. Mutation applied, attestation emitted to witness log

### 2.6 ruvix-vecgraph

```
ruvix-vecgraph/
├── Cargo.toml
└── src/
    ├── lib.rs              # Public API
    ├── vector/
    │   ├── mod.rs
    │   ├── store.rs        # KernelVectorStore (from ADR-087)
    │   ├── hnsw.rs         # HNSW in slab regions (extracted from ruvector-core)
    │   ├── get.rs          # vector_get syscall implementation
    │   ├── put.rs          # vector_put_proved syscall implementation
    │   ├── coherence.rs    # CoherenceMeta co-located with vectors
    │   └── simd.rs         # In-kernel SIMD distance computations
    ├── graph/
    │   ├── mod.rs
    │   ├── store.rs        # KernelGraphStore (from ADR-087)
    │   ├── apply.rs        # graph_apply_proved syscall implementation
    │   ├── mutation.rs     # GraphMutation: AddNode, RemoveNode, AddEdge, etc.
    │   ├── mincut.rs       # MinCut partition metadata
    │   └── partition.rs    # Proof composition boundaries
    └── witness.rs          # Per-store witness regions (append-only)
```

**Vector Store Structure** (from ADR-087 Section 4.3):
```rust
pub struct KernelVectorStore {
    hnsw_region: RegionHandle,       // slab region for HNSW graph nodes
    data_region: RegionHandle,       // slab region for vector data (f32 or quantized)
    witness_region: RegionHandle,    // append-only mutation witness log
    coherence_config: CoherenceConfig,
    proof_policy: ProofPolicy,
    dimensions: u32,
    capacity: u32,
}
```

### 2.7 ruvix-sched

```
ruvix-sched/
├── Cargo.toml
└── src/
    ├── lib.rs              # Public API
    ├── scheduler.rs        # Coherence-aware scheduler implementation
    ├── priority.rs         # SchedulerScore: deadline + novelty - risk
    ├── task.rs             # TaskControlBlock with coherence fields
    ├── deadline.rs         # Earliest-deadline-first within capability partitions
    ├── novelty.rs          # Novelty signal from vector distance
    ├── risk.rs             # Structural risk from pending coherence delta
    ├── partition.rs        # RVF-origin partition scheduling
    ├── preemption.rs       # Bounded preemption at queue boundaries
    └── timer.rs            # timer_wait syscall implementation
```

**Priority Computation** (from ADR-087 Section 5.1):
```rust
fn compute_priority(task: &TaskControlBlock) -> SchedulerScore {
    let deadline_urgency = task.deadline.map_or(0.0, |d| {
        1.0 / (d.saturating_duration_since(now()).as_micros() as f64 + 1.0)
    });
    let novelty_boost = task.pending_input_novelty; // 0.0..1.0
    let risk_penalty = task.pending_coherence_delta.min(0.0).abs() * RISK_WEIGHT;
    SchedulerScore { score: deadline_urgency + novelty_boost - risk_penalty }
}
```

### 2.8 ruvix-boot

```
ruvix-boot/
├── Cargo.toml
└── src/
    ├── lib.rs              # Public API
    ├── manifest.rs         # RVF manifest parsing (4KB boot manifest)
    ├── signature.rs        # ML-DSA-65 signature verification
    ├── component.rs        # Component graph parsing and DAG validation
    ├── memory_schema.rs    # Region declarations from manifest
    ├── proof_policy.rs     # Per-component proof tier requirements
    ├── rollback.rs         # Rollback hook registration
    ├── witness_policy.rs   # Witness log retention, compression, export rules
    ├── mount.rs            # rvf_mount syscall implementation
    ├── capability_dist.rs  # Initial capability distribution per manifest
    ├── wasm.rs             # WASM component loading (Phase A: optional stub)
    └── attestation.rs      # Boot attestation emission
```

**Boot Stages** (from ADR-087 Section 9):
1. Stage 0: Hardware init (N/A for Phase A Linux-hosted)
2. Stage 1: RVF manifest parse + signature verify
3. Stage 2: Kernel object creation (regions, queues, stores)
4. Stage 3: Component mount + capability distribution
5. Stage 4: Boot attestation emission

### 2.9 ruvix-nucleus

```
ruvix-nucleus/
├── Cargo.toml
└── src/
    ├── lib.rs              # Syscall dispatch, public kernel API
    ├── syscall/
    │   ├── mod.rs          # Syscall table (12 syscalls)
    │   ├── task.rs         # task_spawn
    │   ├── cap.rs          # cap_grant
    │   ├── region.rs       # region_map
    │   ├── queue.rs        # queue_send, queue_recv
    │   ├── timer.rs        # timer_wait
    │   ├── rvf.rs          # rvf_mount
    │   ├── attest.rs       # attest_emit
    │   ├── vector.rs       # vector_get, vector_put_proved
    │   ├── graph.rs        # graph_apply_proved
    │   └── sensor.rs       # sensor_subscribe
    ├── init.rs             # Kernel initialization, root task creation
    ├── invariants.rs       # Syscall invariant enforcement
    └── tests/
        ├── mod.rs
        ├── acceptance.rs   # Full acceptance test from ADR-087 Section 17
        └── replay.rs       # Deterministic replay testing
```

---

## 3. Type Definitions for 6 Kernel Primitives

### 3.1 Task

```rust
/// Task handle referencing a kernel task object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct TaskHandle {
    /// Unique object identifier.
    pub object_id: ObjectId,
    /// Epoch for use-after-free detection.
    pub epoch: u64,
}

/// Task scheduling priority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum TaskPriority {
    /// Idle priority (runs when nothing else needs CPU).
    Idle = 0,
    /// Normal priority (default for most tasks).
    Normal = 1,
    /// High priority (elevated scheduling weight).
    High = 2,
    /// Real-time priority (deadline-driven).
    Realtime = 3,
}

/// Task control block (kernel-internal).
#[repr(C)]
pub struct TaskControlBlock {
    /// Task handle for external reference.
    pub handle: TaskHandle,
    /// RVF component ID containing the entry point.
    pub entry: RvfComponentId,
    /// Capability set granted to this task.
    pub capabilities: CapabilityTable,
    /// Base scheduling priority.
    pub priority: TaskPriority,
    /// Optional hard deadline.
    pub deadline: Option<Instant>,
    /// Pending input novelty score (0.0..1.0) for scheduler.
    pub pending_input_novelty: f32,
    /// Pending coherence delta for scheduler risk penalty.
    pub pending_coherence_delta: f32,
    /// RVF mount origin for partition scheduling.
    pub partition_id: RvfMountHandle,
    /// Task state (Running, Ready, Blocked, Terminated).
    pub state: TaskState,
    /// Statistics for profiling.
    pub stats: TaskStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskState {
    /// Task is ready to run.
    Ready = 0,
    /// Task is currently executing.
    Running = 1,
    /// Task is blocked on a queue or timer.
    Blocked = 2,
    /// Task has terminated.
    Terminated = 3,
}
```

### 3.2 Capability

```rust
use bitflags::bitflags;

/// Unique identifier for a kernel object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ObjectId(pub u64);

/// Object type discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ObjectType {
    Task = 0,
    Region = 1,
    Queue = 2,
    Timer = 3,
    VectorStore = 4,
    GraphStore = 5,
    RvfMount = 6,
    WitnessLog = 7,
    TaskFactory = 8,  // Root capability for spawning tasks
}

bitflags! {
    /// Capability rights bitmap.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct CapRights: u32 {
        /// Read access to the object.
        const READ       = 0b0000_0001;
        /// Write access to the object.
        const WRITE      = 0b0000_0010;
        /// Right to grant this capability to other tasks.
        const GRANT      = 0b0000_0100;
        /// Right to revoke derived capabilities.
        const REVOKE     = 0b0000_1000;
        /// Right to execute (for tasks/components).
        const EXECUTE    = 0b0001_0000;
        /// Right to generate proof tokens for this object.
        const PROVE      = 0b0010_0000;
        /// Non-transitive grant (recipient cannot re-grant).
        const GRANT_ONCE = 0b0100_0000;
    }
}

/// A capability is a kernel-managed, unforgeable access token.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Capability {
    /// Unique identifier for the kernel object.
    pub object_id: ObjectId,
    /// The type of kernel object.
    pub object_type: ObjectType,
    /// Rights bitmap: Read, Write, Grant, Revoke, Execute, Prove.
    pub rights: CapRights,
    /// Capability badge for caller-visible demultiplexing.
    pub badge: u64,
    /// Epoch - invalidated if object destroyed or capability revoked.
    pub epoch: u64,
}

/// Handle to a capability in a task's capability table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct CapHandle {
    /// Index into the task's capability table.
    pub index: u32,
    /// Epoch for use-after-free detection.
    pub epoch: u64,
}

impl Capability {
    /// Check if this capability grants the specified rights.
    pub fn has_rights(&self, required: CapRights) -> bool {
        self.rights.contains(required)
    }

    /// Derive a new capability with reduced rights.
    pub fn derive(&self, subset_rights: CapRights) -> Option<Capability> {
        if subset_rights.difference(self.rights).is_empty() {
            Some(Capability {
                object_id: self.object_id,
                object_type: self.object_type,
                rights: subset_rights,
                badge: self.badge,
                epoch: self.epoch,
            })
        } else {
            None // Cannot grant rights we don't hold
        }
    }
}
```

### 3.3 Region

```rust
/// Handle to a memory region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct RegionHandle {
    pub object_id: ObjectId,
    pub epoch: u64,
}

/// Region access policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum RegionPolicy {
    /// Contents are set once at creation and never modified.
    /// The kernel may deduplicate identical immutable regions.
    Immutable,

    /// Contents can only be appended, never overwritten or truncated.
    AppendOnly {
        /// Maximum size in bytes.
        max_size: usize,
    },

    /// Fixed-size slots allocated from a free list.
    /// Slots can be freed and reused. No fragmentation by construction.
    Slab {
        /// Size of each slot in bytes.
        slot_size: usize,
        /// Number of slots.
        slot_count: usize,
    },
}

/// Region descriptor (kernel-internal).
#[repr(C)]
pub struct RegionDescriptor {
    /// Handle for external reference.
    pub handle: RegionHandle,
    /// Memory policy governing access.
    pub policy: RegionPolicy,
    /// Base address in kernel memory (or mmap address for Linux).
    pub base_addr: *mut u8,
    /// Total size in bytes.
    pub size: usize,
    /// For AppendOnly: current write cursor position.
    pub append_cursor: usize,
    /// For Slab: free list head.
    pub slab_free_head: Option<u32>,
    /// Content hash for immutable regions (for deduplication).
    pub content_hash: Option<[u8; 32]>,
    /// Capability handle authorizing this region (for revocation tracking).
    pub auth_cap: CapHandle,
}

/// Slab allocation handle with generation counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SlabSlot {
    /// Slot index within the slab.
    pub index: u32,
    /// Generation counter for use-after-free detection.
    pub generation: u32,
}
```

### 3.4 Queue

```rust
use core::sync::atomic::{AtomicU32, Ordering};

/// Handle to a kernel queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct QueueHandle {
    pub object_id: ObjectId,
    pub epoch: u64,
}

/// Message priority for queue ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum MsgPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Queue descriptor (kernel-internal).
#[repr(C)]
pub struct KernelQueue {
    /// Handle for external reference.
    pub handle: QueueHandle,
    /// Shared region containing the ring buffer.
    pub ring_region: RegionHandle,
    /// Ring size (power of 2).
    pub ring_size: u32,
    /// Submission queue head (sender writes).
    pub sq_head: AtomicU32,
    /// Submission queue tail (kernel advances).
    pub sq_tail: AtomicU32,
    /// Completion queue head (receiver writes).
    pub cq_head: AtomicU32,
    /// Completion queue tail (kernel advances).
    pub cq_tail: AtomicU32,
    /// RVF WIT type ID for message validation.
    pub schema: WitTypeId,
    /// Maximum message size in bytes.
    pub max_msg_size: u32,
    /// Overflow counter (for diagnostics).
    pub overflow_count: AtomicU32,
}

/// WIT type identifier for schema validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct WitTypeId(pub u64);

/// Zero-copy message descriptor.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MsgDescriptor {
    /// Region containing the message data.
    pub region: RegionHandle,
    /// Offset into the region.
    pub offset: u32,
    /// Length of the message data.
    pub length: u32,
    /// Message priority.
    pub priority: MsgPriority,
}
```

### 3.5 Timer

```rust
use core::time::Duration;

/// Handle to a timer object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct TimerHandle {
    pub object_id: ObjectId,
    pub epoch: u64,
}

/// Timer specification (absolute or relative).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub enum TimerSpec {
    /// Wait until an absolute instant.
    Absolute(Instant),
    /// Wait for a relative duration.
    Relative(Duration),
}

/// Kernel instant (nanoseconds since boot).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C)]
pub struct Instant {
    /// Nanoseconds since kernel boot.
    pub nanos: u64,
}

impl Instant {
    /// Get current instant from kernel clock.
    pub fn now() -> Self {
        // Platform-specific implementation
        Instant { nanos: 0 } // Placeholder
    }

    /// Calculate duration until this instant.
    pub fn saturating_duration_since(&self, earlier: Instant) -> Duration {
        if self.nanos > earlier.nanos {
            Duration::from_nanos(self.nanos - earlier.nanos)
        } else {
            Duration::ZERO
        }
    }
}
```

### 3.6 Proof

```rust
/// A proof token authorizing a specific mutation.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct ProofToken {
    /// Hash of the mutation being authorized (SHA3-256).
    pub mutation_hash: [u8; 32],
    /// Proof tier (Reflex, Standard, Deep).
    pub tier: ProofTier,
    /// The proof payload.
    pub payload: ProofPayload,
    /// Expiry - proofs are time-bounded to prevent replay.
    pub valid_until: Instant,
    /// Nonce - prevents proof reuse.
    pub nonce: u64,
}

/// Proof tier from ADR-047.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ProofTier {
    /// Sub-microsecond hash check. For high-frequency vector updates.
    Reflex = 0,
    /// Merkle witness verification. For graph mutations.
    Standard = 1,
    /// Full coherence verification with mincut analysis.
    Deep = 2,
}

/// Proof payload variants.
#[derive(Clone, Debug)]
pub enum ProofPayload {
    /// Fast hash-based proof for Reflex tier.
    HashProof {
        hash: [u8; 32],
    },
    /// Merkle witness for Standard tier.
    MerkleWitness {
        root: [u8; 32],
        path: Vec<[u8; 32]>,
        leaf_index: u32,
    },
    /// Full coherence certificate for Deep tier.
    CoherenceCertificate {
        partition_id: u32,
        coherence_score: f32,
        mincut_witness: Vec<u8>,
    },
}

/// 82-byte attestation compatible with ADR-047 ProofAttestation.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct ProofAttestation {
    /// Epoch when attestation was created.
    pub epoch: u64,              // 8 bytes
    /// Mutation hash that was authorized.
    pub mutation_hash: [u8; 32], // 32 bytes
    /// Hash of the proof token used.
    pub proof_hash: [u8; 32],    // 32 bytes
    /// Timestamp (nanos since boot).
    pub timestamp: u64,          // 8 bytes
    /// Proof tier used.
    pub tier: ProofTier,         // 1 byte
    /// Reserved for alignment.
    pub _reserved: u8,           // 1 byte
}                                // Total: 82 bytes

static_assertions::const_assert_eq!(core::mem::size_of::<ProofAttestation>(), 82);
```

---

## 4. Syscall Interface Design

### 4.1 Syscall Table

The complete RuVix syscall interface consists of exactly 12 syscalls. This is a hard architectural constraint per ADR-087.

| Syscall # | Name | Category | Description |
|-----------|------|----------|-------------|
| 0 | `task_spawn` | Task | Spawn new task with capability set |
| 1 | `cap_grant` | Capability | Grant capability to another task |
| 2 | `region_map` | Region | Map memory region into address space |
| 3 | `queue_send` | Queue | Send typed message to queue |
| 4 | `queue_recv` | Queue | Receive message from queue |
| 5 | `timer_wait` | Timer | Wait until deadline/duration |
| 6 | `rvf_mount` | RVF | Mount signed RVF package |
| 7 | `attest_emit` | Attestation | Emit cryptographic attestation |
| 8 | `vector_get` | Vector | Read vector from kernel store |
| 9 | `vector_put_proved` | Vector | Write vector with proof |
| 10 | `graph_apply_proved` | Graph | Apply graph mutation with proof |
| 11 | `sensor_subscribe` | Sensor | Subscribe to sensor events |

### 4.2 Syscall Signatures

```rust
//! Complete RuVix syscall interface.
//! No syscall may be added without an ADR amendment and ABI version bump.

use crate::types::*;

// --- Syscall 0: Task Management ---

/// Spawn a new task with an explicit capability set.
/// The caller must hold Cap<TaskFactory> to invoke this.
/// Returns a handle to the new task.
pub fn task_spawn(
    entry: RvfComponentId,      // RVF component containing the entry point
    caps: &[CapHandle],         // capabilities granted to the new task
    priority: TaskPriority,     // base scheduling priority
    deadline: Option<Duration>, // optional hard deadline
) -> Result<TaskHandle, KernelError>;

// --- Syscall 1: Capability Management ---

/// Grant a capability to another task.
/// The granting task must hold the capability with the Grant right.
/// Capabilities are unforgeable kernel objects.
pub fn cap_grant(
    target: TaskHandle,
    cap: CapHandle,
    rights: CapRights,          // subset of caller's rights on this cap
) -> Result<CapHandle, KernelError>;

// --- Syscall 2: Region Memory ---

/// Map a memory region into the calling task's address space.
/// Region policy (immutable, append-only, slab) is set at creation.
pub fn region_map(
    size: usize,
    policy: RegionPolicy,       // Immutable | AppendOnly | Slab
    cap: CapHandle,             // capability authorizing the mapping
) -> Result<RegionHandle, KernelError>;

// --- Syscalls 3-4: Queue IPC ---

/// Send a typed message to a queue.
/// The message is zero-copy if sender and receiver share a region.
pub fn queue_send(
    queue: QueueHandle,
    msg: &[u8],                 // serialized message (RVF wire format)
    priority: MsgPriority,
) -> Result<(), KernelError>;

/// Receive a message from a queue.
/// Blocks until a message is available or the timeout expires.
pub fn queue_recv(
    queue: QueueHandle,
    buf: &mut [u8],
    timeout: Duration,
) -> Result<usize, KernelError>;

// --- Syscall 5: Timer ---

/// Wait until a deadline or duration elapses.
/// The scheduler may preempt the task and resume it when the timer fires.
pub fn timer_wait(
    deadline: TimerSpec,        // Absolute(Instant) | Relative(Duration)
) -> Result<(), KernelError>;

// --- Syscall 6: RVF Boot ---

/// Mount a signed RVF package into the component namespace.
/// The kernel verifies the package signature, proof policy, and
/// witness log policy before making components available.
pub fn rvf_mount(
    rvf_data: &[u8],           // raw RVF bytes (or region handle)
    mount_point: &str,          // namespace path (e.g., "/agents/planner")
    cap: CapHandle,             // capability authorizing the mount
) -> Result<RvfMountHandle, KernelError>;

// --- Syscall 7: Attestation ---

/// Emit a cryptographic attestation for a completed operation.
/// The attestation is appended to the kernel's witness log.
/// Returns the 82-byte attestation (compatible with ADR-047).
pub fn attest_emit(
    operation: &AttestPayload,  // what was done
    proof: &ProofToken,         // the proof that authorized it
) -> Result<ProofAttestation, KernelError>;

// --- Syscalls 8-9: Vector Kernel Objects ---

/// Read a vector from a kernel-resident vector store.
/// Returns the vector data and its coherence metadata.
pub fn vector_get(
    store: VectorStoreHandle,
    key: VectorKey,
) -> Result<(Vec<f32>, CoherenceMeta), KernelError>;

/// Write a vector to a kernel-resident vector store.
/// Requires a valid proof token -- no proof, no mutation.
pub fn vector_put_proved(
    store: VectorStoreHandle,
    key: VectorKey,
    data: &[f32],
    proof: ProofToken,
) -> Result<ProofAttestation, KernelError>;

// --- Syscall 10: Graph Kernel Objects ---

/// Apply a graph mutation (add/remove node/edge, update weight).
/// Requires a valid proof token -- no proof, no mutation.
pub fn graph_apply_proved(
    graph: GraphHandle,
    mutation: &GraphMutation,
    proof: ProofToken,
) -> Result<ProofAttestation, KernelError>;

// --- Syscall 11: Sensor / Perception ---

/// Subscribe to a sensor stream (RuView perception events).
/// Events are delivered to the specified queue.
pub fn sensor_subscribe(
    sensor: SensorDescriptor,   // identifies the sensor
    target_queue: QueueHandle,
    cap: CapHandle,
) -> Result<SubscriptionHandle, KernelError>;
```

### 4.3 Syscall Invariants

Every syscall satisfies these invariants from ADR-087 Section 3.2:

1. **Capability-gated**: No syscall succeeds without an appropriate capability handle.
2. **Proof-required for mutation**: `vector_put_proved`, `graph_apply_proved`, and `rvf_mount` require proof tokens.
3. **Bounded latency**: Every syscall has a worst-case execution time expressible in cycles.
4. **Witness-logged**: Every successful mutating syscall emits a witness record.
5. **No allocation in syscall path**: Pre-allocated structures only.

### 4.4 Error Types

```rust
/// Comprehensive kernel error enum.
#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum KernelError {
    // --- Capability Errors (0x01XX) ---
    /// The specified capability handle is invalid.
    InvalidCapability = 0x0100,
    /// The capability does not grant the required rights.
    InsufficientRights = 0x0101,
    /// The capability epoch has expired (object destroyed or revoked).
    CapabilityExpired = 0x0102,
    /// Maximum capability delegation depth exceeded.
    DelegationDepthExceeded = 0x0103,
    /// Cannot grant rights not held by the caller.
    RightsEscalation = 0x0104,

    // --- Region Errors (0x02XX) ---
    /// Out of memory when mapping a region.
    OutOfMemory = 0x0200,
    /// Invalid region policy for the requested operation.
    InvalidRegionPolicy = 0x0201,
    /// Append-only region is full.
    RegionFull = 0x0202,
    /// Slab slot use-after-free detected.
    SlabUseAfterFree = 0x0203,
    /// Invalid slab slot index.
    InvalidSlabSlot = 0x0204,

    // --- Queue Errors (0x03XX) ---
    /// Queue is full, message not sent.
    QueueFull = 0x0300,
    /// Queue is empty (should not occur with blocking recv).
    QueueEmpty = 0x0301,
    /// Message exceeds maximum size.
    MessageTooLarge = 0x0302,
    /// Timeout expired before message arrived.
    Timeout = 0x0303,
    /// Message schema validation failed.
    SchemaValidationFailed = 0x0304,
    /// Zero-copy descriptor points to Slab region (TOCTOU risk).
    InvalidDescriptorRegion = 0x0305,

    // --- Proof Errors (0x04XX) ---
    /// Proof token mutation hash does not match.
    ProofHashMismatch = 0x0400,
    /// Proof token has expired.
    ProofExpired = 0x0401,
    /// Proof nonce has already been used.
    ProofNonceReused = 0x0402,
    /// Caller does not hold PROVE right on the object.
    ProofRightMissing = 0x0403,
    /// Proof verification failed (invalid Merkle witness, etc.).
    ProofVerificationFailed = 0x0404,
    /// Proof tier too low for the operation.
    ProofTierInsufficient = 0x0405,

    // --- RVF Errors (0x05XX) ---
    /// RVF signature verification failed.
    SignatureInvalid = 0x0500,
    /// RVF manifest parsing failed.
    ManifestParseError = 0x0501,
    /// RVF component graph is invalid (cycle detected).
    InvalidComponentGraph = 0x0502,
    /// Mount point already occupied.
    MountPointOccupied = 0x0503,

    // --- Task Errors (0x06XX) ---
    /// Invalid task handle.
    InvalidTask = 0x0600,
    /// Task has terminated.
    TaskTerminated = 0x0601,
    /// Caller does not hold TaskFactory capability.
    TaskFactoryRequired = 0x0602,

    // --- Vector/Graph Errors (0x07XX) ---
    /// Vector key not found in store.
    VectorNotFound = 0x0700,
    /// Vector dimension mismatch.
    DimensionMismatch = 0x0701,
    /// Vector store capacity exceeded.
    StoreCapacityExceeded = 0x0702,
    /// Invalid graph mutation.
    InvalidGraphMutation = 0x0703,
    /// Graph node not found.
    GraphNodeNotFound = 0x0704,

    // --- Sensor Errors (0x08XX) ---
    /// Invalid sensor descriptor.
    InvalidSensor = 0x0800,
    /// Sensor not available.
    SensorUnavailable = 0x0801,

    // --- Internal Errors (0xFFXX) ---
    /// Internal kernel error (should not occur).
    InternalError = 0xFF00,
}
```

---

## 5. Integration Strategy with Existing RuVector Crates

### 5.1 ruvector-verified Integration

**Source Crate**: `crates/ruvector-verified`

**Integration Points**:
- `ProofEnvironment` -> `ruvix-proof::ProofEngine` foundation
- `ProofAttestation` -> `ruvix-types::ProofAttestation` (82-byte compat)
- `VerifiedOp<T>` -> Kernel operation wrapper pattern

**Extraction Strategy**:
```rust
// In ruvix-proof/src/integration/verified.rs

use ruvector_verified::{ProofEnvironment, ProofAttestation as VerifiedAttestation};

/// Bridge between ruvector-verified and ruvix-proof
pub struct VerifiedProofBridge {
    env: ProofEnvironment,
}

impl VerifiedProofBridge {
    pub fn new() -> Self {
        Self { env: ProofEnvironment::new() }
    }

    /// Convert ruvix ProofToken to verification request
    pub fn verify(&mut self, token: &ProofToken) -> Result<u32, KernelError> {
        let proof_id = self.env.alloc_term();
        // Verify proof against mutation_hash
        // ...
        Ok(proof_id)
    }

    /// Emit attestation using ruvector-verified format
    pub fn emit_attestation(&self, token: &ProofToken) -> ProofAttestation {
        ProofAttestation {
            epoch: self.env.terms_allocated() as u64,
            mutation_hash: token.mutation_hash,
            proof_hash: self.hash_proof(token),
            timestamp: Instant::now().nanos,
            tier: token.tier,
            _reserved: 0,
        }
    }
}
```

### 5.2 ruvector-core Integration

**Source Crate**: `crates/ruvector-core`

**Integration Points**:
- HNSW algorithm -> `ruvix-vecgraph::hnsw`
- Distance metrics -> `ruvix-vecgraph::simd`
- Quantization -> Not in kernel (per ADR-087: quantization in-component)

**Extraction Strategy**:
```rust
// In ruvix-vecgraph/src/vector/hnsw.rs

/// HNSW index operating on slab-allocated regions
/// Extracted from ruvector-core but uses kernel regions instead of Vec<T>
pub struct KernelHnsw {
    /// Slab region for HNSW graph nodes
    nodes_region: RegionHandle,
    /// Slab region for neighbor lists
    neighbors_region: RegionHandle,
    /// Configuration
    config: HnswConfig,
    /// Entry point for search
    entry_point: Option<SlabSlot>,
    /// Maximum layer
    max_layer: usize,
}

impl KernelHnsw {
    /// Insert a vector into the index (requires proof)
    pub fn insert_proved(
        &mut self,
        vector_slot: SlabSlot,  // Slot in data region
        proof: &ProofToken,
    ) -> Result<SlabSlot, KernelError> {
        // Verify proof first
        // ...
        // Allocate HNSW node from slab
        // ...
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, KernelError> {
        // HNSW search using slab-allocated structures
        // ...
    }
}
```

### 5.3 ruvector-graph-transformer Integration

**Source Crate**: `crates/ruvector-graph-transformer`

**Integration Points**:
- Graph mutation substrate -> `ruvix-vecgraph::graph`
- Proof-gated mutation -> Already uses `ruvector-verified`

**Extraction Strategy**:
```rust
// In ruvix-vecgraph/src/graph/mutation.rs

/// Graph mutation operations
#[derive(Clone, Debug)]
pub enum GraphMutation {
    /// Add a node with initial state vector
    AddNode {
        node_id: NodeId,
        state: Vec<f32>,
    },
    /// Remove a node (and all incident edges)
    RemoveNode {
        node_id: NodeId,
    },
    /// Add an edge with weight
    AddEdge {
        from: NodeId,
        to: NodeId,
        weight: f64,
    },
    /// Remove an edge
    RemoveEdge {
        from: NodeId,
        to: NodeId,
    },
    /// Update edge weight
    UpdateWeight {
        from: NodeId,
        to: NodeId,
        new_weight: f64,
    },
    /// Update node state vector
    UpdateState {
        node_id: NodeId,
        new_state: Vec<f32>,
    },
}

/// Compute mutation hash for proof verification
impl GraphMutation {
    pub fn hash(&self) -> [u8; 32] {
        use sha3::{Sha3_256, Digest};
        let mut hasher = Sha3_256::new();
        // Serialize and hash the mutation
        // ...
        hasher.finalize().into()
    }
}
```

### 5.4 cognitum-gate-kernel Integration

**Source Crate**: `crates/cognitum-gate-kernel`

**Integration Points**:
- 256-tile coherence fabric -> `ruvix-sched` coherence scoring
- Mincut partitions -> `ruvix-vecgraph::graph::partition`

**Extraction Strategy**:
```rust
// In ruvix-sched/src/coherence.rs

/// Coherence configuration from cognitum-gate-kernel
pub struct CoherenceConfig {
    /// Number of tiles in the coherence fabric
    pub tile_count: usize,
    /// Coherence threshold for scheduler penalty
    pub risk_threshold: f32,
    /// Weight for risk penalty in priority computation
    pub risk_weight: f32,
}

/// Compute coherence delta for a pending mutation
pub fn compute_coherence_delta(
    current_energy: f64,
    mutation: &GraphMutation,
    partition: &PartitionMeta,
) -> f32 {
    // Port from cognitum-gate-kernel
    // ...
}
```

### 5.5 rvf Workspace Integration

**Source Crate**: `crates/rvf/` (workspace with 17+ sub-crates)

**Integration Points**:
- `rvf-manifest` -> `ruvix-boot::manifest`
- `rvf-crypto` -> `ruvix-boot::signature` (ML-DSA-65)
- `rvf-wire` -> Wire format for queue messages

**Extraction Strategy**:
```rust
// In ruvix-boot/src/manifest.rs

use rvf_manifest::{Manifest, ComponentGraph, MemorySchema, ProofPolicy};

/// Parse RVF boot manifest (4KB)
pub fn parse_manifest(data: &[u8]) -> Result<BootManifest, KernelError> {
    if data.len() < 4096 {
        return Err(KernelError::ManifestParseError);
    }

    let manifest = Manifest::parse(&data[..4096])
        .map_err(|_| KernelError::ManifestParseError)?;

    Ok(BootManifest {
        component_graph: manifest.component_graph,
        memory_schema: manifest.memory_schema,
        proof_policy: manifest.proof_policy,
        witness_log_policy: manifest.witness_log_policy,
        rollback_hooks: manifest.rollback_hooks,
    })
}

// In ruvix-boot/src/signature.rs

use rvf_crypto::mldsa65;

/// Verify ML-DSA-65 signature on RVF package
pub fn verify_signature(
    rvf_data: &[u8],
    signature: &[u8],
    public_key: &[u8],
) -> Result<(), KernelError> {
    mldsa65::verify(public_key, rvf_data, signature)
        .map_err(|_| KernelError::SignatureInvalid)
}
```

### 5.6 ruvector-snapshot Integration

**Source Crate**: `crates/ruvector-snapshot`

**Integration Points**:
- Region snapshots -> Deterministic replay support
- Checkpoint/restore -> Acceptance test requirement

**Extraction Strategy**:
```rust
// In ruvix-nucleus/src/checkpoint.rs

/// Checkpoint the kernel state for deterministic replay
pub fn checkpoint(
    regions: &[RegionHandle],
    witness_log: RegionHandle,
) -> Result<Checkpoint, KernelError> {
    let mut region_snapshots = Vec::new();
    for region in regions {
        let snapshot = snapshot_region(*region)?;
        region_snapshots.push(snapshot);
    }

    let witness_log_snapshot = snapshot_region(witness_log)?;

    Ok(Checkpoint {
        regions: region_snapshots,
        witness_log: witness_log_snapshot,
        timestamp: Instant::now(),
    })
}

/// Restore from checkpoint for replay
pub fn restore(checkpoint: &Checkpoint) -> Result<(), KernelError> {
    for snapshot in &checkpoint.regions {
        restore_region(snapshot)?;
    }
    restore_region(&checkpoint.witness_log)?;
    Ok(())
}
```

---

## 6. Acceptance Test Specification

From ADR-087 Section 17, the acceptance test is the single gate for Phase A completion.

### 6.1 Test Implementation

```rust
// In ruvix-nucleus/src/tests/acceptance.rs

#[test]
fn acceptance_test_phase_a() {
    // GIVEN: A signed RVF package "acceptance.rvf" containing:
    //   - sensor_adapter component (simulated)
    //   - vector_store component (kernel-resident, capacity=100)
    //   - proof_engine component
    //   - writer component
    //   - reader component
    //   - Proof policy: Standard tier for all mutations
    //   - Witness log policy: retain all, no compression

    let rvf_data = include_bytes!("../../testdata/acceptance.rvf");

    // Initialize kernel
    let mut kernel = RuvixNucleus::new();
    kernel.init_root_task();

    // Step 1: rvf_mount
    let mount_handle = kernel
        .rvf_mount(rvf_data, "/test", kernel.root_cap())
        .expect("mount should succeed");

    // Step 2: sensor_adapter emits one PerceptionEvent to queue
    let perception_event = PerceptionEvent::simulated();
    let event_queue = kernel.get_queue(mount_handle, "sensor_events");
    kernel.queue_send(event_queue, &perception_event.serialize(), MsgPriority::Normal)
        .expect("send should succeed");

    // Step 3: writer receives event
    let mut buf = [0u8; 4096];
    let len = kernel.queue_recv(event_queue, &mut buf, Duration::from_secs(1))
        .expect("recv should succeed");
    let event = PerceptionEvent::deserialize(&buf[..len]);

    // Step 4: writer computes embedding vector
    let embedding = compute_embedding(&event);

    // Step 5: writer requests proof from proof_engine
    let mutation_hash = compute_mutation_hash(&embedding);
    let proof = kernel.request_proof(mutation_hash, ProofTier::Standard)
        .expect("proof should be issued");

    // Step 6: vector_put_proved
    let vector_store = kernel.get_vector_store(mount_handle);
    let key = VectorKey::new("test_vector_001");
    let attestation = kernel
        .vector_put_proved(vector_store, key.clone(), &embedding, proof)
        .expect("vector_put_proved should succeed");

    // Verify attestation is 82 bytes
    assert_eq!(std::mem::size_of_val(&attestation), 82);

    // Step 7: reader calls vector_get
    let (retrieved_vector, coherence_meta) = kernel
        .vector_get(vector_store, key.clone())
        .expect("vector_get should succeed");
    assert_eq!(retrieved_vector, embedding);

    // Step 8: System checkpoints
    let checkpoint = kernel.checkpoint()
        .expect("checkpoint should succeed");

    // Step 9: System shuts down
    kernel.shutdown();

    // Step 10: System restarts from checkpoint
    let mut kernel2 = RuvixNucleus::new();
    kernel2.restore(&checkpoint)
        .expect("restore should succeed");

    // Step 11: System replays witness log
    kernel2.replay_witness_log()
        .expect("replay should succeed");

    // Step 12: reader calls vector_get again
    let vector_store2 = kernel2.get_vector_store_by_path("/test");
    let (retrieved_vector2, coherence_meta2) = kernel2
        .vector_get(vector_store2, key.clone())
        .expect("vector_get after replay should succeed");

    // THEN: Vectors are EXACTLY the same
    assert_eq!(retrieved_vector2, retrieved_vector);
    assert_eq!(coherence_meta2, coherence_meta);

    // Verify witness log contains exactly 3 attestations:
    // 1 boot + 1 mount + 1 mutation
    let witness_count = kernel2.witness_log_count();
    assert_eq!(witness_count, 3);

    // Verify no proof-less mutation was accepted
    assert!(kernel2.proof_less_mutation_count() == 0);

    // Total replay time < 2x original execution time
    let replay_time = kernel2.last_replay_duration();
    let original_time = checkpoint.creation_duration();
    assert!(replay_time < original_time * 2);
}

#[test]
fn acceptance_test_proof_rejection() {
    let mut kernel = RuvixNucleus::new();
    kernel.init_root_task();

    let vector_store = kernel.create_vector_store(100, 128);

    // Attempt store without proof -> should fail
    let embedding = vec![0.0f32; 128];
    let key = VectorKey::new("test_no_proof");

    // This should not compile - vector_put_proved requires ProofToken
    // kernel.vector_put(vector_store, key, &embedding);

    // Attempt with expired proof -> should fail
    let expired_proof = ProofToken {
        mutation_hash: [0u8; 32],
        tier: ProofTier::Standard,
        payload: ProofPayload::HashProof { hash: [0u8; 32] },
        valid_until: Instant { nanos: 0 }, // Already expired
        nonce: 1,
    };

    let result = kernel.vector_put_proved(vector_store, key, &embedding, expired_proof);
    assert_eq!(result, Err(KernelError::ProofExpired));
}
```

---

## 7. Implementation Timeline

| Week | Milestone | Crate(s) | Key Deliverables |
|------|-----------|----------|------------------|
| 1-2 | Core types | `ruvix-types` | All 6 primitive types, KernelError, handles |
| 3-4 | Region manager | `ruvix-region` | Slab, AppendOnly, Immutable regions on Linux mmap |
| 5-6 | Queue IPC | `ruvix-queue` | io_uring-style ring buffers, zero-copy descriptors |
| 7-8 | Capability manager | `ruvix-cap` | Capability table, derivation tree, revocation |
| 9-10 | Proof engine | `ruvix-proof` | Token gen/verify, witness log, ruvector-verified integration |
| 11-12 | Vector/Graph stores | `ruvix-vecgraph` | Kernel HNSW, graph mutations, proof-gated writes |
| 13-14 | Scheduler | `ruvix-sched` | Coherence-aware priority, deadline scheduling |
| 15-16 | RVF boot | `ruvix-boot` | Manifest parsing, ML-DSA-65 verify, component mount |
| 17-18 | Integration + ABI freeze | `ruvix-nucleus` | All 12 syscalls, acceptance test passes |

---

## 8. Build Configuration

### 8.1 Workspace Cargo.toml

```toml
[workspace]
resolver = "2"
members = [
    "ruvix-types",
    "ruvix-region",
    "ruvix-queue",
    "ruvix-cap",
    "ruvix-proof",
    "ruvix-vecgraph",
    "ruvix-sched",
    "ruvix-boot",
    "ruvix-nucleus",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.77"
license = "MIT OR Apache-2.0"
repository = "https://github.com/ruvnet/ruvector"
authors = ["ruv.io", "RuVector Team"]

[workspace.dependencies]
# Internal
ruvix-types = { path = "ruvix-types" }
ruvix-region = { path = "ruvix-region" }
ruvix-queue = { path = "ruvix-queue" }
ruvix-cap = { path = "ruvix-cap" }
ruvix-proof = { path = "ruvix-proof" }
ruvix-vecgraph = { path = "ruvix-vecgraph" }
ruvix-sched = { path = "ruvix-sched" }
ruvix-boot = { path = "ruvix-boot" }

# External - RuVector ecosystem
ruvector-verified = { path = "../ruvector-verified", features = ["ultra"] }
ruvector-core = { path = "../ruvector-core", default-features = false }
ruvector-graph-transformer = { path = "../ruvector-graph-transformer" }
ruvector-mincut = { path = "../ruvector-mincut", default-features = false }
ruvector-coherence = { path = "../ruvector-coherence" }
ruvector-snapshot = { path = "../ruvector-snapshot" }

# External - RVF
rvf-manifest = { path = "../rvf/rvf-manifest" }
rvf-crypto = { path = "../rvf/rvf-crypto" }
rvf-wire = { path = "../rvf/rvf-wire" }

# External - General
bitflags = "2.4"
static_assertions = "1.1"
sha3 = { version = "0.10", default-features = false }
thiserror = "1.0"

# Linux-specific (Phase A)
memmap2 = { version = "0.9", optional = true }
libc = { version = "0.2", optional = true }

[workspace.features]
default = ["linux-hosted"]
linux-hosted = ["memmap2", "libc"]
no_std = []
```

### 8.2 Individual Crate Cargo.toml Template

```toml
# ruvix-types/Cargo.toml
[package]
name = "ruvix-types"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
description = "Core type definitions for RuVix Cognition Kernel"

[dependencies]
bitflags = { workspace = true }
static_assertions = { workspace = true }

[features]
default = []
std = []
alloc = []

[lib]
# Ensure no_std by default
```

---

## 9. Quality Attributes

### 9.1 Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Syscall latency (simple) | <1us | Capability check + operation |
| Proof verification (Reflex) | <100ns | Hash comparison only |
| Proof verification (Standard) | <10us | Merkle path verification |
| Proof verification (Deep) | <10ms | Full coherence analysis |
| Vector distance (SIMD) | <100ns | 512-dim f32 vectors |
| Queue send/recv (zero-copy) | <500ns | Descriptor only, no data copy |
| Region map | <10us | mmap + capability table update |

### 9.2 Resource Limits

| Resource | Limit | Configuration |
|----------|-------|---------------|
| Capability delegation depth | 8 | Per-RVF configurable |
| Reflex proof cache entries | 64 | Compile-time constant |
| Reflex proof TTL | 100ms | Compile-time constant |
| Max queue ring size | 64KB | Per-queue configurable |
| Max message size | 4KB | Per-queue configurable |
| Witness log entry | 82 bytes | Fixed (ADR-047 compat) |

### 9.3 Security Properties

1. **No ambient authority**: Every syscall requires explicit capability
2. **Proof-gated mutation**: No state change without cryptographic proof
3. **Witness logging**: Complete audit trail for all mutations
4. **Epoch-based invalidation**: Use-after-free detection on all handles
5. **TOCTOU prevention**: Zero-copy only from immutable/append regions
6. **Delegation limits**: Maximum depth prevents capability explosion

---

## 10. References

- ADR-087: RuVix Cognition Kernel (source specification)
- ADR-047: Proof-Gated Mutation Protocol
- ADR-029: RVF Canonical Binary Format
- ADR-030: RVF Cognitive Container
- ADR-014: Coherence Engine Architecture
- seL4 Microkernel (capability model inspiration)
- io_uring (queue design inspiration)

---

*Document Version: 1.0.0*
*Last Updated: 2026-03-14*
*Architecture Reference: ADR-087*
