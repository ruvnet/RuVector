# ADR-087: RuVix Cognition Kernel — An Operating System for the Agentic Age

## Status

**Accepted** — Phase A Implemented

## Date

2026-03-08 (Proposed)
2026-03-14 (Phase A Implemented)

## Implementation Status

### Phase A: Linux-Hosted Nucleus ✅ COMPLETE

| Crate | Status | Tests | Description |
|-------|--------|-------|-------------|
| `ruvix-types` | ✅ | 63 | Core types: 6 primitives, handles, proof tokens, capabilities |
| `ruvix-region` | ✅ | 51 | Memory regions: Immutable, AppendOnly, Slab policies |
| `ruvix-queue` | ✅ | 47 | io_uring-style ring buffers, zero-copy IPC |
| `ruvix-cap` | ✅ | 54 | seL4-inspired capability manager, derivation trees |
| `ruvix-proof` | ✅ | 73 | 3-tier proof engine: Reflex <100ns, Standard <100μs, Deep <10ms |
| `ruvix-sched` | ✅ | 39 | Coherence-aware scheduler with novelty boosting |
| `ruvix-boot` | ✅ | 59 | 5-stage RVF boot loader with ML-DSA-65 signatures |
| `ruvix-vecgraph` | ✅ | 55 | Kernel-resident vector/graph stores with HNSW |
| `ruvix-nucleus` | ✅ | 319 | Unified kernel: 12 syscalls, checkpoint/replay |

**Total: 760 tests passing**

### Security Invariants Implemented

- **SEC-001**: Boot signature failure → PANIC (no fallback)
- **SEC-002**: Proof cache with 100ms TTL, single-use nonces, max 64 entries
- **SEC-003**: Capability delegation depth limit (max 8)
- **SEC-004**: TOCTOU protection for zero-copy IPC

### Phase B: Bare Metal AArch64 — Pending

Target: QEMU virt, Raspberry Pi 4/5

## Deciders

ruv

## Related

- ADR-029 RVF canonical binary format
- ADR-030 RVF cognitive container / self-booting vector files
- ADR-042 Security RVF AIDefence TEE
- ADR-047 Proof-gated mutation protocol
- ADR-014 Coherence engine architecture
- ADR-061 Reasoning kernel architecture
- ADR-006 Unified memory pool and paging strategy
- ADR-005 WASM runtime integration
- ADR-032 RVF WASM integration
- `crates/cognitum-gate-kernel/` — no_std WASM coherence kernel
- `crates/ruvector-verified/` — ProofGate<T>, ProofEnvironment, attestation chain
- `crates/rvf/` — RVF format implementation
- `crates/ruvector-core/` — HNSW vector database
- `crates/ruvector-graph-transformer/` — graph mutation substrate

---

## 1. Context

### 1.1 The Problem with Conventional Operating Systems

Every major operating system today — Linux, Windows, macOS, seL4, Zephyr — was designed for a world where the primary compute actor is a human being operating through a process abstraction. The process model assumes:

1. A single sequential instruction stream per thread
2. File-based persistent state (byte streams with names)
3. POSIX IPC semantics (pipes, sockets, signals)
4. Discretionary or mandatory access control based on user identity
5. A scheduler optimized for interactive latency or batch throughput

None of these assumptions hold for agentic workloads. An AI agent does not think in files. It thinks in vectors, graphs, proofs, and causal event streams. It does not need fork/exec. It needs capability-gated task spawning with proof-of-intent. It does not communicate through byte pipes. It communicates through typed semantic queues where every message carries a coherence score and a witness hash.

Running agentic workloads on Linux is like running a modern web application on a mainframe batch scheduler — technically possible, structurally wrong.

### 1.2 What RuVector Already Provides

The RuVector ecosystem (107 crates, 50+ npm packages) has incrementally built every primitive needed for a cognition kernel, but scattered across userspace libraries:

- **RVF** (ADR-029): A self-describing binary format with segments for vectors, graphs, WASM microkernels, cryptographic witnesses, and TEE attestation quotes.
- **Cognitum Gate Kernel** (`cognitum-gate-kernel`): A 256-tile no_std WASM coherence fabric operating on mincut partitions.
- **Proof-Gated Mutation** (ADR-047): `ProofGate<T>` enforcing "no proof, no mutation" at the type level with 82-byte attestation witnesses.
- **Coherence Engine** (ADR-014): Structural consistency scoring that replaces probabilistic confidence with graph-theoretic guarantees.
- **Cognitive Containers** (ADR-030): Self-booting RVF files that carry their own execution kernel, enabling single-file microservices.
- **Reasoning Kernel** (ADR-061): A brain-augmented reasoning protocol with witnessable artifacts at every step.
- **RVF Security Hardening** (ADR-042): TEE attestation, AIDefence layers, EBPF policy enforcement, and witness chain audit.

These primitives exist. They work. But they run on top of Linux, mediated by POSIX, paying the abstraction tax at every boundary. RuVix promotes them to first-class kernel resources.

### 1.3 Why Not Just Use seL4/Zephyr/Unikernel

**seL4** proves that a capability kernel with formal verification is viable (8,700 lines of C, fully verified). But seL4 has no concept of vectors, graphs, coherence, or proofs. Adding these would require reimplementing the entire RuVector stack as userspace servers communicating through IPC — reintroducing the overhead we want to eliminate.

**Zephyr/FreeRTOS** target microcontrollers with cooperative/preemptive scheduling. They have no memory protection, no capability model, and no concept of attestation.

**Unikernels (Hermit, Unikraft)** eliminate the OS/application boundary but retain POSIX semantics. They make Linux faster, not different.

RuVix is different: it has six kernel primitives, twelve syscalls, and every mutation is proof-gated. Everything else — including the entire AgentDB intelligence runtime, Claude Code adapters, and RuView perception pipeline — lives above the kernel in RVF component space.

---

## 2. Decision

### 2.1 Core Thesis

RuVix is a cognition kernel. It is not a general-purpose operating system. It has exactly six kernel primitives:

| Primitive | Purpose | Analog |
|-----------|---------|--------|
| **Task** | Unit of concurrent execution with capability set | seL4 TCB |
| **Capability** | Unforgeable typed token granting access to a resource | seL4 capability |
| **Region** | Contiguous memory with access policy (immutable, append-only, slab) | seL4 Untyped + frame |
| **Queue** | Typed ring buffer for inter-task communication | io_uring SQ/CQ |
| **Timer** | Deadline-driven scheduling primitive | POSIX timer_create |
| **Proof** | Cryptographic attestation gating state mutation | Novel (from ADR-047) |

Everything else — file systems, networking, device drivers, vector indexes, graph engines, AI inference — is an RVF component running in user space, communicating through queues, accessing resources through capabilities.

### 2.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AGENT CONTROL PLANE                           │
│  Claude Code │ Codex │ Custom Agents │ AgentDB Planner Runtime     │
├─────────────────────────────────────────────────────────────────────┤
│                      RVF COMPONENT SPACE                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ RuView   │ │ AgentDB  │ │ RuVLLM   │ │ Network  │ ...          │
│  │ Percep.  │ │ Intelli. │ │ Infer.   │ │ Stack    │              │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘              │
│       │queue        │queue       │queue        │queue               │
├───────┴─────────────┴────────────┴─────────────┴───────────────────┤
│                      RUVIX COGNITION KERNEL                        │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐    │
│  │ Capability Mgr │  │ Queue IPC      │  │ Coherence-Aware    │    │
│  │ (cap_grant,    │  │ (queue_send,   │  │ Scheduler          │    │
│  │  cap_revoke)   │  │  queue_recv)   │  │ (deadline+novelty  │    │
│  │                │  │  io_uring ring │  │  +structural risk) │    │
│  └────────────────┘  └────────────────┘  └────────────────────┘    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐    │
│  │ Region Memory  │  │ Proof Engine   │  │ Vector/Graph       │    │
│  │ (slabs, immut, │  │ (attest_emit,  │  │ Kernel Objects     │    │
│  │  append-only)  │  │  proof_verify) │  │ (vector_get/put,   │    │
│  │                │  │                │  │  graph_apply)      │    │
│  └────────────────┘  └────────────────┘  └────────────────────┘    │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ RVF Boot Loader — mounts signed RVF packages as root      │    │
│  └────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                      HARDWARE / HYPERVISOR                          │
│  AArch64 (primary) │ x86_64 (secondary) │ WASM (hosted)           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Syscall Surface

RuVix exposes exactly 12 syscalls. This is a hard architectural constraint. New functionality is added through RVF components, not new syscalls.

### 3.1 Syscall Table

```rust
/// The complete RuVix syscall interface.
/// No syscall may be added without an ADR amendment and ABI version bump.

// --- Task Management ---

/// Spawn a new task with an explicit capability set.
/// The caller must hold Cap<TaskFactory> to invoke this.
/// Returns a handle to the new task.
fn task_spawn(
    entry: RvfComponentId,      // RVF component containing the entry point
    caps: &[CapHandle],         // capabilities granted to the new task
    priority: TaskPriority,     // base scheduling priority
    deadline: Option<Duration>, // optional hard deadline
) -> Result<TaskHandle, KernelError>;

// --- Capability Management ---

/// Grant a capability to another task.
/// The granting task must hold the capability with the Grant right.
/// Capabilities are unforgeable kernel objects.
fn cap_grant(
    target: TaskHandle,
    cap: CapHandle,
    rights: CapRights,          // subset of caller's rights on this cap
) -> Result<CapHandle, KernelError>;

// --- Region Memory ---

/// Map a memory region into the calling task's address space.
/// Region policy (immutable, append-only, slab) is set at creation.
fn region_map(
    size: usize,
    policy: RegionPolicy,       // Immutable | AppendOnly | Slab { slot_size }
    cap: CapHandle,             // capability authorizing the mapping
) -> Result<RegionHandle, KernelError>;

// --- Queue IPC ---

/// Send a typed message to a queue.
/// The message is zero-copy if sender and receiver share a region.
fn queue_send(
    queue: QueueHandle,
    msg: &[u8],                 // serialized message (RVF wire format)
    priority: MsgPriority,
) -> Result<(), KernelError>;

/// Receive a message from a queue.
/// Blocks until a message is available or the timeout expires.
fn queue_recv(
    queue: QueueHandle,
    buf: &mut [u8],
    timeout: Duration,
) -> Result<usize, KernelError>;

// --- Timer ---

/// Wait until a deadline or duration elapses.
/// The scheduler may preempt the task and resume it when the timer fires.
fn timer_wait(
    deadline: TimerSpec,        // Absolute(Instant) | Relative(Duration)
) -> Result<(), KernelError>;

// --- RVF Boot ---

/// Mount a signed RVF package into the component namespace.
/// The kernel verifies the package signature, proof policy, and
/// witness log policy before making components available.
fn rvf_mount(
    rvf_data: &[u8],           // raw RVF bytes (or region handle)
    mount_point: &str,          // namespace path (e.g., "/agents/planner")
    cap: CapHandle,             // capability authorizing the mount
) -> Result<RvfMountHandle, KernelError>;

// --- Attestation ---

/// Emit a cryptographic attestation for a completed operation.
/// The attestation is appended to the kernel's witness log.
/// Returns the 82-byte attestation (compatible with ADR-047 ProofAttestation).
fn attest_emit(
    operation: &AttestPayload,  // what was done
    proof: &ProofToken,         // the proof that authorized it
) -> Result<ProofAttestation, KernelError>;

// --- Vector/Graph Kernel Objects ---

/// Read a vector from a kernel-resident vector store.
/// Returns the vector data and its coherence metadata.
fn vector_get(
    store: VectorStoreHandle,
    key: VectorKey,
) -> Result<(Vec<f32>, CoherenceMeta), KernelError>;

/// Write a vector to a kernel-resident vector store.
/// Requires a valid proof token — no proof, no mutation.
fn vector_put_proved(
    store: VectorStoreHandle,
    key: VectorKey,
    data: &[f32],
    proof: ProofToken,
) -> Result<ProofAttestation, KernelError>;

/// Apply a graph mutation (add/remove node/edge, update weight).
/// Requires a valid proof token — no proof, no mutation.
fn graph_apply_proved(
    graph: GraphHandle,
    mutation: &GraphMutation,
    proof: ProofToken,
) -> Result<ProofAttestation, KernelError>;

// --- Sensor / Perception ---

/// Subscribe to a sensor stream (RuView perception events).
/// Events are delivered to the specified queue.
fn sensor_subscribe(
    sensor: SensorDescriptor,   // identifies the sensor (type, device, filter)
    target_queue: QueueHandle,
    cap: CapHandle,
) -> Result<SubscriptionHandle, KernelError>;
```

### 3.2 Syscall Properties

Every syscall satisfies these invariants:

1. **Capability-gated**: No syscall succeeds without an appropriate capability handle. There is no ambient authority.
2. **Proof-required for mutation**: `vector_put_proved`, `graph_apply_proved`, and `rvf_mount` require cryptographic proof tokens. Read-only operations do not.
3. **Bounded latency**: Every syscall has a worst-case execution time expressible in cycles. The kernel contains no unbounded loops.
4. **Witness-logged**: Every successful syscall that mutates state emits a witness record to the kernel's append-only log.
5. **No allocation in syscall path**: The kernel pre-allocates all internal structures. Syscalls operate on pre-mapped regions and pre-created queues.

---

## 4. Memory Model

### 4.1 Region-Based Memory

RuVix replaces virtual memory with regions. A region is a contiguous, capability-protected memory object with one of three policies:

```rust
#[derive(Clone, Copy, Debug)]
pub enum RegionPolicy {
    /// Contents are set once at creation and never modified.
    /// The kernel may deduplicate identical immutable regions.
    /// Ideal for: RVF component code, trained model weights, lookup tables.
    Immutable,

    /// Contents can only be appended, never overwritten or truncated.
    /// A monotonic write cursor tracks the append position.
    /// Ideal for: witness logs, event streams, time-series vectors.
    AppendOnly {
        max_size: usize,
    },

    /// Fixed-size slots allocated from a free list.
    /// Slots can be freed and reused. No fragmentation by construction.
    /// Ideal for: task control blocks, capability tables, queue ring buffers.
    Slab {
        slot_size: usize,
        slot_count: usize,
    },
}
```

### 4.2 No Virtual Memory, No Page Faults

RuVix does not implement demand paging. All regions are physically backed at `region_map` time. This eliminates:

- Page fault handlers (a major source of kernel complexity and timing jitter)
- Swap — if memory is exhausted, `region_map` returns `Err(OutOfMemory)`
- Copy-on-write — immutable regions are shared by reference; mutable regions are explicitly copied through `region_map` with a source handle

This design follows seL4's philosophy: the kernel provides the mechanism (regions), and policy (which regions to create, when to reclaim) is handled by a user-space resource manager running as an RVF component.

### 4.3 Vector Store as Kernel Memory Object

Unlike conventional kernels where all data structures are userspace constructs, RuVix makes vector stores and graph stores kernel-resident objects. Vector data lives in kernel-managed regions with the same protection as capability tables. HNSW index nodes are slab-allocated (fixed-size slots, zero allocator overhead during search). Coherence metadata is co-located with each vector (coherence score, last-mutation epoch, proof attestation hash). On AArch64 with SVE/SME, the kernel performs distance computations in-kernel without context switches.

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

pub struct KernelGraphStore {
    node_region: RegionHandle,       // slab region for graph nodes
    edge_region: RegionHandle,       // slab region for adjacency lists
    witness_region: RegionHandle,    // append-only mutation witness log
    partition_meta: PartitionMeta,   // MinCut partition metadata
    proof_policy: ProofPolicy,
}
```

---

## 5. Scheduling Model

### 5.1 Coherence-Aware Scheduler

The RuVix scheduler is not a conventional priority scheduler. It combines three signals:

1. **Deadline pressure**: Hard real-time tasks with `deadline` set in `task_spawn` get earliest-deadline-first (EDF) scheduling within their capability partition.
2. **Novelty signal**: Tasks processing genuinely new information (measured by vector distance from recent inputs) get a priority boost. This prevents the system from starving exploration in favor of exploitation.
3. **Structural risk**: Tasks whose pending mutations would increase graph incoherence (lowering the coherence score below a threshold) get deprioritized until a proof-verified coherence restoration is scheduled.

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

### 5.2 Scheduling Guarantees

- **No priority inversion**: Capability-based access means tasks cannot block on resources they do not hold capabilities for. The kernel never needs priority inheritance protocols.
- **Bounded preemption**: The kernel preempts at queue boundaries (after a `queue_send` or `queue_recv` completes), not at arbitrary instruction boundaries. This eliminates the need for kernel-level spinlocks.
- **Partition scheduling**: Tasks are grouped by their RVF mount origin. Each partition gets a guaranteed time slice, preventing a misbehaving RVF component from starving others.

---

## 6. Capability Manager

### 6.1 seL4-Inspired Explicit Object Access

Every kernel object (task, region, queue, timer, vector store, graph store, RVF mount) is accessed exclusively through capabilities. A capability is an unforgeable kernel-managed token comprising:

```rust
/// A capability is a kernel-managed, unforgeable access token.
#[derive(Clone)]
pub struct Capability {
    /// Unique identifier for the kernel object.
    object_id: ObjectId,
    /// The type of kernel object (Task, Region, Queue, Timer, VectorStore, GraphStore, RvfMount).
    object_type: ObjectType,
    /// Rights bitmap: Read, Write, Grant, Revoke, Execute, Prove.
    rights: CapRights,
    /// Capability badge — caller-visible identifier for demultiplexing.
    badge: u64,
    /// Epoch — invalidated if the object is destroyed or the capability is revoked.
    epoch: u64,
}

bitflags::bitflags! {
    pub struct CapRights: u32 {
        const READ    = 0b0000_0001;
        const WRITE   = 0b0000_0010;
        const GRANT   = 0b0000_0100;
        const REVOKE  = 0b0000_1000;
        const EXECUTE = 0b0001_0000;
        const PROVE   = 0b0010_0000; // right to generate proof tokens for this object
    }
}
```

### 6.2 Capability Derivation Rules

A task can only grant capabilities it holds, with equal or fewer rights. `PROVE` is required for `vector_put_proved`/`graph_apply_proved`. `GRANT` is required for `cap_grant`. Revoking a capability invalidates all derived capabilities (propagation through the derivation tree).

### 6.3 Initial Capability Set

At boot, the kernel creates a root task holding capabilities for all physical memory (as untyped regions), the boot RVF package, the kernel witness log (append-only), and a root queue for hardware interrupts. The root task creates all other kernel objects and distributes capabilities. Following seL4's principle: the kernel creates nothing after boot.

---

## 7. Queue-First IPC

### 7.1 Shared Ring Queues

All inter-task communication in RuVix goes through queues. There are no synchronous IPC calls, no shared memory without explicit region grants, and no signals.

```rust
pub struct KernelQueue {
    ring_region: RegionHandle,   // shared region containing the ring buffer
    ring_size: u32,              // power of 2
    sq_head: AtomicU32,          // submission queue head (sender writes)
    sq_tail: AtomicU32,          // submission queue tail (kernel advances)
    cq_head: AtomicU32,          // completion queue head (receiver writes)
    cq_tail: AtomicU32,          // completion queue tail (kernel advances)
    schema: WitTypeId,           // RVF WIT type for message validation
    max_msg_size: u32,
}
```

### 7.2 Zero-Copy Semantics

When sender and receiver share a region, `queue_send` places a descriptor (offset + length) in the ring rather than copying bytes. The receiver reads directly from the shared region. This is critical for high-throughput vector streaming where copying 768-dimensional f32 vectors would be prohibitive.

### 7.3 Queue-Based Device Drivers

Hardware interrupts are delivered as messages to designated queues. A device driver is an RVF component that:
1. Holds capabilities for device MMIO regions
2. Subscribes to interrupt queues
3. Translates hardware events into typed messages on application queues

This means all device drivers run in user space with no kernel privileges beyond their capability set.

---

## 8. Proof-Gated Mutation Protocol

### 8.1 Kernel-Enforced Invariant

In RuVix, proof-gated mutation (ADR-047) is not a library convention. It is a kernel invariant. The kernel physically prevents state mutation without a valid proof token.

```rust
/// A proof token authorizing a specific mutation.
/// Generated by the Proof Engine and consumed by a mutating syscall.
pub struct ProofToken {
    /// Hash of the mutation being authorized.
    mutation_hash: [u8; 32],
    /// Proof tier (Reflex, Standard, Deep) — from ADR-047.
    tier: ProofTier,
    /// The proof payload (Merkle witness, ZK proof, or coherence certificate).
    payload: ProofPayload,
    /// Expiry — proofs are time-bounded to prevent replay.
    valid_until: Instant,
    /// Nonce — prevents proof reuse.
    nonce: u64,
}

#[derive(Clone, Copy)]
pub enum ProofTier {
    /// Sub-microsecond hash check. For high-frequency vector updates.
    Reflex,
    /// Merkle witness verification. For graph mutations.
    Standard,
    /// Full coherence verification with mincut analysis. For structural changes.
    Deep,
}
```

### 8.2 Proof Lifecycle

1. A task prepares a mutation (e.g., `GraphMutation::AddEdge { from, to, weight }`).
2. The task computes the mutation hash and requests a proof from the Proof Engine (an RVF component, not in-kernel).
3. The Proof Engine evaluates the mutation against the current coherence state, proof policy, and attestation chain.
4. If approved, the Proof Engine issues a `ProofToken` with a bounded validity window.
5. The task calls `graph_apply_proved(graph, &mutation, proof)`.
6. The kernel verifies: (a) the proof token matches the mutation hash, (b) the token has not expired, (c) the nonce has not been used, (d) the calling task holds `PROVE` rights on the graph.
7. If all checks pass, the mutation is applied and an attestation is emitted to the witness log.
8. If any check fails, the syscall returns `Err(ProofRejected)` and no state changes.

### 8.3 Proof Composition

Regional proofs compose. When multiple mutations within a mincut partition are all proved individually, a partition-level proof can be derived (see ADR-047 Section 4). The kernel maintains partition coherence scores and can fast-path mutations within a coherent partition using `Reflex` tier proofs.

---

## 9. RVF Boot Sequence

### 9.1 Boot Protocol

RuVix boots from a single signed RVF file. The boot sequence is:

```
Power On / Hypervisor Start
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 0: Hardware Init (AArch64)             │
│  - Initialize MMU with identity mapping      │
│  - Initialize UART for early console         │
│  - Detect available memory, cache topology   │
│  - If TEE: initialize realm/enclave          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 1: RVF Manifest Parse                  │
│  - Read 4 KB boot manifest from RVF header   │
│  - Verify ML-DSA-65 signature (post-quantum) │
│  - Parse component graph                     │
│  - Parse memory schema (region requirements) │
│  - Parse proof policy                        │
│  - Parse witness log policy                  │
│  - Parse rollback hooks                      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 2: Kernel Object Creation              │
│  - Create root task with all capabilities    │
│  - Create initial regions from memory schema │
│  - Create boot queue for init messages       │
│  - Initialize kernel witness log (append)    │
│  - Initialize kernel vector store (if spec.) │
│  - Initialize kernel graph store (if spec.)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 3: Component Mount                     │
│  - Mount RVF components per component graph  │
│  - Distribute capabilities per manifest      │
│  - Spawn initial tasks per WIT entry points  │
│  - Connect queues per manifest wiring        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 4: First Attestation                   │
│  - Emit boot attestation to witness log      │
│  - Record: RVF hash, capability table hash,  │
│    region layout hash, timestamp             │
│  - System is now live                        │
└──────────────────────────────────────────────┘
```

### 9.2 RVF as Boot Object

An RVF boot package is a complete cognitive unit containing:

| Section | Purpose | Size Budget |
|---------|---------|-------------|
| Manifest | Component graph, memory schema, proof policy, rollback hooks, witness log policy | 4 KB |
| Signatures | ML-DSA-65 package signature + per-component signatures | 2-8 KB |
| WIT/ABI | Component interface types (WASM Interface Types) | 1-16 KB |
| Component Graph | DAG of components with queue wiring and capability grants | 1-4 KB |
| Memory Schema | Region declarations (size, policy, initial content) | 1-4 KB |
| Proof Policy | Per-component proof tier requirements | 512 B - 2 KB |
| Rollback Hooks | WASM functions for state rollback on proof failure | 1-8 KB |
| Witness Log Policy | Retention, compression, export rules for attestations | 256 B - 1 KB |
| WASM Components | Compiled WASM component binaries | 8 KB - 16 MB |
| Initial Data | Pre-loaded vectors, graph state, model weights | 0 - 1 GB |

### 9.3 Deterministic Replay

Because every mutation is witnessed and every input is queued, a RuVix system can replay from any checkpoint:

1. Load a checkpoint RVF (containing region snapshots + witness log prefix)
2. Replay queued messages in witness-log order
3. Re-verify proofs at each mutation
4. The resulting state must be identical (bit-for-bit) to the original

This is the foundation of the acceptance test (Section 12).

---

## 10. RuView as Perception Plane

### 10.1 Position in Architecture

RuView sits outside the kernel but close — it is the first RVF component layer. Its job is to normalize external signals into typed, coherence-scored events and publish them into kernel queues.

### 10.2 Sensor Abstraction

```rust
/// A sensor descriptor identifies a data source for RuView.
pub struct SensorDescriptor {
    /// Sensor type: Camera, Microphone, NetworkTap, MarketFeed, GitStream, etc.
    sensor_type: SensorType,
    /// Device identifier (hardware address, URL, stream ID).
    device_id: DeviceId,
    /// Filter expression (e.g., "symbol=AAPL" or "file_ext=.rs").
    filter: Option<FilterExpr>,
    /// Requested sampling rate (events per second, 0 = all).
    sample_rate: u32,
}
```

### 10.3 Event Normalization

RuView transforms raw sensor data into `PerceptionEvent` structs carrying: a vector embedding (matching kernel store dimensionality), a coherence score relative to recent context, a causal hash linking to the previous event from the same sensor, and a nanosecond timestamp. Events flow through `sensor_subscribe` into kernel queues where agent tasks consume them.

---

## 11. AgentDB as Planner/Intelligence Runtime

### 11.1 Explicit Non-Kernel Placement

AgentDB, Claude Code, Codex, and all AI reasoning systems are NOT in the trusted kernel. They are RVF components running in user space with:

- Capability-restricted access to vector stores (read + proved write)
- Queue-based communication (no direct kernel memory access)
- Proof-gated mutation (the intelligence runtime cannot modify state without passing through the proof engine)

This is a deliberate security boundary. The kernel trusts mathematics (proofs, hashes, capabilities). It does not trust neural networks.

### 11.2 Control Plane Adapters

Claude Code and Codex connect to RuVix as control plane adapters:

```
┌──────────────┐    queue     ┌──────────────┐    syscall    ┌────────┐
│ Claude Code  │◀────────────▶│ AgentDB      │──────────────▶│ RuVix  │
│ (external)   │  WebSocket   │ (RVF comp.)  │  cap-gated    │ Kernel │
└──────────────┘              └──────────────┘               └────────┘
```

The adapter translates natural language intent into typed mutations with proof requests. The kernel neither knows nor cares that the mutation originated from an LLM.

---

## 12. Build Path

### Phase A: Linux-Hosted Nucleus (Days 1-60)

**Goal**: Implement all 12 syscalls as a Rust library running in Linux userspace. Freeze the ABI.

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1-2 | Core types | `ruvix-types` crate: all kernel object types, capability types, proof types. No_std compatible. |
| 3-4 | Region manager | `ruvix-region` crate: slab allocator, append-only regions, immutable regions. Uses mmap on Linux. |
| 5-6 | Queue IPC | `ruvix-queue` crate: io_uring-style ring buffers in shared memory. Lock-free send/recv. |
| 7-8 | Capability manager | `ruvix-cap` crate: capability table, derivation tree, revocation propagation. |
| 9-10 | Proof engine | `ruvix-proof` crate: proof token generation/verification, witness log. Integrates `ruvector-verified`. |
| 11-12 | Vector/Graph kernel objects | `ruvix-vecgraph` crate: kernel-resident vector and graph stores using regions. Integrates `ruvector-core` and `ruvector-graph-transformer`. |
| 13-14 | Scheduler | `ruvix-sched` crate: coherence-aware task scheduler. Runs as a Linux thread scheduler. |
| 15-16 | RVF boot loader | `ruvix-boot` crate: RVF package parsing, component mounting, capability distribution. |
| 17-18 | Integration + ABI freeze | `ruvix-nucleus` crate: all subsystems integrated. ABI frozen. Acceptance test passes. |

**Phase A Acceptance Test**: A signed RVF boots in the Linux-hosted nucleus, consumes a simulated RuView event from a queue, performs one proof-gated vector mutation, emits an attestation to the witness log, shuts down, restarts from checkpoint, replays to the same state, and the final vector store contents are bit-identical.

### Phase B: Bare Metal AArch64 Microkernel (Days 60-120)

**Goal**: Run the same ABI on bare metal AArch64 (Raspberry Pi 4/5, QEMU virt).

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 19-22 | AArch64 bootstrap | Exception vectors, MMU setup, UART driver, physical memory manager. |
| 23-26 | Region → physical memory | Replace mmap with direct physical page allocation. Implement region policies on hardware page tables. |
| 27-30 | Interrupt → queue | GIC (Generic Interrupt Controller) driver delivering interrupts as queue messages. Timer driver for `timer_wait`. |
| 31-34 | WASM component runtime | Embedded Wasmtime or wasm-micro-runtime for executing RVF WASM components on bare metal. |
| 35-38 | Scheduler on hardware | Coherence-aware scheduler using AArch64 timer interrupts for preemption. |
| 39-42 | Acceptance test on hardware | Same acceptance test as Phase A, running on QEMU AArch64 virt machine. |

### No Phase C

There is no POSIX compatibility layer. RuVix does not implement `open()`, `read()`, `write()`, `fork()`, `exec()`, or any POSIX syscall. Applications that need POSIX run on Linux. RuVix runs cognitive workloads.

---

## 13. Demo Applications

### 13.1 Application Matrix

| # | Application | Category | Kernel Features Exercised | Complexity |
|---|-------------|----------|---------------------------|------------|
| 1 | **Proof-Gated Vector Journal** | Foundational | vector_put_proved, attest_emit, deterministic replay | Low |
| 2 | **Edge ML Inference Pipeline** | Practical | rvf_mount, queue IPC, sensor_subscribe, vector_get, timer_wait | Medium |
| 3 | **Autonomous Drone Swarm Coordinator** | Practical | task_spawn (per-drone), cap_grant (dynamic trust), queue mesh, coherence scheduler | High |
| 4 | **Self-Healing Knowledge Graph** | Practical | graph_apply_proved, coherence scoring, proof composition, rollback hooks | High |
| 5 | **Collective Intelligence Mesh** | Exotic | Multi-kernel federation via queue bridges, cross-kernel cap_grant, distributed proof composition | Very High |
| 6 | **Quantum-Coherent Memory Replay** | Exotic | Superposition-tagged vectors, deferred proof resolution, probabilistic region policies | Very High |
| 7 | **Biological Signal Processor** | Exotic | sensor_subscribe (EEG/EMG), real-time coherence scoring, deadline scheduling, witness-backed diagnostics | High |
| 8 | **Adversarial Reasoning Arena** | Exotic | Competing agent tasks with conflicting proof policies, capability isolation, coherence arbitration | Very High |

### 13.2 Demo 1: Proof-Gated Vector Journal

The simplest demonstration of the kernel's core invariant.

```
RVF Package: vector_journal.rvf
Components:
  - writer: Generates random vectors, requests proofs, calls vector_put_proved
  - reader: Periodically calls vector_get, verifies coherence metadata
  - auditor: Reads witness log, verifies attestation chain integrity

Test scenario:
  1. Writer stores 1000 vectors with valid proofs → all succeed
  2. Writer attempts store without proof → kernel rejects (Err(ProofRejected))
  3. Writer attempts store with expired proof → kernel rejects
  4. System checkpoints, restarts, replays → final state identical
  5. Auditor verifies: witness log contains exactly 1000 attestations
```

### 13.3 Demo 2: Edge ML Inference Pipeline

An RVF package that runs a complete ML inference pipeline on an edge device.

```
RVF Package: edge_inference.rvf
Components:
  - sensor_adapter: Subscribes to camera sensor, emits frame embeddings to queue
  - feature_store: Kernel vector store holding reference embeddings
  - classifier: Receives embeddings, queries feature_store, emits classifications
  - model_updater: Periodically receives new model weights via queue,
                   performs proof-gated update of feature_store

Kernel features:
  - sensor_subscribe for camera frames
  - queue_send/recv for pipeline stages
  - vector_get for nearest-neighbor lookup
  - vector_put_proved for model updates (proof-gated)
  - timer_wait for periodic model refresh
  - Coherence scheduler prioritizes inference over model updates
```

### 13.4 Demo 3: Autonomous Drone Swarm Coordinator

```
RVF Package: drone_swarm.rvf
Components:
  - coordinator: Maintains global mission graph, assigns waypoints
  - drone_agent[N]: Per-drone task with position vectors, local planning
  - trust_manager: Dynamically adjusts capabilities based on drone behavior
  - coherence_monitor: Watches for swarm fragmentation (graph mincut)

Kernel features:
  - task_spawn per drone agent (dynamic fleet scaling)
  - cap_grant/revoke for dynamic trust (misbehaving drone loses capabilities)
  - graph_apply_proved for mission graph updates
  - Coherence scheduler penalizes plans that fragment the swarm graph
  - Deterministic replay for post-mission analysis
```

### 13.5 Demo 4: Self-Healing Knowledge Graph

```
RVF Package: knowledge_graph.rvf
Components:
  - ingestor: Consumes knowledge events, proposes graph mutations
  - coherence_checker: Evaluates proposed mutations against graph invariants
  - healer: Detects coherence drops, proposes compensating mutations
  - checkpoint_manager: Periodic state snapshots with rollback hooks

Kernel features:
  - graph_apply_proved for every knowledge mutation
  - Proof composition across mincut partitions
  - Rollback hooks triggered when coherence drops below threshold
  - Witness log provides complete audit trail for knowledge provenance
```

### 13.6 Demo 5: Collective Intelligence Mesh

Multiple RuVix instances forming a distributed cognitive fabric. Each node runs its own kernel with local vector/graph stores. Queue bridges (network-backed queues) connect nodes. Cross-kernel capability delegation works via attested queue messages. Distributed proof composition allows node A to prove locally while node B verifies. The mesh self-organizes via coherence gradients with no central coordinator. Knowledge migrates toward nodes that use it (vector locality). Proof chains span multiple kernels (federated attestation via `ruvector-raft`).

### 13.7 Demo 6: Quantum-Coherent Memory Replay

A speculative demonstration exploring quantum-inspired memory semantics. Vectors are stored in superposition states (multiple weighted values). Proof resolution collapses superposition to a definite value. Until observed via `vector_get`, mutations accumulate as unresolved proofs. Replay can explore alternative proof resolution paths by checkpointing, resolving proofs differently, and comparing outcomes. Requires an experimental `Superposition` region policy not in the initial kernel.

### 13.8 Demo 7: Biological Signal Processor

EEG and EMG sensor adapters emit neural/muscle signal vectors via `sensor_subscribe`. A fusion engine combines them into intent vectors. Hard deadline scheduling (256 Hz = 3.9ms deadline) ensures real-time processing. Coherence scoring detects anomalous signals (seizure detection). Witness-backed diagnostics provide regulatory compliance audit trails. Proof-gated model updates prevent untested parameter changes in clinical settings.

### 13.9 Demo 8: Adversarial Reasoning Arena

Two competing agent tasks (red/blue) propose mutations to maximize conflicting objectives on a shared graph. An arbiter evaluates competing proofs and grants mutations to the stronger proof. Capability isolation prevents agents from accessing each other's state. The coherence-aware scheduler penalizes agents that lower coherence. The full witness log enables post-hoc analysis of adversarial dynamics and emergent strategies.

---

## 14. Failure Modes and Mitigations

| Failure Mode | Impact | Mitigation |
|---|---|---|
| **Proof engine unavailable** | All mutations blocked (no proof tokens issued) | Kernel maintains a Reflex-tier proof cache for critical paths. Cache entries have short TTL (100ms). If proof engine is down for >1s, kernel emits a diagnostic attestation and suspends non-critical tasks. |
| **Witness log full** | New attestations cannot be written; mutations blocked | Append-only regions have configurable max size. When 90% full, kernel emits a compaction request to the witness manager component. At 100%, kernel rejects mutations until space is freed. Witness log is never truncated — only checkpointed and archived. |
| **Coherence score collapse** | Scheduler deprioritizes all mutation tasks; system stalls | Coherence floor threshold triggers automatic rollback to last checkpoint where coherence was above threshold. Rollback hooks in the RVF manifest execute compensating logic. |
| **Capability leak (over-granting)** | Task gains access to resources beyond its intended scope | Revocation propagates through derivation tree. Periodic capability audit compares held capabilities against manifest-declared permissions. Discrepancies trigger automatic revocation. |
| **Vector store capacity exhausted** | `vector_put_proved` returns `OutOfMemory` | Pre-allocated capacity is declared in RVF manifest. Resource manager component is responsible for eviction policy (LRU by coherence score, quantization-based compression). Kernel enforces capacity limits. |
| **Queue overflow** | `queue_send` returns `QueueFull`; producer back-pressure | Ring buffer size is declared at queue creation. Producers must handle `QueueFull` by retrying or dropping low-priority messages. Kernel never silently drops messages. |
| **Malicious RVF package** | Arbitrary code execution, capability theft | RVF signature verification at `rvf_mount` time. WASM component sandboxing. Component capabilities are limited to what the manifest declares and the mounting task grants. No ambient authority. |
| **AArch64 hardware fault** | Kernel panic, data corruption | Region checksums enable corruption detection. Append-only witness log survives if storage is intact. Replay from last checkpoint recovers state. TEE attestation detects hardware tampering. |

---

## 15. Consequences

### 15.1 Positive

1. **Proof-gated mutation as kernel invariant**: Every state change is auditable, replayable, and formally justified. This eliminates entire categories of bugs (unauthorized writes, silent corruption, untracked state drift).

2. **Zero-overhead vector/graph operations**: Vector and graph stores as kernel objects eliminate the syscall-per-query overhead of running a vector database as a userspace service. Distance computations can use kernel-privileged SIMD/SVE instructions.

3. **Deterministic replay**: Because every input is queued, every mutation is proved, and every effect is witnessed, the system can replay from any checkpoint to reproduce any state. This is invaluable for debugging, auditing, and regulatory compliance.

4. **Minimal attack surface**: 12 syscalls, 6 primitives, capability-only access. The kernel TCB is orders of magnitude smaller than Linux (~30M LOC) or even seL4 (~8.7K LOC verified C + ~600 LOC assembly). Target: <15K LOC Rust.

5. **RVF as universal deployment unit**: A single signed file contains code, data, capabilities, proof policies, and rollback hooks. No package managers, no container runtimes, no dependency hell.

6. **Coherence-aware scheduling**: The scheduler understands semantic content, not just priority numbers. This enables intelligent resource allocation in cognitive workloads.

7. **Natural integration with RuVector**: All 107 existing crates can be compiled as RVF components. The kernel's vector/graph objects use the same formats and algorithms as `ruvector-core` and `ruvector-graph-transformer`.

### 15.2 Negative

1. **No POSIX compatibility**: Existing software cannot run on RuVix without rewriting. This limits the ecosystem to purpose-built RVF components.

2. **Hardware support initially limited**: AArch64-first means no x86_64 bare metal in Phase B. x86_64 support requires a separate BSP effort.

3. **Proof overhead on hot paths**: Even `Reflex` tier proofs add ~100ns per mutation. For workloads with millions of mutations per second, this is measurable. Mitigation: batch mutations under a single partition proof.

4. **No dynamic memory allocation**: Pre-allocated regions mean the system must know its memory requirements at boot time. Dynamic workloads require a resource manager component with eviction/compaction policies.

5. **Unfamiliar programming model**: Developers accustomed to POSIX, threads, and mutexes must learn capabilities, queues, and proof-gated mutation. Documentation and tooling investment is required.

### 15.3 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ABI instability during Phase A | Medium | High — breaks all existing RVF components | Freeze ABI at week 18. No changes without ADR amendment. |
| WASM component model immaturity | Medium | Medium — limits component interoperability | Pin to WASI Preview 2. Maintain a WIT type registry. |
| Performance regression vs. Linux-hosted RuVector | Low | High — undermines the motivation | Benchmark suite comparing Linux-hosted vs. RuVix-native for vector ops, graph mutations, queue throughput. |
| Formal verification infeasibility | High | Medium — limits trust claims | Do not claim formal verification in v1. Focus on testing and deterministic replay as the verification mechanism. |
| Single-developer dependency | High | High — project stalls if key contributor leaves | Document everything in ADRs. Keep kernel small enough for one person to understand fully. |

---

## 16. Integration with Existing RuVector Crates

### 16.1 Crate Mapping

| Existing Crate | RuVix Role | Integration Path |
|---|---|---|
| `ruvector-core` | Kernel vector store implementation | Extract HNSW algorithm into `ruvix-vecgraph`. Use slab regions instead of `Vec<T>`. |
| `ruvector-graph-transformer` | Kernel graph store implementation | Extract graph mutation logic into `ruvix-vecgraph`. Proof-gate via kernel syscalls. |
| `ruvector-verified` | Proof engine foundation | `ProofGate<T>`, `ProofEnvironment`, `ProofAttestation` become kernel types. `ruvix-proof` wraps these. |
| `cognitum-gate-kernel` | Coherence scoring reference | Port 256-tile coherence fabric to operate on kernel graph store regions. |
| `ruvector-coherence` | Scheduler coherence signal | Coherence score computation feeds into `compute_priority()`. |
| `ruvector-mincut` | Graph partitioning for proof composition | MinCut partitions define proof composition boundaries. |
| `rvf` | Boot loader format parser | `ruvix-boot` depends on `rvf` for manifest parsing and signature verification. |
| `ruvector-raft` | Multi-kernel consensus (Demo 5) | Queue-bridge Raft for distributed coherence in collective intelligence mesh. |
| `ruvector-snapshot` | Checkpoint/restore | Region snapshots for deterministic replay and rollback. |
| `sona` | AgentDB intelligence runtime | Runs as RVF component in user space. Communicates via queues. |
| `ruvllm` | ML inference component | Runs as RVF component. Uses `vector_get` for retrieval, proof-gated weight updates. |
| `ruvector-temporal-tensor` | Time-series vector storage | Append-only regions are a natural fit for temporal tensor data. |

### 16.2 Shared No_std Types

A new `ruvix-types` crate (no_std, no alloc) defines all kernel interface types. This crate is depended on by both kernel code and RVF component code, ensuring type-level compatibility across the boundary.

```toml
[package]
name = "ruvix-types"
version = "0.1.0"
edition = "2021"

[features]
default = []
std = []
alloc = []

[dependencies]
# Zero external dependencies for the kernel type crate
```

---

## 17. Acceptance Test Specification

The acceptance test is the single gate for Phase A completion. It is not a unit test — it is a system-level integration test that exercises every kernel subsystem.

### 17.1 Test Procedure

```
GIVEN:
  - A signed RVF package "acceptance.rvf" containing:
    - A sensor_adapter component (simulated)
    - A vector_store component (kernel-resident, capacity=100)
    - A proof_engine component
    - A writer component
    - A reader component
    - Proof policy: Standard tier for all mutations
    - Witness log policy: retain all, no compression

WHEN:
  Step 1: rvf_mount("acceptance.rvf", "/test", root_cap)
  Step 2: sensor_adapter emits one PerceptionEvent to queue
  Step 3: writer receives event, computes embedding vector
  Step 4: writer requests proof from proof_engine
  Step 5: writer calls vector_put_proved(store, key, vector, proof)
  Step 6: kernel verifies proof, applies mutation, emits attestation
  Step 7: reader calls vector_get(store, key)
  Step 8: System checkpoints (region snapshots + witness log)
  Step 9: System shuts down
  Step 10: System restarts from checkpoint
  Step 11: System replays witness log
  Step 12: reader calls vector_get(store, key) again

THEN:
  - Step 5 returns Ok(attestation) with 82-byte witness
  - Step 7 returns the exact vector and coherence metadata
  - Step 12 returns the EXACT SAME vector and coherence metadata as Step 7
  - Witness log contains exactly: 1 boot attestation + 1 mount attestation + 1 mutation attestation
  - No proof-less mutation was accepted at any point
  - Total replay time < 2x original execution time
```

---

## 18. Comparison with Prior Art

| Property | Linux | seL4 | Zephyr | Hermit | RuVix |
|---|---|---|---|---|---|
| Kernel LOC | ~30M | ~8.7K | ~200K | ~50K | <15K (target) |
| Primitives | process, file, socket, signal, pipe | TCB, CNode, endpoint, notification, untyped | thread, semaphore, FIFO, timer | process (unikernel) | task, capability, region, queue, timer, proof |
| Syscalls | ~450 | ~12 | ~150 | POSIX subset | 12 |
| Memory model | virtual memory + demand paging | untyped + retype | flat or MPU regions | single address space | regions (immutable/append/slab) |
| IPC | pipe, socket, signal, shmem | synchronous endpoint | FIFO, mailbox, pipe | POSIX | queue (io_uring-style) |
| Access control | DAC/MAC (users, groups, SELinux) | capabilities | none (trusted code) | none (unikernel) | capabilities + proof |
| Vector/graph native | no | no | no | no | yes (kernel objects) |
| Proof-gated mutation | no | no | no | no | yes (kernel invariant) |
| Formal verification | no | yes (functional correctness) | no | no | no (v1), replay-based |
| POSIX compatible | yes | no (but has CAmkES) | partial | yes | no |
| Hardware target | all | ARM, RISC-V, x86 | MCUs | x86_64, aarch64 | AArch64 (primary) |

---

## 19. Open Questions

1. **Maximum proof verification time?** Should `Deep` tier proofs >10ms be rejected? Position: yes, configurable per-partition timeout.
2. **Vector quantization: in-kernel or in-component?** Position: in-component; kernel stores pre-quantized data.
3. **Multi-kernel conflicting proofs?** Position: Raft consensus (`ruvector-raft`) with coherence score tiebreaker.
4. **Hot-swapping RVF components?** Position: add `rvf_unmount` in a future ABI revision, not in initial 12 syscalls.
5. **Minimum viable Phase B hardware?** Position: Raspberry Pi 4 (Cortex-A72) for accessibility, validate on Pi 5 (Cortex-A76).

6. **Boot signature failure behavior?** Position: boot halts (kernel panic) on signature verification failure — no fallback, diagnostic to UART.
7. **Slab slot use-after-free?** Position: slab allocations return capability-protected handles with per-slot generation counters; stale handles are detected and rejected.
8. **Queue schema validation boundary?** Position: kernel validates message size and WIT type tag at `queue_send` time; deep structural validation is receiver-side. Document this as a trust boundary.

---

## 20. Security Hardening Notes

Identified during pre-release security audit. All are specification clarifications, not structural flaws.

### 20.1 Root Task Privilege Attenuation

The root task holds capabilities for all physical memory at boot. After Stage 3 component mounting completes, the root task MUST drop all capabilities except its own task handle and the kernel witness log (read-only). If the root task is implemented as a persistent init component, it retains only the minimum capability set declared in its RVF manifest. This prevents a compromised root task from owning the entire system post-boot.

### 20.2 Capability Delegation Depth Limit

`cap_grant` with `GRANT` right enables transitive delegation chains. To prevent unbounded delegation: maximum delegation depth is 8 (configurable per-RVF). A `GRANT_ONCE` right (non-transitive) is available for cases where a task should delegate access but not the ability to further delegate. The periodic capability audit (Section 14) flags delegation chains deeper than 4.

### 20.3 Boot RVF Proof Bootstrap

Mounting the Proof Engine itself cannot require a user-space proof (circular dependency). Resolution: Stage 3 component mounting from the cryptographically-signed boot RVF is the single kernel-trusted path that bypasses proof token requirements. Post-boot `rvf_mount` calls always require proof tokens from the now-running Proof Engine. The boot RVF's signature verification at Stage 1 provides equivalent assurance.

### 20.4 Reflex Proof Cache Scoping

The Reflex-tier proof cache (Section 14, proof engine unavailable mitigation) caches proof tokens scoped to a specific `(mutation_hash, nonce)` pair, not to operation classes. Cached proofs are single-use (nonce consumed on first verification). Cache entries have 100ms TTL. The cache size is bounded (default: 64 entries). This prevents window-of-opportunity attacks where coherence degrades between cache insertion and consumption.

### 20.5 Zero-Copy IPC TOCTOU Mitigation

When `queue_send` places a descriptor referencing a shared region, the referenced data segment must be in an `Immutable` or `AppendOnly` region. The kernel rejects `queue_send` descriptors pointing into `Slab` regions (which permit overwrites). This eliminates time-of-check-to-time-of-use attacks where a sender modifies shared data after the receiver reads the descriptor but before it processes the content.

### 20.6 Boot Signature Failure

If ML-DSA-65 signature verification fails at Stage 1, the kernel panics immediately. There is no fallback boot path, no recovery mode, and no unsigned boot option. A diagnostic message is written to UART (if initialized). This is a deliberate design choice: a system that boots unsigned code provides no security guarantees.

---

## 21. References

1. **seL4 Microkernel** — Klein et al., "seL4: Formal Verification of an OS Kernel," SOSP 2009. The capability model and "kernel creates nothing after boot" principle directly inspire RuVix's capability manager.

2. **io_uring** — Axboe, "Efficient IO with io_uring," 2019. The submission/completion ring design inspires RuVix's queue IPC.

3. **WASM Component Model** — W3C WebAssembly CG, "Component Model," 2024. WIT (WASM Interface Types) provides the type system for RVF component interfaces.

4. **RVF Specification** — ADR-029, RuVector project. The canonical binary format that becomes the RuVix boot object.

5. **Proof-Gated Mutation** — ADR-047, RuVector project. The `ProofGate<T>` type and three-tier proof routing that becomes a kernel invariant.

6. **Hermit OS** — Lankes et al., "A Rust-Based Unikernel," VEE 2023. Demonstrates Rust-native kernel development and links against application at build time.

7. **Theseus OS** — Boos et al., "Theseus: an Experiment in Operating System Structure and State Management," OSDI 2020. Safe-language OS using Rust ownership for isolation without hardware privilege rings.

8. **Capability Hardware Enhanced RISC Instructions (CHERI)** — Watson et al., "CHERI: A Hybrid Capability-System Architecture," IEEE S&P 2015. Hardware-enforced capabilities that could accelerate RuVix's capability checks.

9. **Coherence Engine** — ADR-014, RuVector project. Graph-theoretic consistency scoring replacing probabilistic confidence.

10. **Cognitum Gate Kernel** — `crates/cognitum-gate-kernel/`, RuVector project. No_std WASM kernel for 256-tile coherence fabric.

---

## 22. Decision Record

This ADR proposes RuVix as a new architectural layer in the RuVector ecosystem. It does not replace any existing crate or ADR. It promotes existing primitives (RVF, proof-gated mutation, coherence scoring, capability-based access) from library conventions to kernel-enforced invariants.

The build path is intentionally conservative: Phase A delivers a Linux-hosted prototype with the full syscall surface. Phase B delivers bare metal. There is no Phase C because POSIX compatibility would compromise every design principle.

The acceptance test is the single measure of success: a signed RVF boots, consumes an event, performs a proof-gated mutation, emits an attestation, and replays deterministically.
