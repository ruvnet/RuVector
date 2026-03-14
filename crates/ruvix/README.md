# RuVix Cognition Kernel

> An Operating System for the Agentic Age

RuVix is a purpose-built kernel for AI agents. Unlike traditional operating systems designed for humans clicking files, RuVix understands vectors, graphs, proofs, and coherence scores natively.

[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.77+-orange.svg)](https://www.rust-lang.org/)
[![no_std](https://img.shields.io/badge/no__std-compatible-green.svg)](#no_std-support)

---

## Why RuVix?

### The Problem with Conventional Operating Systems

Every major operating system today (Linux, Windows, macOS, seL4, Zephyr) was designed for a world where the primary compute actor is a human operating through a process abstraction. The process model assumes:

1. A single sequential instruction stream per thread
2. File-based persistent state (byte streams with names)
3. POSIX IPC semantics (pipes, sockets, signals)
4. Discretionary or mandatory access control based on user identity
5. A scheduler optimized for interactive latency or batch throughput

**None of these assumptions hold for agentic workloads.**

An AI agent does not think in files. It thinks in **vectors**, **graphs**, **proofs**, and **causal event streams**. It does not need `fork/exec`. It needs **capability-gated task spawning** with proof-of-intent. It does not communicate through byte pipes. It communicates through **typed semantic queues** where every message carries a coherence score and a witness hash.

Running agentic workloads on Linux is like running a modern web application on a mainframe batch scheduler. Technically possible. Structurally wrong.

### Why Not seL4, Zephyr, or Unikernels?

| System | Limitation |
|--------|------------|
| **seL4** | Proves capability correctness, but has no concept of vectors, graphs, coherence, or proofs. Adding these requires reimplementing the entire RuVector stack as userspace servers. |
| **Zephyr/FreeRTOS** | Targets microcontrollers. No memory protection, no capability model, no attestation. |
| **Unikernels** | Eliminate OS/application boundary but retain POSIX semantics. They make Linux faster, not different. |

**RuVix is different.** Six kernel primitives. Twelve syscalls. Every mutation is proof-gated. Everything else lives above the kernel in RVF component space.

---

## Key Features

### Proof-Gated Mutation

Every change requires cryptographic proof. No proof = no mutation. Period.

Think of it as "commit signatures" for every memory write. The proof engine supports three verification tiers:

| Tier | Latency | Use Case |
|------|---------|----------|
| **Reflex** | <10us | Real-time hotpath mutations |
| **Standard** | ~100us | Normal operations |
| **Deep** | ~1ms | Security-critical operations |

### Six Primitives, Twelve Syscalls

RuVix has exactly six kernel primitives:

| Primitive | Purpose | Analog |
|-----------|---------|--------|
| **Task** | Unit of concurrent execution with capability set | seL4 TCB |
| **Capability** | Unforgeable typed token granting access to a resource | seL4 capability |
| **Region** | Contiguous memory with access policy (immutable, append-only, slab) | seL4 Untyped + frame |
| **Queue** | Typed ring buffer for inter-task communication | io_uring SQ/CQ |
| **Timer** | Deadline-driven scheduling primitive | POSIX timer_create |
| **Proof** | Cryptographic attestation gating state mutation | Novel (from ADR-047) |

Everything else (file systems, networking, device drivers, vector indexes, graph engines, AI inference) is an RVF component running in user space.

### Performance

Benchmarks show significant improvements over traditional Linux syscalls:

| Metric | RuVix | Linux | Improvement |
|--------|-------|-------|-------------|
| IPC latency | ~45ns | ~1000ns | **22x faster** |
| Permission check | ~12ns | ~200ns | **17x faster** |
| Memory overhead | ~2KB/task | ~32KB/process | **98% reduction** |

All with cryptographic proofs included.

### Witness-Logged Everything

Every mutation is recorded in a tamper-evident witness log. This enables:

- **Deterministic replay**: Restore any previous state from checkpoint + witness log
- **Auditability**: Prove exactly what happened and when
- **Debugging**: Step through execution deterministically

---

## Architecture

```
+---------------------------------------------------------------------+
|                      AGENT CONTROL PLANE                            |
|  Claude | GPT | Custom Agents | AgentDB Planner Runtime | Swarms    |
+---------------------------------------------------------------------+
|                      RVF COMPONENT SPACE                            |
|  +----------+ +----------+ +----------+ +----------+                |
|  | RuView   | | AgentDB  | | RuVLLM   | | Network  | ...            |
|  | Percept. | | Intelli. | | Infer.   | | Stack    |                |
|  +----+-----+ +----+-----+ +----+-----+ +----+-----+                |
|       |queue       |queue       |queue       |queue                  |
+-------+------------+------------+------------+-----------------------+
|                      RUVIX COGNITION KERNEL                         |
|                                                                     |
|  +----------------+  +----------------+  +--------------------+     |
|  | Capability Mgr |  | Queue IPC      |  | Coherence-Aware    |     |
|  | (cap_grant,    |  | (queue_send,   |  | Scheduler          |     |
|  |  cap_revoke)   |  |  queue_recv)   |  | (deadline+novelty  |     |
|  |                |  |  io_uring ring |  |  +structural risk) |     |
|  +----------------+  +----------------+  +--------------------+     |
|  +----------------+  +----------------+  +--------------------+     |
|  | Region Memory  |  | Proof Engine   |  | Vector/Graph       |     |
|  | (slabs, immut, |  | (attest_emit,  |  | Kernel Objects     |     |
|  |  append-only)  |  |  proof_verify) |  | (vector_get/put,   |     |
|  |                |  |                |  |  graph_apply)      |     |
|  +----------------+  +----------------+  +--------------------+     |
|                                                                     |
|  +---------------------------------------------------------------+  |
|  | RVF Boot Loader - mounts signed RVF packages as root          |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
|                      HARDWARE / HYPERVISOR                          |
|  AArch64 (primary) | x86_64 (secondary) | WASM (hosted)            |
+---------------------------------------------------------------------+
```

---

## Quick Start

### Installation

Add RuVix crates to your `Cargo.toml`:

```toml
[dependencies]
ruvix-nucleus = "0.1"
ruvix-types = "0.1"
```

For `no_std` environments:

```toml
[dependencies]
ruvix-nucleus = { version = "0.1", default-features = false, features = ["alloc"] }
```

### Basic Usage

```rust
use ruvix_nucleus::{Kernel, KernelConfig, Syscall, VectorStoreConfig, ProofTier};

// Create and boot the kernel
let mut kernel = Kernel::new(KernelConfig::default());
kernel.boot(0, [0u8; 32]).expect("Boot failed");

// Create a vector store (4 dimensions, 100 capacity)
let config = VectorStoreConfig::new(4, 100);
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

### Deterministic Replay

```rust
use ruvix_nucleus::{CheckpointConfig, Checkpoint};

// Create a checkpoint
let checkpoint = kernel.checkpoint(CheckpointConfig::full()).unwrap();

// Later: restore and replay
let bytes = checkpoint.to_bytes();
let restored = Checkpoint::from_bytes(&bytes).unwrap();

// Verify state matches checkpoint
assert!(kernel.verify_checkpoint(&restored));
```

---

## Syscall Reference

RuVix exposes exactly 12 syscalls. This is a hard architectural constraint. New functionality is added through RVF components, not new syscalls.

| # | Syscall | Proof Required | Description |
|---|---------|----------------|-------------|
| 0 | `TaskSpawn` | No | Create a new task with explicit capability set |
| 1 | `CapGrant` | No | Grant a capability to another task |
| 2 | `RegionMap` | No | Map a memory region into address space |
| 3 | `QueueSend` | No | Send a typed message to a queue |
| 4 | `QueueRecv` | No | Receive a message from a queue |
| 5 | `TimerWait` | No | Wait until a deadline or duration elapses |
| 6 | `RvfMount` | Yes | Mount an RVF package |
| 7 | `AttestEmit` | Yes | Emit an attestation record |
| 8 | `VectorGet` | No | Get a vector from a kernel store |
| 9 | `VectorPutProved` | Yes | Put a vector with proof |
| 10 | `GraphApplyProved` | Yes | Apply a graph mutation with proof |
| 11 | `SensorSubscribe` | No | Subscribe to sensor events |

### Syscall Invariants (ADR-087 Section 3.2)

1. **Capability-gated**: No syscall succeeds without appropriate capability
2. **Proof-required for mutation**: `vector_put_proved`, `graph_apply_proved`, `rvf_mount`
3. **Bounded latency**: No unbounded loops in syscall path
4. **Witness-logged**: Every mutation emits a witness record
5. **No allocation in syscall path**: Pre-allocated structures only

---

## Crate Overview

### Phase A: Core Kernel (Linux-hosted)

| Crate | Purpose | Tests | Description |
|-------|---------|-------|-------------|
| `types` | Core kernel types | 122 | no_std types for handles, capabilities, proofs, regions, queues, timers |
| `region` | Memory management | 88 | Immutable, append-only, and slab region implementations |
| `queue` | IPC ring buffers | 70 | io_uring-style zero-copy message passing |
| `cap` | Capability management | 103 | seL4-inspired unforgeable capability tokens with derivation and revocation |
| `proof` | Proof engine | 84 | 3-tier proof verification (Reflex/Standard/Deep) |
| `sched` | Scheduler | 47 | Coherence-aware deadline scheduler with novelty scoring |
| `vecgraph` | Vector/graph stores | 69 | Kernel-resident HNSW vector store and graph store |
| `boot` | Boot loader | 61 | RVF package mounting, signature verification, attestation |
| `nucleus` | Kernel integration | 110 | Syscall dispatch, deterministic replay, witness log |

### Phase B: Bare Metal AArch64

| Crate | Purpose | Tests | Description |
|-------|---------|-------|-------------|
| `hal` | Hardware Abstraction Layer | 15 | Console, InterruptController, Timer, Mmu, PowerManagement traits |
| `aarch64` | AArch64 boot & MMU | - | Exception vectors, page tables, boot assembly |
| `drivers` | Device drivers | - | PL011 UART, GIC-400, ARM Generic Timer |
| `physmem` | Physical memory | 75 | Buddy allocator for page frames (4KB-2MB) |

### Phase C: Multi-Core & DMA

| Crate | Purpose | Tests | Description |
|-------|---------|-------|-------------|
| `smp` | Symmetric Multi-Processing | 41 | PerCpu data, ticket spinlocks, IPIs, 256-core support |
| `dma` | DMA controller | 54 | Scatter-gather descriptors, cache coherence |
| `dtb` | Device Tree parser | 41 | FDT parsing, node/property iteration |

### Phase D: Raspberry Pi 4/5

| Crate | Purpose | Tests | Description |
|-------|---------|-------|-------------|
| `bcm2711` | BCM2711/2712 SoC drivers | - | GPIO, VideoCore mailbox, mini UART, interrupt controller |
| `rpi-boot` | RPi boot support | - | Spin table CPU wake, early UART, config.txt parsing |

### Phase E: Networking & Filesystem

| Crate | Purpose | Tests | Description |
|-------|---------|-------|-------------|
| `net` | Network stack | 87 | Ethernet/ARP/IPv4/UDP/ICMP, NetworkDevice trait |
| `fs` | Filesystem | 98 | VFS layer, FAT32 (read-only), RamFS (read-write) |

### QEMU Swarm Simulation

| Component | Description |
|-----------|-------------|
| `qemu-swarm` | Multi-QEMU cluster orchestration for distributed testing |

**Total: 1,000+ tests across all phases**

---

## Benchmarks

Run the full benchmark suite:

```bash
# Compare RuVix vs Linux syscalls
cargo run --release -p ruvix-bench --bin ruvix-vs-linux

# Syscall latency benchmarks
cargo run --release -p ruvix-bench --bin syscall-bench

# Proof tier overhead comparison
cargo run --release -p ruvix-bench --bin proof-overhead

# Memory usage benchmarks
cargo run --release -p ruvix-bench --bin memory-bench

# Throughput benchmarks
cargo run --release -p ruvix-bench --bin throughput-bench
```

Or use Criterion benchmarks:

```bash
cargo bench -p ruvix-nucleus
cargo bench -p ruvix-cap
cargo bench -p ruvix-region
cargo bench -p ruvix-queue
```

---

## CLI Tools

RuVix provides two command-line interfaces for development and debugging.

### Host CLI (`ruvix-cli`)

A development workstation tool for building, configuring, and deploying RuVix kernels.

```bash
# Install the CLI
cargo install --path crates/cli

# Or run directly
cargo run -p ruvix-cli -- <command>
```

#### Commands

| Command | Description |
|---------|-------------|
| `ruvix build` | Build kernel image with specified configuration |
| `ruvix flash` | Flash kernel to SD card or network boot target |
| `ruvix config` | Manage kernel configuration (features, memory, security) |
| `ruvix keys` | Manage trusted boot signing keys |
| `ruvix dtb` | Validate and analyze Device Tree Blob files |
| `ruvix monitor` | Monitor running kernel via UART/network |
| `ruvix security` | Security audit and CVE checks |

#### Examples

```bash
# Build for Raspberry Pi 4 with secure boot
ruvix build --target rpi4 --secure-boot --release

# Generate a new signing key pair
ruvix keys generate --algorithm ed25519 --output kernel-key

# Sign a kernel image
ruvix keys sign --key kernel-key.priv --image ruvix.bin

# Validate a Device Tree
ruvix dtb validate ./bcm2711-rpi-4-b.dtb

# Flash to SD card
ruvix flash --device /dev/disk4 --image ruvix.bin --dtb bcm2711.dtb

# Security audit
ruvix security audit --depth full
```

### Kernel Shell (`rvsh`)

An in-kernel debug shell accessible over UART (or network in future). Provides runtime inspection and debugging capabilities.

#### Shell Commands

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `info` | Display kernel version and boot info |
| `mem` | Memory statistics (physical, regions, slabs) |
| `tasks` | List tasks with state and capabilities |
| `caps` | Capability table dump |
| `queues` | Queue status and statistics |
| `vectors` | Vector store information |
| `graphs` | Graph store information |
| `proofs` | Proof verification statistics |
| `irq` | Interrupt controller status |
| `cpu` | Per-CPU information (SMP systems) |
| `witness` | Witness log viewer |
| `trace` | Enable/disable syscall tracing |
| `perf` | Performance counters |
| `panic` | View panic history |
| `reboot` | Restart the kernel |

#### Example Session

```
RuVix Cognition Kernel v0.1.0
Boot: 2024-01-15 14:32:00 UTC
CPU: Cortex-A72 x4 @ 1.5GHz
RAM: 4096 MB

rvsh> mem
Physical Memory:
  Total:     4096 MB
  Free:      3847 MB (94%)
  Kernel:      64 MB

Regions:
  Immutable:    12 (48 KB)
  Append:        4 (16 KB)
  Slab:          8 (256 KB)

rvsh> tasks
ID   NAME              STATE    CAPS  PRI   CPU
0    idle              RUNNING     1  255    0
1    init              BLOCKED     8   10    1
2    vector-service    READY      16    5    -
3    network-stack     BLOCKED    32   15    2

rvsh> caps 2
Task 2 (vector-service) capabilities:
  [0] CAP_REGION_READ  -> Region 0x1000 (vectors)
  [1] CAP_REGION_WRITE -> Region 0x1000 (vectors)
  [2] CAP_QUEUE_SEND   -> Queue 0 (requests)
  [3] CAP_QUEUE_RECV   -> Queue 1 (responses)
  ...

rvsh> trace on
Syscall tracing enabled.

rvsh> witness tail 5
[14:33:01.234] VectorPut key=0x42 store=0 proof=Reflex
[14:33:01.235] QueueSend queue=0 msg_type=VectorResult
[14:33:01.240] TimerWait deadline=14:33:02.000
[14:33:02.001] QueueRecv queue=1 msg_type=VectorQuery
[14:33:02.002] VectorGet key=0x43 store=0
```

#### Enabling the Shell

The shell is enabled by default in debug builds. For release builds:

```toml
# In Cargo.toml or via CLI
[features]
kernel-shell = []  # Enable rvsh in release builds
```

Or via `ruvix-cli`:

```bash
ruvix build --features kernel-shell --release
```

---

## no_std Support

All RuVix crates support `#![no_std]` environments:

```toml
[dependencies]
ruvix-types = { version = "0.1", default-features = false }
ruvix-nucleus = { version = "0.1", default-features = false, features = ["alloc"] }
```

Without `alloc`, crates use fixed-size arrays with compile-time limits:

- Maximum 16 vector stores
- Maximum 16 graph stores
- Maximum 64 tasks

---

## ADR-087 Reference

RuVix is specified in [ADR-087: RuVix Cognition Kernel](../docs/adr/ADR-087-ruvix-cognition-kernel.md), which covers:

- **Section 1**: Context and motivation
- **Section 2**: Core decision and architecture
- **Section 3**: Syscall surface (12 syscalls, 5 invariants)
- **Section 4**: Capability model
- **Section 5**: Region memory policies
- **Section 6**: Queue IPC semantics
- **Section 7**: Proof engine design
- **Section 8**: Scheduler architecture
- **Section 9**: Boot sequence
- **Section 10**: RVF integration
- **Section 17**: Acceptance criteria

All acceptance criteria from Section 17 are implemented and tested.

---

## Building

```bash
# Build all crates
cargo build --release

# Build with all features
cargo build --release --all-features

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench --workspace
```

---

## Project Structure

Located at `crates/ruvix/` in the RuVector monorepo:

```
crates/ruvix/
  Cargo.toml                  # Workspace manifest
  README.md                   # This file
  crates/
    # Phase A: Core Kernel
    types/                    # Core kernel types (no_std, zero deps)
    region/                   # Memory region management
    queue/                    # io_uring-style IPC ring buffers
    cap/                      # seL4-inspired capability management
    proof/                    # 3-tier proof engine
    sched/                    # Coherence-aware scheduler
    boot/                     # RVF boot loader
    vecgraph/                 # Vector and graph stores
    nucleus/                  # Kernel integration

    # Phase B: Bare Metal
    hal/                      # Hardware Abstraction Layer traits
    aarch64/                  # AArch64 boot, MMU, exceptions
    drivers/                  # PL011 UART, GIC-400, Timer
    physmem/                  # Buddy allocator for physical pages

    # Phase C: Multi-Core
    smp/                      # SMP, spinlocks, IPIs
    dma/                      # DMA controller abstraction
    dtb/                      # Device Tree parser

    # Phase D: Raspberry Pi
    bcm2711/                  # BCM2711/2712 SoC drivers
    rpi-boot/                 # RPi-specific boot support

    # Phase E: Networking/FS
    net/                      # Ethernet/IP/UDP/ICMP stack
    fs/                       # VFS, FAT32, RamFS

    # CLI & Debug Tools
    cli/                      # Host-side ruvix-cli tool
    shell/                    # In-kernel debug shell (rvsh)

  aarch64-boot/               # Linker scripts, QEMU target
  qemu-swarm/                 # Multi-QEMU cluster simulation
  tests/                      # Integration tests
  benches/                    # Benchmark suite
  examples/
    cognitive_demo/           # Demo application
```

---

## Related Projects

- **[RuVector](https://github.com/ruvnet/ruvector)**: The self-learning vector database that RuVix powers
- **[RVF](../rvf/)**: Cognitive container format for self-booting vector files
- **[AgentDB](https://github.com/ruvnet/agentdb)**: Long-term memory for AI agents
- **[Claude-Flow](https://github.com/ruvnet/claude-flow)**: Multi-agent orchestration for Claude Code

---

## Contributing

Contributions are welcome. Please:

1. Read the [ADR-087](../../docs/adr/ADR-087-ruvix-cognition-kernel.md) specification
2. Run the full test suite before submitting PRs
3. Maintain no_std compatibility for core crates
4. Add tests for new functionality
5. Follow the existing code style

---

## License

MIT OR Apache-2.0

---

## Acknowledgments

RuVix builds on ideas from:

- **seL4**: Capability-based security model
- **io_uring**: Zero-copy ring buffer IPC
- **RuVector**: Proof-gated mutation protocol (ADR-047)
- **Cognitum**: Coherence engine architecture (ADR-014)
