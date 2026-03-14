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

---

## Phase B: Bare Metal AArch64 Port

### B.1 Objectives

Phase B transforms the Linux-hosted nucleus into a true bare metal microkernel running natively on AArch64 hardware. Key objectives:

1. **Remove all Linux/std dependencies** — Eliminate libc, pthreads, mmap, and all Linux syscalls
2. **Native AArch64 boot** — Boot directly on QEMU virt machine and Raspberry Pi 4/5 hardware
3. **Hardware capability enforcement** — Use MMU page tables for region protection and capability boundaries
4. **Real interrupt handling** — GIC-400 interrupt controller integration for device drivers
5. **Hardware timer integration** — ARM Generic Timer for scheduling and `timer_wait` syscall

### B.2 New Crates

| Crate | Purpose | Dependencies | Lines of Code (Est.) |
|-------|---------|--------------|---------------------|
| **`ruvix-hal`** | Hardware Abstraction Layer traits | `ruvix-types` | ~500 |
| **`ruvix-aarch64`** | AArch64 boot, MMU, exceptions | `ruvix-hal`, `ruvix-types` | ~2,000 |
| **`ruvix-drivers`** | PL011 UART, GIC-400, ARM Timer | `ruvix-hal`, `ruvix-aarch64` | ~1,500 |
| **`ruvix-physmem`** | Physical memory allocator | `ruvix-types`, `ruvix-aarch64` | ~800 |

These join the Phase A primitives table:

| Crate | Status | Tests | Description |
|-------|--------|-------|-------------|
| `ruvix-hal` | ⏳ Phase B | TBD | HAL traits for UART, timer, interrupt controller, MMU |
| `ruvix-aarch64` | ⏳ Phase B | TBD | Exception vectors, MMU tables, boot sequence |
| `ruvix-drivers` | ⏳ Phase B | TBD | PL011 UART, GIC-400, ARM Timer drivers |
| `ruvix-physmem` | ⏳ Phase B | TBD | Buddy allocator for physical page frames |

### B.3 Memory Map (QEMU virt)

The QEMU AArch64 virt machine provides the following memory layout:

| Region | Start Address | End Address | Size | Purpose |
|--------|--------------|-------------|------|---------|
| **Flash** | `0x0000_0000` | `0x0800_0000` | 128 MB | Boot ROM, RVF package location |
| **GIC Distributor** | `0x0800_0000` | `0x0801_0000` | 64 KB | GIC-400 distributor registers |
| **GIC CPU Interface** | `0x0801_0000` | `0x0802_0000` | 64 KB | GIC-400 CPU interface registers |
| **UART** | `0x0900_0000` | `0x0900_1000` | 4 KB | PL011 UART registers |
| **RTC** | `0x0901_0000` | `0x0901_1000` | 4 KB | PL031 Real-Time Clock |
| **GPIO** | `0x0903_0000` | `0x0903_1000` | 4 KB | PL061 GPIO controller |
| **PCIe Config** | `0x4000_0000` | `0x4010_0000` | 256 MB | PCIe ECAM configuration space |
| **RAM** | `0x4000_0000` | `0x8000_0000` | 1 GB | Main system memory (default) |

**Raspberry Pi 4/5 Differences:**
- Base RAM starts at `0x0000_0000` (not `0x4000_0000`)
- GIC base at `0xFF84_0000` (BCM2711 interrupt controller)
- UART0 at `0xFE20_1000` (mini UART) or UART2 at `0xFE20_1400` (PL011)
- Device tree required for full peripheral discovery

### B.4 Boot Sequence

The bare metal boot sequence consists of five stages:

#### Stage 1: Assembly Entry (_start)

```
_start:
    // Executed at EL1 (kernel mode)
    // x0 = DTB address (from bootloader)

    1. Check execution level (ensure EL1)
    2. Set up exception vector table (VBAR_EL1)
    3. Initialize stack pointer (SP_EL1 = kernel_stack_top)
    4. Clear BSS segment (zero-fill static data)
    5. Save DTB address to known location
    6. Jump to Rust entry point (kernel_entry)
```

**Registers on entry:**
- `x0` — Device Tree Blob (DTB) address
- `x1-x3` — Reserved (bootloader-specific)
- `PC` — Entry point (`_start` in boot ROM or RAM)

**Assembly file:** `ruvix-aarch64/src/boot.S` (~150 lines)

#### Stage 2: MMU Initialization

Before Rust code can execute, the kernel must set up virtual memory:

```rust
// ruvix-aarch64/src/mmu.rs
pub unsafe fn init_mmu() {
    // 1. Identity map kernel code/data (VA = PA)
    //    0x4000_0000 - 0x4010_0000 (16 MB)
    //    Read-only for code, RW for data, no user access

    // 2. Map kernel heap region
    //    VA: 0xFFFF_FF00_0000_0000 (canonical high)
    //    PA: (allocated physical pages)
    //    RW, no user access

    // 3. Map device MMIO regions
    //    GIC, UART, timers — uncacheable, device memory

    // 4. Enable MMU (SCTLR_EL1.M = 1)
}
```

**Page table structure:**
- 4 KB pages, 4-level translation (TTBR0_EL1 for user, TTBR1_EL1 for kernel)
- Kernel mappings in top canonical half (`0xFFFF_FF00_0000_0000+`)
- Physical memory allocator initialized from DTB memory nodes

#### Stage 3: Exception Vectors and GIC Initialization

```rust
// ruvix-aarch64/src/exceptions.rs
#[repr(C, align(2048))]
pub struct ExceptionVectorTable {
    sync_el1_sp0:  extern "C" fn(),  // Synchronous from EL1 using SP_EL0
    irq_el1_sp0:   extern "C" fn(),  // IRQ from EL1 using SP_EL0
    fiq_el1_sp0:   extern "C" fn(),  // FIQ from EL1 using SP_EL0
    serror_el1_sp0: extern "C" fn(), // SError from EL1 using SP_EL0
    // ... (16 total vectors for all EL/SP combinations)
}

pub unsafe fn init_exceptions() {
    let vector_table = &EXCEPTION_VECTORS as *const _ as u64;
    asm!("msr VBAR_EL1, {}", in(reg) vector_table);
}
```

**GIC-400 initialization:**
```rust
// ruvix-drivers/src/gic400.rs
pub unsafe fn init_gic() {
    // 1. Enable distributor (GICD_CTLR)
    // 2. Configure interrupt priorities (GICD_IPRIORITYR)
    // 3. Set interrupt targets (GICD_ITARGETSR) — all to CPU0
    // 4. Enable CPU interface (GICC_CTLR)
    // 5. Set priority mask (GICC_PMR) — allow all priorities
    // 6. Enable specific interrupts (GICD_ISENABLER)
    //    - UART RX interrupt (IRQ 33)
    //    - Timer interrupt (IRQ 27 for secure physical timer)
}
```

#### Stage 4: Kernel Entry and Capability Table Initialization

```rust
// ruvix-nucleus/src/main.rs
#[no_mangle]
pub extern "C" fn kernel_entry(dtb_addr: usize) -> ! {
    // 1. Parse device tree to discover memory layout
    let dtb = unsafe { DeviceTree::from_addr(dtb_addr) };
    let memory_nodes = dtb.memory_nodes();

    // 2. Initialize physical memory allocator
    let mut phys_allocator = PhysicalAllocator::new(memory_nodes);

    // 3. Initialize region manager (convert mmap to physical pages)
    let region_mgr = RegionManager::new(&mut phys_allocator);

    // 4. Initialize capability manager
    let cap_mgr = CapabilityManager::new();

    // 5. Create root task with all initial capabilities
    let root_task = Task::new_root(&cap_mgr, &region_mgr);

    // 6. Initialize queue manager
    let queue_mgr = QueueManager::new(&mut phys_allocator);

    // 7. Initialize proof engine
    let proof_engine = ProofEngine::new(&cap_mgr);

    // 8. Initialize vector/graph kernel objects
    let vecgraph_mgr = VecGraphManager::new(&mut phys_allocator, &proof_engine);

    // 9. Initialize scheduler
    let scheduler = Scheduler::new();
    scheduler.add_task(root_task);

    // 10. Load boot RVF and jump to first component
    load_boot_rvf_and_start(&cap_mgr, &region_mgr, &queue_mgr);

    // 11. Enter scheduler loop (never returns)
    scheduler.run()
}
```

#### Stage 5: First RVF Component Load

```rust
// ruvix-boot/src/loader.rs
pub fn load_boot_rvf_and_start(
    cap_mgr: &CapabilityManager,
    region_mgr: &RegionManager,
    queue_mgr: &QueueManager,
) {
    // 1. Read RVF from flash at 0x0000_0000
    let rvf_bytes = unsafe { read_flash(0x0000_0000, RVF_MAX_SIZE) };

    // 2. Verify ML-DSA-65 signature (Stage 1 from Phase A)
    let manifest = rvf::verify_and_parse(&rvf_bytes)
        .expect("Boot RVF signature verification failed");

    // 3. Create regions per memory schema
    for region_spec in manifest.memory_schema {
        region_mgr.create_region(region_spec.size, region_spec.policy);
    }

    // 4. Load WASM components into executable regions
    for component in manifest.components {
        let code_region = region_mgr.map_immutable(&component.wasm_bytes);
        // WASM runtime initialization happens here (Wasmtime or wasm-micro-runtime)
    }

    // 5. Wire queues per manifest
    for queue_spec in manifest.queue_wiring {
        queue_mgr.create_queue(queue_spec.size, queue_spec.schema);
    }

    // 6. Spawn initial tasks
    for entry_point in manifest.entry_points {
        let task = Task::spawn(
            entry_point.component_id,
            entry_point.caps,
            TaskPriority::Normal,
            None, // no deadline
        );
        scheduler.add_task(task);
    }

    // 7. Emit boot attestation
    kernel.attest_emit(
        &AttestPayload::Boot {
            rvf_hash: manifest.hash,
            timestamp: timer.now(),
        },
        &ProofToken::trusted_boot(),
    );
}
```

### B.5 Security Model Enhancements

#### MMU-Enforced Capability Boundaries

In Phase A (Linux-hosted), region protection relied on `mprotect()`. In Phase B, the kernel directly controls page table entries:

```rust
// ruvix-region/src/region.rs (Phase B version)
impl RegionManager {
    pub fn map_region(&mut self, policy: RegionPolicy, size: usize) -> Result<RegionHandle> {
        let phys_pages = self.phys_allocator.allocate_pages(pages_for(size))?;

        let pte_flags = match policy {
            RegionPolicy::Immutable => {
                // User-accessible, read-only, cacheable
                PTE_USER | PTE_RO | PTE_CACHEABLE
            }
            RegionPolicy::AppendOnly { .. } => {
                // Kernel-only, write-append enforced via capability checks
                PTE_KERNEL_RW | PTE_CACHEABLE
            }
            RegionPolicy::Slab { .. } => {
                // Kernel-only, full RW
                PTE_KERNEL_RW | PTE_CACHEABLE
            }
        };

        // Map into kernel address space with appropriate permissions
        let virt_addr = self.mmu.map_pages(phys_pages, pte_flags)?;

        Ok(RegionHandle {
            virt_addr,
            phys_pages,
            policy,
            size,
        })
    }
}
```

**Page table entry flags:**
```rust
const PTE_VALID:      u64 = 1 << 0;   // Valid entry
const PTE_USER:       u64 = 1 << 6;   // User-accessible (EL0)
const PTE_RO:         u64 = 1 << 7;   // Read-only (AP[2])
const PTE_KERNEL_RW:  u64 = 0 << 6;   // Kernel-only, RW
const PTE_CACHEABLE:  u64 = 0b11 << 2; // Normal memory, write-back
const PTE_DEVICE:     u64 = 0b00 << 2; // Device memory, strongly-ordered
```

#### EL1/EL0 Separation for Kernel/User

- **EL1 (Kernel mode):** All kernel code, syscall handlers, interrupt handlers, scheduler
- **EL0 (User mode):** All RVF components, WASM runtime, AgentDB, RuView

Syscalls trigger synchronous exceptions (SVC instruction) and transition EL0 → EL1. The exception handler validates capabilities before dispatching to syscall implementation.

```rust
// ruvix-aarch64/src/syscall.rs
#[no_mangle]
pub extern "C" fn svc_handler(syscall_num: u64, args: &SyscallArgs) -> SyscallResult {
    let current_task = scheduler::current_task();

    match syscall_num {
        0 => task_spawn(current_task, args),
        1 => cap_grant(current_task, args),
        2 => region_map(current_task, args),
        // ... (12 total syscalls)
        _ => Err(KernelError::InvalidSyscall),
    }
}
```

#### Secure Monitor Calls for Attestation

On hardware with Arm TrustZone (Raspberry Pi 4/5), the kernel can invoke Secure Monitor calls (SMC instruction) to request cryptographic operations from the Trusted Execution Environment (TEE):

```rust
// ruvix-aarch64/src/smc.rs
pub fn secure_hash(data: &[u8]) -> [u8; 32] {
    // SMC call to Secure World for hardware-backed SHA-256
    let result = unsafe {
        smc_call(SMC_HASH_SHA256, data.as_ptr(), data.len())
    };
    result.hash
}

pub fn secure_sign_attestation(attestation: &ProofAttestation) -> Signature {
    // SMC call to Secure World for ML-DSA-65 signing using device key
    unsafe {
        smc_call(SMC_SIGN_MLDSA65, attestation as *const _, size_of::<ProofAttestation>())
    }.signature
}
```

**Use cases:**
- Boot attestation signed by device-unique key (burned into eFUSE)
- Witness log entries signed by TEE to prevent kernel compromise from tampering
- Remote attestation for distributed RuVix mesh (Demo 5)

### B.6 Build Path (Weeks 19-42)

| Week | Milestone | Deliverables | Tests |
|------|-----------|--------------|-------|
| **19-20** | AArch64 bootstrap | `boot.S`, exception vectors, minimal UART output | QEMU boots, prints "RuVix" |
| **21-22** | MMU setup | 4-level page tables, identity + kernel mappings | Virtual memory functional |
| **23-24** | Physical allocator | Buddy allocator for 4KB pages from DTB | Unit tests, fuzz tests |
| **25-26** | Region → MMU | `RegionManager` using page tables, not mmap | Phase A region tests pass |
| **27-28** | GIC-400 driver | Interrupt enable/disable, IRQ routing | Timer IRQ delivered to kernel |
| **29-30** | UART driver | PL011 TX/RX with interrupt support | Console I/O functional |
| **31-32** | ARM Timer driver | Generic Timer for `timer_wait`, scheduler ticks | Deadline scheduling works |
| **33-34** | Queue → interrupts | Device interrupts delivered as queue messages | `sensor_subscribe` functional |
| **35-36** | WASM runtime port | Wasmtime or wasm-micro-runtime on bare metal | "Hello World" WASM executes |
| **37-38** | Scheduler on hardware | Coherence-aware scheduler using timer IRQs | Multi-task preemption works |
| **39-40** | RVF boot on QEMU | Full boot sequence from flash RVF | Phase A acceptance test (QEMU) |
| **41-42** | Raspberry Pi 4 port | Device tree parsing, BCM2711 peripherals | Acceptance test (real hardware) |

### B.7 Testing Strategy

#### QEMU Integration Tests

```bash
# ruvix-test/qemu_integration.sh
qemu-system-aarch64 \
  -machine virt,gic-version=3 \
  -cpu cortex-a72 \
  -m 1G \
  -kernel target/aarch64-unknown-none/release/ruvix-kernel \
  -drive file=boot.rvf,format=raw,if=pflash \
  -nographic \
  -semihosting-config enable=on,target=native
```

**Test cases:**
1. Boot sequence completes without panic
2. MMU maps kernel and user regions correctly
3. Timer interrupt fires at 100 Hz
4. UART TX/RX works (loopback test)
5. GIC delivers interrupts to correct handlers
6. RVF signature verification passes/fails as expected
7. Phase A acceptance test (vector mutation + replay)

#### Raspberry Pi 4 Hardware Tests

```bash
# Copy kernel to SD card FAT32 partition
cp target/aarch64-unknown-none/release/ruvix-kernel /media/boot/kernel8.img
cp boot.rvf /media/boot/

# config.txt
arm_64bit=1
kernel=kernel8.img
enable_uart=1
```

**Test cases:**
1. UART console output appears on GPIO 14/15
2. LED blink test using GPIO driver
3. USB keyboard input via queue subscription
4. Network packet reception via queue (if Ethernet driver implemented)
5. Thermal monitoring via RPi-specific sensors

### B.8 Known Limitations

1. **No symmetric multiprocessing (SMP)** — Phase B targets single-core. Multi-core support requires per-CPU interrupt handling and scheduler affinity (future ADR).

2. **No DMA** — Device drivers use programmed I/O. DMA requires IOMMU configuration and physical address constraints (future enhancement).

3. **No floating-point in kernel** — AArch64 NEON/SVE disabled in kernel mode. Vector distance computations may use integer quantized formats or userspace WASM components (decision pending).

4. **Fixed memory layout** — No dynamic memory expansion. All regions declared at boot in RVF manifest.

5. **No power management** — No CPU idle states, frequency scaling, or suspend/resume. Target: always-on edge devices.

---

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

---

## Phase C: Multi-Core and DMA Support

### C.1 Objectives

Phase C extends the bare metal kernel with symmetric multi-processing (SMP) and DMA capabilities, enabling parallel execution across multiple CPU cores and zero-copy I/O operations.

1. **Symmetric Multi-Processing (SMP)** — Support up to 256 cores with per-CPU data structures, spinlocks, and inter-processor interrupts (IPIs)
2. **DMA Controller Abstraction** — Zero-copy I/O for high-bandwidth peripherals (network, storage, sensors)
3. **Device Tree Parsing** — Runtime hardware discovery for portable multi-platform support
4. **Memory Coherence** — Proper cache and memory barrier usage for SMP correctness
5. **Per-CPU Scheduling** — Load balancing with coherence-aware task migration

### C.2 New Crates

| Crate | Purpose | Dependencies | Lines of Code (Est.) |
|-------|---------|--------------|---------------------|
| **`ruvix-smp`** | Multi-core boot, per-CPU data, spinlocks, IPIs | `ruvix-aarch64`, `ruvix-types` | ~1,000 |
| **`ruvix-dma`** | DMA controller abstraction for zero-copy I/O | `ruvix-hal`, `ruvix-physmem` | ~500 |
| **`ruvix-dtb`** | Device tree blob (DTB) parser for hardware discovery | `ruvix-types` | ~600 |

Updated primitives table:

| Crate | Status | Tests | Description |
|-------|--------|-------|-------------|
| `ruvix-smp` | Phase C | TBD | SMP boot, per-CPU data, spinlocks, IPIs |
| `ruvix-dma` | Phase C | TBD | DMA controller abstraction, scatter-gather lists |
| `ruvix-dtb` | Phase C | TBD | Flattened Device Tree parser, memory/peripheral discovery |

### C.3 SMP Boot Sequence

Secondary CPU bring-up follows the ARM PSCI (Power State Coordination Interface) v0.2 specification, with spin-table fallback for platforms without PSCI.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SMP BOOT SEQUENCE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PRIMARY CPU (CPU 0)                    SECONDARY CPUs (1..N)            │
│  ═════════════════                      ═══════════════════════          │
│                                                                          │
│  1. Execute _start                      1. Held in firmware/spin-table   │
│     │                                                                    │
│  2. Initialize MMU, GIC                                                  │
│     │                                                                    │
│  3. Parse DTB → discover CPUs                                            │
│     │                                                                    │
│  4. Allocate per-CPU data                                                │
│     │                                                                    │
│  5. For each secondary CPU:             2. PSCI: CPU_ON or spin-release  │
│     │   ┌──────────────────────────────────────────────────────────┐     │
│     └──▶│ smc #0 (PSCI CPU_ON)  OR  write spin-table release addr │     │
│         └──────────────────────────────────────────────────────────┘     │
│                                              │                           │
│                                         3. Jump to secondary_entry       │
│                                              │                           │
│                                         4. Initialize per-CPU state      │
│                                              │                           │
│                                         5. Enable local GIC interface    │
│                                              │                           │
│                                         6. Signal ready via IPI          │
│  6. Wait for all CPUs ready                  │                           │
│     ◀────────────────────────────────────────┘                           │
│     │                                                                    │
│  7. Enter scheduler                     7. Enter scheduler               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### PSCI-Based Boot (Preferred)

```rust
// ruvix-smp/src/boot.rs
pub unsafe fn boot_secondary_psci(cpu_id: usize, entry_point: usize) -> Result<(), PsciError> {
    let result = smc_call(
        PSCI_CPU_ON,           // Function ID
        cpu_id as u64,         // Target CPU MPIDR
        entry_point as u64,    // Entry point address
        0,                     // Context ID (passed in x0)
    );

    match result {
        0 => Ok(()),                          // SUCCESS
        PSCI_E_ALREADY_ON => Ok(()),          // Already running
        PSCI_E_ON_PENDING => Ok(()),          // Boot in progress
        err => Err(PsciError::from(err)),
    }
}
```

#### Spin-Table Boot (Fallback)

```rust
// ruvix-smp/src/boot.rs
pub unsafe fn boot_secondary_spintable(
    cpu_id: usize,
    spin_table_addr: *mut u64,
    entry_point: usize,
) {
    // Write entry point to spin-table release address
    core::ptr::write_volatile(spin_table_addr, entry_point as u64);

    // Data synchronization barrier ensures write is visible
    asm!("dsb sy");

    // Send event to wake sleeping CPUs
    asm!("sev");
}
```

#### Per-CPU Data Structure

```rust
// ruvix-smp/src/percpu.rs
#[repr(C, align(64))]  // Cache-line aligned to prevent false sharing
pub struct PerCpuData {
    /// CPU identifier (MPIDR_EL1 affinity)
    pub cpu_id: u64,

    /// Current running task (if any)
    pub current_task: Option<TaskHandle>,

    /// Per-CPU scheduler run queue
    pub run_queue: RunQueue,

    /// Per-CPU timer state
    pub timer: TimerState,

    /// Per-CPU GIC interface state
    pub gic_cpu: GicCpuState,

    /// Per-CPU exception stack pointer
    pub exception_stack: *mut u8,

    /// Boot synchronization flag
    pub ready: AtomicBool,

    /// Padding to fill cache line
    _pad: [u8; 16],
}

impl PerCpuData {
    /// Access current CPU's data via TPIDR_EL1 register
    pub fn current() -> &'static mut PerCpuData {
        let ptr: *mut PerCpuData;
        unsafe {
            asm!("mrs {}, TPIDR_EL1", out(reg) ptr);
            &mut *ptr
        }
    }
}
```

### C.4 Memory Barriers and Synchronization

SMP correctness requires careful use of AArch64 memory barriers. The kernel enforces these invariants:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEMORY BARRIER USAGE GUIDE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  BARRIER        │ INSTRUCTION │ USE CASE                                 │
│  ═══════════════│═════════════│══════════════════════════════════════    │
│                 │             │                                          │
│  DMB (Data      │ dmb sy      │ Before reading shared data after         │
│  Memory         │ dmb ish     │ acquiring a lock. After writing          │
│  Barrier)       │ dmb ishst   │ shared data before releasing a lock.     │
│                 │             │                                          │
│  DSB (Data      │ dsb sy      │ Before executing WFI/WFE (wait for       │
│  Synchronization│ dsb ish     │ interrupt/event). After memory-mapped    │
│  Barrier)       │ dsb ishst   │ I/O writes to ensure completion.         │
│                 │             │                                          │
│  ISB (Instr.    │ isb         │ After modifying system registers         │
│  Synchronization│             │ (TTBR, VBAR, etc.). After self-          │
│  Barrier)       │             │ modifying code or cache maintenance.     │
│                 │             │                                          │
│  Shareability   │             │                                          │
│  Domains:       │             │                                          │
│    sy  = Full   │             │ All observers in the system              │
│    ish = Inner  │             │ Inner-shareable (all CPUs)               │
│    osh = Outer  │             │ Outer-shareable (clusters + DMA)         │
│                 │             │                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Spinlock Implementation

```rust
// ruvix-smp/src/spinlock.rs

/// Ticket spinlock with proper memory ordering
pub struct SpinLock<T> {
    next_ticket: AtomicU32,
    now_serving: AtomicU32,
    data: UnsafeCell<T>,
}

impl<T> SpinLock<T> {
    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        // Acquire ticket atomically
        let ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed);

        // Spin until our ticket is served
        while self.now_serving.load(Ordering::Acquire) != ticket {
            // WFE: wait for event, reduces power consumption
            unsafe { asm!("wfe") };
        }

        // Acquired — DMB ensures all prior writes are visible
        SpinLockGuard { lock: self }
    }
}

impl<T> Drop for SpinLockGuard<'_, T> {
    fn drop(&mut self) {
        // Release: increment now_serving with Release ordering
        self.lock.now_serving.fetch_add(1, Ordering::Release);

        // SEV: signal event to wake waiting CPUs
        unsafe { asm!("sev") };
    }
}
```

#### Inter-Processor Interrupts (IPIs)

```rust
// ruvix-smp/src/ipi.rs

/// IPI types used by the kernel
#[repr(u8)]
pub enum IpiType {
    /// Reschedule request — target CPU should check run queue
    Reschedule = 0,

    /// TLB shootdown — invalidate TLB entries for an address range
    TlbInvalidate = 1,

    /// Stop CPU — for kernel panic or shutdown
    Stop = 2,

    /// Function call — execute a closure on target CPU
    Call = 3,
}

pub fn send_ipi(target_cpu: usize, ipi_type: IpiType) {
    let gic = GicDistributor::get();

    // Use SGI (Software Generated Interrupt) for IPIs
    // SGI ID 0-15 are available for software use
    let sgi_id = ipi_type as u8;

    gic.send_sgi(target_cpu, sgi_id);
}

pub fn broadcast_ipi(ipi_type: IpiType, include_self: bool) {
    let gic = GicDistributor::get();
    let target = if include_self {
        SgiTarget::AllIncludingSelf
    } else {
        SgiTarget::AllExcludingSelf
    };

    gic.send_sgi_broadcast(target, ipi_type as u8);
}
```

### C.5 DMA Controller Abstraction

```rust
// ruvix-dma/src/lib.rs

/// DMA transfer descriptor
pub struct DmaDescriptor {
    /// Source physical address (for mem-to-dev)
    pub src_addr: PhysAddr,

    /// Destination physical address (for dev-to-mem)
    pub dst_addr: PhysAddr,

    /// Transfer length in bytes
    pub length: usize,

    /// Transfer direction
    pub direction: DmaDirection,

    /// Completion callback (optional)
    pub callback: Option<fn(DmaResult)>,
}

#[derive(Clone, Copy)]
pub enum DmaDirection {
    MemToDevice,
    DeviceToMem,
    MemToMem,
}

/// DMA controller trait implemented by platform-specific drivers
pub trait DmaController {
    /// Allocate a DMA channel
    fn allocate_channel(&self) -> Result<DmaChannel, DmaError>;

    /// Submit a scatter-gather list for transfer
    fn submit_sg(
        &self,
        channel: DmaChannel,
        descriptors: &[DmaDescriptor],
    ) -> Result<DmaTransferId, DmaError>;

    /// Poll for completion (non-blocking)
    fn poll(&self, transfer_id: DmaTransferId) -> DmaStatus;

    /// Wait for completion (blocking)
    fn wait(&self, transfer_id: DmaTransferId) -> DmaResult;

    /// Cancel a pending transfer
    fn cancel(&self, transfer_id: DmaTransferId);
}
```

#### Scatter-Gather DMA for Zero-Copy Queue IPC

```rust
// Integration with ruvix-queue for zero-copy network packet reception

pub fn receive_packet_zero_copy(
    dma: &dyn DmaController,
    queue: &KernelQueue,
    device: &NetworkDevice,
) -> Result<(), DmaError> {
    // Allocate receive buffer from queue's shared region
    let buffer = queue.ring_region.allocate_slot()?;

    // Build DMA descriptor pointing directly to queue buffer
    let desc = DmaDescriptor {
        src_addr: device.rx_fifo_addr(),  // Device FIFO address
        dst_addr: buffer.phys_addr(),      // Queue buffer physical address
        length: MTU,
        direction: DmaDirection::DeviceToMem,
        callback: Some(|result| {
            // On completion, advance queue tail
            queue.advance_cq_tail(result.bytes_transferred);
        }),
    };

    // Submit DMA transfer — no CPU copying involved
    dma.submit_sg(channel, &[desc])?;

    Ok(())
}
```

### C.6 Device Tree Parser

```rust
// ruvix-dtb/src/lib.rs

/// Parsed device tree structure
pub struct DeviceTree<'a> {
    root: DtNode<'a>,
    strings_block: &'a [u8],
    reserved_memory: Vec<MemoryRange>,
}

impl<'a> DeviceTree<'a> {
    /// Parse DTB from a raw pointer (passed by bootloader in x0)
    pub unsafe fn from_ptr(ptr: *const u8) -> Result<Self, DtbError> {
        let header = &*(ptr as *const FdtHeader);

        // Validate magic number
        if u32::from_be(header.magic) != FDT_MAGIC {
            return Err(DtbError::InvalidMagic);
        }

        // Parse structure block, strings block, reserved memory
        // ...
    }

    /// Enumerate all CPUs from /cpus node
    pub fn cpus(&self) -> impl Iterator<Item = CpuNode> + '_ {
        self.root
            .find_node("/cpus")
            .into_iter()
            .flat_map(|cpus| cpus.children())
            .filter(|n| n.name().starts_with("cpu@"))
            .map(CpuNode::from)
    }

    /// Enumerate memory regions from /memory nodes
    pub fn memory_regions(&self) -> impl Iterator<Item = MemoryRange> + '_ {
        self.root
            .find_nodes_by_type("memory")
            .flat_map(|n| n.reg_property())
    }

    /// Find a device by compatible string
    pub fn find_compatible(&self, compat: &str) -> Option<DtNode<'a>> {
        self.root.find_by_compatible(compat)
    }
}

/// CPU node information
pub struct CpuNode {
    pub cpu_id: u64,           // reg property
    pub enable_method: EnableMethod,
    pub spin_table_addr: Option<PhysAddr>,
    pub psci_method: Option<PsciMethod>,
}

pub enum EnableMethod {
    Psci,
    SpinTable,
}
```

### C.7 Build Path (Weeks 43-54)

| Week | Milestone | Deliverables | Tests |
|------|-----------|--------------|-------|
| **43-44** | DTB parser | `ruvix-dtb` crate, CPU/memory enumeration | Unit tests with real Pi4/Pi5 DTBs |
| **45-46** | Per-CPU data | TPIDR_EL1 setup, per-CPU allocator | Each CPU accesses correct data |
| **47-48** | Spinlocks | Ticket spinlock, reader-writer locks | Stress test across all CPUs |
| **49-50** | Secondary boot | PSCI CPU_ON, spin-table fallback | All CPUs reach scheduler |
| **51-52** | IPIs | SGI-based IPIs, TLB shootdown | Cross-CPU reschedule, TLB invalidate |
| **53-54** | DMA controller | GICv3 DMA, scatter-gather | Zero-copy packet reception |

---

## Phase D: Raspberry Pi 4/5 Support

### D.1 Hardware Differences

Phase D provides dedicated support for Raspberry Pi 4 (BCM2711) and Raspberry Pi 5 (BCM2712) hardware. These platforms differ significantly from the QEMU virt machine.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI 4 vs PI 5 COMPARISON                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  FEATURE            │ Raspberry Pi 4 (BCM2711)  │ Raspberry Pi 5 (BCM2712)│
│  ═══════════════════│═══════════════════════════│═════════════════════════│
│  CPU                │ 4x Cortex-A72 @ 1.8GHz    │ 4x Cortex-A76 @ 2.4GHz   │
│  RAM                │ 1/2/4/8 GB LPDDR4         │ 4/8 GB LPDDR4X           │
│  GPU                │ VideoCore VI              │ VideoCore VII            │
│                     │                           │                          │
│  PERIPHERAL BASE    │ 0xFE00_0000               │ 0x1F00_0000_0000         │
│  RAM BASE           │ 0x0000_0000               │ 0x0000_0000              │
│                     │                           │                          │
│  INTERRUPT          │ GIC-400 (GICv2)           │ GIC-500 (GICv3)          │
│  CONTROLLER         │ @ 0xFF84_0000             │ @ 0x10_7FFF_0000         │
│                     │                           │                          │
│  UART               │ PL011 @ 0xFE20_1000       │ PL011 @ 0x1F00_0003_0000 │
│                     │ Mini @ 0xFE21_5000        │                          │
│                     │                           │                          │
│  GPIO               │ @ 0xFE20_0000             │ @ 0x1F00_00D0_0000       │
│                     │                           │                          │
│  MAILBOX            │ @ 0xFE00_B880             │ @ 0x1F00_0000_0880       │
│  (VideoCore IPC)    │                           │                          │
│                     │                           │                          │
│  PCIe               │ 1x Gen 2.0 lane           │ 1x Gen 3.0 lane          │
│                     │                           │ (M.2 NVMe support)       │
│                     │                           │                          │
│  BOOT               │ Start4.elf → kernel8.img │ RP1 chip → kernel_2712.img│
│                     │                           │                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Memory Maps

**Raspberry Pi 4 (BCM2711):**

| Region | Start Address | End Address | Size | Purpose |
|--------|--------------|-------------|------|---------|
| **RAM (Low)** | `0x0000_0000` | `0x3C00_0000` | 960 MB | Main memory (below GPU split) |
| **Mailbox** | `0xFE00_B880` | `0xFE00_B8BF` | 64 B | VideoCore mailbox registers |
| **GPIO** | `0xFE20_0000` | `0xFE20_00F4` | 244 B | GPIO registers |
| **UART0** | `0xFE20_1000` | `0xFE20_1FFF` | 4 KB | PL011 UART |
| **Mini UART** | `0xFE21_5000` | `0xFE21_507F` | 128 B | Aux mini UART |
| **GIC Dist** | `0xFF84_1000` | `0xFF84_1FFF` | 4 KB | GIC-400 distributor |
| **GIC CPU** | `0xFF84_2000` | `0xFF84_2FFF` | 4 KB | GIC-400 CPU interface |
| **Local Periph** | `0xFF80_0000` | `0xFF80_0FFF` | 4 KB | Per-core timers, mailboxes |

**Raspberry Pi 5 (BCM2712):**

| Region | Start Address | End Address | Size | Purpose |
|--------|--------------|-------------|------|---------|
| **RAM** | `0x0000_0000` | `0x1_0000_0000` | 4 GB | Main memory |
| **High Periph** | `0x1F00_0000_0000` | `0x1F00_FFFF_FFFF` | 4 GB | Peripheral space |
| **GIC-500** | `0x10_7FFF_0000` | `0x10_7FFF_FFFF` | 64 KB | GICv3 |
| **RP1 Periph** | `0x1F00_0000_0000` | varies | varies | RP1 south bridge (USB, Ethernet) |

### D.2 New Crates

| Crate | Purpose | Dependencies | Lines of Code (Est.) |
|-------|---------|--------------|---------------------|
| **`ruvix-bcm2711`** | BCM2711/2712 SoC drivers (GPIO, mailbox, UART) | `ruvix-hal`, `ruvix-aarch64` | ~1,200 |
| **`ruvix-rpi-boot`** | RPi-specific boot support (config.txt parsing, stub) | `ruvix-aarch64`, `ruvix-dtb` | ~400 |

### D.3 Boot Process

Raspberry Pi boot differs significantly from QEMU:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI BOOT SEQUENCE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STAGE 0: On-Chip ROM (Raspberry Pi 4/5)                                 │
│  ════════════════════════════════════════                                │
│  1. Power-on reset                                                       │
│  2. Load second-stage bootloader from SD/eMMC/USB                        │
│     - Pi 4: bootcode.bin (from boot partition)                           │
│     - Pi 5: RP1 chip handles initial boot                                │
│                                                                          │
│  STAGE 1: VideoCore Firmware                                             │
│  ════════════════════════════════════════                                │
│  1. Load start4.elf (or start.elf) from boot partition                   │
│  2. Parse config.txt for boot configuration                              │
│  3. Initialize DRAM, clocks, voltage                                     │
│  4. Load kernel8.img (AArch64 kernel) to 0x80000                         │
│  5. Load device tree (bcm2711-rpi-4-b.dtb) to memory                     │
│  6. Release ARM cores with DTB address in x0                             │
│                                                                          │
│  STAGE 2: RuVix Entry (kernel8.img)                                      │
│  ════════════════════════════════════════                                │
│  1. _start at 0x80000 (or address from config.txt)                       │
│  2. Parse DTB to discover hardware                                       │
│  3. Initialize UART for early console                                    │
│  4. Continue standard RuVix boot sequence                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### config.txt Configuration

```ini
# /boot/config.txt for RuVix on Raspberry Pi 4/5

# Enable 64-bit mode
arm_64bit=1

# Kernel filename
kernel=ruvix-kernel8.img

# Enable UART for early console
enable_uart=1

# Disable Bluetooth to free up PL011 UART
dtoverlay=disable-bt

# Fixed kernel load address
kernel_address=0x80000

# Pass DTB to kernel
device_tree=bcm2711-rpi-4-b.dtb

# Disable GPU boot splash
disable_splash=1

# Minimum GPU memory (more for kernel)
gpu_mem=16

# For Pi 5: use correct kernel name
[pi5]
kernel=ruvix-kernel_2712.img
device_tree=bcm2712-rpi-5-b.dtb
```

### D.4 BCM2711/2712 Drivers

#### GPIO Driver

```rust
// ruvix-bcm2711/src/gpio.rs

const GPIO_BASE_PI4: usize = 0xFE20_0000;
const GPIO_BASE_PI5: usize = 0x1F00_00D0_0000;

pub struct BcmGpio {
    base: *mut u32,
}

impl BcmGpio {
    pub fn new(platform: Platform) -> Self {
        let base = match platform {
            Platform::Pi4 => GPIO_BASE_PI4,
            Platform::Pi5 => GPIO_BASE_PI5,
        } as *mut u32;

        Self { base }
    }

    /// Set pin function (input, output, alt0-5)
    pub fn set_function(&self, pin: u8, function: GpioFunction) {
        let reg = (pin / 10) as usize;
        let shift = ((pin % 10) * 3) as u32;

        unsafe {
            let ptr = self.base.add(reg);  // GPFSEL0-5
            let mut val = ptr.read_volatile();
            val &= !(0b111 << shift);
            val |= (function as u32) << shift;
            ptr.write_volatile(val);
        }
    }

    /// Set output pin high/low
    pub fn set(&self, pin: u8, high: bool) {
        let reg = if high { 7 } else { 10 };  // GPSET0 or GPCLR0
        let offset = (pin / 32) as usize;
        let bit = 1u32 << (pin % 32);

        unsafe {
            self.base.add(reg + offset).write_volatile(bit);
        }
    }

    /// Read input pin level
    pub fn read(&self, pin: u8) -> bool {
        let offset = (pin / 32) as usize;
        let bit = 1u32 << (pin % 32);

        unsafe {
            (self.base.add(13 + offset).read_volatile() & bit) != 0
        }
    }
}

#[repr(u32)]
pub enum GpioFunction {
    Input = 0b000,
    Output = 0b001,
    Alt0 = 0b100,
    Alt1 = 0b101,
    Alt2 = 0b110,
    Alt3 = 0b111,
    Alt4 = 0b011,
    Alt5 = 0b010,
}
```

#### VideoCore Mailbox

```rust
// ruvix-bcm2711/src/mailbox.rs

const MAILBOX_BASE_PI4: usize = 0xFE00_B880;

/// Property tag IDs for VideoCore communication
pub mod tags {
    pub const GET_BOARD_REVISION: u32 = 0x0001_0002;
    pub const GET_ARM_MEMORY: u32 = 0x0001_0005;
    pub const GET_VC_MEMORY: u32 = 0x0001_0006;
    pub const SET_POWER_STATE: u32 = 0x0002_8001;
    pub const GET_CLOCK_RATE: u32 = 0x0003_0002;
    pub const SET_CLOCK_RATE: u32 = 0x0003_8002;
}

pub struct Mailbox {
    base: *mut u32,
}

impl Mailbox {
    /// Send a property tag request to VideoCore
    pub fn call(&self, channel: u8, message: &mut PropertyMessage) -> Result<(), MailboxError> {
        // Ensure 16-byte alignment
        let ptr = message as *mut _ as usize;
        assert!(ptr & 0xF == 0, "Mailbox message must be 16-byte aligned");

        // Wait for mailbox to be ready
        while self.status() & MAILBOX_FULL != 0 {
            core::hint::spin_loop();
        }

        // Write message address (with channel in low 4 bits)
        let value = (ptr as u32) | (channel as u32);
        unsafe {
            self.base.add(8).write_volatile(value);  // WRITE register
        }

        // Wait for response
        loop {
            while self.status() & MAILBOX_EMPTY != 0 {
                core::hint::spin_loop();
            }

            let response = unsafe { self.base.read_volatile() };  // READ register
            if response & 0xF == channel as u32 {
                break;
            }
        }

        // Check response code
        if message.code == MAILBOX_RESPONSE_SUCCESS {
            Ok(())
        } else {
            Err(MailboxError::ResponseFailed)
        }
    }
}
```

### D.5 Build Path (Weeks 55-62)

| Week | Milestone | Deliverables | Tests |
|------|-----------|--------------|-------|
| **55-56** | RPi boot stub | Linker script, kernel8.img generation, UART output | Boot to console on Pi 4 |
| **57-58** | GPIO driver | BCM GPIO, LED blink, button input | GPIO functional tests |
| **59-60** | VideoCore mailbox | Property tags, memory query, clock control | Query board info |
| **61-62** | Pi 5 port | BCM2712 addresses, GICv3, RP1 | Boot on Pi 5 hardware |

---

## Phase E: Networking and Filesystem

### E.1 Network Stack Architecture

Phase E implements a minimal network stack optimized for agentic workloads: high-throughput, low-latency vector streaming and RVF package distribution.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      NETWORK STACK ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                    RVF COMPONENT SPACE                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐    │   │
│  │  │ AgentDB     │  │ RVF Package │  │ Mesh Coordinator        │    │   │
│  │  │ Replication │  │ Distributor │  │ (Demo 5)                │    │   │
│  │  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘    │   │
│  │         │                │                     │                  │   │
│  │         └────────────────┼─────────────────────┘                  │   │
│  │                          │ queue                                   │   │
│  └──────────────────────────┼────────────────────────────────────────┘   │
│                             │                                            │
│  ┌──────────────────────────┼────────────────────────────────────────┐   │
│  │  ruvix-net (RVF component)                                        │   │
│  │  ═════════════════════════                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │                     SOCKET LAYER                             │  │   │
│  │  │   UdpSocket   │   TcpListener (future)   │   RawSocket      │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  │                             │                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │                    TRANSPORT LAYER                           │  │   │
│  │  │        UDP         │        ICMP        │     (TCP future)   │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  │                             │                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │                    NETWORK LAYER (IPv4)                      │  │   │
│  │  │   IP Header   │   Routing Table   │   ICMP Echo/Reply       │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  │                             │                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │                    LINK LAYER                                │  │   │
│  │  │   ARP Cache   │   Ethernet Frame   │   MAC Address          │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  │                             │                                      │   │
│  └─────────────────────────────┼─────────────────────────────────────┘   │
│                                │ queue_send/recv                         │
│  ┌─────────────────────────────┼─────────────────────────────────────┐   │
│  │  KERNEL                     │                                      │   │
│  │  ═══════                    │                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │                 NETWORK DEVICE DRIVER                        │  │   │
│  │  │   virtio-net (QEMU)  │   bcmgenet (Pi4)  │   RP1 (Pi5)      │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  │                                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### virtio-net Driver

```rust
// ruvix-drivers/src/virtio_net.rs

const VIRTIO_NET_F_MAC: u64 = 1 << 5;
const VIRTIO_NET_F_MRG_RXBUF: u64 = 1 << 15;

pub struct VirtioNet {
    common: VirtioCommon,
    rx_queue: VirtQueue,
    tx_queue: VirtQueue,
    mac_address: [u8; 6],
}

impl VirtioNet {
    pub fn init(base: *mut u8) -> Result<Self, VirtioError> {
        let common = VirtioCommon::new(base)?;

        // Negotiate features
        let features = common.read_features();
        let negotiated = features & (VIRTIO_NET_F_MAC | VIRTIO_NET_F_MRG_RXBUF);
        common.write_features(negotiated);

        // Initialize virtqueues
        let rx_queue = VirtQueue::new(&common, 0, 256)?;  // Queue 0 = RX
        let tx_queue = VirtQueue::new(&common, 1, 256)?;  // Queue 1 = TX

        // Read MAC address
        let mut mac = [0u8; 6];
        for i in 0..6 {
            mac[i] = common.read_config(i);
        }

        common.set_driver_ok();

        Ok(Self { common, rx_queue, tx_queue, mac_address: mac })
    }

    /// Receive a packet (DMA zero-copy into queue buffer)
    pub fn receive(&mut self, buffer: &mut [u8]) -> Result<usize, NetError> {
        let desc_id = self.rx_queue.pop_used()?;
        let len = self.rx_queue.get_used_len(desc_id);

        // Copy from virtqueue buffer to caller's buffer
        self.rx_queue.read_buffer(desc_id, buffer)?;

        // Recycle the descriptor
        self.rx_queue.add_available(desc_id);

        Ok(len)
    }

    /// Transmit a packet
    pub fn transmit(&mut self, packet: &[u8]) -> Result<(), NetError> {
        let desc_id = self.tx_queue.alloc_descriptor()?;

        self.tx_queue.write_buffer(desc_id, packet)?;
        self.tx_queue.add_available(desc_id);
        self.tx_queue.notify();

        Ok(())
    }
}
```

#### ARP Cache

```rust
// ruvix-net/src/arp.rs

pub struct ArpCache {
    entries: [Option<ArpEntry>; 64],
    pending: [Option<ArpPending>; 16],
}

pub struct ArpEntry {
    ip: Ipv4Addr,
    mac: [u8; 6],
    expires: Instant,
}

impl ArpCache {
    pub fn lookup(&self, ip: Ipv4Addr) -> Option<[u8; 6]> {
        self.entries.iter()
            .flatten()
            .find(|e| e.ip == ip && e.expires > Instant::now())
            .map(|e| e.mac)
    }

    pub fn insert(&mut self, ip: Ipv4Addr, mac: [u8; 6]) {
        let entry = ArpEntry {
            ip,
            mac,
            expires: Instant::now() + Duration::from_secs(300),
        };

        // Find empty slot or oldest entry
        let slot = self.entries.iter_mut()
            .enumerate()
            .min_by_key(|(_, e)| e.as_ref().map(|e| e.expires))
            .map(|(i, _)| i)
            .unwrap();

        self.entries[slot] = Some(entry);
    }
}
```

### E.2 Filesystem Architecture

RuVix implements a minimal VFS layer optimized for RVF package storage and retrieval, not general-purpose file operations.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FILESYSTEM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                           VFS LAYER                                │   │
│  │  ════════════════════════════════════════════════════════════════ │   │
│  │                                                                    │   │
│  │  open(path) → FileHandle       read(fh, buf) → usize              │   │
│  │  close(fh)                     write(fh, buf) → usize (limited)   │   │
│  │  stat(path) → Metadata         readdir(path) → [DirEntry]         │   │
│  │                                                                    │   │
│  └────────────────────────────┬──────────────────────────────────────┘   │
│                               │                                          │
│         ┌─────────────────────┼─────────────────────┐                    │
│         │                     │                     │                    │
│         ▼                     ▼                     ▼                    │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│  │   FAT32     │      │   RamFS     │      │  (Future)   │               │
│  │  (read-only)│      │    /tmp     │      │   ext4      │               │
│  ├─────────────┤      ├─────────────┤      ├─────────────┤               │
│  │ SD card     │      │ Kernel heap │      │ NVMe/USB    │               │
│  │ boot part.  │      │ regions     │      │             │               │
│  └─────────────┘      └─────────────┘      └─────────────┘               │
│                                                                          │
│  Mount points:                                                           │
│    /boot  → FAT32 (SD card boot partition, RVF packages)                │
│    /tmp   → RamFS (temporary files, checkpoints)                        │
│    /rvf   → Virtual (mounted RVF component namespaces)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### VFS Layer

```rust
// ruvix-fs/src/vfs.rs

pub trait Filesystem: Send + Sync {
    fn mount(&mut self, device: &dyn BlockDevice) -> Result<(), FsError>;
    fn open(&self, path: &str, flags: OpenFlags) -> Result<FileHandle, FsError>;
    fn read(&self, handle: FileHandle, buf: &mut [u8]) -> Result<usize, FsError>;
    fn write(&self, handle: FileHandle, buf: &[u8]) -> Result<usize, FsError>;
    fn close(&self, handle: FileHandle);
    fn stat(&self, path: &str) -> Result<Metadata, FsError>;
    fn readdir(&self, path: &str) -> Result<Vec<DirEntry>, FsError>;
}

pub struct Vfs {
    mounts: BTreeMap<PathBuf, Box<dyn Filesystem>>,
}

impl Vfs {
    pub fn mount(&mut self, path: &str, fs: Box<dyn Filesystem>) -> Result<(), FsError> {
        self.mounts.insert(PathBuf::from(path), fs);
        Ok(())
    }

    pub fn open(&self, path: &str, flags: OpenFlags) -> Result<FileHandle, FsError> {
        let (mount_path, relative_path) = self.resolve_mount(path)?;
        let fs = self.mounts.get(mount_path).ok_or(FsError::NotFound)?;
        fs.open(relative_path, flags)
    }
}
```

#### FAT32 (Read-Only)

```rust
// ruvix-fs/src/fat32.rs

pub struct Fat32 {
    device: Box<dyn BlockDevice>,
    bpb: BiosParameterBlock,
    fat_start: u64,
    data_start: u64,
    root_cluster: u32,
}

impl Fat32 {
    pub fn mount(device: Box<dyn BlockDevice>) -> Result<Self, FsError> {
        let mut sector = [0u8; 512];
        device.read_block(0, &mut sector)?;

        let bpb = BiosParameterBlock::parse(&sector)?;

        // Validate FAT32 signature
        if bpb.sectors_per_fat_16 != 0 {
            return Err(FsError::InvalidFilesystem);
        }

        let fat_start = bpb.reserved_sectors as u64;
        let data_start = fat_start + (bpb.num_fats as u64 * bpb.sectors_per_fat_32 as u64);

        Ok(Self {
            device,
            bpb,
            fat_start,
            data_start,
            root_cluster: bpb.root_cluster,
        })
    }

    fn read_cluster(&self, cluster: u32, buf: &mut [u8]) -> Result<(), FsError> {
        let sector = self.cluster_to_sector(cluster);
        for i in 0..self.bpb.sectors_per_cluster {
            let offset = i as usize * 512;
            self.device.read_block(sector + i as u64, &mut buf[offset..offset + 512])?;
        }
        Ok(())
    }

    fn cluster_to_sector(&self, cluster: u32) -> u64 {
        self.data_start + ((cluster - 2) as u64 * self.bpb.sectors_per_cluster as u64)
    }
}
```

#### RamFS

```rust
// ruvix-fs/src/ramfs.rs

pub struct RamFs {
    root: RamFsNode,
    allocator: RegionAllocator,
}

enum RamFsNode {
    Directory {
        name: String,
        children: Vec<RamFsNode>,
    },
    File {
        name: String,
        data: RegionHandle,
        size: usize,
    },
}

impl Filesystem for RamFs {
    fn open(&self, path: &str, flags: OpenFlags) -> Result<FileHandle, FsError> {
        let node = self.resolve_path(path)?;
        match node {
            RamFsNode::File { data, size, .. } => {
                Ok(FileHandle::new(data.clone(), *size, flags))
            }
            RamFsNode::Directory { .. } => Err(FsError::IsDirectory),
        }
    }

    fn write(&self, handle: FileHandle, buf: &[u8]) -> Result<usize, FsError> {
        // RamFS supports writes for checkpoints
        let region = handle.region();
        region.write(handle.position(), buf)?;
        Ok(buf.len())
    }
}
```

### E.3 New Crates

| Crate | Purpose | Dependencies | Lines of Code (Est.) |
|-------|---------|--------------|---------------------|
| **`ruvix-net`** | Ethernet/IP/UDP/ICMP network stack | `ruvix-types`, `ruvix-queue` | ~1,500 |
| **`ruvix-fs`** | VFS layer, FAT32, RamFS | `ruvix-types`, `ruvix-region` | ~1,800 |

### E.4 Build Path (Weeks 63-74)

| Week | Milestone | Deliverables | Tests |
|------|-----------|--------------|-------|
| **63-64** | Block device layer | SD card driver, virtio-blk | Read/write sectors |
| **65-66** | FAT32 read-only | BPB parsing, cluster chain, directory traversal | Read files from SD |
| **67-68** | VFS layer | Mount points, path resolution, file handles | Mount FAT32 at /boot |
| **69-70** | RamFS | In-memory filesystem for /tmp | Checkpoint read/write |
| **71-72** | Ethernet driver | virtio-net, bcmgenet (Pi4) | Packet TX/RX |
| **73-74** | IP/UDP/ICMP | ARP, IP routing, UDP sockets, ping | Ping reply, UDP echo |

---

## QEMU Swarm Simulation

### Swarm.1 Architecture

The QEMU Swarm Simulation system enables testing of distributed RuVix deployments without physical hardware. Multiple QEMU instances run in parallel, connected via virtual networking.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QEMU SWARM SIMULATION ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                       SWARM ORCHESTRATOR                           │   │
│  │  ═══════════════════════════════════════════════════════════════  │   │
│  │                                                                    │   │
│  │   swarm-ctl                                                        │   │
│  │   ├── launch     : Start N QEMU nodes                              │   │
│  │   ├── connect    : Establish virtual network                       │   │
│  │   ├── deploy     : Push RVF packages to nodes                      │   │
│  │   ├── monitor    : Aggregate console output                        │   │
│  │   ├── inject     : Inject faults (network, disk, CPU)             │   │
│  │   └── teardown   : Graceful shutdown all nodes                     │   │
│  │                                                                    │   │
│  └──────────────────────────────┬────────────────────────────────────┘   │
│                                 │                                        │
│                     ┌───────────┼───────────┐                            │
│                     │           │           │                            │
│              ┌──────▼──────┐ ┌──▼───┐ ┌─────▼─────┐                      │
│              │   Node 0    │ │Node 1│ │  Node N   │                      │
│              │  (QEMU)     │ │(QEMU)│ │  (QEMU)   │                      │
│              │             │ │      │ │           │                      │
│              │ ┌─────────┐ │ │ ...  │ │ ┌───────┐ │                      │
│              │ │ RuVix   │ │ │      │ │ │RuVix  │ │                      │
│              │ │ Kernel  │ │ │      │ │ │Kernel │ │                      │
│              │ └────┬────┘ │ │      │ │ └───┬───┘ │                      │
│              │      │      │ │      │ │     │     │                      │
│              │ ┌────▼────┐ │ │      │ │ ┌───▼───┐ │                      │
│              │ │ virtio  │ │ │      │ │ │virtio │ │                      │
│              │ │   net   │ │ │      │ │ │ net   │ │                      │
│              │ └────┬────┘ │ │      │ │ └───┬───┘ │                      │
│              └──────┼──────┘ └──────┘ └─────┼─────┘                      │
│                     │                       │                            │
│              ┌──────▼───────────────────────▼──────┐                     │
│              │         VIRTUAL NETWORK              │                     │
│              │  ════════════════════════════════   │                     │
│              │                                      │                     │
│              │   TAP interfaces + bridge            │                     │
│              │   OR                                 │                     │
│              │   QEMU user-mode networking          │                     │
│              │   OR                                 │                     │
│              │   vde_switch for complex topologies  │                     │
│              │                                      │                     │
│              └──────────────────────────────────────┘                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Orchestrator Implementation

```rust
// tools/swarm-ctl/src/main.rs

use std::process::{Command, Stdio};
use std::collections::HashMap;

pub struct SwarmOrchestrator {
    nodes: Vec<QemuNode>,
    network: VirtualNetwork,
    console_mux: ConsoleMux,
}

pub struct QemuNode {
    id: usize,
    process: std::process::Child,
    console: UnixStream,
    monitor: UnixStream,
    mac_address: [u8; 6],
    ip_address: Ipv4Addr,
}

impl SwarmOrchestrator {
    pub fn launch(&mut self, config: &SwarmConfig) -> Result<(), SwarmError> {
        // Create virtual network
        self.network = VirtualNetwork::create(&config.topology)?;

        for i in 0..config.node_count {
            let node = self.spawn_node(i, &config)?;
            self.nodes.push(node);
        }

        // Wait for all nodes to boot
        self.wait_for_boot_complete()?;

        Ok(())
    }

    fn spawn_node(&self, id: usize, config: &SwarmConfig) -> Result<QemuNode, SwarmError> {
        let mac = generate_mac(id);
        let console_sock = format!("/tmp/ruvix-swarm-{}-console.sock", id);
        let monitor_sock = format!("/tmp/ruvix-swarm-{}-monitor.sock", id);

        let mut cmd = Command::new("qemu-system-aarch64");
        cmd.args([
            "-machine", "virt,gic-version=3",
            "-cpu", "cortex-a72",
            "-m", &format!("{}M", config.memory_mb),
            "-smp", &format!("{}", config.cpus),
            "-kernel", &config.kernel_path,
            "-drive", &format!("file={},format=raw,if=pflash", config.rvf_path),
            // Networking
            "-netdev", &format!("tap,id=net0,ifname=tap{},script=no,downscript=no", id),
            "-device", &format!("virtio-net-pci,netdev=net0,mac={}", mac_to_string(&mac)),
            // Console
            "-chardev", &format!("socket,id=console,path={},server=on,wait=off", console_sock),
            "-serial", "chardev:console",
            // Monitor
            "-chardev", &format!("socket,id=monitor,path={},server=on,wait=off", monitor_sock),
            "-monitor", "chardev:monitor",
            // No graphics
            "-nographic",
        ]);

        let process = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;

        // Connect to console and monitor sockets
        std::thread::sleep(Duration::from_millis(500));
        let console = UnixStream::connect(&console_sock)?;
        let monitor = UnixStream::connect(&monitor_sock)?;

        Ok(QemuNode {
            id,
            process,
            console,
            monitor,
            mac_address: mac,
            ip_address: Ipv4Addr::new(10, 0, 0, (id + 1) as u8),
        })
    }
}
```

### Swarm.2 Configuration

#### Node Resource Allocation

```yaml
# swarm-config.yaml

cluster:
  name: "ruvix-test-cluster"
  node_count: 5

nodes:
  default:
    cpus: 2
    memory_mb: 512
    kernel: "target/aarch64-unknown-none/release/ruvix-kernel"
    rvf: "boot.rvf"

  # Override for specific nodes
  coordinator:
    index: 0
    cpus: 4
    memory_mb: 1024
    rvf: "coordinator.rvf"

  workers:
    indices: [1, 2, 3, 4]
    cpus: 2
    memory_mb: 512
    rvf: "worker.rvf"
```

#### Network Topologies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SUPPORTED NETWORK TOPOLOGIES                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  RING TOPOLOGY                    MESH TOPOLOGY                          │
│  ══════════════                   ══════════════                         │
│                                                                          │
│      ┌───┐                            ┌───┐                              │
│      │ 0 │◄──────┐                    │ 0 │                              │
│      └─┬─┘      │                    └─┬─┘                              │
│        │        │                  ╱   │   ╲                            │
│        ▼        │                ╱     │     ╲                          │
│      ┌───┐      │           ┌───┐    ┌─┴─┐    ┌───┐                      │
│      │ 1 │      │           │ 1 │◄──▶│   │◄──▶│ 2 │                      │
│      └─┬─┘      │           └─┬─┘    └───┘    └─┬─┘                      │
│        │        │             │  ╲           ╱  │                        │
│        ▼        │             │    ╲       ╱    │                        │
│      ┌───┐      │             │      ╲   ╱      │                        │
│      │ 2 │      │           ┌─┴─┐    ┌─┴─┐    ┌─┴─┐                      │
│      └─┬─┘      │           │ 3 │◄──▶│ 4 │◄──▶│   │                      │
│        │        │           └───┘    └───┘    └───┘                      │
│        ▼        │                                                        │
│      ┌───┐      │                                                        │
│      │ 3 │──────┘                                                        │
│      └───┘                                                               │
│                                                                          │
│  STAR TOPOLOGY                    HIERARCHICAL TOPOLOGY                  │
│  ══════════════                   ═════════════════════                  │
│                                                                          │
│      ┌───┐  ┌───┐                        ┌───┐                           │
│      │ 1 │  │ 2 │                        │ 0 │  (Coordinator)            │
│      └─┬─┘  └─┬─┘                        └─┬─┘                           │
│        │      │                        ╱   │   ╲                         │
│        └──┬───┘                      ╱     │     ╲                       │
│           ▼                        ╱       │       ╲                     │
│         ┌───┐                  ┌───┐     ┌───┐     ┌───┐                 │
│         │ 0 │  (Hub)           │ 1 │     │ 2 │     │ 3 │  (Leaders)      │
│         └─┬─┘                  └─┬─┘     └─┬─┘     └─┬─┘                 │
│        ╱  │  ╲                   │         │         │                   │
│      ╱    │    ╲              ┌──┴──┐   ┌──┴──┐   ┌──┴──┐                │
│  ┌───┐  ┌───┐  ┌───┐         │4│5│  │6│7│  │8│9│  (Workers)              │
│  │ 3 │  │ 4 │  │ 5 │         └─┴─┘  └─┴─┘  └─┴─┘                         │
│  └───┘  └───┘  └───┘                                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

```yaml
# Network topology configuration

network:
  topology: "mesh"  # ring | mesh | star | hierarchical

  # Mesh-specific: which nodes connect to which
  mesh:
    connectivity: 0.7  # 70% of possible edges exist

  # Star-specific: hub node index
  star:
    hub: 0

  # Hierarchical-specific
  hierarchical:
    coordinator: 0
    leaders: [1, 2, 3]
    workers_per_leader: 3

  # Network parameters
  latency_ms: 1
  bandwidth_mbps: 1000
  packet_loss_percent: 0.0  # For fault injection
```

#### Fault Injection Scenarios

```yaml
# fault-injection.yaml

scenarios:
  network_partition:
    description: "Partition cluster into two halves"
    trigger: "manual"
    action:
      type: "network_partition"
      groups:
        - [0, 1, 2]
        - [3, 4]
    duration_s: 30

  node_crash:
    description: "Kill a random worker node"
    trigger: "random"
    probability: 0.01  # per second
    action:
      type: "kill_node"
      target: "random_worker"
    recovery_s: 10  # restart after

  network_delay:
    description: "Add latency to specific link"
    trigger: "manual"
    action:
      type: "add_latency"
      from: 0
      to: 1
      latency_ms: 100
      jitter_ms: 20
    duration_s: 60

  disk_failure:
    description: "Simulate disk I/O errors"
    trigger: "manual"
    action:
      type: "disk_error"
      target: 2
      error_rate: 0.1  # 10% of reads/writes fail
    duration_s: 30
```

### Swarm.3 Use Cases

#### Distributed Consensus Testing

```bash
# Test Raft consensus under network partition
./swarm-ctl launch --config swarm-5-node.yaml
./swarm-ctl deploy --rvf consensus-test.rvf

# Wait for cluster formation
./swarm-ctl wait --condition "consensus_formed"

# Inject network partition
./swarm-ctl inject --scenario network_partition

# Verify:
# - Majority partition elects new leader
# - Minority partition becomes read-only
# - After healing, cluster converges

./swarm-ctl verify --invariant "no_split_brain"
./swarm-ctl inject --scenario heal_network

./swarm-ctl verify --invariant "single_leader"
./swarm-ctl verify --invariant "log_consistency"
```

#### RVF Package Deployment Across Cluster

```bash
# Deploy an updated RVF package to all nodes
./swarm-ctl launch --config swarm-10-node.yaml

# Deploy initial package
./swarm-ctl deploy --rvf agent-v1.rvf --all

# Rolling update to v2
./swarm-ctl deploy --rvf agent-v2.rvf --strategy rolling --batch-size 2

# Verify no service disruption
./swarm-ctl verify --invariant "service_available" --throughout-deploy

# Rollback if needed
./swarm-ctl rollback --to agent-v1.rvf --all
```

#### Performance Benchmarking Under Load

```bash
# Launch benchmark cluster
./swarm-ctl launch --config swarm-benchmark.yaml

# Deploy benchmark workload
./swarm-ctl deploy --rvf vector-benchmark.rvf

# Generate load
./swarm-ctl benchmark \
  --workload "vector_insert" \
  --rate 10000  \  # ops/sec
  --duration 60 \
  --collect-metrics

# Results:
# - Throughput: ops/sec achieved
# - Latency: p50, p95, p99 in microseconds
# - CPU utilization per node
# - Memory usage per node
# - Network bandwidth utilization
```

### Swarm.4 Console Multiplexing

```rust
// tools/swarm-ctl/src/console.rs

pub struct ConsoleMux {
    nodes: Vec<NodeConsole>,
    output_mode: OutputMode,
}

pub enum OutputMode {
    /// All output interleaved with node prefixes
    Interleaved,
    /// Each node to separate file
    SeparateFiles,
    /// TUI with panes per node
    Tui,
}

impl ConsoleMux {
    pub fn run(&mut self) {
        match self.output_mode {
            OutputMode::Interleaved => {
                // Poll all consoles, prefix output with node ID
                loop {
                    for node in &mut self.nodes {
                        if let Some(line) = node.read_line() {
                            println!("[node-{}] {}", node.id, line);
                        }
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            OutputMode::Tui => {
                // Use crossterm/tui-rs for multi-pane display
                self.run_tui();
            }
            OutputMode::SeparateFiles => {
                // Write each node's output to node-N.log
                self.run_file_mode();
            }
        }
    }
}
```

### Swarm.5 Build Path (Weeks 75-82)

| Week | Milestone | Deliverables | Tests |
|------|-----------|--------------|-------|
| **75-76** | QEMU launcher | Node spawning, console sockets, monitor control | Launch/teardown 5 nodes |
| **77-78** | Virtual networking | TAP interfaces, bridge setup, vde_switch | Ping between nodes |
| **79-80** | Orchestrator CLI | swarm-ctl commands, config parsing, status | Full CLI functional |
| **81-82** | Fault injection | Network partition, node kill, latency injection | Consensus survives faults |

---

## Phase Timeline Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RUVIX DEVELOPMENT TIMELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE         │ WEEKS   │ DURATION │ KEY DELIVERABLES                  │
│  ══════════════│═════════│══════════│═══════════════════════════════    │
│                │         │          │                                    │
│  Phase A       │  1-18   │ 18 weeks │ Linux-hosted nucleus, 12 syscalls │
│  (Complete)    │         │          │ 760 tests passing                  │
│                │         │          │                                    │
│  Phase B       │ 19-42   │ 24 weeks │ Bare metal AArch64, QEMU virt      │
│                │         │          │ MMU, GIC, timer, WASM runtime      │
│                │         │          │                                    │
│  Phase C       │ 43-54   │ 12 weeks │ SMP (256 cores), DMA, DTB parser   │
│                │         │          │ Spinlocks, IPIs, per-CPU data      │
│                │         │          │                                    │
│  Phase D       │ 55-62   │  8 weeks │ Raspberry Pi 4/5 support           │
│                │         │          │ BCM2711/2712 drivers               │
│                │         │          │                                    │
│  Phase E       │ 63-74   │ 12 weeks │ Networking (UDP/ICMP), Filesystem  │
│                │         │          │ virtio-net, FAT32, RamFS           │
│                │         │          │                                    │
│  QEMU Swarm    │ 75-82   │  8 weeks │ Multi-QEMU orchestration           │
│                │         │          │ Fault injection, benchmarking      │
│                │         │          │                                    │
│  ══════════════│═════════│══════════│═══════════════════════════════    │
│  TOTAL         │  1-82   │ 82 weeks │ ~19 months                         │
│                │         │          │                                    │
└─────────────────────────────────────────────────────────────────────────┘
```
