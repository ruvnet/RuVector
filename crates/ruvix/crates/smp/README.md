# ruvix-smp

Symmetric Multi-Processing support for the RuVix Cognition Kernel (ADR-087 Phase C).

## Overview

This crate provides the fundamental primitives for multi-core operation in the RuVix kernel:

| Component | Purpose |
|-----------|---------|
| `CpuId` | Strongly-typed CPU identifier (0-255) |
| `CpuState` | CPU lifecycle state machine |
| `PerCpu<T>` | Per-CPU data storage |
| `CpuTopology` | System-wide CPU state tracking |
| `SpinLock<T>` | Fair ticket-based spinlock |
| `IpiMessage` | Inter-processor interrupt messages |

## Features

- **no_std** compatible with zero external dependencies
- Up to 256 CPUs supported
- Fair FIFO spinlock ordering via ticket algorithm
- Power-efficient WFE/SEV synchronization
- Cache-line aligned per-CPU data to prevent false sharing

## Multi-Core Boot Sequence

### Overview

```
                    +------------------+
                    |   Power On       |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+        +----------v----------+
    |  Primary CPU (0)  |        | Secondary CPUs (1-N)|
    |   - EL3 firmware  |        |   - Spin in WFE     |
    |   - EL2 hypervisor|        |   - Wait for release|
    +--------+----------+        +----------+----------+
             |                              |
    +--------v----------+                   |
    |  kernel_main()    |                   |
    |  - Init topology  |                   |
    |  - Boot self      |                   |
    +--------+----------+                   |
             |                              |
    +--------v----------+                   |
    |  Release via PSCI |-------------------+
    |  or spin-table    |
    +--------+----------+
             |                   +----------v----------+
             |                   | secondary_entry()   |
             |                   |  - Read MPIDR_EL1   |
             |                   |  - Init per-CPU     |
             |                   |  - Call boot_cpu()  |
             |                   +----------+----------+
             |                              |
    +--------v----------+        +----------v----------+
    | wait_for_all_cpus |<-------|  SEV to primary     |
    +--------+----------+        +----------+----------+
             |                              |
    +--------v----------+        +----------v----------+
    |  All CPUs Online  |        |  Enter scheduler    |
    +-------------------+        +---------------------+
```

### Detailed Steps

#### 1. Primary CPU Initialization

The primary CPU (CPU 0) starts first and initializes the system:

```rust
use ruvix_smp::{CpuTopology, CpuId};

static TOPOLOGY: CpuTopology = CpuTopology::new();

fn kernel_main() {
    // 1. Detect number of CPUs (from DTB or ACPI)
    let num_cpus = detect_cpu_count();

    // 2. Initialize topology
    TOPOLOGY.init(num_cpus);

    // 3. Boot primary CPU
    TOPOLOGY.boot_cpu(CpuId::BOOT_CPU);

    // 4. Initialize shared resources
    init_memory_allocator();
    init_interrupt_controller();

    // 5. Start secondary CPUs
    for i in 1..num_cpus {
        let cpu = CpuId::new(i as u8).unwrap();
        start_secondary_cpu(cpu);
    }

    // 6. Wait for all CPUs
    TOPOLOGY.wait_for_all_cpus();

    // 7. All CPUs online - start scheduler
    scheduler_start();
}
```

#### 2. Secondary CPU Entry

Each secondary CPU runs this code after being released:

```rust
use ruvix_smp::{current_cpu, CpuTopology, PerCpu};

// Per-CPU stack (must be initialized before boot)
static STACKS: PerCpu<[u8; 16384]> = PerCpu::new([[0; 16384]; MAX_CPUS]);

extern "C" fn secondary_entry() {
    // 1. Get CPU ID from hardware
    let cpu = current_cpu();

    // 2. Set up per-CPU stack
    let stack = STACKS.get(cpu);
    set_stack_pointer(stack.as_ptr() as usize + stack.len());

    // 3. Initialize per-CPU state
    init_cpu_local_state(cpu);

    // 4. Enable interrupts for this CPU
    enable_local_interrupts(cpu);

    // 5. Mark as online (sends SEV to primary)
    TOPOLOGY.boot_cpu(cpu);

    // 6. Enter scheduler
    scheduler_enter();
}
```

#### 3. Starting Secondary CPUs

On ARM64, secondaries are started via PSCI or spin-table:

```rust
// PSCI method (recommended)
fn start_secondary_cpu_psci(cpu: CpuId) {
    let entry_point = secondary_entry as usize;
    let context_id = cpu.as_usize();

    psci_cpu_on(cpu.as_u8() as u64, entry_point, context_id);
}

// Spin-table method (legacy)
fn start_secondary_cpu_spintable(cpu: CpuId, spin_table: &mut [u64]) {
    let entry_point = secondary_entry as usize;

    // Write entry point to spin table
    spin_table[cpu.as_usize()] = entry_point as u64;

    // Memory barrier to ensure write is visible
    unsafe { dsb(); }

    // Wake the CPU
    unsafe { sev(); }
}
```

### Memory Barriers in Boot

Correct ordering is critical during boot:

```rust
// Primary CPU: Release secondary
store_entry_point(secondary_entry);
unsafe { dsb(); }  // Ensure store completes
unsafe { sev(); }  // Wake secondaries

// Secondary CPU: Wait for release
loop {
    let entry = load_entry_point();
    if entry != 0 {
        unsafe { isb(); }  // Flush pipeline
        jump_to(entry);
    }
    unsafe { wfe(); }  // Power-efficient wait
}
```

## Usage Examples

### SpinLock

```rust
use ruvix_smp::SpinLock;

// Static initialization
static COUNTER: SpinLock<u64> = SpinLock::new(0);

fn increment() {
    let mut guard = COUNTER.lock();
    *guard += 1;
} // Lock released here

fn try_increment() -> bool {
    if let Some(mut guard) = COUNTER.try_lock() {
        *guard += 1;
        true
    } else {
        false
    }
}
```

### Per-CPU Data

```rust
use ruvix_smp::{PerCpu, CpuId, current_cpu, MAX_CPUS};

// Per-CPU statistics
static STATS: PerCpu<CpuStats> = PerCpu::new([CpuStats::new(); MAX_CPUS]);

fn record_event() {
    let cpu = current_cpu();
    STATS.get_mut(cpu).events += 1;
}

fn total_events() -> u64 {
    STATS.iter().map(|s| s.events).sum()
}
```

### CPU Topology

```rust
use ruvix_smp::{CpuTopology, CpuId, CpuState};

static TOPOLOGY: CpuTopology = CpuTopology::new();

fn show_cpu_status() {
    for cpu in TOPOLOGY.online_cpus() {
        println!("{}: {}", cpu, TOPOLOGY.state(cpu));
    }
    println!("Total online: {}", TOPOLOGY.online_count());
}

fn halt_cpu(cpu: CpuId) {
    if TOPOLOGY.halt_cpu(cpu) {
        println!("{} halted", cpu);
    }
}
```

### Inter-Processor Interrupts

```rust
use ruvix_smp::{send_ipi, IpiMessage, IpiTarget, CpuId};

// Request reschedule on another CPU
fn wake_cpu(cpu: CpuId) {
    send_ipi(IpiTarget::Cpu(cpu), IpiMessage::Reschedule);
}

// TLB shootdown to all other CPUs
fn tlb_flush_all(asid: u16) {
    send_ipi(IpiTarget::AllOther, IpiMessage::TlbFlush { asid: Some(asid) });
}

// Halt all CPUs for shutdown
fn system_halt() {
    send_ipi(IpiTarget::AllOther, IpiMessage::Halt);
}
```

## Memory Barriers

### When to Use Each Barrier

| Barrier | Use Case |
|---------|----------|
| `dmb()` | Order memory accesses (most common) |
| `dmb_ish()` | Order accesses between CPUs only |
| `dsb()` | Wait for memory operations to complete |
| `isb()` | Flush instruction pipeline |
| `sev()` | Wake CPUs waiting in WFE |
| `wfe()` | Sleep until event (power-efficient spin) |

### Common Patterns

```rust
use ruvix_smp::barriers::*;

// Pattern 1: Lock release
fn release_lock() {
    store_unlock_value();
    unsafe {
        dsb();   // Ensure store completes
        sev();   // Wake waiters
    }
}

// Pattern 2: Page table update
fn update_page_table() {
    write_pte(new_entry);
    unsafe {
        dsb();   // Ensure PTE write completes
        isb();   // Flush pipeline
    }
    tlbi();      // Invalidate TLB
    unsafe {
        dsb();   // Ensure TLBI completes
        isb();   // Flush pipeline
    }
}

// Pattern 3: Power-efficient spin
fn spin_wait(condition: &AtomicBool) {
    unsafe { sevl(); }  // Set local event flag
    while !condition.load(Ordering::Acquire) {
        unsafe { wfe(); }
    }
}
```

## Safety

Most unsafe code is isolated to specific areas:

1. **Memory barriers** (`barriers` module) - Use inline assembly
2. **SpinLock internals** - Atomic operations with custom memory ordering
3. **MPIDR_EL1 access** - Reading CPU ID register
4. **PerCpu mutable access** - Interior mutability pattern

All unsafe operations are well-documented with safety comments.

## Testing

Run tests with:

```bash
# Unit tests (uses test-mode feature)
cargo test --features test-mode

# With proptest for property-based testing
cargo test --features test-mode,proptest
```

Note: ARM64-specific code cannot be tested on non-ARM platforms. The `test-mode` feature replaces inline assembly with atomics and `spin_loop()`.

## License

MIT OR Apache-2.0
