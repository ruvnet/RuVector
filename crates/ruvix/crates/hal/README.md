# ruvix-hal

Hardware Abstraction Layer (HAL) for the RuVix Cognition Kernel (ADR-087).

## Overview

`ruvix-hal` provides platform-independent traits that hardware-specific implementations must satisfy. This crate is the foundation for building portable kernel code that works across ARM64, RISC-V, and x86_64 platforms.

## Design Principles

- **`#![no_std]`** - No standard library dependency
- **`#![forbid(unsafe_code)]`** - Trait definitions are completely safe
- **Zero dependencies** - Only depends on `ruvix-types`
- **Platform-agnostic** - Traits work across all supported architectures
- **Result-based errors** - All fallible operations return `Result`

## HAL Subsystems

### 1. Console (`console.rs`)

Serial I/O for kernel debugging and early boot logging.

```rust
use ruvix_hal::Console;

fn kernel_log<C: Console>(console: &mut C, msg: &str) -> Result<(), ConsoleError> {
    console.write_str("[ ")?;
    console.write_str(msg)?;
    console.write_str(" ]\n")?;
    console.flush()?;
    Ok(())
}
```

**Key Features:**
- Blocking and buffered writes
- IRQ-safe operations
- ASCII and UTF-8 support

### 2. Interrupt Controller (`interrupt.rs`)

IRQ/FIQ management and routing for ARM64 GICv3 and similar controllers.

```rust
use ruvix_hal::{InterruptController, interrupt::TriggerMode};

fn setup_uart_irq<I: InterruptController>(irq: &mut I) -> Result<(), InterruptError> {
    const UART_IRQ: u32 = 33;

    irq.set_priority(UART_IRQ, 1)?;
    irq.set_trigger_mode(UART_IRQ, TriggerMode::Level)?;
    irq.enable(UART_IRQ)?;

    Ok(())
}
```

**Key Features:**
- Priority-based routing
- Edge and level triggering
- Multi-core affinity support
- RAII interrupt masking

### 3. Timer (`timer.rs`)

Monotonic time and deadline scheduling for ARM Generic Timer and similar.

```rust
use ruvix_hal::Timer;

fn schedule_wakeup<T: Timer>(timer: &mut T, delay_ms: u64) -> Result<(), TimerError> {
    let deadline = timer.now_ns() + (delay_ms * 1_000_000);
    timer.set_deadline(deadline)?;
    Ok(())
}
```

**Key Features:**
- Nanosecond precision (u64, 584 years before wrap)
- Deadline interrupts
- Stopwatch helpers
- Busy-wait utilities

### 4. MMU (`mmu.rs`)

Virtual memory and page table management with capability-aware extensions.

```rust
use ruvix_hal::{Mmu, mmu::PagePermissions};

fn map_kernel_code<M: Mmu>(mmu: &mut M, code_start: u64, code_size: usize) -> Result<(), MmuError> {
    let pages = (code_size + 4095) / 4096;

    for i in 0..pages {
        let virt = code_start + (i * 4096) as u64;
        let phys = virt; // Identity mapping for kernel

        mmu.map_page(virt, phys, PagePermissions::KERNEL_RX)?;
    }

    mmu.flush_tlb();
    Ok(())
}
```

**Key Features:**
- 4 KiB pages (ARM64 standard)
- Capability-tagged page table entries
- TLB management
- Permission bitflags (READ, WRITE, EXECUTE, USER)

### 5. Power Management (`power.rs`)

CPU power states, reset control, and multi-core coordination via ARM PSCI.

```rust
use ruvix_hal::{PowerManagement, power::ResetType};

fn panic_handler<P: PowerManagement>(pm: &P, msg: &str) -> ! {
    // Log panic message...

    match pm.reset(ResetType::Warm) {
        Ok(_) => unreachable!(),
        Err(_) => loop { pm.wait_for_interrupt() },
    }
}
```

**Key Features:**
- WFI/WFE low-power modes
- CPU hotplug (cpu_on/cpu_off)
- System reset and shutdown
- Multi-core coordination

## Usage

Add to your kernel's `Cargo.toml`:

```toml
[dependencies]
ruvix-hal = { version = "0.1.0", path = "../hal" }
```

Implement the HAL traits for your target platform:

```rust
use ruvix_hal::*;

struct MyConsole { /* platform-specific fields */ }
impl Console for MyConsole { /* implementation */ }

struct MyMmu { /* platform-specific fields */ }
impl Mmu for MyMmu { /* implementation */ }

// Generic kernel code works across platforms
fn kernel_main<C: Console, M: Mmu>(console: &mut C, mmu: &mut M) {
    console.write_str("RuVix booting...\n").unwrap();
    // ... rest of kernel initialization
}
```

## Architecture

```
ruvix-hal (traits only, #![forbid(unsafe_code)])
    ↓
ruvix-hal-aarch64 (ARM64 implementation, uses unsafe)
    ↓
ruvix-kernel (generic kernel code)
```

Platform-specific implementations (like `ruvix-hal-aarch64`) will implement these traits using `unsafe` code to access hardware registers.

## Error Handling

All HAL operations use `Result<T, E>` for error handling:

- `ConsoleError` - Serial I/O errors
- `InterruptError` - IRQ controller errors
- `TimerError` - Timer operation errors
- `MmuError` - Virtual memory errors
- `PowerError` - Power management errors

## Thread Safety

All HAL traits are designed for single-threaded kernel use. Multi-core kernels must:

1. Use spinlocks or disable interrupts when modifying shared state
2. Use per-CPU data structures where possible
3. Follow platform-specific synchronization requirements

## Testing

Mock implementations are provided in `#[cfg(test)]` modules for unit testing kernel code without hardware.

```rust
#[cfg(test)]
mod tests {
    use ruvix_hal::Console;

    struct MockConsole { buffer: Vec<u8> }
    impl Console for MockConsole { /* ... */ }

    #[test]
    fn test_kernel_log() {
        let mut console = MockConsole::new();
        kernel_log(&mut console, "test message").unwrap();
        assert_eq!(console.buffer, b"[ test message ]\n");
    }
}
```

## Examples

See `crates/ruvix/crates/hal-aarch64/` for a complete ARM64 implementation.

## License

MIT OR Apache-2.0

## Contributing

Contributions are welcome! Please ensure:

- Trait definitions remain `#![forbid(unsafe_code)]`
- All public APIs are documented with examples
- Error types implement `Display`
- Platform-specific details go in implementation crates, not here

## References

- [ARM Generic Interrupt Controller v3 (GICv3)](https://developer.arm.com/documentation/198123/0302/What-is-the-Arm-Generic-Interrupt-Controller-)
- [ARM Generic Timer](https://developer.arm.com/documentation/ddi0595/2021-12/AArch64-Registers/CNTPCT-EL0--Counter-timer-Physical-Count-register)
- [ARM Power State Coordination Interface (PSCI)](https://developer.arm.com/documentation/den0022/latest/)
- [ARM Architecture Reference Manual (ARMv8-A)](https://developer.arm.com/documentation/ddi0487/latest/)
