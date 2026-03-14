# ruvix-drivers

Device drivers for the RuVix Cognition Kernel (ADR-087).

## Overview

This crate provides hardware drivers for the QEMU virt machine on AArch64. It implements the Hardware Abstraction Layer (HAL) traits defined in `ruvix-hal`.

## Supported Devices

| Device | Description | Base Address | Driver |
|--------|-------------|--------------|--------|
| PL011 UART | ARM PrimeCell UART | 0x0900_0000 | `pl011::Pl011` |
| GICv2 | ARM Generic Interrupt Controller | 0x0800_0000 (GICD)<br>0x0800_1000 (GICC) | `gic::Gic` |
| ARM Generic Timer | System timer with deadline scheduling | System registers | `timer::ArmGenericTimer` |

## Memory Map (QEMU virt)

```
0x0000_0000 - 0x0800_0000  Flash/ROM
0x0800_0000 - 0x0800_0FFF  GIC Distributor (GICD)
0x0800_1000 - 0x0800_1FFF  GIC CPU Interface (GICC)
0x0900_0000 - 0x0900_0FFF  PL011 UART
0x4000_0000 - 0x????_????  RAM (configurable, typically 128MB - 8GB)
```

## Usage

### PL011 UART

```rust
use ruvix_drivers::pl011::Pl011;
use core::fmt::Write;

// Initialize UART at base address 0x0900_0000
let mut uart = Pl011::new(0x0900_0000);
uart.init().expect("UART init failed");

// Write a string (blocking)
write!(uart, "Hello, RuVix!\n").expect("Write failed");

// Write bytes directly
uart.write_byte(b'A').expect("Write failed");

// Read a byte (non-blocking)
if let Some(byte) = uart.read_byte() {
    uart.write_byte(byte).expect("Echo failed");
}

// Enable receive interrupt
uart.enable_rx_interrupt().expect("Failed to enable RX IRQ");

// Flush transmit buffer
uart.flush().expect("Flush failed");
```

### GICv2 (Interrupt Controller)

```rust
use ruvix_drivers::gic::Gic;

// Initialize GIC
let mut gic = Gic::new(0x0800_0000, 0x0800_1000);
gic.init().expect("GIC init failed");

// Enable UART interrupt (IRQ 33 for QEMU virt)
gic.enable(33).expect("Failed to enable IRQ 33");

// Set priority (0 = highest, 15 = lowest)
gic.set_priority(33, 1).expect("Failed to set priority");

// In interrupt handler:
if let Some(irq) = gic.acknowledge() {
    match irq {
        33 => {
            // Handle UART interrupt
            println!("UART interrupt!");
        }
        30 => {
            // Handle timer interrupt
            println!("Timer interrupt!");
        }
        _ => {
            println!("Unknown IRQ: {}", irq);
        }
    }

    // Signal end of interrupt
    gic.end_of_interrupt(irq).expect("Failed to EOI");
}

// Check if interrupt is pending
if gic.is_pending(33).unwrap_or(false) {
    println!("IRQ 33 is pending");
}

// Get number of supported interrupts
let max_irqs = gic.max_interrupts();
println!("GIC supports {} interrupts", max_irqs);
```

### ARM Generic Timer

```rust
use ruvix_drivers::timer::ArmGenericTimer;

// Initialize timer
let timer = ArmGenericTimer::new();

// Get current time in nanoseconds (monotonic)
let now = timer.now_ns();
println!("Current time: {} ns", now);

// Get raw counter value
let ticks = timer.now_ticks();
println!("Counter: {} ticks", ticks);

// Get timer frequency
let freq = timer.frequency();
println!("Timer frequency: {} Hz", freq);

// Set timer to fire in 1 second (relative)
timer.set_timeout_ns(1_000_000_000).expect("Failed to set timeout");

// Set timer to fire at absolute time (absolute)
let deadline = now + 1_000_000_000; // 1 second from now
timer.set_deadline_ns(deadline).expect("Failed to set deadline");

// Enable timer interrupt
timer.enable().expect("Failed to enable timer");

// In interrupt handler:
if timer.is_pending() {
    println!("Timer fired!");
    timer.acknowledge().expect("Failed to ACK timer");
}

// Disable timer
timer.disable().expect("Failed to disable timer");
```

## Interrupt Numbers (QEMU virt)

| IRQ | Device | Description |
|-----|--------|-------------|
| 0-15 | SGI | Software Generated Interrupts |
| 16-31 | PPI | Private Peripheral Interrupts |
| 27 | EL1 Physical Timer | Generic Timer (secure) |
| 30 | EL1 Non-secure Physical Timer | Generic Timer (non-secure) |
| 33 | UART0 | PL011 UART receive/transmit |

## MMIO Utilities

The `mmio` module provides type-safe MMIO operations:

```rust
use ruvix_drivers::mmio::{read_volatile, write_volatile, dsb, dmb, isb, MmioReg};

// Low-level volatile operations
unsafe {
    let uart_dr = 0x0900_0000 as *mut u32;
    write_volatile(uart_dr, 0x41); // Write 'A'
    dsb(); // Ensure write completes

    dmb(); // Memory barrier
    let value = read_volatile(uart_dr);
    isb(); // Instruction synchronization
}

// Type-safe register wrapper
unsafe {
    let mut dr = MmioReg::<u32>::new(0x0900_0000);
    dr.write(0x41); // Automatic barriers

    let value = dr.read(); // Automatic barriers

    dr.modify(|val| val | 0x10); // Read-modify-write with barriers
}
```

## Memory Barriers

All MMIO operations use appropriate memory barriers:

- **DMB (Data Memory Barrier)** - Ensures memory accesses complete before next instruction
- **DSB (Data Synchronization Barrier)** - Stronger than DMB, waits for all instructions to complete
- **ISB (Instruction Synchronization Barrier)** - Flushes pipeline, ensures subsequent instructions see barrier effects

Read operations use `DMB` before and after the read.
Write operations use `DSB` before and after the write.

## Safety

All drivers follow these safety principles:

1. **All MMIO accesses are volatile** - Prevents compiler optimizations
2. **Memory barriers are used correctly** - Ensures ordering guarantees
3. **Register addresses are validated** - Bounds checking where possible
4. **All `unsafe` blocks have SAFETY comments** - Documents safety invariants

## Features

- `qemu-virt` - Enable QEMU virt machine support (default)

## Dependencies

- `ruvix-types` - Kernel interface types
- `ruvix-hal` - Hardware abstraction layer traits

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## References

- [ARM PrimeCell PL011 UART Technical Reference Manual](https://developer.arm.com/documentation/ddi0183/latest/)
- [ARM Generic Interrupt Controller Architecture Specification](https://developer.arm.com/documentation/ihi0048/latest/)
- [ARM Architecture Reference Manual ARMv8](https://developer.arm.com/documentation/ddi0487/latest/)
- [QEMU virt Machine Documentation](https://www.qemu.org/docs/master/system/arm/virt.html)
