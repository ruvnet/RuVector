# ruvix-aarch64

AArch64 architecture support for the RuVix Cognition Kernel.

## Overview

This crate provides low-level AArch64 support for RuVix, implementing the hardware abstraction layer defined in `ruvix-hal`. It includes:

- **Boot sequence**: Assembly entry point, BSS initialization, early setup
- **Memory Management Unit (MMU)**: 4-level page tables, capability-based permissions
- **Exception handling**: Synchronous exceptions, IRQ, FIQ, SError
- **System registers**: Safe accessors for AArch64 system registers

## Memory Layout

```
User space (TTBR0_EL1):
  0x0000_0000_0000_0000 - 0x0000_FFFF_FFFF_FFFF

Kernel space (TTBR1_EL1):
  0xFFFF_0000_0000_0000 - 0xFFFF_FFFF_FFFF_FFFF
```

The kernel uses a split address space with TTBR0 for user mappings and TTBR1 for kernel mappings. This enables efficient context switching (only TTBR0 needs to be updated).

## Boot Sequence

1. **Assembly entry** (`_start` in `src/asm/boot.S`):
   - Disable interrupts (DAIF)
   - Drop from EL2/EL3 to EL1 if necessary
   - Set up stack pointer
   - Clear BSS section
   - Jump to `early_init()`

2. **Early initialization** (`early_init()` in `src/boot.rs`):
   - Configure MMU (MAIR, TCR, page tables)
   - Configure exception vectors (VBAR)
   - Enable MMU and caches (SCTLR)
   - Jump to `kernel_main()`

3. **Kernel main** (`kernel_main()` in `src/boot.rs`):
   - Entry point for high-level kernel initialization
   - Currently just halts with WFI

## MMU Configuration

### Page Tables

The MMU uses 4KB pages with 4-level page tables (48-bit virtual addresses):

- **Level 0**: 512GB regions (bits 47-39)
- **Level 1**: 1GB blocks (bits 38-30)
- **Level 2**: 2MB blocks (bits 29-21)
- **Level 3**: 4KB pages (bits 20-12)

### Memory Attributes

Three memory attribute indices are configured in MAIR_EL1:

- **Index 0**: Device-nGnRnE (non-cacheable device memory)
- **Index 1**: Normal Write-Back (cacheable, write-back)
- **Index 2**: Normal Write-Through (cacheable, write-through)

### Capability Mapping

RuVix capabilities map to AArch64 page table permissions:

| Capability | PTE Flags |
|------------|-----------|
| `READ` | VALID, RO, XN |
| `READ+WRITE` | VALID, XN |
| `READ+EXECUTE` | VALID, RO |
| `READ+WRITE+EXECUTE` | VALID |
| `USER` | USER flag set |

## Exception Handling

### Vector Table

The exception vector table is 2KB aligned and contains 16 entries (4 exception types × 4 sources):

```
Exception types: Sync, IRQ, FIQ, SError
Sources: Current EL SP0, Current EL SPx, Lower EL AArch64, Lower EL AArch32
```

### Exception Context

When an exception occurs, the assembly wrapper saves the full CPU context:

```rust
#[repr(C)]
pub struct ExceptionContext {
    pub gpr: [u64; 31],  // x0-x30
    pub sp: u64,          // Stack pointer
    pub elr: u64,         // Exception link register
    pub spsr: u64,        // Saved program status
}
```

### Handlers

- `handle_sync_exception()`: Synchronous exceptions (syscalls, page faults)
- `handle_irq()`: IRQ interrupts
- `handle_fiq()`: FIQ interrupts
- `handle_serror()`: SError interrupts

## System Registers

All system register access is `unsafe` and documented with safety requirements:

```rust
// Read SCTLR_EL1
let sctlr = unsafe { sctlr_el1_read() };

// Write SCTLR_EL1 (enable MMU and caches)
unsafe { sctlr_el1_write(sctlr | 0x1005) };
```

Key registers:

- **SCTLR_EL1**: System control (MMU, caches, alignment)
- **TCR_EL1**: Translation control (page size, address size)
- **MAIR_EL1**: Memory attribute indirection
- **TTBR0_EL1/TTBR1_EL1**: Page table base registers
- **VBAR_EL1**: Exception vector base address
- **ESR_EL1**: Exception syndrome (exception information)
- **FAR_EL1**: Fault address (for page faults)

## Features

- `qemu-virt`: Target QEMU virt platform (RAM at 0x4000_0000)

## Safety

This crate uses extensive `unsafe` code for:

- Assembly code (boot, exception vectors)
- System register access
- Memory-mapped I/O
- Direct memory manipulation (BSS clearing, page table setup)

All `unsafe` blocks are documented with `SAFETY` comments explaining:

1. **Why the operation is unsafe**
2. **What invariants must hold**
3. **Why the invariants are satisfied**

## Testing

Unit tests verify:

- Exception class extraction from ESR
- Capability to PTE flag conversion
- Register accessor compilation
- Memory layout constants

Integration tests (in `ruvix-kernel`) verify:

- Boot sequence on QEMU
- MMU initialization
- Exception handling
- System call dispatch

## License

Dual licensed under MIT or Apache-2.0 (see repository root).
