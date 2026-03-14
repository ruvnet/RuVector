# RuVix AArch64 Boot Infrastructure

Bare metal boot infrastructure for RuVix on AArch64 architecture.

## Overview

This directory contains the build infrastructure for creating a bootable RuVix kernel for AArch64 systems, specifically targeting QEMU's `virt` machine with Cortex-A72 CPU.

## Architecture

### Memory Layout

- **Base Address**: `0x40000000` (1 GiB)
- **RAM Size**: 128 MiB
- **Stack Size**: 64 KiB
- **Heap**: Dynamic, from `__heap_start` to end of RAM

### Memory Sections

```
0x40000000  .text.boot    Boot code (KEEP enforced)
            .text         Code section
            .rodata       Read-only data (4K aligned)
            .data         Initialized data (4K aligned)
            .bss          Uninitialized data (4K aligned)
            __stack       64K stack
            __heap        Heap to end of RAM
```

## Files

### Build Configuration

- **linker.ld**: Linker script defining memory layout and sections
- **aarch64-ruvix.json**: Custom Rust target specification for bare metal AArch64
- **.cargo/config.toml**: Cargo build configuration
- **build.rs**: Build script to link with custom linker script
- **Makefile**: Build automation with QEMU integration

## Prerequisites

### Required Tools

```bash
# Rust nightly (for build-std)
rustup default nightly

# QEMU for AArch64
# macOS
brew install qemu

# Ubuntu/Debian
sudo apt-get install qemu-system-aarch64

# Fedora
sudo dnf install qemu-system-aarch64
```

### Rust Components

```bash
rustup component add rust-src
rustup component add llvm-tools-preview
```

## Building

### Quick Build

```bash
make build
```

This runs:
```bash
cargo build --release
```

### Build Details

The build process:

1. Compiles with custom target `aarch64-ruvix.json`
2. Links using `linker.ld` script via `rust-lld`
3. Builds Rust core libraries from source (`build-std`)
4. Creates a bare metal binary at `target/aarch64-ruvix/release/ruvix-kernel`

## Running

### Run in QEMU

```bash
make run
```

This executes:
```bash
qemu-system-aarch64 \
    -machine virt \
    -cpu cortex-a72 \
    -m 128M \
    -nographic \
    -kernel target/aarch64-ruvix/release/ruvix-kernel
```

### Debug in QEMU

```bash
make debug
```

This starts QEMU with GDB server on port 1234:
```bash
qemu-system-aarch64 \
    -machine virt \
    -cpu cortex-a72 \
    -m 128M \
    -nographic \
    -s -S \
    -kernel target/aarch64-ruvix/release/ruvix-kernel
```

Then in another terminal:
```bash
gdb-multiarch target/aarch64-ruvix/release/ruvix-kernel
(gdb) target remote :1234
(gdb) continue
```

## Target Specification

The `aarch64-ruvix.json` target:

- **Architecture**: AArch64
- **OS**: None (bare metal)
- **Panic**: Abort (no unwinding)
- **Red Zone**: Disabled (required for bare metal)
- **Features**: Strict alignment, NEON, FP-ARMv8
- **Linker**: rust-lld (LLD linker from LLVM)

## Memory Map (QEMU virt)

The QEMU `virt` machine provides:

| Address Range | Device |
|---------------|--------|
| 0x00000000 - 0x08000000 | Flash (128 MiB) |
| 0x08000000 - 0x09000000 | Device memory |
| 0x09000000 - 0x09010000 | UART (PL011) |
| 0x40000000 - 0x48000000 | RAM (128 MiB) |

Our kernel is loaded at `0x40000000` (RAM base).

## Customization

### Changing RAM Size

Edit `linker.ld`:
```ld
MEMORY {
    RAM (rwx) : ORIGIN = 0x40000000, LENGTH = 256M  /* 256 MiB */
}
```

And `Makefile`:
```makefile
-m 256M
```

### Changing Stack Size

Edit `linker.ld`:
```ld
. = . + 128K;  /* 128 KiB stack */
```

### Adding Sections

Edit `linker.ld` to add custom sections:
```ld
.custom_section : ALIGN(4K) {
    __custom_start = .;
    *(.custom)
    __custom_end = .;
} > RAM
```

## Troubleshooting

### Build Errors

**Error**: `error: linking with 'rust-lld' failed`
- Ensure `llvm-tools-preview` is installed: `rustup component add llvm-tools-preview`

**Error**: `error: requires nightly`
- Switch to nightly: `rustup default nightly`

**Error**: `can't find crate for 'core'`
- Install rust-src: `rustup component add rust-src`

### Runtime Issues

**QEMU doesn't start**
- Verify QEMU is installed: `qemu-system-aarch64 --version`
- Check kernel exists: `ls -lh target/aarch64-ruvix/release/ruvix-kernel`

**Kernel crashes immediately**
- Ensure `_start` symbol is defined in your kernel
- Check that `.text.boot` section contains the entry point
- Verify stack is properly initialized before calling Rust code

## Next Steps

1. Create `src/main.rs` with `_start` entry point
2. Implement UART driver for console output
3. Set up page tables and MMU
4. Initialize heap allocator
5. Add interrupt handling

## References

- [ARM Cortex-A Series Programmer's Guide](https://developer.arm.com/documentation/den0024/latest)
- [QEMU virt Machine Documentation](https://www.qemu.org/docs/master/system/arm/virt.html)
- [Rust Embedded Book](https://rust-embedded.github.io/book/)
- [OSDev Wiki - ARM](https://wiki.osdev.org/ARM)

## License

Part of the RuVector project.
