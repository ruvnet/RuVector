# ruvix-rpi-boot

Raspberry Pi boot support for the RuVix Cognition Kernel (ADR-087 Phase D).

## Overview

This crate provides early boot support for Raspberry Pi 4/5, including:

- DTB (Device Tree Blob) parsing
- Early UART initialization
- Secondary CPU wake via spin table
- Boot configuration parsing
- EL2 to EL1 transition

## Boot Process

### Raspberry Pi Boot Sequence

1. **GPU ROM** - Loads `bootcode.bin` from SD card
2. **bootcode.bin** - Initializes SDRAM, loads `start4.elf`
3. **start4.elf** - GPU firmware, parses `config.txt`, loads `kernel8.img`
4. **kernel8.img** - Our kernel starts executing

### Entry Point Requirements

When the kernel starts:

| Register | Value |
|----------|-------|
| x0 | DTB physical address |
| x1-x3 | Reserved (0) |

CPU State:
- MMU is OFF
- D-cache is OFF
- I-cache may be ON or OFF
- Interrupts are masked (DAIF)
- Running in EL2 (hypervisor) or EL1 (kernel)

## Usage

### Early UART Output

```rust
use ruvix_rpi_boot::{early_uart_init, early_print, early_println};

// Initialize UART (must be done once)
early_uart_init();

// Print messages
early_print_banner();
early_diagnostics();
early_println("Kernel initializing...");
```

### DTB Parsing

```rust
use ruvix_rpi_boot::dtb::{parse_dtb_header, is_valid_dtb};

// Validate and parse DTB
if let Some(info) = unsafe { parse_dtb_header(dtb_addr) } {
    early_print("DTB size: ");
    early_print_dec(info.total_size as u64);
    early_println(" bytes");
}
```

### Secondary CPU Wake

```rust
use ruvix_rpi_boot::spin_table::{wake_secondary_cpus, get_cpu_id};

// Define entry point for secondary CPUs
extern "C" fn secondary_entry() -> ! {
    let cpu_id = get_cpu_id();
    // Initialize per-CPU data...
    loop { wait_for_interrupt(); }
}

// Wake all secondary CPUs
wake_secondary_cpus(secondary_entry as usize);
```

### Boot Configuration

```rust
use ruvix_rpi_boot::config::{parse_cmdline, BootConfig};

let cmdline = b"console=ttyS0,115200 quiet ruvix.heap_size=128M\0";
let config = parse_cmdline(cmdline);

if config.quiet {
    // Suppress messages
}

if let Some(baud) = config.console_baud {
    // Use specified baud rate
}
```

### EL2 to EL1 Transition

```rust
use ruvix_rpi_boot::{current_el, is_el2};

if is_el2() {
    // Configure EL2 before dropping
    unsafe { ruvix_rpi_boot::drop_to_el1(); }
}

assert!(current_el() == 1);
```

## Memory Layout

### RPi 4 (4GB model)

```
0x0000_0000 - 0x0000_0FFF : Reserved (interrupt vectors)
0x0000_1000 - 0x0007_FFFF : Low memory / spin tables
0x0008_0000 - ...         : Kernel load address (default)
...
0x3B40_0000 - 0x3FFF_FFFF : GPU memory (configurable)
```

### Spin Table Addresses

| CPU | Address |
|-----|---------|
| 0 | Primary (boot CPU) |
| 1 | 0xE0 |
| 2 | 0xE8 |
| 3 | 0xF0 |

## config.txt Settings

Required settings for RuVix:

```
# Enable 64-bit kernel
arm_64bit=1

# Enable mini UART for console
enable_uart=1

# Optional: Set GPU memory
gpu_mem=64

# Optional: Set kernel filename
kernel=kernel8.img
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `console=ttyS0,115200` | Serial console device |
| `quiet` | Suppress kernel messages |
| `debug` | Enable debug messages |
| `nosmp` | Disable SMP (single CPU) |
| `maxcpus=N` | Limit to N CPUs |
| `ruvix.heap_size=64M` | Set heap size |
| `ruvix.log_level=debug` | Set log level |

## Building

### For Raspberry Pi 4

```bash
cargo build --target aarch64-unknown-none --features rpi4
```

### For Raspberry Pi 5

```bash
cargo build --target aarch64-unknown-none --features rpi5
```

## Creating kernel8.img

1. Build the kernel as an ELF binary
2. Extract the binary sections:
   ```bash
   aarch64-none-elf-objcopy -O binary kernel.elf kernel8.img
   ```
3. Copy to SD card boot partition

## Dependencies

- `ruvix-types` - Kernel type definitions
- `ruvix-bcm2711` - BCM2711/BCM2712 SoC drivers

## Cargo Features

- `rpi4` - Raspberry Pi 4 (BCM2711) support
- `rpi5` - Raspberry Pi 5 (BCM2712) support

## Safety

This crate contains unsafe code for:

- MMIO register access
- Assembly instructions (barriers, SEV/WFE)
- Pointer operations for DTB parsing
- Exception level transitions

All unsafe operations are documented with safety comments.

## License

MIT OR Apache-2.0

## References

- [Raspberry Pi Firmware Wiki](https://www.raspberrypi.com/documentation/computers/config_txt.html)
- [ARM64 Linux Kernel Boot Protocol](https://www.kernel.org/doc/Documentation/arm64/booting.txt)
- [DeviceTree Specification](https://devicetree-specification.readthedocs.io/)
- [BCM2711 ARM Peripherals](https://datasheets.raspberrypi.com/bcm2711/bcm2711-peripherals.pdf)
