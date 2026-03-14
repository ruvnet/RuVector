# ruvix-bcm2711

BCM2711/BCM2712 SoC drivers for Raspberry Pi 4/5 as part of the RuVix Cognition Kernel (ADR-087).

## Overview

This crate provides low-level, `no_std` drivers for the Broadcom BCM2711 (Raspberry Pi 4) and BCM2712 (Raspberry Pi 5) System-on-Chip devices.

## Supported Hardware

- **Raspberry Pi 4 Model B** (1GB, 2GB, 4GB, 8GB)
- **Raspberry Pi 400**
- **Raspberry Pi Compute Module 4**
- **Raspberry Pi 5** (BCM2712, with `rpi5` feature)

## Features

### GPIO Controller (`gpio`)

- 58 GPIO pins with configurable functions
- Pull-up/pull-down resistor control (BCM2711 new register layout)
- Edge detection and event status
- Digital input/output

```rust
use ruvix_bcm2711::gpio::{Gpio, Function, Pull};

let mut gpio = Gpio::new();

// Configure GPIO 14 for mini UART TX
gpio.set_function(14, Function::Alt5).unwrap();
gpio.set_pull(14, Pull::None).unwrap();

// Use GPIO 21 as output LED
gpio.set_function(21, Function::Output).unwrap();
gpio.write(21, true);  // Turn on
```

### Mini UART (`mini_uart`)

- Serial console I/O via auxiliary UART
- Configurable baud rate (default 115200)
- 8N1 configuration
- Implements `core::fmt::Write`

```rust
use ruvix_bcm2711::mini_uart::MiniUart;
use core::fmt::Write;

let mut uart = MiniUart::new();
uart.init().unwrap();

writeln!(uart, "Hello from RuVix!").unwrap();
```

### VideoCore Mailbox (`mailbox`)

- Communication with GPU firmware
- Hardware information queries
- Clock rate management
- Temperature monitoring

```rust
use ruvix_bcm2711::mailbox::Mailbox;

let mailbox = Mailbox::new();

let revision = mailbox.get_board_revision().unwrap();
let (arm_base, arm_size) = mailbox.get_arm_memory().unwrap();
let temp = mailbox.get_temperature().unwrap(); // millidegrees C
```

### Interrupt Controller (`interrupt`)

- BCM2711 legacy interrupt controller
- GPU IRQs (0-63) and ARM basic IRQs (64-71)
- Enable/disable/pending status

```rust
use ruvix_bcm2711::interrupt::{BcmInterruptController, IRQ_AUX};

let mut irq = BcmInterruptController::new();
irq.init().unwrap();
irq.enable(IRQ_AUX).unwrap(); // Enable mini UART interrupt
```

## Memory Map

### RPi 4 (BCM2711)

| Bus Address | ARM Physical | Description |
|-------------|--------------|-------------|
| 0x7E00_0000 | 0xFE00_0000 | Main peripherals |
| 0x7C00_0000 | 0xFC00_0000 | PCIe / xHCI |
| 0xFF80_0000 | 0x6000_0000 | ARM local peripherals |

### Peripheral Offsets

| Offset | Device |
|--------|--------|
| 0x00003000 | System Timer |
| 0x0000B000 | Interrupt Controller |
| 0x0000B880 | VideoCore Mailbox |
| 0x00200000 | GPIO |
| 0x00201000 | UART0 (PL011) |
| 0x00215000 | Mini UART (AUX) |

## GPIO Pin Assignments

### UART

| Pin | Function | Alt |
|-----|----------|-----|
| 14 | UART0 TXD / UART1 TXD | Alt0 / Alt5 |
| 15 | UART0 RXD / UART1 RXD | Alt0 / Alt5 |

### I2C

| Pin | Function | Alt |
|-----|----------|-----|
| 2 | SDA1 | Alt0 |
| 3 | SCL1 | Alt0 |

### SPI

| Pin | Function | Alt |
|-----|----------|-----|
| 7 | SPI0 CE1 | Alt0 |
| 8 | SPI0 CE0 | Alt0 |
| 9 | SPI0 MISO | Alt0 |
| 10 | SPI0 MOSI | Alt0 |
| 11 | SPI0 SCLK | Alt0 |

## Boot Configuration

Before using the mini UART, ensure your `config.txt` includes:

```
enable_uart=1
```

For RPi 4/5 with 64-bit kernel:

```
arm_64bit=1
enable_uart=1
```

## Clock Frequencies

| Clock | Frequency |
|-------|-----------|
| Core | 500 MHz |
| System | 250 MHz |
| UART (Mini UART ref) | 500 MHz |
| ARM Timer | 54 MHz |

## Cargo Features

- `default` - No special features
- `rpi4` - Raspberry Pi 4 (BCM2711) specific code
- `rpi5` - Raspberry Pi 5 (BCM2712) specific code (different peripheral base)

## Building

```bash
# For RPi 4
cargo build --target aarch64-unknown-none

# For RPi 5
cargo build --target aarch64-unknown-none --features rpi5
```

## Safety

All drivers use MMIO with volatile operations and proper memory barriers (DMB, DSB, ISB). The `unsafe` keyword is only used for:

- Reading/writing hardware registers
- Executing barrier instructions
- VideoCore mailbox property calls

## License

MIT OR Apache-2.0

## References

- [BCM2711 ARM Peripherals](https://datasheets.raspberrypi.com/bcm2711/bcm2711-peripherals.pdf)
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [ARM Cortex-A72 TRM](https://developer.arm.com/documentation/100095/latest)
