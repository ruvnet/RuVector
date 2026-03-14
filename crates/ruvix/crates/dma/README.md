# ruvix-dma

DMA (Direct Memory Access) controller abstraction for the RuVix Cognition Kernel.

## Overview

This crate provides a hardware-agnostic DMA controller interface for efficient
zero-copy data transfers between memory regions and peripheral devices. It is
designed as part of ADR-087 for the RuVix Cognition Kernel.

## Features

- **No-std compatible**: Works in bare-metal environments
- **Zero unsafe code**: All public APIs are safe Rust
- **Scatter-gather support**: Descriptor chains for non-contiguous transfers
- **Cache coherent buffers**: Proper cache management for DMA operations
- **Platform-agnostic**: Works across ARM64, RISC-V, and x86_64

## Core Types

### DmaChannel

Represents a single DMA channel with state tracking:

```rust
use ruvix_dma::{DmaChannel, DmaChannelId, DmaStatus};

let channel = DmaChannel::new(DmaChannelId::new(0));
assert!(!channel.is_busy());
assert_eq!(channel.status(), DmaStatus::Idle);
```

### DmaDirection

Transfer direction enumeration:

```rust
use ruvix_dma::DmaDirection;

let dir = DmaDirection::MemToDevice;
assert!(dir.reads_memory());
assert!(!dir.writes_memory());
```

### DmaBuffer

Cache-coherent memory buffer for DMA transfers:

```rust
use ruvix_dma::{DmaBuffer, DmaBufferFlags};

let buffer = DmaBuffer::new(
    0x1000,              // Physical address
    0xFFFF_0000_1000,    // Virtual address
    4096,                // Size
    DmaBufferFlags::read_write(),
).unwrap();

assert_eq!(buffer.physical_addr(), 0x1000);
assert!(buffer.is_cache_coherent());
```

### DmaConfig

Transfer configuration:

```rust
use ruvix_dma::{DmaConfig, DmaTransferWidth, DmaBurstSize};

let config = DmaConfig::mem_to_mem(0x1000, 0x2000, 4096)
    .with_width(DmaTransferWidth::DoubleWord)
    .with_burst_size(DmaBurstSize::Burst16)
    .with_priority(5);

assert!(config.is_valid());
```

### DmaDescriptor

Scatter-gather descriptor for linked transfers:

```rust
use ruvix_dma::{DmaDescriptor, DmaDescriptorChain};

let mut chain = DmaDescriptorChain::new(0x1_0000);
chain.add_segment(0x1000, 0x2000, 4096).unwrap();
chain.add_segment(0x2000, 0x3000, 2048).unwrap();

assert_eq!(chain.total_length(), 6144);
assert!(chain.is_valid());
```

### DmaController Trait

Interface that platform implementations must satisfy:

```rust
use ruvix_dma::{DmaController, DmaControllerExt};

fn transfer<D: DmaController>(dma: &mut D) -> Result<(), ruvix_dma::DmaError> {
    // Simple memory copy using convenience method
    dma.memcpy(0x1000, 0x2000, 4096)?;

    // Or manual channel management
    let channel = dma.allocate_channel()?;
    let config = ruvix_dma::DmaConfig::mem_to_mem(0x1000, 0x2000, 4096);

    dma.configure(&channel, &config)?;
    dma.start_transfer(&channel, None)?;

    // Poll for completion
    loop {
        match dma.poll_completion(&channel) {
            ruvix_dma::DmaStatus::Complete => break,
            ruvix_dma::DmaStatus::Running => continue,
            _ => return Err(ruvix_dma::DmaError::timeout()),
        }
    }

    dma.release_channel(&channel)?;
    Ok(())
}
```

## DmaStatus

Transfer status enumeration:

| Status | Description |
|--------|-------------|
| `Idle` | Channel is idle, no transfer in progress |
| `Running` | Transfer is currently running |
| `Complete` | Transfer completed successfully |
| `Error(kind)` | Transfer encountered an error |
| `Paused` | Transfer was paused |
| `Aborted` | Transfer was aborted |

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_DMA_CHANNELS` | 16 | Maximum supported channels |
| `MAX_DESCRIPTOR_CHAIN_LENGTH` | 256 | Maximum descriptors per chain |
| `DEFAULT_TRANSFER_TIMEOUT_US` | 1,000,000 | Default timeout (1 second) |
| `DMA_BUFFER_ALIGNMENT` | 64 | Minimum buffer alignment (cache line) |

## Cache Coherency

The crate provides proper cache management for non-coherent systems:

```rust
use ruvix_dma::{DmaBuffer, DmaBufferFlags};

let mut buffer = DmaBuffer::new(
    0x1000,
    0xFFFF_0000_1000,
    4096,
    DmaBufferFlags::read_write(),
).unwrap();

// Before writing to device
buffer.sync_for_device();

// After reading from device
buffer.sync_for_cpu();
```

## License

MIT OR Apache-2.0
