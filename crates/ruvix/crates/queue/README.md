# ruvix-queue

io_uring-style ring buffer IPC for the RuVix Cognition Kernel (ADR-087).

## Overview

This crate implements the Queue primitive from ADR-087 Section 7. All inter-task communication in RuVix goes through queues. There are no synchronous IPC calls, no shared memory without explicit region grants, and no signals.

## Architecture

Queues use io_uring-style ring buffers with separate submission (SQ) and completion (CQ) queues:

```
┌─────────────────────────────────────────────────┐
│                  KernelQueue                     │
├────────────────────┬────────────────────────────┤
│   Submission (SQ)  │     Completion (CQ)        │
│  ┌──┬──┬──┬──┬──┐ │   ┌──┬──┬──┬──┬──┐        │
│  │  │  │  │  │  │ │   │  │  │  │  │  │        │
│  └──┴──┴──┴──┴──┘ │   └──┴──┴──┴──┴──┘        │
│     head→ ←tail   │      head→ ←tail          │
└────────────────────┴────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Lock-free** | Using atomic head/tail pointers |
| **Zero-copy** | Descriptors point to shared regions |
| **Typed** | WIT schema validation at send time |
| **Priority** | Higher priority messages delivered first |

## Components

### KernelQueue

Main queue implementation:

```rust
use ruvix_queue::{KernelQueue, QueueConfig};
use ruvix_types::MsgPriority;

// Create a queue with 64 entries and 4KB max message
let config = QueueConfig::new(64, 4096);
let mut queue = KernelQueue::new(config, region_handle)?;

// Send a message
queue.send(b"hello", MsgPriority::Normal)?;

// Receive with timeout
let mut buf = [0u8; 4096];
let len = queue.recv(&mut buf, Duration::from_millis(100))?;
```

### RingBuffer

Lock-free ring buffer with atomic operations:

```rust
use ruvix_queue::{RingBuffer, RingEntry};

let mut ring = RingBuffer::<64>::new();

// Push entry
ring.push(entry)?;

// Pop entry
let entry = ring.pop()?;

// Check stats
println!("Head: {}, Tail: {}", ring.head(), ring.tail());
```

### MessageDescriptor

Zero-copy message passing:

```rust
use ruvix_queue::MessageDescriptor;

// Descriptor points to data in a shared region
let descriptor = MessageDescriptor {
    region: region_handle,
    offset: 0,
    length: 768 * 4,  // 768 floats
    flags: 0,
};

// Send descriptor instead of copying
queue.send_descriptor(descriptor)?;
```

## Zero-Copy Semantics

When sender and receiver share a region, `queue_send` places a descriptor (offset + length) in the ring rather than copying bytes. This is critical for high-throughput vector streaming where copying 768-dimensional f32 vectors would be prohibitive.

**TOCTOU Protection** (ADR-087 Section 20.5): Only Immutable or AppendOnly regions can use descriptors. The kernel rejects descriptors pointing into Slab regions to prevent time-of-check-to-time-of-use attacks.

## Priority Levels

```rust
use ruvix_types::MsgPriority;

// Four priority levels
let critical = MsgPriority::Critical;  // Sensor alerts
let high = MsgPriority::High;          // Control messages
let normal = MsgPriority::Normal;      // Data messages
let low = MsgPriority::Low;            // Background tasks
```

## Statistics

```rust
use ruvix_queue::RingStats;

let stats = queue.stats();
println!("Messages sent: {}", stats.messages_sent);
println!("Messages received: {}", stats.messages_received);
println!("Bytes transferred: {}", stats.bytes_transferred);
```

## Features

- `std` (default): Enable standard library support
- `alloc`: Enable alloc crate support

## Integration with RuVix

This crate integrates with:

- `ruvix-types`: Core type definitions (`QueueHandle`, `MsgPriority`)
- `ruvix-cap`: Capability checking for queue access
- `ruvix-region`: Region handles for zero-copy descriptors

## License

MIT OR Apache-2.0
