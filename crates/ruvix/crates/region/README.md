# ruvix-region

Memory region management for the RuVix Cognition Kernel (ADR-087).

## Overview

This crate implements the Region primitive. A region is a contiguous, capability-protected memory object with one of three policies. RuVix does not implement demand paging—all regions are physically backed at `region_map` time, eliminating page faults, swap, and copy-on-write complexity.

## Region Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| **Immutable** | Set once at creation, never modified, deduplicatable | Read-only data, constants |
| **AppendOnly** | Only append, never overwrite, with max_size and write cursor | Logs, event streams |
| **Slab** | Fixed-size slots from free list, no fragmentation | Vector storage, HNSW nodes |

## Components

### RegionManager

Creates and manages regions:

```rust
use ruvix_region::{RegionManager, RegionConfig};
use ruvix_types::{RegionPolicy, CapHandle};

let mut manager = RegionManager::new();

// Create a slab region for fixed-size allocations
let slab_handle = manager.create_region(
    RegionPolicy::slab(64, 1024), // 64-byte slots, 1024 slots
    CapHandle::null(),
)?;
```

### SlabAllocator

Fast fixed-size allocation with zero fragmentation:

```rust
use ruvix_region::SlabAllocator;

let mut slab = SlabAllocator::new(64, 1024); // 64-byte slots

// Allocate a slot
let slot = slab.alloc()?;

// Free the slot
slab.free(slot)?;
```

### AppendOnlyRegion

Append-only semantics for logs and event streams:

```rust
use ruvix_region::AppendOnlyRegion;

let mut region = AppendOnlyRegion::new(4096); // 4KB max

// Append data
let offset = region.append(&data)?;

// Read back (no modification allowed)
let slice = region.read(offset, data.len())?;
```

### ImmutableRegion

Write-once semantics for constant data:

```rust
use ruvix_region::ImmutableRegion;

// Create and initialize in one step
let region = ImmutableRegion::new(&initial_data);

// Read-only access
let slice = region.as_slice();
```

## Security Properties

From ADR-087 Section 5:

- **No shared memory without explicit grants**: Regions require capability to access
- **TOCTOU protection**: Slab regions cannot be used for zero-copy IPC descriptors
- **No demand paging**: Regions are fully backed, no page faults possible
- **Physically contiguous**: Memory is contiguous for DMA compatibility

## Statistics

With the `stats` feature enabled:

```rust
use ruvix_region::RegionStats;

// Track region operations
let stats = manager.stats();
println!("Regions created: {}", stats.regions_created);
println!("Bytes allocated: {}", stats.bytes_allocated);
println!("Slab allocs: {}", stats.slab_allocs);
```

## Features

- `std` (default): Enable standard library support
- `alloc`: Enable alloc crate support
- `mmap`: Use mmap for backing memory on Linux
- `stats`: Enable statistics collection

## Integration with RuVix

This crate integrates with:

- `ruvix-types`: Core type definitions (`RegionHandle`, `RegionPolicy`)
- `ruvix-cap`: Capability checking for region access
- `ruvix-vecgraph`: Slab regions for vector storage

## License

MIT OR Apache-2.0
