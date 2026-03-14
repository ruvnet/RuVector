# ruvix-physmem

Physical memory allocator for the RuVix Cognition Kernel (ADR-087).

## Overview

This crate provides a **buddy allocator** for physical page frame allocation. The buddy system uses power-of-two block sizes (4KB to 2MB) for efficient allocation and deallocation with minimal external fragmentation.

## Features

- **Buddy Allocation**: Power-of-two block sizes with automatic splitting and coalescing
- **No External Dependencies**: Pure `no_std` implementation
- **Type-Safe Addresses**: `PhysAddr` newtype wrapper for physical addresses
- **Comprehensive Statistics**: Track allocations, frees, splits, and coalesces
- **Page Frame Abstraction**: `PageFrame` type for safer memory management

## Block Sizes

| Order | Pages | Size |
|-------|-------|------|
| 0 | 1 | 4KB |
| 1 | 2 | 8KB |
| 2 | 4 | 16KB |
| 3 | 8 | 32KB |
| 4 | 16 | 64KB |
| 5 | 32 | 128KB |
| 6 | 64 | 256KB |
| 7 | 128 | 512KB |
| 8 | 256 | 1MB |
| 9 | 512 | 2MB |

## Usage

### Basic Allocation

```rust
use ruvix_physmem::{BuddyAllocator, PhysAddr, PAGE_SIZE};

// Create allocator for 16MB of memory starting at physical address 0x1000_0000
let mut allocator = BuddyAllocator::new(PhysAddr::new(0x1000_0000), 4096);

// Allocate a single page (4KB)
let addr = allocator.alloc_pages(1).expect("allocation failed");
assert!(addr.is_page_aligned());

// Allocate 4 contiguous pages (16KB)
let addr4 = allocator.alloc_pages(4).expect("allocation failed");

// Free the allocations
allocator.dealloc_pages(addr, 1);
allocator.dealloc_pages(addr4, 4);
```

### Using PageFrame

```rust
use ruvix_physmem::{BuddyAllocator, PhysAddr};

let mut allocator = BuddyAllocator::new(PhysAddr::new(0x1000_0000), 1024);

// Allocate and get a PageFrame
if let Some(frame) = allocator.alloc_frame(8) {
    println!("Allocated {} pages at {}", frame.pages(), frame.addr());

    // Check if an address is within the frame
    assert!(frame.contains(frame.addr()));

    // Free the frame
    allocator.free_frame(&frame);
}
```

### Error Handling

```rust
use ruvix_physmem::{BuddyAllocator, PhysAddr, PhysMemError};

let mut allocator = BuddyAllocator::new(PhysAddr::new(0x1000_0000), 64);

// Try to allocate with explicit error handling
match allocator.try_alloc_pages(32) {
    Ok(addr) => {
        println!("Allocated at {}", addr);
        allocator.dealloc_pages(addr, 32);
    }
    Err(PhysMemError::OutOfMemory) => {
        println!("No memory available");
    }
    Err(e) => {
        println!("Allocation failed: {}", e);
    }
}
```

### Statistics

```rust
use ruvix_physmem::{BuddyAllocator, PhysAddr};

let mut allocator = BuddyAllocator::new(PhysAddr::new(0x1000_0000), 1024);

// Perform some allocations
let a1 = allocator.alloc_pages(10).unwrap();
let a2 = allocator.alloc_pages(5).unwrap();
allocator.dealloc_pages(a1, 10);

// Check statistics
let stats = allocator.stats();
println!("Allocations: {}", stats.allocations());
println!("Frees: {}", stats.frees());
println!("Splits: {}", stats.splits());
println!("Coalesces: {}", stats.coalesces());
println!("Utilization: {}%", stats.utilization_percent());

// Per-order statistics
let order_stats = allocator.order_stats();
for (order, stat) in order_stats.iter().enumerate() {
    if stat.free_blocks > 0 {
        println!("Order {}: {} free blocks", order, stat.free_blocks);
    }
}
```

### Physical Address Operations

```rust
use ruvix_physmem::PhysAddr;

let addr = PhysAddr::new(0x1234_5678);

// Check alignment
assert!(!addr.is_page_aligned());
assert_eq!(addr.align_down().as_u64(), 0x1234_5000);
assert_eq!(addr.align_up().as_u64(), 0x1234_6000);

// Page frame number
assert_eq!(addr.pfn(), 0x12345);
assert_eq!(addr.page_offset(), 0x678);

// Address arithmetic
let next = addr.add_pages(1);
assert_eq!(next.as_u64(), 0x1234_6678);

// From page frame number
let addr2 = PhysAddr::from_pfn(256);
assert_eq!(addr2.as_u64(), 0x10_0000); // 1MB
```

### Page Order Calculations

```rust
use ruvix_physmem::{PageOrder, pages_to_order, order_to_pages, order_to_bytes};

// Calculate order from page count
assert_eq!(pages_to_order(1), 0);
assert_eq!(pages_to_order(3), 2); // rounds up to 4 pages
assert_eq!(pages_to_order(512), 9);

// Calculate pages from order
assert_eq!(order_to_pages(0), 1);
assert_eq!(order_to_pages(3), 8);
assert_eq!(order_to_pages(9), 512);

// Calculate bytes from order
assert_eq!(order_to_bytes(0), 4096);      // 4KB
assert_eq!(order_to_bytes(9), 2097152);   // 2MB

// Using PageOrder type
let order = PageOrder::from_pages(5).unwrap();
assert_eq!(order.as_usize(), 3); // rounds up: 5 pages -> order 3 (8 pages)
assert_eq!(order.pages(), 8);
assert_eq!(order.bytes(), 32768);
```

### Deferred Initialization

```rust
use ruvix_physmem::{BuddyAllocator, PhysAddr};

// Create uninitialized allocator (useful for static allocation)
let mut allocator = BuddyAllocator::uninit();

// Later, initialize with actual memory
allocator.init(PhysAddr::new(0x2000_0000), 2048).unwrap();

// Now ready for use
let addr = allocator.alloc_pages(1).unwrap();
```

## Features

- `std`: Enable standard library support (enables `std::error::Error` impl)
- `alloc`: Enable alloc crate support
- `stats`: Enable detailed statistics collection
- `debug-alloc`: Enable debug assertions for allocation tracking

## Implementation Details

### Buddy System Algorithm

The buddy allocator works by:

1. **Initialization**: Memory is organized into the largest possible power-of-two blocks
2. **Allocation**:
   - Find the smallest free block that fits the request
   - Split larger blocks as needed (creating "buddies")
3. **Deallocation**:
   - Mark the block as free
   - Coalesce with its buddy if the buddy is also free
   - Repeat coalescing until no more merges are possible

### Buddy Address Calculation

For a block at address `A` with order `n`:
- Block size: `2^n * PAGE_SIZE` bytes
- Buddy address: `A XOR (2^n * PAGE_SIZE)`

This XOR property ensures that buddies always have the correct alignment for the next higher order.

## Performance Characteristics

| Operation | Time Complexity |
|-----------|-----------------|
| alloc_pages | O(MAX_ORDER) |
| free_pages | O(MAX_ORDER) |
| Split | O(1) per split |
| Coalesce | O(1) per merge |

Space overhead: O(n) where n is the number of free blocks (stored in free lists).

## Integration with RuVix

This allocator is designed to be used by the RuVix kernel for managing physical memory. It integrates with:

- `ruvix-types`: Uses `KernelError` for error propagation
- `ruvix-region`: Provides backing memory for virtual regions
- `ruvix-nucleus`: Called from the kernel's memory management syscalls

## License

MIT OR Apache-2.0
