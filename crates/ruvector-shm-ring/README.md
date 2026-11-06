# ruvector-shm-ring

A record ring for one writer and many readers. The head and each slot's sequence live in the mapping, so another process that maps the same bytes sees them.

`Geometry::fit(max_record, ring_bytes)` chooses the slot grid from a payload limit and a byte budget. On Linux, `allocate` creates a memfd sealed against resize and returns a read-only descriptor beside the read-write one. On macOS it uses POSIX shared memory and removes the name before returning. A reader that tries to map the read-only descriptor writable is refused by the kernel.

Each reader keeps its own cursor. A reader that falls behind skips to the oldest record still intact. The writer does not wait.

This crate does not name streams, decide who may hold a descriptor, or choose what a slow reader should do. Those belong to the process that hands the descriptor out.

`AtomicU64` must be lock-free. Sequence numbers stop before `u64::MAX`, which marks a slot the writer is filling.

```rust
use ruvector_shm_ring::Geometry;

let geometry = Geometry::fit(640, 1024 * 1024).expect("two slots fit");
assert!(geometry.bytes() <= 1024 * 1024);
```

License: MIT
