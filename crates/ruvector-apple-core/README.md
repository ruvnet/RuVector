# ruvector-apple-core

`ruvector-apple-core` is a dependency-free, bounded exact-vector index for
native Apple applications. It exposes both a safe Rust API and a versioned C
ABI that can be consumed by Swift or Objective-C without importing RuVector's
server, storage, network, or application-policy layers.

The initial implementation supports cosine similarity, negative squared L2,
and dot-product ordering; deterministic checksummed snapshots; bounded top-k;
finite-value validation; concurrent readers; and panic containment at the C
boundary. It does not claim approximate-search scale, persistent storage, or a
measured iPhone performance advantage.

## Rust

```rust
use ruvector_apple_core::{
    DistanceMetric, ExactVectorIndex, IndexConfig,
};

let mut index = ExactVectorIndex::new(IndexConfig {
    dimensions: 3,
    capacity: 128,
    metric: DistanceMetric::Cosine,
})?;
index.upsert(7, &[1.0, 0.0, 0.0])?;
let hits = index.search(&[1.0, 0.0, 0.0], 8)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## C and Swift interoperability

The reviewed header is `include/ruvector_apple_core.h`. Search results and
snapshots use self-describing opaque owners: callers borrow their pointer and
length through an accessor, then free the owner without reconstructing a Rust
allocation or supplying its length.

Every non-null opaque pointer supplied to the ABI must be a live, matching
handle returned by this library. Do not forge, reuse, double-free, mix handle
kinds, or destroy a handle concurrently with another call using it. Borrowed
result and snapshot pointers are immutable and remain valid only until their
owner is freed.

The frozen ABI-v1 symbol list lives beside the header. CI compiles and runs the
C consumer fixture on the host, compiles and links it for arm64 iPhoneOS, and
compares the header, static library, and symbol manifest exactly.

## Validation

```bash
cargo test -p ruvector-apple-core
cargo clippy -p ruvector-apple-core --all-targets -- -D warnings
cargo build -p ruvector-apple-core --release --target aarch64-apple-ios
cargo build -p ruvector-apple-core --release --target aarch64-apple-ios-sim
```

Binary archives and crates.io publication are produced only by the protected
exact-SHA Apple release workflow after source, ABI, consumer, and reviewed
physical-device evidence gates pass.
