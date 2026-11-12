# RuVector Rust tutorial

![Animated tutorial: install, write, reopen, verify](../../assets/ruvector/tutorial-steps.svg)

[All examples](../README.md) · [Core crate](../../crates/ruvector-core/README.md) · [API reference](../../docs/api/RUST_API.md)

Use the `ruvector-core` crate for embedded vector search. You need a Rust toolchain and platform build tools.

## Create and run an application

Create a small application with the persistent storage feature. This example uses exact search and disables the default optional features.

```bash
cargo new ruvector-memory
cd ruvector-memory
cargo add ruvector-core --no-default-features --features storage
```

Replace `src/main.rs` with:

```rust
use ruvector_core::{DbOptions, SearchQuery, VectorDB, VectorEntry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = VectorDB::new(DbOptions {
        dimensions: 3,
        storage_path: "./agent-memory.db".into(),
        hnsw_config: None,
        quantization: None,
        ..Default::default()
    })?;

    db.insert(VectorEntry {
        id: Some("example-1".into()),
        vector: vec![1.0, 0.0, 0.0],
        metadata: None,
    })?;

    let hits = db.search(SearchQuery {
        vector: vec![1.0, 0.0, 0.0],
        k: 1,
        filter: None,
        ef_search: None,
    })?;

    assert_eq!(hits[0].id, "example-1");
    println!("Nearest memory: {} (score: {})", hits[0].id, hits[0].score);
    Ok(())
}
```

```bash
cargo run --release
```

The fixed vector demonstrates insertion and retrieval; supply embeddings for semantic search. Keep `Cargo.lock` for reproducible dependency resolution. [Rust API reference](../../docs/api/RUST_API.md) · [Current core types](../../crates/ruvector-core/src/types.rs) · [Storage and search implementation](../../crates/ruvector-core/src/vector_db.rs).



## Verify persistence

After the first successful run, remove only the `db.insert(...)?;` statement and run `cargo run --release` again from the same directory. The existing assertion should still find `example-1`. This checks recovery without reinserting the record.

## Explore the wider stack

| Goal | Guide |
| :--- | :--- |
| Graph relationships | [Graph examples](../graph/README.md) |
| Graph partitioning | [MinCut examples](../mincut/README.md) |
| Contrastive training | [Training guide](../../crates/ruvllm/src/training/README.md) |
| Memory reconstruction | [MRAgent](../mragent/README.md) |
| Portable vector artifacts | [RVF examples](../rvf/README.md) |

Repository example files can target different crates and feature sets. Use each owning crate's Cargo configuration rather than assuming every file is a root workspace `--example` target.
