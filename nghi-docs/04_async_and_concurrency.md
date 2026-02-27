# 04. Async and Concurrency

## 1. Async/Await
Rust's async model is cooperative. Futures are lazy; they do nothing until polled.

```rust
async fn hello_world() {
    println!("Hello, world!");
}

// In main
// block_on(hello_world());
```

### Tokio
Rust needs a runtime to execute async code. `ruvector` uses `tokio`.
Look at `Cargo.toml` dependencies. You'll see `tokio`.

### In Ruvector
The server component (`crates/ruvector-server`) heavily relies on async to handle multiple connections efficiently.
Handlers for API requests are likely `async fn`.

## 2. Shared State and Concurrency
When multiple threads (or async tasks) need to access the same data, we need synchronization.

### `Arc` (Atomic Reference Counting)
Allows multiple owners of the same data across threads.

### `Mutex` (Mutual Exclusion)
Allows only one thread to access the data at a time.

### The Pattern: `Arc<Mutex<T>>`
This is a common pattern to share mutable state.
```rust
let counter = Arc::new(Mutex::new(0));
let c = Arc::clone(&counter);
```

### In Ruvector
The vector index itself needs to be accessed by multiple readers (searches) and writers (inserts).
You might see `RwLock` (Read-Write Lock) used instead of `Mutex` to allow multiple concurrent readers.

## Challenge
Search for `Arc<` or `RwLock<` in the codebase.
See how the main index is shared between the API server and the background workers.

## Next Steps
We've covered the core Rust features. Now let's see how Rust interacts with the outside world.
Go to **[05_advanced_systems.md](./05_advanced_systems.md)**.
