# 03. Error Handling and Traits

## 1. Error Handling
Rust doesn't use exceptions. It uses the `Result` enum.
```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

### Handling Results
You must handle the result.
```rust
match file.open("hello.txt") {
    Ok(file) => println!("File opened!"),
    Err(e) => println!("Error: {}", e),
}
```

### The `?` Operator
This is syntactic sugar for propagating errors.
```rust
fn read_username_from_file() -> Result<String, io::Error> {
    let mut s = String::new();
    File::open("hello.txt")?.read_to_string(&mut s)?;
    Ok(s)
}
```

### In Ruvector
Ruvector uses libraries like `thiserror` and `anyhow` to simplify error handling.
-   `thiserror`: Used in libraries (like `ruvector-core`) to define custom error types.
-   `anyhow`: Used in applications (like `ruvector-cli`) for easy error reporting.

Check `crates/ruvector-core/src/error.rs` (if it exists) or look for `#[derive(Error)]`.

## 2. Traits
Traits are like interfaces in other languages. They define shared behavior.
```rust
trait Summary {
    fn summarize(&self) -> String;
}
```

### Implementing Traits
```rust
impl Summary for Vector {
    fn summarize(&self) -> String {
        format!("Vector ID: {}", self.id)
    }
}
```

### Generics
Traits are often used with Generics to write flexible code.
```rust
fn notify<T: Summary>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}
```

### In Ruvector
Traits are everywhere.
-   **Storage**: A trait might define how vectors are stored (e.g., in-memory vs. disk).
-   **Distance**: A trait might define how to calculate distance between two vectors.

## Challenge
Find a trait definition in the codebase (look for `trait`).
Find where that trait is implemented (`impl TraitName for TypeName`).

## Next Steps
Now that we can handle errors and abstract behavior, let's look at doing things in parallel.
Go to **[04_async_and_concurrency.md](./04_async_and_concurrency.md)**.
