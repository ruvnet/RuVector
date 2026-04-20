# 02. Core Concepts: Ownership, Structs, and Enums

## 1. Ownership and Borrowing
This is what makes Rust unique. It ensures memory safety without a garbage collector.

### The Rules
1.  Each value in Rust has a variable that’s called its **owner**.
2.  There can only be one owner at a time.
3.  When the owner goes out of scope, the value will be dropped.

### Borrowing
You can access data without taking ownership by **borrowing** it using references (`&`).
- `&T`: Immutable reference (read-only).
- `&mut T`: Mutable reference (read-write).

**Rule**: You can have *either* one mutable reference *or* any number of immutable references.

### In Ruvector
Look at `crates/ruvector-core`. You will see functions taking `&self` or `&mut self`.
- `&self`: The method borrows the instance immutably.
- `&mut self`: The method borrows the instance mutably (to modify it).

## 2. Structs
Structs are custom data types.
```rust
struct Vector {
    id: u64,
    data: Vec<f32>,
}
```

### In Ruvector
Search for `struct` in `crates/ruvector-core`. You'll find the core data structures defining what a vector is, how the index is stored, etc.

## 3. Enums and Pattern Matching
Enums allow a value to be one of several variants.
```rust
enum Command {
    Insert(Vector),
    Delete(u64),
}
```

### Pattern Matching (`match`)
Rust's `match` is powerful.
```rust
match command {
    Command::Insert(vec) => println!("Inserting vector {}", vec.id),
    Command::Delete(id) => println!("Deleting vector {}", id),
}
```

### In Ruvector
Enums are often used for:
-   **Errors**: `enum Error { ... }`
-   **Configuration**: Different types of distance metrics (e.g., `Euclidean`, `Cosine`).

## Challenge
Find a `struct` definition in `crates/ruvector-core` and identify its fields.
Find a `match` statement and see how it handles different cases.

## Next Steps
Now that we understand how data is structured and owned, let's look at how to handle errors and define shared behavior.
Go to **[03_error_handling_and_traits.md](./03_error_handling_and_traits.md)**.
