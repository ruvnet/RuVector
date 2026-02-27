# Rust Learning Roadmap for Ruvector

Welcome to your journey of learning Rust through the `ruvector` codebase! This series of documents is designed to take you from a Rust beginner to understanding the complex systems within this repository.

## The Path

1.  **[01_setup_and_basics.md](./01_setup_and_basics.md)**
    *   **Goal**: Get your environment ready and understand the basic syntax.
    *   **Topics**: Installation, Cargo, Variables, Functions, Basic Types.
    *   **Ruvector Context**: Building the project, looking at `ruvector-cli`.

2.  **[02_core_concepts.md](./02_core_concepts.md)**
    *   **Goal**: Master the unique features of Rust.
    *   **Topics**: Ownership, Borrowing, Structs, Enums, Pattern Matching.
    *   **Ruvector Context**: Data structures in `ruvector-core`.

3.  **[03_error_handling_and_traits.md](./03_error_handling_and_traits.md)**
    *   **Goal**: Write robust and reusable code.
    *   **Topics**: `Result`, `Option`, Traits, Generics, `thiserror`, `anyhow`.
    *   **Ruvector Context**: Error definitions and trait usage across crates.

4.  **[04_async_and_concurrency.md](./04_async_and_concurrency.md)**
    *   **Goal**: Understand modern asynchronous programming.
    *   **Topics**: `async`/`await`, Tokio runtime, Shared State (`Arc`, `Mutex`).
    *   **Ruvector Context**: The server implementation and concurrent vector operations.

5.  **[05_advanced_systems.md](./05_advanced_systems.md)**
    *   **Goal**: Interface with other languages and systems.
    *   **Topics**: FFI (Node.js bindings), WASM.
    *   **Ruvector Context**: `ruvector-node`, `ruvector-wasm`.

6.  **[06_ruvector_architecture.md](./06_ruvector_architecture.md)**
    *   **Goal**: Put it all together.
    *   **Topics**: Workspace structure, Crate dependencies, Key data flows.

## Prerequisites

*   A curiosity to learn!
*   Basic programming knowledge in another language (like JavaScript, Python, or C++) is helpful but not strictly required.

Let's get started! Go to **[01_setup_and_basics.md](./01_setup_and_basics.md)**.
