# 01. Setup and Basics

## 1. Setting Up Your Environment

Before we dive into code, let's ensure you have Rust installed.

### Install Rust
Run the following in your terminal:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Follow the on-screen instructions. After installation, restart your terminal and check:
```bash
rustc --version
cargo --version
```

### Clone and Build Ruvector
You are already in the `ruvector` repo. Let's make sure it builds.
```bash
# In the root of the repo
cargo build
```
This might take a while as it compiles all dependencies.

## 2. Rust Basics with Ruvector

Rust uses `cargo` as its build system and package manager.

### The `Cargo.toml` File
Open `Cargo.toml` in the root. This is the workspace definition.
- `[workspace]`: Defines that this repo contains multiple crates.
- `members`: Lists all the crates (e.g., `crates/ruvector-core`, `crates/ruvector-cli`).

### Variables and Mutability
In Rust, variables are immutable by default.
```rust
let x = 5;
// x = 6; // This would cause a compile error!
```
To make them mutable, use `mut`:
```rust
let mut y = 5;
y = 6; // This is okay
```

### Functions
Functions are declared with `fn`.
```rust
fn add(a: i32, b: i32) -> i32 {
    a + b // No semicolon means this is the return value
}
```

### Looking at `ruvector-cli`
Let's look at a real example. Open `crates/ruvector-cli/src/main.rs` (or similar entry point).
You'll likely see a `main` function.
```rust
fn main() {
    // ... code ...
}
```
This is the entry point of the CLI application. It likely parses arguments using `clap` (a popular CLI argument parser library).

## Challenge
1.  Navigate to `crates/ruvector-cli`.
2.  Try to run it: `cargo run -- --help`.
3.  See what commands are available.

## Next Steps
Now that you can build the code and understand the basic syntax, let's dive into the most unique feature of Rust: Ownership.
Go to **[02_core_concepts.md](./02_core_concepts.md)**.
