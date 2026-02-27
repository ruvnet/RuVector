# 05. Advanced Systems: FFI and WASM

Rust is great because it can go where other languages can't, or it can speed them up.

## 1. FFI (Foreign Function Interface) with Node.js
`ruvector` has a Node.js binding (`crates/ruvector-node`).
This allows JavaScript code to call Rust functions directly.

### NAPI-RS
The project uses `napi-rs` to build these bindings.
Look for `#[napi]` macros in `crates/ruvector-node`.
These macros automatically generate the glue code needed for Node.js to understand Rust structs and functions.

## 2. WASM (WebAssembly)
Rust can compile to WebAssembly to run in the browser.
Check `crates/ruvector-wasm`.

### `wasm-bindgen`
This library facilitates communication between WASM and JavaScript.
Look for `#[wasm_bindgen]`.

## 3. Unsafe Code
Rust guarantees memory safety, but sometimes you need to bypass checks (e.g., for performance or FFI).
This is done in `unsafe` blocks.
```rust
unsafe {
    // scary raw pointer manipulation
}
```
`ruvector` likely minimizes this, but it might exist in performance-critical sections (like SIMD vector operations).

## Challenge
1.  Go to `crates/ruvector-node`.
2.  Find a function marked with `#[napi]`.
3.  Imagine how you would call this from JavaScript.

## Next Steps
You have the tools. Now let's look at the map of the castle.
Go to **[06_ruvector_architecture.md](./06_ruvector_architecture.md)**.
