#!/bin/bash
# Check if musica compiles to WASM
set -e
echo "Checking WASM compilation..."
# Try to build with wasm target (may need rustup target add)
rustup target add wasm32-unknown-unknown 2>/dev/null || true
cargo build --manifest-path docs/examples/musica/Cargo.toml --target wasm32-unknown-unknown --features wasm --release 2>&1
echo "WASM build: SUCCESS"
echo "Binary size: $(ls -lh target/wasm32-unknown-unknown/release/*.wasm 2>/dev/null | awk '{print $5}')"
