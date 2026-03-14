#!/bin/bash
set -e

echo "=== rvAgent Test Suite ==="

echo "Building all crates..."
cargo build -p rvagent-core -p rvagent-backends -p rvagent-middleware -p rvagent-tools -p rvagent-subagents -p rvagent-cli -p rvagent-acp

echo "Running tests..."
cargo test -p rvagent-core
cargo test -p rvagent-backends
cargo test -p rvagent-middleware
cargo test -p rvagent-tools
cargo test -p rvagent-subagents
cargo test -p rvagent-cli
cargo test -p rvagent-acp

echo "Running benchmarks (dry run)..."
cargo bench -p rvagent-core --bench state_bench -- --test

echo "=== All tests passed ==="
