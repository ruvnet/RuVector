#!/bin/bash
# Quick check and publish script for router-wasm
# Run this manually when router-core v0.1.1 is confirmed published

set -e

# Resolve repo root from script location (issue #359: don't hardcode paths).
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Checking router-core v0.1.1 availability..."
if cargo search router-core 2>&1 | grep -q "router-core.*0\.1\.1"; then
    echo "✓ router-core v0.1.1 is available!"
    echo ""
    echo "Proceeding with router-wasm publication..."
    echo ""

    # Load API key
    export $(grep "^CRATES_API_KEY=" "$REPO_ROOT"/.env | xargs)

    # Login
    cargo login "$CRATES_API_KEY"

    # Publish
    cd "$REPO_ROOT"/crates/router-wasm
    cargo publish --allow-dirty

    echo ""
    echo "✓ router-wasm v0.1.1 published successfully!"
else
    echo "✗ router-core v0.1.1 not yet available on crates.io"
    echo "  Current version: $(cargo search router-core 2>&1 | grep 'router-core =' | head -1)"
    echo ""
    echo "Please wait for router-core v0.1.1 to be published first."
    exit 1
fi
