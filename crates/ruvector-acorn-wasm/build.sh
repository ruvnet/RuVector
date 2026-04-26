#!/bin/bash
set -e

# Clear any host-only linker flags (the workspace dev shell may export
# `-fuse-ld=mold` for fast native builds; rust-lld for wasm32 rejects
# that flag).
unset RUSTFLAGS

echo "Building RuVector ACORN WASM..."

# Build for web (default — emits at root of npm/packages/acorn-wasm)
echo "Building for web target..."
wasm-pack build --target web --out-dir ../../npm/packages/acorn-wasm

# Build for Node.js
echo "Building for Node.js target..."
wasm-pack build --target nodejs --out-dir ../../npm/packages/acorn-wasm/node

# Build for bundlers (webpack, rollup, vite)
echo "Building for bundler target..."
wasm-pack build --target bundler --out-dir ../../npm/packages/acorn-wasm/bundler

echo "Build complete!"
echo "Web:     npm/packages/acorn-wasm/"
echo "Node.js: npm/packages/acorn-wasm/node/"
echo "Bundler: npm/packages/acorn-wasm/bundler/"

# wasm-pack regenerates `package.json` from `Cargo.toml` metadata, but we
# need the scoped name `@ruvector/acorn-wasm` and a richer description /
# keyword set. Keep the canonical package.json under git as
# `package.scoped.json` and copy it over after the build.
if [ -f ../../npm/packages/acorn-wasm/package.scoped.json ]; then
  cp ../../npm/packages/acorn-wasm/package.scoped.json \
     ../../npm/packages/acorn-wasm/package.json
  echo "(restored scoped package.json from package.scoped.json)"
fi
