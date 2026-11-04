#!/bin/bash
set -e

# Clear any host-only linker flags (the workspace dev shell may export
# `-fuse-ld=mold` for fast native builds; rust-lld for wasm32 rejects
# that flag).
unset RUSTFLAGS

echo "Building RuVector RaBitQ WASM..."

# Build for web (default — emits at root of npm/packages/rabitq-wasm)
echo "Building for web target..."
wasm-pack build --target web --out-dir ../../npm/packages/rabitq-wasm

# Build for Node.js
echo "Building for Node.js target..."
wasm-pack build --target nodejs --out-dir ../../npm/packages/rabitq-wasm/node

# Build for bundlers (webpack, rollup, vite)
echo "Building for bundler target..."
wasm-pack build --target bundler --out-dir ../../npm/packages/rabitq-wasm/bundler

# wasm-pack writes a `.gitignore` containing `*` into every out-dir. In the
# nested node/ and bundler/ dirs npm honours it and silently drops them from
# the tarball (0.1.0 shipped without the node/ and bundler/ entry points the
# README documents). At the package root it clobbers the committed ignore
# file. Remove the nested ones and restore the root one.
rm -f ../../npm/packages/rabitq-wasm/node/.gitignore \
      ../../npm/packages/rabitq-wasm/bundler/.gitignore
git checkout -- ../../npm/packages/rabitq-wasm/.gitignore 2>/dev/null || true

echo "Build complete!"
echo "Web:     npm/packages/rabitq-wasm/"
echo "Node.js: npm/packages/rabitq-wasm/node/"
echo "Bundler: npm/packages/rabitq-wasm/bundler/"

# wasm-pack regenerates `package.json` from `Cargo.toml` metadata, but we
# need the scoped name `@ruvector/rabitq-wasm` and a richer description /
# keyword set. The canonical package.json + README live alongside the
# generated artifacts and are kept under git; restore them after the build
# so subsequent `wasm-pack build` runs don't clobber them.
if [ -f ../../npm/packages/rabitq-wasm/package.scoped.json ]; then
  cp ../../npm/packages/rabitq-wasm/package.scoped.json \
     ../../npm/packages/rabitq-wasm/package.json
  echo "(restored scoped package.json from package.scoped.json)"
fi
