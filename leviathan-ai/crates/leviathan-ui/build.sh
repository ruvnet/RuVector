#!/bin/bash
set -e

echo "🔨 Building Leviathan UI for WASM..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build for web
echo "📦 Building WASM module..."
wasm-pack build --target web --out-dir web/pkg

echo "✅ Build complete!"
echo "🌐 To serve locally, run:"
echo "   python3 -m http.server -d web 8080"
echo "   Then open http://localhost:8080"
