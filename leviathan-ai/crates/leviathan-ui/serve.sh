#!/bin/bash
set -e

# Build first
echo "🔨 Building WASM..."
./build.sh

echo ""
echo "🌐 Starting local server on http://localhost:8080"
echo "   Press Ctrl+C to stop"
echo ""

# Try different servers in order of preference
if command -v python3 &> /dev/null; then
    python3 -m http.server -d web 8080
elif command -v python &> /dev/null; then
    python -m http.server -d web 8080
elif command -v npx &> /dev/null; then
    cd web && npx serve -p 8080
else
    echo "❌ No suitable HTTP server found."
    echo "   Please install Python 3 or Node.js"
    exit 1
fi
