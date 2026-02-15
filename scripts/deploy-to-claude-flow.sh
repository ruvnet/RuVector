#!/usr/bin/env bash
# Deploy RVF META_SEG patches to claude-flow-local
# Re-applies node_modules patches after npm install or manual setup
#
# Usage: ./scripts/deploy-to-claude-flow.sh [claude-flow-local-path]

set -euo pipefail

RUVECTOR_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CF_LOCAL="${1:-/Users/danielalberttis/Desktop/Projects/claude-flow-local}"

echo "=== RVF META_SEG Deployment ==="
echo "Source:  $RUVECTOR_DIR"
echo "Target:  $CF_LOCAL"
echo ""

# 1. Symlink native binary (MUST be symlink, not copy — macOS SIGKILL on copied .node)
BINARY_SRC="$RUVECTOR_DIR/node_modules/@ruvector/rvf-node/rvf-node.darwin-arm64.node"
BINARY_DST="$CF_LOCAL/node_modules/@ruvector/rvf-node/rvf-node.darwin-arm64.node"

if [ ! -f "$BINARY_SRC" ]; then
    echo "ERROR: Source binary not found: $BINARY_SRC"
    echo "Build it first: cd crates/rvf/rvf-node && cargo build --release"
    exit 1
fi

mkdir -p "$(dirname "$BINARY_DST")"
rm -f "$BINARY_DST"
ln -s "$BINARY_SRC" "$BINARY_DST"
echo "[1/4] Native binary symlinked"

# 2. Patch @ruvector/rvf/dist/database.js — add putMeta, getMeta, deleteMeta, listMetaKeys
DB_JS="$CF_LOCAL/node_modules/@ruvector/rvf/dist/database.js"
if [ -f "$DB_JS" ] && ! grep -q "putMeta" "$DB_JS"; then
    # Insert META_SEG methods before ensureOpen()
    sed -i '' '/ensureOpen() {/i\
    // --- META_SEG KV methods (Phase 5) ---\
    async putMeta(key, value) {\
        this.ensureOpen();\
        return this.backend.putMeta(key, value);\
    }\
    async getMeta(key) {\
        this.ensureOpen();\
        return this.backend.getMeta(key);\
    }\
    async deleteMeta(key) {\
        this.ensureOpen();\
        return this.backend.deleteMeta(key);\
    }\
    async listMetaKeys() {\
        this.ensureOpen();\
        return this.backend.listMetaKeys();\
    }' "$DB_JS"
    echo "[2/4] database.js patched"
else
    echo "[2/4] database.js already patched or not found"
fi

# 3. Patch @ruvector/rvf/dist/backend.js — add NodeBackend + WasmBackend META_SEG methods
BE_JS="$CF_LOCAL/node_modules/@ruvector/rvf/dist/backend.js"
if [ -f "$BE_JS" ] && ! grep -q "putMeta" "$BE_JS"; then
    # Add NodeBackend methods before the closing } of NodeBackend class
    # We insert before 'exports.NodeBackend'
    sed -i '' '/^exports.NodeBackend/i\
    // --- META_SEG KV methods (Phase 5) ---\
    async putMeta(key, value) {\
        this.ensureHandle();\
        try { const buf = Buffer.isBuffer(value) ? value : Buffer.from(value); this.handle.putMeta(key, buf); }\
        catch (err) { throw errors_1.RvfError.fromNative(err); }\
    }\
    async getMeta(key) {\
        this.ensureHandle();\
        try { const result = this.handle.getMeta(key); return result ? Buffer.from(result) : null; }\
        catch (err) { throw errors_1.RvfError.fromNative(err); }\
    }\
    async deleteMeta(key) {\
        this.ensureHandle();\
        try { return this.handle.deleteMeta(key); }\
        catch (err) { throw errors_1.RvfError.fromNative(err); }\
    }\
    async listMetaKeys() {\
        this.ensureHandle();\
        try { return this.handle.listMetaKeys(); }\
        catch (err) { throw errors_1.RvfError.fromNative(err); }\
    }\
}' "$BE_JS"

    # Add WasmBackend stubs before 'exports.WasmBackend'
    sed -i '' '/^exports.WasmBackend/i\
    async putMeta() { throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, "putMeta not supported in WASM backend"); }\
    async getMeta() { throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, "getMeta not supported in WASM backend"); }\
    async deleteMeta() { throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, "deleteMeta not supported in WASM backend"); }\
    async listMetaKeys() { throw new errors_1.RvfError(errors_1.RvfErrorCode.BackendNotFound, "listMetaKeys not supported in WASM backend"); }\
}' "$BE_JS"
    echo "[3/4] backend.js patched"
else
    echo "[3/4] backend.js already patched or not found"
fi

# 4. Patch memory-initializer.js — META_SEG loading, putMeta on ingest, skip JSON sidecar
MI_JS="$CF_LOCAL/node_modules/@claude-flow/cli/dist/src/memory/memory-initializer.js"
if [ -f "$MI_JS" ] && ! grep -q "META_SEG" "$MI_JS"; then
    echo "[4/4] WARNING: memory-initializer.js needs manual patching"
    echo "       See MEMORY.md for the 3 required edits"
    echo "       (too complex for sed — use Edit tool in Claude Code)"
else
    echo "[4/4] memory-initializer.js already patched or not found"
fi

echo ""
echo "=== Deployment complete ==="
echo "Restart claude-flow MCP to pick up changes."
