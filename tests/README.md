# RuVector MCP Server Tests

This directory contains comprehensive test scripts and documentation for the RuVector MCP (Model Context Protocol) server.

## Quick Start

```bash
# 1. Run the test suite (no build required)
node tests/run-tests.js

# 2. Build the MCP server
cargo build --release -p ruvector-cli --bin ruvector-mcp

# 3. Run the simple test
node tests/mcp-simple-test.js

# 4. Run the full demo
node tests/mcp-demo.js
```

## Test Scripts

### 🧪 Test Runners

| Script | Description | Dependencies | Runtime |
|--------|-------------|--------------|---------|
| `run-tests.js` | Comprehensive test suite runner | None | ~1s |
| `validate-tests.js` | Test validation and verification | None | ~1s |

### 🎯 MCP Demos

| Script | Description | Dependencies | Runtime |
|--------|-------------|--------------|---------|
| `mcp-simple-test.js` | Lightweight JSON-RPC test | None | 2-5s |
| `mcp-demo.js` | Full-featured MCP demo | @modelcontextprotocol/sdk | 5-10s |

### 📚 Documentation

| File | Description |
|------|-------------|
| `QUICKSTART.md` | Quick start guide with examples |
| `MCP_TESTS.md` | Comprehensive test documentation |
| `README.md` | This file |

## Features

### Vector Operations Tested
- ✅ Database creation with configurable dimensions
- ✅ Vector insertion/upsert with metadata
- ✅ Semantic similarity search
- ✅ Database statistics
- ✅ Batch operations

### Distance Metrics Supported
- `cosine` - Cosine similarity (recommended)
- `euclidean` - L2 distance
- `dotproduct` - Dot product similarity
- `manhattan` - L1 distance

### MCP Tools Demonstrated
1. `vector_db_create` - Create vector database
2. `vector_db_insert` - Insert/upsert vectors
3. `vector_db_search` - Semantic search
4. `vector_db_stats` - Database statistics
5. `vector_db_backup` - Database backup

## Test Results

```bash
$ node tests/run-tests.js

======================================================================
🧪 RuVector MCP Test Suite
======================================================================

Test 1: File Structure Validation
  ✅ All test files present and valid

Test 2: Module Loading
  ✅ Modules load correctly
  ✅ Embedding generation works
  ✅ Embeddings are normalized

Test 3: Dependencies Check
  ⚠️  Optional dependencies noted

Test 4: MCP Server Binary Check
  ⚠️  Build with: cargo build --release -p ruvector-cli --bin ruvector-mcp

Test 5: Documentation Quality
  ✅ Complete documentation

Test 6: Script Permissions
  ✅ All scripts executable

======================================================================
📊 Test Summary
======================================================================
  ✅ Passed:   13/13 (100%)
  ⚠️  Warnings: 2 (optional)
```

## Architecture

```
┌─────────────────────┐
│  Test Scripts       │
│  (Node.js)          │
├─────────────────────┤
│  • run-tests.js     │  ← Comprehensive test suite
│  • validate-tests.js│  ← Validation only
│  • mcp-simple-test  │  ← JSON-RPC direct
│  • mcp-demo.js      │  ← MCP SDK client
└──────────┬──────────┘
           │ JSON-RPC/stdio
┌──────────▼──────────┐
│   MCP Server        │
│   (Rust binary)     │
│   ruvector-mcp      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   RuVector Core     │
│   (Rust crates)     │
└─────────────────────┘
```

## Usage Examples

### Example 1: Simple Test

```bash
# Test without building MCP server (validation only)
node tests/run-tests.js

# Build and test
cargo build --release -p ruvector-cli --bin ruvector-mcp
node tests/mcp-simple-test.js
```

### Example 2: Full Demo

```bash
# Install optional MCP SDK
npm install @modelcontextprotocol/sdk

# Run full demo
node tests/mcp-demo.js
```

### Example 3: Custom Test

```javascript
const { SimpleMCPClient, generateEmbedding } = require('./tests/mcp-simple-test.js');

async function customTest() {
  const client = new SimpleMCPClient({
    cmd: './target/release/ruvector-mcp',
    args: []
  });

  await client.start();

  // Create database
  await client.callTool('vector_db_create', {
    path: './custom.db',
    dimensions: 256,
    distance_metric: 'cosine'
  });

  // Your custom operations...

  await client.close();
}
```

## Integration

### With Claude Code

```bash
# Add to Claude Code MCP servers
claude mcp add ruvector -- ./target/release/ruvector-mcp
```

### With Other Tools

The MCP server can be used with any MCP-compatible client:

```javascript
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

const transport = new StdioClientTransport({
  command: './target/release/ruvector-mcp'
});

const client = new Client({ name: 'my-client', version: '1.0.0' }, {});
await client.connect(transport);
```

## Troubleshooting

### Server not found
```bash
# Build the server
cargo build --release -p ruvector-cli --bin ruvector-mcp

# Or use debug build (faster to compile)
cargo build -p ruvector-cli --bin ruvector-mcp
MCP_SERVER_PATH=./target/debug/ruvector-mcp node tests/mcp-simple-test.js
```

### Dependencies missing
```bash
# For full demo only
npm install @modelcontextprotocol/sdk

# Simple test has no dependencies
node tests/mcp-simple-test.js
```

### Permission denied
```bash
chmod +x tests/*.js
```

### Tests timeout
```bash
# Check if server starts
./target/release/ruvector-mcp --help

# Test JSON-RPC manually
echo '{"jsonrpc":"2.0","id":1,"method":"initialize"}' | ./target/release/ruvector-mcp
```

## Performance Tips

1. **Use batch operations** - Insert multiple vectors at once
2. **Choose right dimensions** - Common: 128, 384, 768, 1536
3. **Normalize vectors** - For cosine similarity
4. **Use release build** - Much faster than debug
5. **Enable GNN caching** - ~250-500x speedup for graph ops

## Next Steps

1. ✅ Run validation: `node tests/run-tests.js`
2. ✅ Build server: `cargo build --release -p ruvector-cli --bin ruvector-mcp`
3. ✅ Run tests: `node tests/mcp-simple-test.js`
4. 🎯 Integrate with real embeddings
5. 📈 Benchmark with your data
6. 🚀 Deploy to production

## Learn More

- [MCP Specification](https://github.com/modelcontextprotocol/specification)
- [RuVector Documentation](../README.md)
- [MCP Server Source](../crates/ruvector-cli/src/mcp_server.rs)
- [MCP Handlers](../crates/ruvector-cli/src/mcp/handlers.rs)

## Contributing

To add new tests:

1. Create test script in `tests/`
2. Add to `run-tests.js`
3. Update documentation
4. Ensure 100% pass rate

## License

MIT - See [../LICENSE](../LICENSE)
