# RuVector MCP Server Test Scripts

This directory contains test scripts demonstrating how to use the RuVector MCP (Model Context Protocol) server.

## Test Scripts

### 1. `mcp-demo.js` - Full Featured Demo

A comprehensive demonstration of the MCP server capabilities using the official MCP SDK.

**Features:**
- Creates a vector database with configurable dimensions
- Inserts sample document vectors with metadata
- Performs semantic search queries
- Demonstrates batch operations
- Shows database statistics

**Prerequisites:**
```bash
# Install dependencies (if not already installed)
npm install

# Build the MCP server binary
cargo build --release -p ruvector-cli --bin ruvector-mcp
```

**Usage:**
```bash
node tests/mcp-demo.js
```

### 2. `mcp-simple-test.js` - Simple JSON-RPC Test

A lightweight test script using direct JSON-RPC communication without requiring the full SDK.

**Features:**
- Minimal dependencies
- Direct JSON-RPC over stdio
- Simple vector operations (create, insert, search)
- Database statistics

**Usage:**
```bash
node tests/mcp-simple-test.js
```

**Custom server path:**
```bash
MCP_SERVER_PATH=/path/to/ruvector-mcp node tests/mcp-simple-test.js
```

## Available MCP Tools

The RuVector MCP server provides the following tools:

### Vector Database Operations
- `vector_db_create` - Create a new vector database
- `vector_db_insert` - Insert/upsert vectors with metadata
- `vector_db_search` - Perform semantic similarity search
- `vector_db_stats` - Get database statistics
- `vector_db_backup` - Backup database to file

### GNN (Graph Neural Network) Operations
- `gnn_layer_create` - Create/cache GNN layers
- `gnn_forward` - Forward pass through GNN layer
- `gnn_batch_forward` - Batch GNN operations
- `gnn_cache_stats` - Get GNN cache statistics
- `gnn_compress` - Compress embeddings
- `gnn_decompress` - Decompress embeddings

## Example: Creating a Collection and Searching

```javascript
const { SimpleMCPClient, generateEmbedding } = require('./mcp-simple-test.js');

async function example() {
  const client = new SimpleMCPClient({
    cmd: './target/release/ruvector-mcp',
    args: []
  });

  await client.start();

  // Create database
  await client.callTool('vector_db_create', {
    path: './my-vectors.db',
    dimensions: 128,
    distance_metric: 'cosine'
  });

  // Insert vectors
  await client.callTool('vector_db_insert', {
    db_path: './my-vectors.db',
    vectors: [{
      id: 'doc-1',
      vector: generateEmbedding(128, 1),
      metadata: { title: 'My Document' }
    }]
  });

  // Search
  const results = await client.callTool('vector_db_search', {
    db_path: './my-vectors.db',
    query: generateEmbedding(128, 1.1),
    k: 5
  });

  console.log(JSON.parse(results.content[0].text));

  await client.close();
}
```

## Distance Metrics

Supported distance metrics:
- `cosine` - Cosine similarity (recommended for normalized vectors)
- `euclidean` - Euclidean distance (L2)
- `dotproduct` - Dot product similarity
- `manhattan` - Manhattan distance (L1)

## Integration with Claude Code

To use the MCP server with Claude Code:

```bash
# Add to Claude Code MCP configuration
claude mcp add ruvector -- /path/to/ruvector-mcp
```

Then Claude can use the vector database tools directly.

## Performance Tips

1. **Batch Operations**: Use batch insert for better performance
2. **Dimensions**: Choose dimensions based on your embedding model (common: 128, 384, 768, 1536)
3. **Distance Metric**: Use `cosine` for normalized embeddings, `euclidean` for unnormalized
4. **GNN Caching**: The server automatically caches GNN layers for ~250-500x speedup

## Troubleshooting

### Server binary not found
```bash
# Build the release version
cargo build --release -p ruvector-cli --bin ruvector-mcp

# Or use debug version (slower)
cargo build -p ruvector-cli --bin ruvector-mcp
MCP_SERVER_PATH=./target/debug/ruvector-mcp node tests/mcp-simple-test.js
```

### Dependencies missing
```bash
# Install Node.js dependencies
npm install

# Or install just the MCP SDK
npm install @modelcontextprotocol/sdk
```

### Permission denied
```bash
# Make the test script executable
chmod +x tests/mcp-demo.js
chmod +x tests/mcp-simple-test.js
```

## Architecture

```
┌─────────────────┐
│  Test Script    │
│  (Node.js)      │
└────────┬────────┘
         │ JSON-RPC
         │ over stdio
┌────────▼────────┐
│   MCP Server    │
│  (Rust binary)  │
└────────┬────────┘
         │
┌────────▼────────┐
│   RuVector DB   │
│  (Rust crate)   │
└─────────────────┘
```

## Further Reading

- [Model Context Protocol Specification](https://github.com/modelcontextprotocol/specification)
- [RuVector Documentation](../README.md)
- [MCP Server Implementation](../crates/ruvector-cli/src/mcp_server.rs)
- [MCP Handler Reference](../crates/ruvector-cli/src/mcp/handlers.rs)
