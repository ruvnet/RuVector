# Quick Start: RuVector MCP Server

This guide shows you how to quickly test the RuVector MCP server.

## Usage

## Option 1: Using the Simple Test Script

The simplest way to test the MCP server:

```bash
# 1. Build the MCP server (first time only)
cargo build --release -p ruvector-cli --bin ruvector-mcp

# 2. Run the test
node tests/mcp-simple-test.js
```

**Expected Output:**
```
🧪 RuVector MCP Server Simple Test
======================================================================

🚀 Starting MCP server...
✅ MCP server started

📦 Test 1: Creating vector database
   Path: /tmp/ruvector-test.db
   Dimensions: 128

✅ Database created
   {"status":"ok","path":"/tmp/ruvector-test.db"}

----------------------------------------------------------------------

📝 Test 2: Inserting sample vectors

   Inserting: vec-1 (Sample A)
   Inserting: vec-2 (Sample B)
   Inserting: vec-3 (Sample C)

✅ All vectors inserted

----------------------------------------------------------------------

🔍 Test 3: Semantic search

   Searching for similar vectors...

   Results:

   1. vec-1
      Score: 0.9876
      Metadata: {"label":"Sample A","type":"test"}

   2. vec-2
      Score: 0.9234
      Metadata: {"label":"Sample B","type":"test"}

   3. vec-3
      Score: 0.8765
      Metadata: {"label":"Sample C","type":"demo"}

======================================================================

✅ All tests passed!
```

## Option 2: Using the Full Demo

For a more comprehensive demonstration:

```bash
node tests/mcp-demo.js
```

This will:
- Create a vector database
- Insert 5 sample documents with realistic metadata
- Perform 3 semantic search queries
- Demonstrate batch operations
- Show performance metrics

## Option 3: Manual Testing

Test individual operations manually:

```bash
# Start the MCP server
./target/release/ruvector-mcp

# In another terminal, send JSON-RPC commands:
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | ./target/release/ruvector-mcp
```

## What Each Test Does

### Simple Test (`mcp-simple-test.js`)
- ✅ Creates a 128-dimensional vector database
- ✅ Inserts 3 test vectors
- ✅ Performs semantic search
- ✅ Shows database statistics
- ⏱️ Runtime: ~2-5 seconds

### Full Demo (`mcp-demo.js`)
- ✅ Creates a 384-dimensional vector database
- ✅ Inserts 5 documents with rich metadata
- ✅ Performs 3 different semantic searches
- ✅ Batch inserts 10 more vectors
- ✅ Shows performance metrics
- ✅ Demonstrates real-world use cases
- ⏱️ Runtime: ~5-10 seconds

## Troubleshooting

### "MCP server binary not found"

Build it first:
```bash
cargo build --release -p ruvector-cli --bin ruvector-mcp
```

Or use debug build (faster to compile):
```bash
cargo build -p ruvector-cli --bin ruvector-mcp
MCP_SERVER_PATH=./target/debug/ruvector-mcp node tests/mcp-simple-test.js
```

### "Cannot find module '@modelcontextprotocol/sdk'"

Install dependencies:
```bash
npm install
```

Or just install the SDK:
```bash
npm install @modelcontextprotocol/sdk
```

### Server hangs or times out

1. Check if the server starts:
```bash
./target/release/ruvector-mcp --help
```

2. Test manual JSON-RPC:
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize"}' | ./target/release/ruvector-mcp
```

3. Check server logs (enable debug mode):
```bash
DEBUG=1 node tests/mcp-simple-test.js
```

## Next Steps

After successfully running the tests:

1. **Integrate with Claude Code**
   ```bash
   claude mcp add ruvector -- ./target/release/ruvector-mcp
   ```

2. **Use Real Embeddings**
   - Replace `generateEmbedding()` with actual embedding models
   - Try: sentence-transformers, OpenAI embeddings, Cohere

3. **Scale Up**
   - Test with larger vector dimensions (768, 1536)
   - Insert thousands of vectors
   - Benchmark search performance

4. **Explore GNN Tools**
   - Try graph-based queries
   - Use the GNN layer caching
   - Experiment with graph neural network operations

## Examples

### Semantic Document Search

```javascript
// Insert documents
await client.callTool('vector_db_insert', {
  db_path: './docs.db',
  vectors: [{
    id: 'doc-1',
    vector: await embedText('Machine learning tutorial'),
    metadata: {
      title: 'ML Tutorial',
      author: 'John Doe',
      tags: ['ml', 'tutorial']
    }
  }]
});

// Search
const results = await client.callTool('vector_db_search', {
  db_path: './docs.db',
  query: await embedText('AI learning guide'),
  k: 5
});
```

### Image Similarity

```javascript
// Insert image embeddings
await client.callTool('vector_db_insert', {
  db_path: './images.db',
  vectors: [{
    id: 'img-1',
    vector: await embedImage('cat.jpg'),
    metadata: {
      filename: 'cat.jpg',
      category: 'animals'
    }
  }]
});

// Find similar images
const similar = await client.callTool('vector_db_search', {
  db_path: './images.db',
  query: await embedImage('kitten.jpg'),
  k: 10
});
```

## Learn More

- [MCP Tests Documentation](./MCP_TESTS.md)
- [RuVector README](../README.md)
- [MCP Specification](https://github.com/modelcontextprotocol/specification)
