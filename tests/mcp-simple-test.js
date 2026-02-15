#!/usr/bin/env node

/**
 * Simple RuVector MCP Test Script
 * 
 * This is a simplified test that uses JSON-RPC directly to interact
 * with the MCP server without requiring the full SDK client setup.
 * 
 * Usage:
 *   node tests/mcp-simple-test.js
 */

const { spawn } = require('child_process');
const readline = require('readline');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Configuration
const config = {
  dbPath: path.join(os.tmpdir(), 'ruvector-test.db'),
  dimensions: 128,
  distanceMetric: 'cosine'
};

/**
 * Generate a simple mock embedding
 */
function generateEmbedding(dim, seed = 0) {
  const vector = [];
  for (let i = 0; i < dim; i++) {
    const angle = (i + seed) * Math.PI / dim;
    vector.push(Math.cos(angle) * (1 + seed * 0.1));
  }
  // Normalize
  const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
  return vector.map(v => v / magnitude);
}

/**
 * Simple MCP client using JSON-RPC over stdio
 */
class SimpleMCPClient {
  constructor(serverCommand) {
    this.serverCommand = serverCommand;
    this.process = null;
    this.requestId = 0;
    this.callbacks = new Map();
  }

  async start() {
    console.log('🚀 Starting MCP server...');
    
    this.process = spawn(this.serverCommand.cmd, this.serverCommand.args, {
      stdio: ['pipe', 'pipe', 'inherit']
    });

    this.rl = readline.createInterface({
      input: this.process.stdout,
      crlfDelay: Infinity
    });

    this.rl.on('line', (line) => {
      try {
        const response = JSON.parse(line);
        if (response.id !== undefined && this.callbacks.has(response.id)) {
          const callback = this.callbacks.get(response.id);
          this.callbacks.delete(response.id);
          
          if (response.error) {
            callback.reject(new Error(response.error.message));
          } else {
            callback.resolve(response.result);
          }
        }
      } catch (e) {
        // Ignore parse errors for non-JSON lines
      }
    });

    this.process.on('error', (err) => {
      console.error('Server process error:', err);
    });

    // Initialize
    await this.request('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: {
        name: 'simple-test',
        version: '1.0.0'
      }
    });

    console.log('✅ MCP server started\n');
  }

  async request(method, params = null) {
    const id = ++this.requestId;
    
    const request = {
      jsonrpc: '2.0',
      id,
      method,
      ...(params && { params })
    };

    return new Promise((resolve, reject) => {
      this.callbacks.set(id, { resolve, reject });
      
      const timeout = setTimeout(() => {
        if (this.callbacks.has(id)) {
          this.callbacks.delete(id);
          reject(new Error(`Request timeout: ${method}`));
        }
      }, 30000);

      this.callbacks.get(id).timeout = timeout;

      this.process.stdin.write(JSON.stringify(request) + '\n');
    });
  }

  async callTool(name, args) {
    return this.request('tools/call', {
      name,
      arguments: args
    });
  }

  async close() {
    if (this.process) {
      this.process.kill();
    }
  }
}

/**
 * Main test function
 */
async function runTest() {
  console.log('\n' + '='.repeat(70));
  console.log('🧪 RuVector MCP Server Simple Test');
  console.log('='.repeat(70) + '\n');

  // Determine server path
  const mcpServerPath = process.env.MCP_SERVER_PATH || 
                        path.join(__dirname, '../target/release/ruvector-mcp');
  
  if (!fs.existsSync(mcpServerPath)) {
    console.log('ℹ️  MCP server binary not found at:', mcpServerPath);
    console.log('\n📝 To build the MCP server:');
    console.log('   cargo build --release -p ruvector-cli --bin ruvector-mcp');
    console.log('\n💡 Or set MCP_SERVER_PATH environment variable\n');
    
    // Try alternative paths
    const altPaths = [
      path.join(__dirname, '../target/debug/ruvector-mcp'),
      'ruvector-mcp' // Try in PATH
    ];
    
    let found = false;
    for (const altPath of altPaths) {
      if (fs.existsSync(altPath)) {
        console.log(`✅ Found server at: ${altPath}\n`);
        found = true;
        break;
      }
    }
    
    if (!found) {
      console.log('❌ Cannot proceed without MCP server binary\n');
      process.exit(1);
    }
  }

  // Clean up old database
  if (fs.existsSync(config.dbPath)) {
    fs.unlinkSync(config.dbPath);
  }

  const client = new SimpleMCPClient({
    cmd: mcpServerPath,
    args: []
  });

  try {
    await client.start();

    // Test 1: Create database
    console.log('📦 Test 1: Creating vector database');
    console.log(`   Path: ${config.dbPath}`);
    console.log(`   Dimensions: ${config.dimensions}\n`);

    const createResult = await client.callTool('vector_db_create', {
      path: config.dbPath,
      dimensions: config.dimensions,
      distance_metric: config.distanceMetric
    });

    console.log('✅ Database created');
    console.log(`   ${JSON.stringify(createResult)}\n`);
    console.log('-'.repeat(70) + '\n');

    // Test 2: Insert vectors
    console.log('📝 Test 2: Inserting sample vectors\n');

    const vectors = [
      {
        id: 'vec-1',
        vector: generateEmbedding(config.dimensions, 1),
        metadata: { label: 'Sample A', type: 'test' }
      },
      {
        id: 'vec-2',
        vector: generateEmbedding(config.dimensions, 2),
        metadata: { label: 'Sample B', type: 'test' }
      },
      {
        id: 'vec-3',
        vector: generateEmbedding(config.dimensions, 3),
        metadata: { label: 'Sample C', type: 'demo' }
      }
    ];

    for (const vec of vectors) {
      console.log(`   Inserting: ${vec.id} (${vec.metadata.label})`);
      await client.callTool('vector_db_insert', {
        db_path: config.dbPath,
        vectors: [vec]
      });
    }

    console.log('\n✅ All vectors inserted\n');
    console.log('-'.repeat(70) + '\n');

    // Test 3: Search
    console.log('🔍 Test 3: Semantic search\n');

    const queryVector = generateEmbedding(config.dimensions, 1.5);
    
    console.log('   Searching for similar vectors...');
    const searchResult = await client.callTool('vector_db_search', {
      db_path: config.dbPath,
      query: queryVector,
      k: 3
    });

    console.log('\n   Results:');
    const results = JSON.parse(searchResult.content[0].text);
    results.forEach((result, idx) => {
      const vec = vectors.find(v => v.id === result.id);
      console.log(`\n   ${idx + 1}. ${result.id}`);
      console.log(`      Score: ${result.score.toFixed(4)}`);
      console.log(`      Metadata: ${JSON.stringify(vec.metadata)}`);
    });

    console.log('\n\n' + '-'.repeat(70) + '\n');

    // Test 4: Statistics
    console.log('📊 Test 4: Database statistics\n');

    const statsResult = await client.callTool('vector_db_stats', {
      db_path: config.dbPath
    });

    const stats = JSON.parse(statsResult.content[0].text);
    console.log('   Database Info:');
    console.log(`   - Vectors: ${stats.count || 'N/A'}`);
    console.log(`   - Dimensions: ${stats.dimensions || config.dimensions}`);
    console.log(`   - Metric: ${stats.distance_metric || config.distanceMetric}\n`);

    console.log('='.repeat(70));
    console.log('\n✅ All tests passed!\n');

  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  } finally {
    await client.close();
  }
}

// Run tests
if (require.main === module) {
  runTest().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}

module.exports = { SimpleMCPClient, generateEmbedding };
