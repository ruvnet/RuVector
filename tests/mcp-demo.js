#!/usr/bin/env node

/**
 * RuVector MCP Server Test Script
 *
 * This script demonstrates using the ruvector MCP server to:
 * 1. Create a vector collection (database)
 * 2. Upsert sample vectors with metadata
 * 3. Perform semantic search queries
 *
 * Prerequisites:
 * - Build the MCP server: cargo build --release -p ruvector-cli --bin ruvector-mcp
 * - Install MCP SDK: npm install @modelcontextprotocol/sdk
 * - Or run via npm: npm run mcp
 *
 * Usage:
 *   node tests/mcp-demo.js
 */

// Try to load MCP SDK, provide helpful error if not found
let Client, StdioClientTransport;
try {
  const clientModule = require('@modelcontextprotocol/sdk/client/index.js');
  const transportModule = require('@modelcontextprotocol/sdk/client/stdio.js');
  Client = clientModule.Client;
  StdioClientTransport = transportModule.StdioClientTransport;
} catch (error) {
  console.error('\n❌ Error: @modelcontextprotocol/sdk not found!\n');
  console.error('Please install it first:');
  console.error('  npm install @modelcontextprotocol/sdk\n');
  console.error('Or use the simple test script instead:');
  console.error('  node tests/mcp-simple-test.js\n');
  process.exit(1);
}

const path = require('path');
const fs = require('fs');
const os = require('os');
const { spawn } = require('child_process');

// Configuration
const config = {
  dbPath: path.join(os.tmpdir(), 'ruvector-demo.db'),
  dimensions: 384, // Common dimension for sentence embeddings
  distanceMetric: 'cosine'
};

// Sample document embeddings (simulated)
// In a real scenario, these would come from an embedding model
const sampleDocuments = [
  {
    id: 'doc-1',
    vector: generateMockEmbedding(384, 'machine learning'),
    metadata: {
      title: 'Introduction to Machine Learning',
      category: 'AI',
      author: 'John Doe',
      tags: ['ml', 'ai', 'tutorial']
    }
  },
  {
    id: 'doc-2',
    vector: generateMockEmbedding(384, 'deep learning neural networks'),
    metadata: {
      title: 'Deep Learning with Neural Networks',
      category: 'AI',
      author: 'Jane Smith',
      tags: ['deep-learning', 'neural-networks', 'ai']
    }
  },
  {
    id: 'doc-3',
    vector: generateMockEmbedding(384, 'web development javascript'),
    metadata: {
      title: 'Modern Web Development with JavaScript',
      category: 'Web',
      author: 'Bob Johnson',
      tags: ['javascript', 'web', 'frontend']
    }
  },
  {
    id: 'doc-4',
    vector: generateMockEmbedding(384, 'database design sql'),
    metadata: {
      title: 'Database Design Principles',
      category: 'Database',
      author: 'Alice Williams',
      tags: ['database', 'sql', 'design']
    }
  },
  {
    id: 'doc-5',
    vector: generateMockEmbedding(384, 'python programming'),
    metadata: {
      title: 'Python Programming Fundamentals',
      category: 'Programming',
      author: 'Charlie Brown',
      tags: ['python', 'programming', 'basics']
    }
  }
];

/**
 * Generate a mock embedding vector
 * In production, use a real embedding model (e.g., sentence-transformers, OpenAI)
 */
function generateMockEmbedding(dimensions, text) {
  // Simple hash-based embedding for demo purposes
  const seed = hashString(text);
  const random = seededRandom(seed);
  const vector = [];
  
  for (let i = 0; i < dimensions; i++) {
    vector.push(random() * 2 - 1); // Range: -1 to 1
  }
  
  // Normalize the vector (important for cosine similarity)
  const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  return vector.map(val => val / magnitude);
}

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
}

function seededRandom(seed) {
  let state = seed;
  return function() {
    state = (state * 9301 + 49297) % 233280;
    return state / 233280;
  };
}

/**
 * MCP Client wrapper for easier interaction
 */
class MCPVectorClient {
  constructor() {
    this.client = null;
    this.transport = null;
  }

  async connect(serverPath) {
    console.log('🔌 Connecting to MCP server...');
    console.log(`   Server: ${serverPath}`);
    
    // Start the MCP server process
    const serverProcess = spawn(serverPath, [], {
      stdio: ['pipe', 'pipe', 'inherit']
    });

    this.transport = new StdioClientTransport({
      command: serverPath,
      args: []
    });

    this.client = new Client(
      {
        name: 'ruvector-test-client',
        version: '1.0.0'
      },
      {
        capabilities: {}
      }
    );

    await this.client.connect(this.transport);
    console.log('✅ Connected to MCP server\n');
  }

  async listTools() {
    const response = await this.client.listTools();
    return response.tools;
  }

  async callTool(name, args) {
    const response = await this.client.callTool({ name, arguments: args });
    return response;
  }

  async close() {
    if (this.client) {
      await this.client.close();
    }
    if (this.transport) {
      await this.transport.close();
    }
  }
}

/**
 * Main test script
 */
async function main() {
  console.log('🚀 RuVector MCP Server Demo\n');
  console.log('='.repeat(70));
  console.log('\nThis demo will:');
  console.log('  1. Create a vector database');
  console.log('  2. Insert sample document vectors');
  console.log('  3. Perform semantic searches');
  console.log('  4. Display database statistics\n');
  console.log('='.repeat(70) + '\n');

  // Clean up old database if exists
  if (fs.existsSync(config.dbPath)) {
    console.log(`🧹 Cleaning up old database: ${config.dbPath}\n`);
    fs.unlinkSync(config.dbPath);
  }

  const client = new MCPVectorClient();

  try {
    // Determine server path
    const serverPath = process.env.MCP_SERVER_PATH || 
                      path.join(__dirname, '../target/release/ruvector-mcp');
    
    if (!fs.existsSync(serverPath)) {
      console.error('❌ Error: MCP server binary not found!');
      console.error(`   Expected: ${serverPath}`);
      console.error('\n   Please build it first:');
      console.error('   cargo build --release -p ruvector-cli --bin ruvector-mcp\n');
      process.exit(1);
    }

    // Connect to server
    await client.connect(serverPath);

    // List available tools
    console.log('📋 Available MCP Tools:\n');
    const tools = await client.listTools();
    tools.forEach((tool, idx) => {
      console.log(`   ${idx + 1}. ${tool.name} - ${tool.description}`);
    });
    console.log('\n' + '='.repeat(70) + '\n');

    // Step 1: Create vector database
    console.log('📦 Step 1: Creating vector database\n');
    console.log(`   Path: ${config.dbPath}`);
    console.log(`   Dimensions: ${config.dimensions}`);
    console.log(`   Distance Metric: ${config.distanceMetric}\n`);

    const createResult = await client.callTool('vector_db_create', {
      path: config.dbPath,
      dimensions: config.dimensions,
      distance_metric: config.distanceMetric
    });

    console.log('✅ Database created successfully');
    console.log(`   ${JSON.stringify(createResult.content[0].text)}\n`);
    console.log('='.repeat(70) + '\n');

    // Step 2: Insert vectors
    console.log('📝 Step 2: Inserting sample vectors\n');
    console.log(`   Number of documents: ${sampleDocuments.length}\n`);

    for (let i = 0; i < sampleDocuments.length; i++) {
      const doc = sampleDocuments[i];
      console.log(`   Inserting: ${doc.metadata.title}`);
      
      await client.callTool('vector_db_insert', {
        db_path: config.dbPath,
        vectors: [{
          id: doc.id,
          vector: doc.vector,
          metadata: doc.metadata
        }]
      });
    }

    console.log('\n✅ All vectors inserted successfully\n');
    console.log('='.repeat(70) + '\n');

    // Step 3: Get database statistics
    console.log('📊 Step 3: Database Statistics\n');

    const statsResult = await client.callTool('vector_db_stats', {
      db_path: config.dbPath
    });

    const stats = JSON.parse(statsResult.content[0].text);
    console.log('   Database Info:');
    console.log(`   - Total vectors: ${stats.count || 'N/A'}`);
    console.log(`   - Dimensions: ${stats.dimensions || config.dimensions}`);
    console.log(`   - Distance metric: ${stats.distance_metric || config.distanceMetric}\n`);
    console.log('='.repeat(70) + '\n');

    // Step 4: Semantic searches
    console.log('🔍 Step 4: Semantic Search Queries\n');

    const queries = [
      {
        name: 'AI/ML Content',
        text: 'artificial intelligence machine learning',
        k: 3
      },
      {
        name: 'Web Development',
        text: 'web development javascript',
        k: 2
      },
      {
        name: 'Programming Languages',
        text: 'python programming',
        k: 3
      }
    ];

    for (const query of queries) {
      console.log(`\n   Query: "${query.name}" (${query.text})`);
      console.log('   ' + '-'.repeat(65));

      const queryVector = generateMockEmbedding(config.dimensions, query.text);
      
      const searchResult = await client.callTool('vector_db_search', {
        db_path: config.dbPath,
        query: queryVector,
        k: query.k
      });

      const results = JSON.parse(searchResult.content[0].text);
      
      console.log(`   Results (top ${query.k}):\n`);
      
      if (results && results.length > 0) {
        results.forEach((result, idx) => {
          const doc = sampleDocuments.find(d => d.id === result.id);
          if (doc) {
            console.log(`   ${idx + 1}. ${doc.metadata.title}`);
            console.log(`      ID: ${result.id}`);
            console.log(`      Score: ${result.score.toFixed(4)}`);
            console.log(`      Category: ${doc.metadata.category}`);
            console.log(`      Tags: ${doc.metadata.tags.join(', ')}\n`);
          }
        });
      } else {
        console.log('   No results found\n');
      }
    }

    console.log('='.repeat(70) + '\n');

    // Step 5: Demonstrate batch operations
    console.log('⚡ Step 5: Batch Insert Performance\n');

    const batchSize = 10;
    const batchDocs = [];
    
    for (let i = 0; i < batchSize; i++) {
      batchDocs.push({
        id: `batch-doc-${i}`,
        vector: generateMockEmbedding(config.dimensions, `document ${i}`),
        metadata: {
          title: `Batch Document ${i}`,
          category: 'Batch',
          index: i
        }
      });
    }

    console.log(`   Batch inserting ${batchSize} vectors...`);
    const batchStart = Date.now();
    
    await client.callTool('vector_db_insert', {
      db_path: config.dbPath,
      vectors: batchDocs
    });

    const batchTime = Date.now() - batchStart;
    console.log(`   ✅ Batch insert completed in ${batchTime}ms`);
    console.log(`   ⚡ Rate: ${(batchSize / (batchTime / 1000)).toFixed(2)} vectors/sec\n`);

    // Final statistics
    const finalStatsResult = await client.callTool('vector_db_stats', {
      db_path: config.dbPath
    });
    const finalStats = JSON.parse(finalStatsResult.content[0].text);
    console.log(`   Final database size: ${finalStats.count || 'N/A'} vectors\n`);
    console.log('='.repeat(70) + '\n');

    // Success summary
    console.log('✅ Demo completed successfully!\n');
    console.log('📝 Summary:');
    console.log(`   • Created vector database with ${config.dimensions} dimensions`);
    console.log(`   • Inserted ${sampleDocuments.length + batchSize} vectors total`);
    console.log(`   • Performed ${queries.length} semantic search queries`);
    console.log(`   • Demonstrated batch operations\n`);
    console.log('💡 Next steps:');
    console.log('   • Try the GNN tools for graph-based queries');
    console.log('   • Experiment with different distance metrics');
    console.log('   • Use real embeddings from models like sentence-transformers');
    console.log('   • Explore metadata filtering in searches\n');

  } catch (error) {
    console.error('\n❌ Error during demo:', error.message);
    if (error.stack) {
      console.error('\nStack trace:');
      console.error(error.stack);
    }
    process.exit(1);
  } finally {
    // Clean up
    await client.close();
    console.log('👋 Connection closed\n');
  }
}

// Run the demo
if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

module.exports = { MCPVectorClient, generateMockEmbedding };
