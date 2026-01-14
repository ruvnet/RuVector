# RuVector Project Integration Playbook

**Last Updated**: 2026-01-14
**Version**: 0.1.31

---

## Quick Start by Integration Method

### Integration Decision Matrix

| Method | Best For | Latency | Setup Time |
|--------|----------|---------|------------|
| **npm/Node.js** | Backend services, APIs | <1ms | 2 min |
| **WASM/Browser** | Client-side search, offline apps | <5ms | 5 min |
| **Rust Native** | High-performance, embedded | <0.5ms | 10 min |
| **REST API** | Language-agnostic, microservices | 5-20ms | 15 min |
| **PostgreSQL** | Existing Postgres, pgvector replacement | <2ms | 20 min |
| **MCP (Claude)** | AI agents, Claude Code | N/A | 5 min |

---

## 1. Node.js Integration

### Installation

```bash
npm install ruvector
# or
yarn add ruvector
```

### Basic Usage

```javascript
const { VectorDB } = require('ruvector');

// Create database
const db = new VectorDB({
    dimensions: 384,
    storagePath: './vectors.db',
    distanceMetric: 'cosine'
});

// Insert vectors
const id = await db.insert({
    vector: new Float32Array(384).fill(0.1),
    metadata: { text: 'Example document' }
});

// Search
const results = await db.search({
    vector: queryEmbedding,
    k: 10
});

results.forEach((r, i) => {
    console.log(`${i+1}. ID: ${r.id}, Score: ${r.distance}`);
});
```

### With Graph Queries (Cypher)

```javascript
const { GraphDB } = require('@ruvector/graph-node');

const graph = new GraphDB();

// Create nodes
await graph.execute(`
    CREATE (a:Person {name: 'Alice', embedding: $emb1})
    CREATE (b:Person {name: 'Bob', embedding: $emb2})
    CREATE (a)-[:KNOWS]->(b)
`, { emb1: embedding1, emb2: embedding2 });

// Query
const friends = await graph.execute(`
    MATCH (p:Person)-[:KNOWS]->(friend)
    WHERE vector.similarity(p.embedding, $query) > 0.8
    RETURN friend.name
`, { query: queryEmbedding });
```

### With GNN Enhancement

```javascript
const { GNNLayer } = require('@ruvector/gnn');

// Create GNN layer
const layer = new GNNLayer(384, 256, 4);  // input, hidden, heads

// Forward pass enhances search results
const enhanced = layer.forward(query, neighbors, weights);

// Train from feedback
layer.train({
    queries: trainingQueries,
    positives: relevantResults,
    negatives: irrelevantResults
});
```

### With AI Routing (Tiny Dancer)

```javascript
const { Router } = require('@ruvector/tiny-dancer');

const router = new Router();

// Add routes
router.addRoute('technical', ['How do I...', 'What is the error...']);
router.addRoute('billing', ['Invoice', 'Payment', 'Subscription']);
router.addRoute('general', ['Hello', 'Thanks']);

// Route a query
const decision = router.route('How do I reset my password?');
console.log(`Route: ${decision.route}, Confidence: ${decision.confidence}`);
```

---

## 2. WASM/Browser Integration

### Installation

```bash
npm install ruvector-wasm
# or include from CDN
```

### Basic Usage (ES Modules)

```javascript
import init, { VectorDB } from 'ruvector-wasm';

async function main() {
    await init();

    const db = new VectorDB(384);  // dimensions

    // Insert
    const id = db.insert(new Float32Array([0.1, 0.2, ...]));

    // Search
    const results = db.search(queryVector, 10);

    // Results: [{ id, distance }, ...]
    console.log(results);
}
```

### With IndexedDB Persistence

```javascript
import init, { VectorDB } from 'ruvector-wasm';

async function main() {
    await init();

    const db = new VectorDB(384);

    // Load from IndexedDB
    const stored = localStorage.getItem('ruvector-state');
    if (stored) {
        db.loadFromJson(stored);
    }

    // ... use db ...

    // Save to IndexedDB
    localStorage.setItem('ruvector-state', db.toJson());
}
```

### React Integration

```jsx
import { useEffect, useState } from 'react';
import init, { VectorDB } from 'ruvector-wasm';

function useVectorDB(dimensions) {
    const [db, setDb] = useState(null);

    useEffect(() => {
        let mounted = true;
        init().then(() => {
            if (mounted) {
                setDb(new VectorDB(dimensions));
            }
        });
        return () => { mounted = false; };
    }, [dimensions]);

    return db;
}

function SearchComponent() {
    const db = useVectorDB(384);
    const [results, setResults] = useState([]);

    const handleSearch = async (query) => {
        if (!db) return;
        const embedding = await getEmbedding(query);
        setResults(db.search(embedding, 10));
    };

    return (
        <div>
            <input onChange={(e) => handleSearch(e.target.value)} />
            {results.map((r) => <div key={r.id}>{r.distance}</div>)}
        </div>
    );
}
```

### Web Worker for Large Datasets

```javascript
// worker.js
import init, { VectorDB } from 'ruvector-wasm';

let db = null;

self.onmessage = async (e) => {
    const { type, payload } = e.data;

    if (type === 'init') {
        await init();
        db = new VectorDB(payload.dimensions);
        self.postMessage({ type: 'ready' });
    }

    if (type === 'search' && db) {
        const results = db.search(payload.vector, payload.k);
        self.postMessage({ type: 'results', payload: results });
    }
};
```

---

## 3. Rust Native Integration

### Cargo.toml

```toml
[dependencies]
ruvector-core = "0.1.31"
ruvector-graph = "0.1.31"  # Optional: Cypher queries
ruvector-gnn = "0.1.31"    # Optional: GNN layers
```

### Basic Usage

```rust
use ruvector_core::{VectorDB, VectorEntry, SearchQuery, DbOptions};

fn main() -> anyhow::Result<()> {
    // Create database
    let options = DbOptions {
        dimensions: 384,
        storage_path: "./vectors.db".into(),
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let db = VectorDB::new(options)?;

    // Insert
    let entry = VectorEntry {
        id: None,
        vector: vec![0.1; 384],
        metadata: Some(serde_json::json!({"text": "example"})),
    };
    let id = db.insert(entry)?;

    // Search
    let query = SearchQuery {
        vector: vec![0.1; 384],
        k: 10,
        filter: None,
        include_vectors: false,
    };
    let results = db.search(&query)?;

    for result in results {
        println!("ID: {}, Distance: {}", result.id, result.distance);
    }

    Ok(())
}
```

### With Graph Queries

```rust
use ruvector_graph::{GraphDB, NodeBuilder};

let db = GraphDB::new();

// Create node
let doc = NodeBuilder::new("doc1")
    .label("Document")
    .property("embedding", vec![0.1, 0.2, 0.3])
    .property("text", "Hello world")
    .build();
db.create_node(doc)?;

// Cypher query
let results = db.execute_cypher(r#"
    MATCH (d:Document)
    WHERE d.text CONTAINS 'Hello'
    RETURN d
"#)?;
```

### With Distributed Cluster

```rust
use ruvector_raft::{RaftNode, RaftNodeConfig};
use ruvector_cluster::{ClusterManager, ConsistentHashRing};

// Configure Raft cluster
let config = RaftNodeConfig {
    node_id: "node-1".into(),
    cluster_members: vec!["node-1", "node-2", "node-3"]
        .into_iter().map(Into::into).collect(),
    election_timeout_min: 150,
    election_timeout_max: 300,
    heartbeat_interval: 50,
};
let raft = RaftNode::new(config);

// Auto-sharding
let ring = ConsistentHashRing::new(64, 3);  // 64 shards, RF=3
let shard = ring.get_shard("my-key");
```

---

## 4. REST API Server

### Start Server

```bash
# Install CLI
cargo install ruvector-cli

# Start server
ruvector server start --port 8080 --db ./vectors.db

# Or with Docker
docker run -p 8080:8080 ruvector/server:latest
```

### API Endpoints

```bash
# Insert vector
curl -X POST http://localhost:8080/vectors \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "metadata": {"text": "example"}}'

# Search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "k": 10}'

# Cypher query
curl -X POST http://localhost:8080/cypher \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n) RETURN n LIMIT 10"}'
```

### Client Libraries (Any Language)

```python
# Python
import requests

def search(query_vector, k=10):
    response = requests.post(
        "http://localhost:8080/search",
        json={"vector": query_vector, "k": k}
    )
    return response.json()
```

```go
// Go
func Search(queryVector []float32, k int) ([]Result, error) {
    body, _ := json.Marshal(map[string]interface{}{
        "vector": queryVector,
        "k":      k,
    })
    resp, err := http.Post("http://localhost:8080/search",
        "application/json", bytes.NewReader(body))
    // ...
}
```

---

## 5. PostgreSQL Extension

### Installation

```bash
# Docker (recommended)
docker run -d \
  -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 \
  ruvector/postgres:latest

# From source
cargo install cargo-pgrx --version "0.12.9" --locked
cargo pgrx init
cd crates/ruvector-postgres
cargo pgrx install --release

# Enable extension
psql -c "CREATE EXTENSION ruvector;"
```

### Basic Usage (pgvector-compatible)

```sql
-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(384)
);

-- Insert
INSERT INTO documents (content, embedding)
VALUES ('Hello world', '[0.1, 0.2, ...]');

-- Create HNSW index
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Search (top 10 nearest)
SELECT id, content, embedding <=> '[0.1, 0.2, ...]' AS distance
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;
```

### Advanced Features (77+ Functions)

```sql
-- Local embeddings (no API needed)
SELECT ruvector_embed('all-MiniLM-L6-v2', 'Hello world') AS embedding;

-- Hybrid search (vector + BM25)
SELECT * FROM ruvector_hybrid_search(
    'documents',
    'embedding',
    '[0.1, ...]',
    'content',
    'search terms',
    10,       -- k
    0.7       -- vector_weight
);

-- Graph traversal
SELECT * FROM ruvector_graph_neighbors('node_id', 2);

-- GNN enhancement
SELECT ruvector_gnn_enhance(embedding, neighbors, weights)
FROM documents;

-- Attention mechanisms
SELECT ruvector_flash_attention(query, key, value);

-- SPARQL queries
SELECT ruvector_sparql('SELECT ?s WHERE { ?s rdf:type ex:Doc }');
```

---

## 6. MCP Integration (Claude Code)

### Setup

```bash
# Add MCP server
claude mcp add ruvector npx ruvector mcp start

# Or with ruv-swarm for enhanced features
claude mcp add ruv-swarm npx ruv-swarm mcp start
```

### Available MCP Tools

```javascript
// Swarm coordination
mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 6 }
mcp__ruv-swarm__agent_spawn { type: "researcher" }
mcp__ruv-swarm__task_orchestrate { task: "analyze code", strategy: "parallel" }

// Memory
mcp__ruv-swarm__memory_usage { action: "store", key: "context", value: "..." }

// Neural features
mcp__ruv-swarm__neural_train { patterns: [...] }
mcp__ruv-swarm__neural_patterns { query: "..." }
```

### Self-Learning Hooks

```bash
# Initialize hooks in project
npx @ruvector/cli hooks init
npx @ruvector/cli hooks install

# Hooks fire automatically on:
# - PreToolUse: Get agent routing suggestions
# - PostToolUse: Record outcomes for learning
# - SessionStart/End: Manage session state
```

---

## Common Task Examples

### Semantic Search with Filtering

```javascript
// Node.js
const results = await db.search({
    vector: queryEmbedding,
    k: 10,
    filter: {
        $and: [
            { category: { $eq: 'tech' } },
            { date: { $gte: '2024-01-01' } }
        ]
    }
});
```

```rust
// Rust
let query = SearchQuery {
    vector: query_embedding,
    k: 10,
    filter: Some(Filter::and(vec![
        Filter::eq("category", "tech"),
        Filter::gte("date", "2024-01-01"),
    ])),
    include_vectors: false,
};
```

### RAG Pipeline

```javascript
// Node.js
const { VectorDB } = require('ruvector');
const { OpenAI } = require('openai');

async function rag(question) {
    // 1. Embed question
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: question
    });

    // 2. Search for context
    const context = await db.search({
        vector: embedding.data[0].embedding,
        k: 5
    });

    // 3. Generate answer
    const response = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [{
            role: 'user',
            content: `Context: ${context.map(c => c.metadata.text).join('\n')}\n\nQuestion: ${question}`
        }]
    });

    return response.choices[0].message.content;
}
```

### Recommendation System

```javascript
// Node.js with Cypher
const recommendations = await graph.execute(`
    MATCH (user:User {id: $userId})-[:PURCHASED]->(item:Product)
    MATCH (item)-[:SIMILAR_TO]->(rec:Product)
    WHERE NOT (user)-[:PURCHASED]->(rec)
    RETURN rec
    ORDER BY vector.similarity(item.embedding, rec.embedding) DESC
    LIMIT 10
`, { userId: 'user123' });
```

---

## Performance Tuning

### HNSW Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `m` | 32 | 16-64 | More = higher recall, more memory |
| `ef_construction` | 200 | 100-400 | Higher = better index quality |
| `ef_search` | 100 | 50-500 | Higher = better recall, slower |

```javascript
// Tune for speed
const db = new VectorDB({
    dimensions: 384,
    hnswM: 16,
    hnswEfConstruction: 100,
    hnswEfSearch: 50
});

// Tune for accuracy
const db = new VectorDB({
    dimensions: 384,
    hnswM: 64,
    hnswEfConstruction: 400,
    hnswEfSearch: 200
});
```

### Quantization

```javascript
// Enable compression for memory savings
const db = new VectorDB({
    dimensions: 384,
    quantization: 'scalar',  // 4x compression
    // quantization: 'pq8',  // 8x compression
    // quantization: 'binary', // 32x compression
});
```

### Batch Operations

```javascript
// Instead of individual inserts
for (const vec of vectors) {
    await db.insert(vec);  // Slow
}

// Use batch insert
await db.insertBatch(vectors);  // 10-100x faster
```

### Native CPU Features

```bash
# Build with SIMD optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Check enabled features
cargo rustc -- --print cfg | grep target_feature
```

---

## Deployment Considerations

### Memory Estimation

```
Memory = vectors * dimensions * bytes_per_element + index_overhead

Examples (1M vectors, 384 dim):
- f32: 1M * 384 * 4 = 1.5GB + ~500MB overhead = ~2GB
- int8: 1M * 384 * 1 = 384MB + ~500MB overhead = ~900MB
- PQ8: 1M * 48 * 1 = 48MB + ~500MB overhead = ~550MB
```

### Scaling Guidelines

| Vectors | Recommended |
|---------|-------------|
| <100K | Single node, in-memory |
| 100K-10M | Single node, mmap |
| 10M-100M | Cluster (3-5 nodes) |
| >100M | Sharded cluster |

### Production Checklist

- [ ] Enable native CPU features (`-C target-cpu=native`)
- [ ] Configure appropriate quantization
- [ ] Set up memory-mapped storage for large datasets
- [ ] Configure HNSW parameters for your recall/speed tradeoff
- [ ] Enable monitoring (Prometheus metrics available)
- [ ] Set up backup strategy (snapshots)
- [ ] Plan for index rebuild time

---

## Quick Reference Commands

```bash
# CLI
ruvector --help                      # Show help
ruvector create --path ./db --dim 384  # Create database
ruvector insert --db ./db --input data.json  # Insert vectors
ruvector search --db ./db --query "[...]" --top-k 10  # Search
ruvector info --db ./db              # Show database info
ruvector bench --db ./db             # Run benchmarks
ruvector server start --port 8080    # Start REST server

# npm
npx ruvector                         # Interactive CLI
npx ruvector hooks init              # Initialize learning hooks
npx ruvector hooks install           # Install into Claude settings

# PostgreSQL
psql -c "CREATE EXTENSION ruvector;" # Enable extension
psql -c "SELECT ruvector_version();" # Check version
```

---

*For detailed institutional knowledge, see [PLAYBOOK-INSTITUTIONAL.md](./PLAYBOOK-INSTITUTIONAL.md)*
