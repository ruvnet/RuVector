# RuVector-Postgres

[![Crates.io](https://img.shields.io/crates/v/ruvector-postgres.svg)](https://crates.io/crates/ruvector-postgres)
[![Documentation](https://docs.rs/ruvector-postgres/badge.svg)](https://docs.rs/ruvector-postgres)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14--17-blue.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-available-blue.svg)](https://hub.docker.com/r/ruvnet/ruvector-postgres)
[![npm](https://img.shields.io/npm/v/@ruvector/core.svg)](https://www.npmjs.com/package/@ruvector/core)
[![Security](https://img.shields.io/badge/Security-Audited-green.svg)](docs/SECURITY_AUDIT_REPORT.md)

**The most advanced PostgreSQL vector database extension.** A drop-in pgvector replacement with **230+ SQL functions**, SIMD acceleration, 39 attention mechanisms, GNN layers, hyperbolic embeddings, mincut-gated transformers, hybrid search, multi-tenancy, self-healing, and self-learning capabilities.

## v2.0.0 Highlights (December 2025)

- **Mincut-Gated Transformers**: Ultra-low-latency inference with coherence control via λ signals
- **Hybrid Search**: Vector + BM25 fusion with RRF and linear blending
- **Multi-Tenancy**: Row-level security with automatic tenant isolation
- **Self-Healing**: Automated index repair with integrity validation
- **Integrity Control**: Stoer-Wagner mincut-based quality assurance
- **IVFFlat Index**: Full inverted list storage with proper page management
- **HNSW Index**: Fixed query execution with heap scan integration
- **Security Audit**: 3 critical SQL injection vulnerabilities fixed
- **GNN Module**: Complete Graph Neural Network operators

## Why RuVector?

| Feature | pgvector | RuVector-Postgres |
|---------|----------|-------------------|
| Vector Search | HNSW, IVFFlat | HNSW, IVFFlat (optimized) |
| Distance Metrics | 3 | 8+ (including hyperbolic) |
| **Local Embeddings** | - | **6 models (fastembed)** |
| **Attention Mechanisms** | - | **39 types** |
| **Gated Transformers** | - | **Mincut-coherence control** |
| **Hybrid Search** | - | **RRF + Linear fusion** |
| **Graph Neural Networks** | - | **GCN, GraphSAGE, GAT** |
| **Hyperbolic Embeddings** | - | **Poincare, Lorentz** |
| **Sparse Vectors / BM25** | Partial | **Full support** |
| **Multi-Tenancy** | - | **Row-level isolation** |
| **Self-Healing** | - | **Auto index repair** |
| **Integrity Control** | - | **Stoer-Wagner mincut** |
| **Self-Learning** | - | **ReasoningBank** |
| **Agent Routing** | - | **Tiny Dancer** |
| **Graph/Cypher** | - | **Full support** |
| **SPARQL/RDF** | - | **W3C SPARQL 1.1** |
| AVX-512/NEON SIMD | Partial | **Full** |
| Quantization | No | **Scalar, Product, Binary** |

## Installation

### Docker (Recommended)

```bash
# Start the container
docker run -d --name ruvector-pg \
  -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 \
  ruvnet/ruvector-postgres:latest

# Connect with psql
PGPASSWORD=secret psql -h localhost -p 5432 -U postgres

# Or use the ruvector app user (created automatically)
PGPASSWORD=ruvector psql -h localhost -p 5432 -U ruvector -d postgres
```

The container initializes with:
- Extension `ruvector` pre-installed and tested
- User `ruvector` with password `ruvector` for application use
- SIMD acceleration (AVX2/AVX-512) auto-detected

### npm (Node.js Bindings)

```bash
# Install the core package with native bindings
npm install @ruvector/core

# Or install the full ruvector package
npm install ruvector
```

```javascript
const { VectorDB, cosineDistance } = require('@ruvector/core');

// Create a vector database
const db = new VectorDB({ dimensions: 384 });

// Add vectors
db.add([0.1, 0.2, 0.3, ...]);

// Search
const results = db.search(queryVector, { k: 10 });
```

### From Source

```bash
# Install pgrx
cargo install cargo-pgrx --version "0.12.9" --locked
cargo pgrx init --pg16 $(which pg_config)

# Build and install
cd crates/ruvector-postgres
cargo pgrx install --release
```

### CLI Tool

```bash
npm install -g @ruvector/postgres-cli
ruvector-pg -c "postgresql://localhost:5432/mydb" install
```

## Quick Start

```sql
-- Create the extension
CREATE EXTENSION ruvector;

-- Create a table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding ruvector(1536)
);

-- Create an HNSW index
CREATE INDEX ON documents USING ruhnsw (embedding ruvector_l2_ops);

-- Find similar documents
SELECT content, embedding <-> '[0.15, 0.25, ...]'::ruvector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

## 230+ SQL Functions

RuVector exposes all advanced AI capabilities as native PostgreSQL functions.

### Core Vector Operations

```sql
-- Distance metrics
SELECT ruvector_cosine_distance(a, b);
SELECT ruvector_l2_distance(a, b);
SELECT ruvector_inner_product(a, b);
SELECT ruvector_manhattan_distance(a, b);

-- Vector operations
SELECT ruvector_normalize(embedding);
SELECT ruvector_add(a, b);
SELECT ruvector_scalar_mul(embedding, 2.0);
```

### Hyperbolic Geometry (8 functions)

Perfect for hierarchical data like taxonomies, knowledge graphs, and org charts.

```sql
-- Poincare ball model
SELECT ruvector_poincare_distance(a, b, -1.0);  -- curvature -1

-- Lorentz hyperboloid model
SELECT ruvector_lorentz_distance(a, b, -1.0);

-- Hyperbolic operations
SELECT ruvector_mobius_add(a, b, -1.0);       -- Hyperbolic translation
SELECT ruvector_exp_map(base, tangent, -1.0); -- Tangent to manifold
SELECT ruvector_log_map(base, target, -1.0);  -- Manifold to tangent

-- Model conversion
SELECT ruvector_poincare_to_lorentz(poincare_vec, -1.0);
SELECT ruvector_lorentz_to_poincare(lorentz_vec, -1.0);

-- Minkowski inner product
SELECT ruvector_minkowski_dot(a, b);
```

### Sparse Vectors & BM25 (14 functions)

Full sparse vector support with text scoring.

```sql
-- Create sparse vectors
SELECT ruvector_sparse_create(ARRAY[0, 5, 10], ARRAY[0.5, 0.3, 0.2], 100);
SELECT ruvector_sparse_from_dense(dense_vector, 0.01);  -- threshold

-- Sparse operations
SELECT ruvector_sparse_dot(a, b);
SELECT ruvector_sparse_cosine(a, b);
SELECT ruvector_sparse_l2_distance(a, b);
SELECT ruvector_sparse_add(a, b);
SELECT ruvector_sparse_scale(vec, 2.0);
SELECT ruvector_sparse_normalize(vec);
SELECT ruvector_sparse_topk(vec, 10);  -- Top-k elements

-- Text scoring
SELECT ruvector_bm25_score(query_terms, doc_freqs, doc_len, avg_doc_len, total_docs);
SELECT ruvector_tf_idf(term_freq, doc_freq, total_docs);
```

### 39 Attention Mechanisms

Full transformer-style attention in PostgreSQL.

```sql
-- Scaled dot-product attention
SELECT ruvector_attention_scaled_dot(query, keys, values);

-- Multi-head attention
SELECT ruvector_attention_multi_head(query, keys, values, num_heads);

-- Flash attention (memory efficient)
SELECT ruvector_attention_flash(query, keys, values, block_size);

-- Sparse attention patterns
SELECT ruvector_attention_sparse(query, keys, values, sparsity_pattern);

-- Linear attention (O(n) complexity)
SELECT ruvector_attention_linear(query, keys, values);

-- Causal/masked attention
SELECT ruvector_attention_causal(query, keys, values);

-- Cross attention
SELECT ruvector_attention_cross(query, context_keys, context_values);

-- Self attention
SELECT ruvector_attention_self(input, num_heads);
```

### Graph Neural Networks (5 functions)

GNN layers for graph-structured data.

```sql
-- GCN (Graph Convolutional Network)
SELECT ruvector_gnn_gcn_layer(features, adjacency, weights);

-- GraphSAGE (inductive learning)
SELECT ruvector_gnn_graphsage_layer(features, neighbor_features, weights);

-- GAT (Graph Attention Network)
SELECT ruvector_gnn_gat_layer(features, adjacency, attention_weights);

-- Message passing
SELECT ruvector_gnn_message_pass(node_features, edge_index, edge_weights);

-- Aggregation
SELECT ruvector_gnn_aggregate(messages, aggregation_type);  -- mean, max, sum
```

### Agent Routing - Tiny Dancer (11 functions)

Intelligent query routing to specialized AI agents.

```sql
-- Route query to best agent
SELECT ruvector_route_query(query_embedding, agent_registry);

-- Route with context
SELECT ruvector_route_with_context(query, context, agents);

-- Multi-agent routing
SELECT ruvector_multi_agent_route(query, agents, top_k);

-- Agent management
SELECT ruvector_register_agent(name, capabilities, embedding);
SELECT ruvector_update_agent_performance(agent_id, metrics);
SELECT ruvector_get_routing_stats();

-- Affinity calculation
SELECT ruvector_calculate_agent_affinity(query, agent);
SELECT ruvector_select_best_agent(query, agent_list);

-- Adaptive routing
SELECT ruvector_adaptive_route(query, context, learning_rate);

-- FastGRNN acceleration
SELECT ruvector_fastgrnn_forward(input, hidden, weights);
```

### Local Embeddings (6 functions)

Generate embeddings directly in PostgreSQL - no external API calls needed.

```sql
-- Generate embedding from text (default: all-MiniLM-L6-v2)
SELECT ruvector_embed('Hello, world!');

-- Use specific model
SELECT ruvector_embed('Hello, world!', 'bge-small-en-v1.5');

-- Batch embedding (efficient for multiple texts)
SELECT ruvector_embed_batch(ARRAY['First doc', 'Second doc', 'Third doc']);

-- List available models
SELECT ruvector_list_models();

-- Get model information (dimensions, description)
SELECT ruvector_model_info('all-MiniLM-L6-v2');

-- Preload model into cache for faster subsequent calls
SELECT ruvector_preload_model('bge-base-en-v1.5');
```

**Supported Models:**

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast, general-purpose (default) |
| `bge-small-en-v1.5` | 384 | MTEB #1, English |
| `bge-base-en-v1.5` | 768 | Higher accuracy, English |
| `bge-large-en-v1.5` | 1024 | Highest accuracy, English |
| `nomic-embed-text-v1` | 768 | Long context (8192 tokens) |
| `nomic-embed-text-v1.5` | 768 | Updated long context |

**Example: Automatic Embedding on Insert**

```sql
-- Create table with trigger for auto-embedding
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding ruvector(384)
);

-- Insert with automatic embedding generation
INSERT INTO articles (title, content, embedding)
VALUES (
    'Introduction to AI',
    'Artificial intelligence is transforming...',
    ruvector_embed('Artificial intelligence is transforming...')
);

-- Semantic search
SELECT title, embedding <=> ruvector_embed('machine learning basics') AS distance
FROM articles
ORDER BY distance
LIMIT 5;
```

### Self-Learning / ReasoningBank (7 functions)

Adaptive search parameter optimization.

```sql
-- Record learning trajectory
SELECT ruvector_record_trajectory(input, output, success, context);

-- Get verdict on approach
SELECT ruvector_get_verdict(trajectory_id);

-- Memory distillation
SELECT ruvector_distill_memory(trajectories, compression_ratio);

-- Adaptive search
SELECT ruvector_adaptive_search(query, context, ef_search);

-- Learning feedback
SELECT ruvector_learning_feedback(search_id, relevance_scores);

-- Get learned patterns
SELECT ruvector_get_learning_patterns(context);

-- Optimize search parameters
SELECT ruvector_optimize_search_params(query_type, historical_data);
```

### Graph Storage & Cypher (8 functions)

Graph operations with Cypher query support.

```sql
-- Create graph elements
SELECT ruvector_graph_create_node(labels, properties, embedding);
SELECT ruvector_graph_create_edge(from_node, to_node, edge_type, properties);

-- Graph queries
SELECT ruvector_graph_get_neighbors(node_id, edge_type, depth);
SELECT ruvector_graph_shortest_path(start_node, end_node);
SELECT ruvector_graph_pagerank(edge_table, damping, iterations);

-- Cypher queries
SELECT ruvector_cypher_query('MATCH (n:Person)-[:KNOWS]->(m) RETURN n, m');

-- Traversal
SELECT ruvector_graph_traverse(start_node, direction, max_depth);

-- Similarity search on graph
SELECT ruvector_graph_similarity_search(query_embedding, node_type, top_k);
```

### SPARQL & RDF (14 functions)

W3C-standard SPARQL 1.1 query language for RDF data.

```sql
-- Create RDF triple store
SELECT ruvector_create_rdf_store('knowledge_graph');

-- Insert triples
SELECT ruvector_insert_triple(
    'knowledge_graph',
    '<http://example.org/person/1>',
    '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>',
    '<http://example.org/Person>'
);

-- Bulk load N-Triples
SELECT ruvector_load_ntriples('knowledge_graph', '
    <http://example.org/person/1> <http://xmlns.com/foaf/0.1/name> "Alice" .
    <http://example.org/person/1> <http://xmlns.com/foaf/0.1/knows> <http://example.org/person/2> .
');

-- SPARQL SELECT query
SELECT ruvector_sparql('knowledge_graph', '
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?person ?name
    WHERE {
        ?person a <http://example.org/Person> .
        ?person foaf:name ?name .
    }
', 'json');

-- SPARQL ASK query
SELECT ruvector_sparql('knowledge_graph',
    'ASK { <http://example.org/person/1> ?p ?o }',
    'json'
);

-- Get store statistics
SELECT ruvector_rdf_stats('knowledge_graph');

-- Query triples by pattern (NULL = wildcard)
SELECT ruvector_query_triples('knowledge_graph',
    NULL, -- any subject
    '<http://xmlns.com/foaf/0.1/name>', -- predicate
    NULL  -- any object
);

-- SPARQL UPDATE operations
SELECT ruvector_sparql_update('knowledge_graph', '
    INSERT DATA {
        <http://example.org/person/3> <http://xmlns.com/foaf/0.1/name> "Charlie" .
    }
');
```

**SPARQL Features:**
- SELECT, CONSTRUCT, ASK, DESCRIBE query forms
- Property paths (sequence `/`, alternative `|`, inverse `^`, transitive `*`, `+`)
- FILTER expressions with 50+ built-in functions
- Aggregates (COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT)
- OPTIONAL, UNION, MINUS graph patterns
- Named graphs support
- Result formats: JSON, XML, CSV, TSV
- **~198K triples/sec** insertion, **~5.5M queries/sec** lookups

### Gated Transformers (13 functions)

Ultra-low-latency transformer inference with mincut-gated coherence control.

```sql
-- Get gate decision from integrity mincut signals
SELECT gated_transformer_gate_decision(
    lambda := 150,        -- Current mincut value
    lambda_prev := 160,   -- Previous mincut
    boundary_count := 5,  -- Witness edge count
    layer := 3            -- Current transformer layer
);
-- Returns: {"decision": "Allow", "reason": "None", "tier": 3, ...}

-- Check early exit conditions
SELECT gated_transformer_early_exit_check(
    lambda := 180,
    layer := 8,
    total_layers := 12
);
-- Returns: {"can_exit": true, "confidence": 0.92, "exit_layer": 8, ...}

-- Mixture-of-Depths token routing (50% FLOPs reduction)
SELECT gated_transformer_route_tokens(
    lambda := 150,
    token_count := 512,
    layer_capacity := 0.5  -- Route only 50% of tokens through compute
);
-- Returns: [{"index": 0, "route": "Compute"}, {"index": 1, "route": "Skip"}, ...]

-- Configuration management
SELECT gated_transformer_config();  -- Get current config
SELECT gated_transformer_set_config(
    lambda_min := 50,
    lambda_critical := 20,
    check_interval := 64
);

-- Policy management
SELECT gated_transformer_gate_policy();  -- Get current policy
SELECT gated_transformer_set_policy(
    enable_tiering := true,
    enable_kv_flush := true,
    enable_freeze := false
);

-- Bridge with integrity module
SELECT gated_transformer_from_integrity('my_hnsw_index');

-- Get combined coherence score
SELECT gated_transformer_coherence_score(
    lambda := 150,
    lambda_prev := 160,
    boundary_count := 5
);
-- Returns: 0.875 (normalized 0-1 coherence)
```

**Gated Transformer Features:**
- **Dynamic Compute Allocation**: Mixture-of-Depths routes tokens for 50% FLOPs reduction
- **Early Exit**: Layer-skipping with 30-50% latency reduction when coherence is high
- **Tiered Decisions**: 5 tiers from Full→Reduced→Conservative→Minimal→Critical
- **KV-Cache Management**: Automatic flush/freeze based on coherence signals
- **Boundary Detection**: Witness edge tracking for structural integrity

### Hybrid Search (7 functions)

Vector + keyword fusion with multiple ranking strategies.

```sql
-- Linear fusion (alpha blending)
SELECT ruvector_hybrid_linear(
    vector_results,   -- Array of (id, score) from vector search
    keyword_results,  -- Array of (id, score) from BM25
    alpha := 0.7      -- 0.7 vector weight, 0.3 keyword weight
);

-- Reciprocal Rank Fusion (RRF)
SELECT ruvector_hybrid_rrf(
    vector_results,
    keyword_results,
    k := 60  -- RRF constant
);

-- Combined search with auto-fusion
SELECT ruvector_hybrid_search(
    query_text := 'machine learning optimization',
    query_embedding := $embedding,
    table_name := 'documents',
    text_column := 'content',
    vector_column := 'embedding',
    limit_k := 10
);

-- Get/Set hybrid search parameters
SELECT ruvector_get_hybrid_alpha();  -- Returns current alpha
SELECT ruvector_set_hybrid_alpha(0.6);
SELECT ruvector_get_hybrid_rrf_k();
SELECT ruvector_set_hybrid_rrf_k(40);
```

### Multi-Tenancy (17 functions)

Row-level security with automatic tenant isolation.

```sql
-- Set current tenant context
SELECT ruvector_set_tenant('tenant_123');
SELECT ruvector_get_tenant();

-- Create tenant-isolated table
SELECT ruvector_create_tenant_table(
    'documents',
    'id SERIAL PRIMARY KEY, content TEXT, embedding ruvector(384)'
);

-- Automatic tenant filtering (via RLS policies)
INSERT INTO documents (content, embedding)
VALUES ('Hello', '[0.1, 0.2, ...]'::ruvector);
-- Automatically tagged with tenant_id

-- Query only sees current tenant's data
SELECT * FROM documents
WHERE embedding <-> $query < 0.5;

-- Tenant management
SELECT ruvector_list_tenants();
SELECT ruvector_tenant_stats('tenant_123');
SELECT ruvector_migrate_tenant('old_tenant', 'new_tenant');

-- Cross-tenant queries (admin only)
SELECT ruvector_admin_query_all_tenants('documents', 'SELECT count(*) FROM documents');
```

### Self-Healing (23 functions)

Automated index repair with integrity validation.

```sql
-- Check index health
SELECT ruvector_index_health('documents_embedding_idx');
-- Returns: {"status": "healthy", "fragmentation": 0.05, "orphaned_nodes": 0}

-- Automatic repair
SELECT ruvector_auto_repair('documents_embedding_idx');

-- Schedule maintenance
SELECT ruvector_schedule_maintenance(
    'documents_embedding_idx',
    interval := '1 day',
    repair_threshold := 0.1  -- Repair if fragmentation > 10%
);

-- Self-healing operations
SELECT ruvector_compact_index('documents_embedding_idx');
SELECT ruvector_rebalance_hnsw('documents_embedding_idx');
SELECT ruvector_rebuild_ivf_centroids('documents_embedding_idx');
SELECT ruvector_validate_graph_connectivity('documents_embedding_idx');

-- Monitor healing status
SELECT ruvector_healing_status();
SELECT ruvector_last_repair_log('documents_embedding_idx');

-- Integrity checks
SELECT ruvector_check_orphaned_vectors('documents');
SELECT ruvector_check_duplicate_vectors('documents', threshold := 0.001);
```

### Integrity Control (4 functions)

Stoer-Wagner mincut-based quality assurance for vector indices.

```sql
-- Get integrity status
SELECT ruvector_integrity_status();
-- Returns: {"enabled": true, "active_contracts": 1, "contracts": ["default"]}

-- Create integrity contract (SLA)
SELECT ruvector_integrity_create_contract(
    id := 'production_sla',
    name := 'Production SLA',
    min_recall := 0.95,        -- Minimum recall requirement
    max_latency_ms := 100,     -- Maximum query latency
    min_mincut := 0.1          -- Minimum graph connectivity
);

-- Validate against contract
SELECT ruvector_integrity_validate(
    'production_sla',
    recall := 0.97,
    latency_ms := 45,
    mincut := 0.15
);
-- Returns: {"passed": true, "recall": 0.97, "latency_ms": 45, "mincut": 0.15, "failures": []}

-- Compute mincut for graph connectivity
SELECT ruvector_mincut(
    n := 100,  -- Number of nodes
    edges_json := '[{"u": 0, "v": 1, "w": 1.0}, ...]'::jsonb
);
-- Returns minimum cut value (Stoer-Wagner algorithm)
```

## Vector Types

### `ruvector(n)` - Dense Vector

```sql
CREATE TABLE items (embedding ruvector(1536));
-- Storage: 8 + (4 x dimensions) bytes
```

### `halfvec(n)` - Half-Precision Vector

```sql
CREATE TABLE items (embedding halfvec(1536));
-- Storage: 8 + (2 x dimensions) bytes (50% savings)
```

### `sparsevec(n)` - Sparse Vector

```sql
CREATE TABLE items (embedding sparsevec(50000));
INSERT INTO items VALUES ('{1:0.5, 100:0.8, 5000:0.3}/50000');
-- Storage: 12 + (8 x non_zero_elements) bytes
```

## Distance Operators

| Operator | Distance | Use Case |
|----------|----------|----------|
| `<->` | L2 (Euclidean) | General similarity |
| `<=>` | Cosine | Text embeddings |
| `<#>` | Inner Product | Normalized vectors |
| `<+>` | Manhattan (L1) | Sparse features |

## Index Types

### HNSW (Hierarchical Navigable Small World)

```sql
CREATE INDEX ON items USING ruhnsw (embedding ruvector_l2_ops)
WITH (m = 16, ef_construction = 64);

SET ruvector.ef_search = 100;  -- Tune search quality
```

### IVFFlat (Inverted File Flat)

```sql
CREATE INDEX ON items USING ruivfflat (embedding ruvector_l2_ops)
WITH (lists = 100);

SET ruvector.ivfflat_probes = 10;  -- Tune search quality
```

## Performance Benchmarks

*AMD EPYC 7763 (64 cores), 256GB RAM:*

| Operation | 10K vectors | 100K vectors | 1M vectors |
|-----------|-------------|--------------|------------|
| HNSW Build | 0.8s | 8.2s | 95s |
| HNSW Search (top-10) | 0.3ms | 0.5ms | 1.2ms |
| Cosine Distance | 0.01ms | 0.01ms | 0.01ms |
| Poincare Distance | 0.02ms | 0.02ms | 0.02ms |
| GCN Forward | 2.1ms | 18ms | 180ms |
| BM25 Score | 0.05ms | 0.08ms | 0.15ms |

*Single distance calculation (1536 dimensions):*

| Metric | AVX2 Time | Speedup vs Scalar |
|--------|-----------|-------------------|
| L2 (Euclidean) | 38 ns | 3.7x |
| Cosine | 51 ns | 3.7x |
| Inner Product | 36 ns | 3.7x |

## Use Cases

### Semantic Search with RAG

```sql
SELECT content, embedding <=> $query_embedding AS similarity
FROM documents
WHERE category = 'technical'
ORDER BY similarity
LIMIT 5;
```

### Knowledge Graph with Hierarchical Embeddings

```sql
-- Use hyperbolic embeddings for taxonomy
SELECT name, ruvector_poincare_distance(embedding, $query, -1.0) AS distance
FROM taxonomy_nodes
ORDER BY distance
LIMIT 10;
```

### Hybrid Search (Vector + BM25)

```sql
SELECT
    content,
    0.7 * (1.0 / (1.0 + embedding <-> $query_vector)) +
    0.3 * ruvector_bm25_score(terms, doc_freqs, length, avg_len, total) AS score
FROM documents
ORDER BY score DESC
LIMIT 10;
```

### Multi-Agent Query Routing

```sql
SELECT ruvector_route_query(
    $user_query_embedding,
    (SELECT array_agg(row(name, capabilities)) FROM agents)
) AS best_agent;
```

### Graph Neural Network Inference

```sql
SELECT ruvector_gnn_gcn_layer(
    node_features,
    adjacency_matrix,
    trained_weights
) AS updated_features
FROM graph_nodes;
```

## CLI Tool

Install the CLI for easy management:

```bash
npm install -g @ruvector/postgres-cli

# Commands
ruvector-pg install                    # Install extension
ruvector-pg vector create table --dim 384 --index hnsw
ruvector-pg hyperbolic poincare-distance --a "[0.1,0.2]" --b "[0.3,0.4]"
ruvector-pg gnn gcn --features "[[...]]" --adj "[[...]]"
ruvector-pg graph query "MATCH (n) RETURN n"
ruvector-pg routing route --query "[...]" --agents agents.json
ruvector-pg learning adaptive-search --context "[...]"
ruvector-pg bench run --type all --size 10000
```

## Related Packages

- [`@ruvector/postgres-cli`](https://www.npmjs.com/package/@ruvector/postgres-cli) - CLI for RuVector PostgreSQL
- [`ruvector-core`](https://crates.io/crates/ruvector-core) - Core vector operations library
- [`ruvector-tiny-dancer`](https://crates.io/crates/ruvector-tiny-dancer) - Agent routing library

## Documentation

| Document | Description |
|----------|-------------|
| [docs/API.md](docs/API.md) | Complete SQL API reference |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture |
| [docs/SIMD_OPTIMIZATION.md](docs/SIMD_OPTIMIZATION.md) | SIMD details |
| [docs/guides/ATTENTION_QUICK_REFERENCE.md](docs/guides/ATTENTION_QUICK_REFERENCE.md) | Attention mechanisms |
| [docs/GNN_QUICK_REFERENCE.md](docs/GNN_QUICK_REFERENCE.md) | GNN layers |
| [docs/ROUTING_QUICK_REFERENCE.md](docs/ROUTING_QUICK_REFERENCE.md) | Tiny Dancer routing |
| [docs/LEARNING_MODULE_README.md](docs/LEARNING_MODULE_README.md) | ReasoningBank |

## Requirements

- PostgreSQL 14, 15, 16, or 17
- x86_64 (AVX2/AVX-512) or ARM64 (NEON)
- Linux, macOS, or Windows (WSL)

## License

MIT License - See [LICENSE](../../LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md)
