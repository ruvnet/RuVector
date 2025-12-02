# RuVector-Postgres

**High-Performance PostgreSQL Vector Similarity Search Extension**

A drop-in replacement for pgvector, built in Rust with SIMD-optimized distance calculations, advanced indexing algorithms, and quantization support for memory-efficient vector storage.

## Features

- **pgvector API Compatibility** - 100% compatible SQL interface, seamless migration
- **SIMD Acceleration** - AVX-512, AVX2, and ARM NEON optimized distance calculations (2-10x faster)
- **Multiple Index Types** - HNSW and IVFFlat indexes for approximate nearest neighbor search
- **Quantization Support** - Scalar, product, and binary quantization (up to 32x memory reduction)
- **Multiple Vector Types** - Dense (`ruvector`), half-precision (`halfvec`), and sparse (`sparsevec`)
- **Zero-Copy Operations** - Direct memory access for minimal overhead
- **Neon Compatible** - Designed for serverless PostgreSQL environments


## Comparison with pgvector

| Feature | pgvector 0.8.0 | RuVector-Postgres |
|---------|---------------|-------------------|
| Max dimensions | 16,000 | 16,000 |
| HNSW index | Yes | Yes (optimized) |
| IVFFlat index | Yes | Yes (optimized) |
| Half-precision vectors | Yes | Yes |
| Sparse vectors | Yes | Yes |
| **AVX-512 optimized** | Partial | **Full** |
| **ARM NEON optimized** | No | **Yes** |
| **Zero-copy access** | No | **Yes** |
| **Product quantization** | No | **Yes** |
| **Scalar quantization** | No | **Yes** |
| Hybrid search | No | Yes |
| Filtered HNSW | Partial | Yes |

### Performance Benchmarks

*Single distance calculation (1536 dimensions):*

| Metric | AVX2 Time | Speedup vs Scalar |
|--------|-----------|-------------------|
| L2 (Euclidean) | 38 ns | 3.7x |
| Cosine | 51 ns | 3.7x |
| Inner Product | 36 ns | 3.7x |
| Manhattan | 42 ns | 3.7x |

*Batch processing (10K vectors x 384 dimensions):*

| Operation | Time | Throughput |
|-----------|------|------------|
| Sequential | 3.8 ms | 2.6M distances/sec |
| Parallel (16 cores) | 0.28 ms | 35.7M distances/sec |


## Quick Start

### Installation

**Option 1: Quick Install Script**

```bash
# Auto-detects platform and installs dependencies
curl -sSL https://raw.githubusercontent.com/ruvnet/ruvector/main/crates/ruvector-postgres/install/quick-start.sh | bash
```

**Option 2: Full Installation**

```bash
# Clone repository
git clone https://github.com/ruvnet/ruvector.git
cd ruvector/crates/ruvector-postgres

# Install with auto-detection
./install/install.sh --build-from-source

# Or specify PostgreSQL version
./install/install.sh --build-from-source --pg-version 16
```

See [install/install.sh](install/install.sh) for all options including `--dry-run`, `--verbose`, and platform-specific configurations.



### Basic Usage

```sql
-- Create the extension
CREATE EXTENSION ruvector;

-- Create a table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding ruvector(1536)  -- OpenAI ada-002 dimensions
);

-- Insert vectors
INSERT INTO documents (content, embedding) VALUES
    ('First document', '[0.1, 0.2, 0.3, ...]'),
    ('Second document', '[0.4, 0.5, 0.6, ...]');

-- Create an HNSW index for fast similarity search
CREATE INDEX ON documents USING ruhnsw (embedding ruvector_l2_ops);

-- Find similar documents
SELECT content, embedding <-> '[0.15, 0.25, 0.35, ...]'::ruvector AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

## Vector Types

### `ruvector(n)` - Dense Vector

Standard 32-bit floating point vector for maximum precision.

```sql
CREATE TABLE items (embedding ruvector(1536));
-- Storage: 8 + (4 × dimensions) bytes
```

### `halfvec(n)` - Half-Precision Vector

16-bit floating point for 50% memory savings with minimal accuracy loss.

```sql
CREATE TABLE items (embedding halfvec(1536));
-- Storage: 8 + (2 × dimensions) bytes
```

### `sparsevec(n)` - Sparse Vector

For high-dimensional sparse data (BM25, TF-IDF).

```sql
CREATE TABLE items (embedding sparsevec(50000));
-- Storage: 12 + (8 × non_zero_elements) bytes
INSERT INTO items VALUES ('{1:0.5, 100:0.8, 5000:0.3}/50000');
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

Best for high recall and fast queries.

```sql
CREATE INDEX ON items USING ruhnsw (embedding ruvector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- Tune search quality
SET ruvector.ef_search = 100;
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `m` | 16 | Max connections per layer (2-100) |
| `ef_construction` | 64 | Build-time search breadth (4-1000) |

### IVFFlat (Inverted File Flat)

Best for memory-constrained environments and large datasets.

```sql
CREATE INDEX ON items USING ruivfflat (embedding ruvector_l2_ops)
WITH (lists = 100);

-- Tune search quality
SET ruvector.ivfflat_probes = 10;
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lists` | 100 | Number of clusters (1-10000) |

### When to Use Each Index

| Criteria | HNSW | IVFFlat |
|----------|------|---------|
| Build time | Slower | Faster |
| Search speed | Faster | Fast |
| Memory usage | Higher | Lower |
| Recall | 95-99% | 80-95% |
| Best for | High-recall queries | Large static datasets |

## Tutorials

### Semantic Search with OpenAI Embeddings

```sql
-- Create table for documents
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    embedding ruvector(1536)
);

-- Create index
CREATE INDEX ON documents USING ruhnsw (embedding ruvector_cosine_ops);

-- Search (after inserting embeddings from OpenAI API)
SELECT title, content, embedding <=> $query_embedding AS similarity
FROM documents
ORDER BY similarity
LIMIT 5;
```

### Image Similarity with CLIP Embeddings

```sql
-- CLIP produces 512-dimensional vectors
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    embedding ruvector(512)
);

CREATE INDEX ON images USING ruhnsw (embedding ruvector_l2_ops)
WITH (m = 32, ef_construction = 200);

-- Find similar images
SELECT filename, embedding <-> $query_embedding AS distance
FROM images
ORDER BY distance
LIMIT 10;
```

### Memory-Efficient Large-Scale Search

```sql
-- Use half-precision for 50% memory savings
CREATE TABLE large_dataset (
    id SERIAL PRIMARY KEY,
    embedding halfvec(1536)
);

-- IVFFlat for memory efficiency
CREATE INDEX ON large_dataset USING ruivfflat (embedding ruvector_l2_ops)
WITH (lists = 1000);

-- Increase probes for better recall
SET ruvector.ivfflat_probes = 20;
```

### Hybrid Search (Vector + Text)

```sql
SELECT
    content,
    embedding <-> $query_vector AS vector_score,
    ts_rank(to_tsvector(content), to_tsquery($search_terms)) AS text_score,
    (0.7 * (1.0 / (1.0 + embedding <-> $query_vector)) +
     0.3 * ts_rank(to_tsvector(content), to_tsquery($search_terms))) AS combined
FROM documents
WHERE to_tsvector(content) @@ to_tsquery($search_terms)
ORDER BY combined DESC
LIMIT 10;
```

## Configuration

### GUC Variables

```sql
-- HNSW search quality (higher = better recall, slower)
SET ruvector.ef_search = 100;

-- IVFFlat probes (higher = better recall, slower)
SET ruvector.ivfflat_probes = 10;
```

### Performance Tuning

```sql
-- Enable parallel index builds
SET maintenance_work_mem = '8GB';
SET max_parallel_maintenance_workers = 8;

-- Enable parallel queries
SET max_parallel_workers_per_gather = 4;
```

## Installation Options

The [install.sh](install/install.sh) script supports:

| Option | Description |
|--------|-------------|
| `--pg-version VERSION` | PostgreSQL version (14, 15, 16, 17) |
| `--pg-config PATH` | Path to pg_config |
| `--simd MODE` | SIMD mode: auto, avx512, avx2, neon, scalar |
| `--build-from-source` | Build from source |
| `--skip-tests` | Skip installation tests |
| `--dry-run` | Show what would be done |
| `--verbose` | Verbose output |
| `--uninstall` | Uninstall extension |

Platform-specific setup scripts are available in [install/scripts/](install/scripts/):

- `setup-debian.sh` - Debian/Ubuntu
- `setup-rhel.sh` - RHEL/CentOS/Fedora
- `setup-macos.sh` - macOS (Homebrew)

## Documentation

| Document | Description |
|----------|-------------|
| [docs/API.md](docs/API.md) | Complete SQL API reference |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and design |
| [docs/SIMD_OPTIMIZATION.md](docs/SIMD_OPTIMIZATION.md) | SIMD implementation details |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Detailed installation guide |
| [docs/MIGRATION.md](docs/MIGRATION.md) | Migrating from pgvector |
| [docs/NEON_COMPATIBILITY.md](docs/NEON_COMPATIBILITY.md) | Serverless PostgreSQL deployment |
| [docs/guides/IVFFLAT.md](docs/guides/IVFFLAT.md) | IVFFlat index guide |
| [docs/implementation/](docs/implementation/) | Implementation details |

## Building from Source

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs))
- PostgreSQL 14-17 with development headers
- Build tools (gcc/clang, make)

### Build Steps

```bash
cd crates/ruvector-postgres

# Install pgrx
cargo install cargo-pgrx --version "0.12.9" --locked

# Initialize pgrx for your PostgreSQL version
cargo pgrx init --pg16 $(which pg_config)

# Build and install
cargo pgrx install --release
```

### Running Tests

```bash
# Rust tests
cargo test

# SQL integration tests
psql -f tests/ivfflat_am_test.sql
```

## Requirements

- PostgreSQL 14, 15, 16, or 17
- x86_64 (with AVX2/AVX-512) or ARM64 (with NEON)
- Linux, macOS, or Windows (via WSL)

## License

MIT License - See [LICENSE](../../LICENSE) in the repository root.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/ruvnet/ruvector/issues)
- Examples: [examples/](examples/)
