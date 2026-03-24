# RuVector Documentation

Complete documentation for RuVector, the high-performance Rust vector database with global scale capabilities.

## 📚 Documentation Structure

```
docs/
├── adr/                    # Architecture Decision Records
├── analysis/               # Research & analysis docs
├── api/                    # API references (Rust, Node.js, Cypher)
├── architecture/           # System design docs
├── benchmarks/             # Performance benchmarks & results
├── cloud-architecture/     # Cloud deployment guides
├── code-reviews/           # Code review documentation
├── dag/                    # DAG implementation
├── development/            # Developer guides
├── examples/               # SQL examples
├── gnn/                    # GNN/Graph implementation
├── guides/                 # User guides & tutorials
├── hnsw/                   # HNSW index documentation
├── hooks/                  # Hooks system documentation
├── implementation/         # Implementation details & summaries
├── integration/            # Integration guides
├── nervous-system/         # Nervous system architecture
├── optimization/           # Performance optimization guides
├── plans/                  # Implementation plans
├── postgres/               # PostgreSQL extension docs
├── project-phases/         # Development phases
├── publishing/             # NPM publishing guides
├── research/               # Research documentation
├── ruvllm/                 # RuVLLM documentation
├── security/               # Security audits & reports
├── sparse-inference/       # Sparse inference docs
├── sql/                    # SQL examples
├── testing/                # Testing documentation
└── training/               # Training & LoRA docs
```

### Getting Started
- **[guides/GETTING_STARTED.md](./guides/GETTING_STARTED.md)** - Getting started guide
- **[guides/BASIC_TUTORIAL.md](./guides/BASIC_TUTORIAL.md)** - Basic tutorial
- **[guides/INSTALLATION.md](./guides/INSTALLATION.md)** - Installation instructions
- **[guides/AGENTICDB_QUICKSTART.md](./guides/AGENTICDB_QUICKSTART.md)** - AgenticDB quick start
- **[guides/wasm-api.md](./guides/wasm-api.md)** - WebAssembly API documentation

### Architecture & Design
- **[architecture/](./architecture/)** - System architecture details
- **[cloud-architecture/](./cloud-architecture/)** - Global cloud deployment
- **[adr/](./adr/)** - Architecture Decision Records
- **[nervous-system/](./nervous-system/)** - Nervous system architecture

### API Reference
- **[api/RUST_API.md](./api/RUST_API.md)** - Rust API reference
- **[api/NODEJS_API.md](./api/NODEJS_API.md)** - Node.js API reference
- **[api/CYPHER_REFERENCE.md](./api/CYPHER_REFERENCE.md)** - Cypher query reference

### Performance & Benchmarks
- **[benchmarks/](./benchmarks/)** - Performance benchmarks & results
- **[optimization/](./optimization/)** - Performance optimization guides
- **[analysis/](./analysis/)** - Research & analysis docs

### Security
- **[security/](./security/)** - Security audits & reports

### Implementation
- **[implementation/](./implementation/)** - Implementation details & summaries
- **[integration/](./integration/)** - Integration guides
- **[code-reviews/](./code-reviews/)** - Code review documentation

### Specialized Topics
- **[gnn/](./gnn/)** - GNN/Graph implementation
- **[hnsw/](./hnsw/)** - HNSW index documentation
- **[postgres/](./postgres/)** - PostgreSQL extension docs
- **[ruvllm/](./ruvllm/)** - RuVLLM documentation
- **[training/](./training/)** - Training & LoRA docs

### Development
- **[development/CONTRIBUTING.md](./development/CONTRIBUTING.md)** - Contribution guidelines
- **[development/MIGRATION.md](./development/MIGRATION.md)** - Migration guide
- **[testing/](./testing/)** - Testing documentation
- **[publishing/](./publishing/)** - NPM publishing guides

### Research
- **[research/](./research/)** - Research documentation
  - cognitive-frontier/ - Cognitive frontier research
  - gnn-v2/ - GNN v2 research
  - latent-space/ - HNSW & attention research
  - mincut/ - MinCut algorithm research

---

## 🚀 Quick Links

### For New Users
1. Start with [Getting Started Guide](./guides/GETTING_STARTED.md)
2. Try the [Basic Tutorial](./guides/BASIC_TUTORIAL.md)
3. Review [API Documentation](./api/)

### For Cloud Deployment
1. Read [Architecture Overview](./cloud-architecture/architecture-overview.md)
2. Follow [Deployment Guide](./cloud-architecture/DEPLOYMENT_GUIDE.md)
3. Apply [Performance Optimizations](./cloud-architecture/PERFORMANCE_OPTIMIZATION_GUIDE.md)

### For Contributors
1. Read [Contributing Guidelines](./development/CONTRIBUTING.md)
2. Review [Architecture Decisions](./adr/)
3. Check [Migration Guide](./development/MIGRATION.md)

### For Performance Tuning
1. Review [Optimization Guide](./optimization/PERFORMANCE_TUNING_GUIDE.md)
2. Run [Benchmarks](./benchmarks/BENCHMARKING_GUIDE.md)
3. Check [Analysis](./analysis/)

---

## 📊 Documentation Status

| Category | Directory | Status |
|----------|-----------|--------|
| Getting Started | guides/ | ✅ Complete |
| Architecture | architecture/, adr/ | ✅ Complete |
| API Reference | api/ | ✅ Complete |
| Performance | benchmarks/, optimization/, analysis/ | ✅ Complete |
| Security | security/ | ✅ Complete |
| Implementation | implementation/, integration/ | ✅ Complete |
| Development | development/, testing/ | ✅ Complete |
| Research | research/ | 📚 Ongoing |

**Total Documentation**: 460+ documents across 60+ directories

---

## 🔗 External Resources

- **GitHub Repository**: https://github.com/ruvnet/ruvector
- **Main README**: [../README.md](../README.md)
- **Changelog**: [../CHANGELOG.md](../CHANGELOG.md)
- **License**: [../LICENSE](../LICENSE)

---

**Last Updated**: 2026-02-26 | **Version**: 2.0.4 (core) / 0.1.100 (npm) | **Status**: Production Ready
