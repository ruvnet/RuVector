# ADR-071: npx ruvector Ecosystem Gap Analysis

**Status**: Proposed
**Date**: 2026-02-28
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-065 (npm Publishing Strategy), ADR-070 (npx ruvector Unified Integration)

## 1. Context

The ruvector project produces **79 npm packages** and **27 WASM crates** spanning vector databases, graph engines, LLM orchestration, quantum simulation, spiking neural networks, cryptographic primitives, and distributed compute. The primary CLI entry point — `npx ruvector` (v0.1.100) — exposes only a fraction of this surface: vector CRUD, GNN layers, attention mechanisms, and system diagnostics.

Meanwhile, significant capabilities exist as Rust crates with no npm wrapper, as WASM crates with no JavaScript bindings published, or as npm packages that are disconnected from the CLI. This ADR catalogs every gap and proposes a roadmap to make the full ecosystem accessible through `npx ruvector`.

## 2. Current State

### 2.1 What `npx ruvector` Exposes Today

| Command Group | Subcommands | Backend |
|--------------|-------------|---------|
| `create` | Create vector DB | @ruvector/core |
| `insert` | Insert vectors from JSON | @ruvector/core |
| `search` | ANN search with filters | @ruvector/core |
| `stats` | Database statistics | @ruvector/core |
| `benchmark` | Performance benchmarks | @ruvector/core |
| `info` | System info (backends, versions) | built-in |
| `install` | Install optional packages | built-in |
| `doctor` | Health check | built-in |
| `gnn layer` | Create/test GNN layers | @ruvector/gnn |
| `gnn compress` | Adaptive tensor compression | @ruvector/gnn |
| `gnn search` | Differentiable search | @ruvector/gnn |
| `gnn info` | GNN module info | @ruvector/gnn |
| `attention compute` | 5 attention mechanisms | @ruvector/attention |
| `attention benchmark` | Benchmark attention types | @ruvector/attention |
| `attention hyperbolic` | Hyperbolic geometry ops | @ruvector/attention |
| `attention list` | List mechanisms | @ruvector/attention |
| `attention info` | Module details | @ruvector/attention |

**Total: 17 commands across 4 groups.**

### 2.2 What Exists as npm Packages but NOT in the CLI

| npm Package | Version | Capability | CLI Integration |
|-------------|---------|-----------|----------------|
| `@ruvector/pi-brain` | 0.1.0 | Shared brain CLI + SDK + MCP | **Missing** — has own `npx pi-brain` CLI |
| `@ruvector/sona` | 0.1.4 | Self-optimizing neural architecture | Bundled dep but **no CLI commands** |
| `@ruvector/rvf` | 0.2.0 | RuVector Format SDK (read/write/validate) | Optional dep but **no CLI commands** |
| `@ruvector/rvf-solver` | 0.1.7 | Temporal constraint solver | **No CLI** |
| `@ruvector/rvf-wasm` | 0.1.5 | RVF WASM microkernel | **No CLI** |
| `@ruvector/rvf-node` | 0.1.7 | RVF Node.js bindings | **No CLI** |
| `@ruvector/rvf-mcp-server` | 0.1.3 | RVF MCP server | **No CLI** |
| `@ruvector/ruvllm` | 2.5.1 | LLM orchestration + SONA + HNSW | **Separate CLI** (`npx ruvllm`) |
| `@ruvector/ruvllm-cli` | 0.1.0 | LLM inference CLI | **Separate binary** |
| `@ruvector/ruvllm-wasm` | 0.1.0 | WASM LLM inference | **No CLI** |
| `@ruvector/graph-node` | 2.0.2 | Native hypergraph bindings | **No CLI** |
| `@ruvector/graph-wasm` | 2.0.2 | Neo4j-compatible hypergraph WASM | **No CLI** |
| `@ruvector/graph-data-generator` | 0.1.0 | Synthetic graph data generation | **No CLI** |
| `@ruvector/ruqu-wasm` | 2.0.5 | Quantum simulations | **No CLI** |
| `@ruvector/spiking-neural` | 1.0.1 | Spiking neural networks (SIMD) | **No CLI** |
| `@ruvector/ospipe` | 0.1.2 | Personal AI memory | **No CLI** |
| `@ruvector/ospipe-wasm` | 0.1.0 | Personal AI memory WASM | **No CLI** |
| `@ruvector/rvdna` | 0.3.0 | Genomic analysis (20-SNP biomarker) | **No CLI** |
| `@ruvector/scipix` | 0.1.0 | OCR for scientific documents | **No CLI** |
| `@ruvector/tiny-dancer` | 0.1.17 | Neural router (FastGRNN) | **No CLI** |
| `@ruvector/router` | 0.1.28 | Semantic router for AI agents | **No CLI** |
| `@ruvector/ruvbot` | 0.3.1 | Self-learning AI assistant | **Separate CLI** |
| `@ruvector/rvlite` | 0.2.4 | Lightweight DB (SQL/SPARQL/Cypher) | **No CLI** |
| `@ruvector/agentic-integration` | 1.0.0 | Distributed agent coordination | **No CLI** |
| `@ruvector/agentic-synth` | 0.1.6 | Synthetic data generator | **Has own CLI** |
| `@ruvector/burst-scaling` | 1.0.0 | Adaptive burst scaling | **No CLI** |
| `@ruvector/cognitum-gate-wasm` | 0.1.0 | AI coherence gate | **No CLI** |
| `@ruvector/raft` | 0.1.0 | Raft consensus | **No CLI** |
| `@ruvector/replication` | 0.1.0 | Data replication & sync | **No CLI** |
| `@ruvector/postgres-cli` | 0.2.7 | PostgreSQL pgvector CLI | **Separate CLI** |
| `@ruvector/ruvector-extensions` | 0.1.0 | Embeddings, UI, exports, temporal | **No CLI** |
| `@ruvector/ruvector-wasm-unified` | 1.0.0 | Unified TypeScript WASM API | **No CLI** |

**31 packages with no CLI integration.**

### 2.3 What Exists as WASM Crates but NOT as npm Packages

| WASM Crate | Version | Capability | npm Package |
|-----------|---------|-----------|-------------|
| `ruvector-attention-unified-wasm` | 0.1.0 | Unified attention (46 mechanisms) | **Missing** |
| `ruvector-attention-wasm` | — | Attention WASM bindings | Partial (`@ruvector/attention`) |
| `ruvector-dag-wasm` | 0.1.0 | DAG operations | **Missing** |
| `ruvector-delta-wasm` | 0.1.0 | Delta consensus/behavior tracking | **Missing** |
| `ruvector-domain-expansion-wasm` | 0.1.0 | Transfer learning, domain expansion | **Missing** |
| `ruvector-economy-wasm` | 0.1.0 | Economic engine (reputation, AMM) | **Missing** |
| `ruvector-exotic-wasm` | — | Exotic neural architectures | **Missing** |
| `ruvector-fpga-transformer-wasm` | 0.1.0 | FPGA-optimized transformers | **Missing** |
| `ruvector-gnn-wasm` | — | GNN WASM bindings | Partial (`@ruvector/gnn`) |
| `ruvector-graph-transformer-wasm` | — | Graph transformer | **Missing** |
| `ruvector-hyperbolic-hnsw-wasm` | 0.1.0 | Hyperbolic HNSW search | **Missing** |
| `ruvector-learning-wasm` | 0.1.0 | Online learning | **Missing** |
| `ruvector-math-wasm` | — | Math primitives | **Missing** |
| `ruvector-mincut-gated-transformer` | 0.1.0 | MinCut-gated transformer | **Missing** |
| `ruvector-mincut-wasm` | — | MinCut graph partitioning | **Missing** |
| `ruvector-nervous-system-wasm` | 0.1.0 | Nervous system architecture | **Missing** |
| `ruvector-sparse-inference-wasm` | — | Sparse inference engine | **Missing** |
| `ruvector-temporal-tensor-wasm` | — | Temporal tensor operations | **Missing** |

**18 WASM crates with no npm package.**

### 2.4 What Exists as Rust Crates Only (No WASM, No npm)

| Crate | Capability | Why It Matters |
|-------|-----------|---------------|
| `mcp-brain` | MCP stdio server for shared brain | Core brain MCP — only accessible via `cargo run` |
| `mcp-brain-server` | Cloud Run REST backend | Server-side only |
| `mcp-gate` | MCP coherence gate | Core MCP — only via `cargo run` |
| `cognitum-gate-kernel` | AI coherence gate kernel | Core reasoning engine |
| `cognitum-gate-tilezero` | TileZero game engine | Specialized |
| `prime-radiant` | Prime Radiant visualization | Specialized |
| `ruvector-delta-core` | Delta behavior tracking | Core capability, no JS access |
| `ruvector-delta-runtime` | Delta runtime | Runtime only |
| `ruvector-delta-serde` | Delta serialization | Utility |
| `ruvector-domain-expansion` | Transfer learning engine | Core brain capability |
| `ruvector-mincut` | SubpolynomialMinCut partitioning | Core graph capability |
| `ruvector-attention` | 46 attention mechanisms | Partially exposed via `@ruvector/attention` |
| `sona` | SONA learning engine | Partially exposed via `@ruvector/sona` |
| `rvf-federation` | Federated learning (PII strip, DP) | Core brain pipeline |
| `rvf-crypto` | Witness chains, Ed25519, SHAKE-256 | Core security |
| `agentic-robotics-*` (6 crates) | Autonomous robotics | Entire subsystem missing |
| `thermorust` | Thermal/energy modeling | Specialized |
| `ruvector-dither` | Dithering algorithms | Specialized |
| `ruvector-profiler` | Performance profiler | Dev tool (publish=false) |

**19+ crates with no JavaScript access at all.**

### 2.5 Fragmented CLI Entry Points

Users currently face 7+ separate CLI binaries:

| Binary | Package | Install |
|--------|---------|---------|
| `ruvector` | `ruvector` | `npx ruvector` |
| `pi-brain` / `π` | `@ruvector/pi-brain` | `npx pi-brain` |
| `ruvllm` | `@ruvector/ruvllm-cli` | `npx ruvllm` |
| `ruvbot` | `@ruvector/ruvbot` | `npx ruvbot` |
| `agentic-synth` | `@ruvector/agentic-synth` | `npx agentic-synth` |
| `rvf` | `@ruvector/rvf` | `npx rvf` |
| `postgres-cli` | `@ruvector/postgres-cli` | `npx @ruvector/postgres-cli` |

Each has its own install, auth, and configuration. There is no single `npx ruvector <anything>` that reaches them all.

## 3. Decision

Extend `npx ruvector` to be the **universal entry point** for the entire ecosystem. Every capability — whether it's a Rust crate, WASM binding, or npm package — should be reachable through `npx ruvector <group> <command>`. Missing npm packages for WASM crates should be published. Fragmented CLIs should be consolidated.

## 4. Proposed Command Hierarchy

```
npx ruvector
  │
  ├── [EXISTING] Vector Database
  │   ├── create         Create vector database
  │   ├── insert         Insert vectors
  │   ├── search         ANN search with filters
  │   ├── stats          Database statistics
  │   ├── benchmark      Performance benchmarks
  │   ├── info           System info
  │   ├── install        Install optional packages
  │   └── doctor         Health check
  │
  ├── [EXISTING] GNN
  │   ├── gnn layer      Create/test GNN layers
  │   ├── gnn compress   Adaptive tensor compression
  │   ├── gnn search     Differentiable search
  │   └── gnn info       Module info
  │
  ├── [EXISTING] Attention
  │   ├── attention compute     5 attention mechanisms
  │   ├── attention benchmark   Benchmark all types
  │   ├── attention hyperbolic  Hyperbolic geometry
  │   ├── attention list        List mechanisms
  │   └── attention info        Module details
  │
  ├── [ADR-070] Brain (lazy: @ruvector/pi-brain)
  │   ├── brain share     Share knowledge
  │   ├── brain search    Semantic search
  │   ├── brain get       Retrieve by ID
  │   ├── brain vote      Quality vote
  │   ├── brain list      List memories
  │   ├── brain delete    Delete own
  │   ├── brain transfer  Domain transfer
  │   ├── brain drift     Drift detection
  │   ├── brain partition Knowledge topology
  │   ├── brain status    System health
  │   ├── brain sync      LoRA weight sync
  │   └── brain page      Brainpedia CRUD
  │
  ├── [ADR-070] Edge (lazy: @ruvector/edge-net)
  │   ├── edge status     Network status
  │   ├── edge join       Join as compute node
  │   ├── edge balance    rUv balance
  │   ├── edge tasks      Available compute tasks
  │   └── edge dashboard  Open dashboard
  │
  ├── [ADR-070] MCP
  │   ├── mcp start       Start MCP server (stdio/SSE)
  │   ├── mcp tools       List available tools
  │   └── mcp test        Test connection
  │
  ├── [ADR-070] Identity
  │   ├── identity generate  Generate π key
  │   ├── identity show      Display pseudonym
  │   ├── identity export    Encrypted backup
  │   └── identity import    Restore from backup
  │
  ├── [NEW] LLM (lazy: @ruvector/ruvllm)
  │   ├── llm chat        Interactive chat
  │   ├── llm embed       Generate embeddings
  │   ├── llm complete    Text completion
  │   ├── llm models      List available models
  │   ├── llm benchmark   Inference benchmark
  │   └── llm serve       Start LLM server
  │
  ├── [NEW] RVF (lazy: @ruvector/rvf)
  │   ├── rvf read        Read .rvf container
  │   ├── rvf write       Create .rvf container
  │   ├── rvf validate    Validate integrity
  │   ├── rvf inspect     Show segment layout
  │   ├── rvf merge       Merge containers
  │   └── rvf convert     Format conversions
  │
  ├── [NEW] Graph (lazy: @ruvector/graph-wasm)
  │   ├── graph create    Create hypergraph
  │   ├── graph query     Cypher/SPARQL query
  │   ├── graph import    Import from CSV/JSON/Neo4j
  │   ├── graph export    Export to various formats
  │   ├── graph visualize Text-based visualization
  │   └── graph stats     Graph statistics
  │
  ├── [NEW] SONA (lazy: @ruvector/sona)
  │   ├── sona train      Train with trajectory
  │   ├── sona patterns   Search learned patterns
  │   ├── sona optimize   Run optimization
  │   ├── sona export     Export learned weights
  │   └── sona stats      Learning statistics
  │
  ├── [NEW] Router (lazy: @ruvector/router)
  │   ├── router classify Classify input to route
  │   ├── router train    Train router on examples
  │   ├── router serve    Start router server
  │   └── router benchmark  Route throughput test
  │
  ├── [NEW] Quantum (lazy: @ruvector/ruqu-wasm)
  │   ├── quantum sim     Run quantum simulation
  │   ├── quantum circuit Build quantum circuit
  │   └── quantum stats   Simulation statistics
  │
  ├── [NEW] SNN (lazy: @ruvector/spiking-neural)
  │   ├── snn train       Train spiking network
  │   ├── snn inference   Run inference
  │   └── snn benchmark   SIMD performance test
  │
  ├── [NEW] Delta (lazy: @ruvector/delta-wasm — TO PUBLISH)
  │   ├── delta track     Track behavior changes
  │   ├── delta compare   Compare two snapshots
  │   └── delta report    Drift report
  │
  ├── [NEW] MinCut (lazy: @ruvector/mincut-wasm — TO PUBLISH)
  │   ├── mincut partition  Partition graph
  │   ├── mincut certificate  Verify cut certificate
  │   └── mincut visualize  Text visualization
  │
  ├── [NEW] Synth (lazy: @ruvector/agentic-synth)
  │   ├── synth generate  Generate synthetic data
  │   ├── synth validate  Validate generated data
  │   └── synth config    Configure generators
  │
  ├── [NEW] DNA (lazy: @ruvector/rvdna)
  │   ├── dna analyze     Analyze genomic data
  │   ├── dna biomarker   20-SNP biomarker panel
  │   └── dna report      Generate report
  │
  ├── [NEW] OCR (lazy: @ruvector/scipix)
  │   ├── ocr extract     Extract text from images
  │   ├── ocr table       Extract tables
  │   └── ocr equations   Extract equations
  │
  ├── [NEW] DB (lazy: @ruvector/rvlite)
  │   ├── db query        SQL/SPARQL/Cypher query
  │   ├── db import       Import data
  │   └── db export       Export data
  │
  └── [NEW] Postgres (lazy: @ruvector/postgres-cli)
      ├── pg connect      Connect to PostgreSQL
      ├── pg vector       pgvector operations
      └── pg migrate      Schema migrations
```

**Total: ~100+ commands across 20+ groups** (up from 17 commands across 4 groups).

## 5. npm Packages to Publish

### 5.1 Priority 1 — WASM crates with brain/edge integration value

| WASM Crate | Proposed npm Package | Why |
|-----------|---------------------|-----|
| `ruvector-delta-wasm` | `@ruvector/delta-wasm` | Brain drift detection via `npx ruvector delta` |
| `ruvector-mincut-wasm` | `@ruvector/mincut-wasm` | Brain knowledge partitioning |
| `ruvector-domain-expansion-wasm` | `@ruvector/domain-expansion-wasm` | Brain transfer learning |
| `ruvector-economy-wasm` | `@ruvector/economy-wasm` | Edge-net economics (AMM, reputation) |
| `ruvector-learning-wasm` | `@ruvector/learning-wasm` | Online learning for edge nodes |
| `edge-net` (examples/) | `@ruvector/edge-net` | Edge-net WASM for `npx ruvector edge` |

### 5.2 Priority 2 — Advanced capabilities

| WASM Crate | Proposed npm Package | Why |
|-----------|---------------------|-----|
| `ruvector-attention-unified-wasm` | `@ruvector/attention-unified-wasm` | All 46 attention mechanisms |
| `ruvector-hyperbolic-hnsw-wasm` | `@ruvector/hyperbolic-hnsw-wasm` | Hyperbolic space search |
| `ruvector-nervous-system-wasm` | `@ruvector/nervous-system-wasm` | Full nervous system architecture |
| `ruvector-fpga-transformer-wasm` | `@ruvector/fpga-transformer-wasm` | FPGA-optimized inference |
| `ruvector-sparse-inference-wasm` | `@ruvector/sparse-inference-wasm` | Sparse model inference |
| `ruvector-graph-transformer-wasm` | `@ruvector/graph-transformer-wasm` | Graph transformers |

### 5.3 Priority 3 — Specialized

| WASM Crate | Proposed npm Package | Why |
|-----------|---------------------|-----|
| `ruvector-dag-wasm` | `@ruvector/dag-wasm` | DAG operations |
| `ruvector-math-wasm` | `@ruvector/math-wasm` | Math primitives |
| `ruvector-temporal-tensor-wasm` | `@ruvector/temporal-tensor-wasm` | Temporal operations |
| `ruvector-exotic-wasm` | `@ruvector/exotic-wasm` | Exotic neural architectures |
| `ruvector-mincut-gated-transformer` | `@ruvector/mincut-gated-wasm` | MinCut-gated attention |

## 6. Version Landscape

### 6.1 Mature (v2.x — stable API)

| Package | Version | Notes |
|---------|---------|-------|
| `@ruvector/ruvllm` | 2.5.1 | Self-learning LLM orchestration |
| `@ruvector/graph-node` | 2.0.2 | Native hypergraph (NAPI) |
| `@ruvector/graph-wasm` | 2.0.2 | WASM hypergraph |
| `@ruvector/ruqu-wasm` | 2.0.5 | Quantum simulations |
| `@ruvector/ruvllm-wasm` | 2.0.0 | WASM LLM |
| `micro-hnsw-wasm` | 2.3.2 | HNSW core |

### 6.2 Stable (v1.x — production-ready)

| Package | Version | Notes |
|---------|---------|-------|
| `@ruvector/agentic-integration` | 1.0.0 | Agent coordination |
| `@ruvector/burst-scaling` | 1.0.0 | Adaptive scaling |
| `@ruvector/spiking-neural` | 1.0.1 | SNN with SIMD |
| `@ruvector/ruvector-wasm-unified` | 1.0.0 | Unified WASM API |

### 6.3 Development (v0.x — breaking changes expected)

66 packages at v0.1.x-0.3.x, including the core `ruvector` CLI at v0.1.100.

### 6.4 Version Recommendation

The core `ruvector` package should target **v0.2.0** for the unified CLI expansion (ADR-070 commands), and **v1.0.0** when all Priority 1 packages are published and integrated.

## 7. Implementation Roadmap

### Phase 1: Consolidate Existing (2 weeks)

**Goal**: Bring existing npm packages into `npx ruvector` via lazy loading.

| Task | Effort | Packages |
|------|--------|----------|
| Add `brain` commands | 3 days | @ruvector/pi-brain |
| Add `llm` commands | 2 days | @ruvector/ruvllm |
| Add `rvf` commands | 2 days | @ruvector/rvf |
| Add `graph` commands | 2 days | @ruvector/graph-wasm |
| Add `sona` commands | 1 day | @ruvector/sona |
| Add `router` commands | 1 day | @ruvector/router |
| Add `quantum` commands | 1 day | @ruvector/ruqu-wasm |
| Add `snn` commands | 1 day | @ruvector/spiking-neural |
| Add `synth` commands | 1 day | @ruvector/agentic-synth |
| Add `db` commands | 1 day | @ruvector/rvlite |
| Add `pg` commands | 1 day | @ruvector/postgres-cli |

### Phase 2: Publish Missing WASM (3 weeks)

**Goal**: Build and publish Priority 1 WASM crates to npm.

| Task | Effort | Crate → Package |
|------|--------|-----------------|
| Build + publish delta-wasm | 2 days | ruvector-delta-wasm → @ruvector/delta-wasm |
| Build + publish mincut-wasm | 2 days | ruvector-mincut-wasm → @ruvector/mincut-wasm |
| Build + publish domain-expansion-wasm | 2 days | ruvector-domain-expansion-wasm → @ruvector/domain-expansion-wasm |
| Build + publish economy-wasm | 2 days | ruvector-economy-wasm → @ruvector/economy-wasm |
| Build + publish learning-wasm | 2 days | ruvector-learning-wasm → @ruvector/learning-wasm |
| Build + publish edge-net WASM | 3 days | edge-net → @ruvector/edge-net |
| Add `delta`, `mincut`, `edge` CLI groups | 3 days | CLI integration |
| Add `identity` commands | 2 days | Pi-Key management |
| Add `mcp` commands | 2 days | MCP server lifecycle |

### Phase 3: Publish Advanced WASM (2 weeks)

**Goal**: Build and publish Priority 2 WASM crates.

6 WASM packages to build with `wasm-pack` and publish.

### Phase 4: Polish and Release (1 week)

| Task | Effort |
|------|--------|
| `npx ruvector help` — comprehensive help with all groups | 1 day |
| `npx ruvector list` — list installed vs available packages | 1 day |
| `npx ruvector upgrade` — upgrade all @ruvector packages | 1 day |
| JSON output mode for all commands | 1 day |
| Pipe detection (auto-JSON when not TTY) | 0.5 day |
| Bump to v0.2.0, update README | 0.5 day |

## 8. Dependency Strategy

### 8.1 Bundled (always installed)

```
ruvector (core)
  ├── @ruvector/core          (HNSW vector DB)
  ├── @ruvector/gnn           (GNN layers)
  ├── @ruvector/attention     (attention mechanisms)
  ├── @ruvector/sona          (SONA learning)
  ├── commander, chalk, ora   (CLI utilities)
  └── @modelcontextprotocol/sdk  (MCP protocol)
```

### 8.2 Optional Peer (lazy-loaded on first use)

```
@ruvector/pi-brain            → brain commands
@ruvector/edge-net            → edge commands
@ruvector/ruvllm              → llm commands
@ruvector/rvf                 → rvf commands
@ruvector/graph-wasm          → graph commands
@ruvector/ruqu-wasm           → quantum commands
@ruvector/spiking-neural      → snn commands
@ruvector/router              → router commands
@ruvector/delta-wasm          → delta commands
@ruvector/mincut-wasm         → mincut commands
@ruvector/agentic-synth       → synth commands
@ruvector/rvdna               → dna commands
@ruvector/scipix              → ocr commands
@ruvector/rvlite              → db commands
@ruvector/postgres-cli        → pg commands
```

### 8.3 Lazy Loading Pattern

```typescript
async function requirePackage(name: string): Promise<any> {
  try {
    return await import(name);
  } catch {
    console.error(chalk.red(`${name} is not installed.`));
    console.error(chalk.yellow(`  npm install ${name}`));
    console.error(chalk.dim(`  or: npx ruvector install ${name}`));
    process.exit(1);
  }
}
```

Each command group registers itself but defers the `import()` until the command is actually invoked. This keeps `npx ruvector` startup fast (~200ms) regardless of how many optional packages are installed.

## 9. Gap Summary

### By the Numbers

| Category | Available | In CLI | Gap |
|----------|-----------|--------|-----|
| npm packages | 79 | 4 bundled | **75 packages not in CLI** |
| WASM crates | 27 | 2 via npm | **18 without npm packages** |
| Rust-only crates | 19+ | 0 | **19+ with no JS access** |
| CLI entry points | 7 separate | 1 unified | **6 fragmented CLIs** |
| Commands | ~100 possible | 17 | **~83 missing commands** |

### Critical Gaps

1. **No brain access from CLI** — The shared intelligence at pi.ruv.io has no CLI path (ADR-070 proposed, not implemented)
2. **No edge network CLI** — Edge-net compute network unreachable from Node.js CLI
3. **No LLM commands** — ruvllm (v2.5.1, the most mature package) is a separate CLI
4. **No RVF commands** — The core file format has no CLI tooling
5. **No graph commands** — Hypergraph engine (v2.0.2) invisible to CLI users
6. **No identity management** — Pi-Key generation/management only in Rust
7. **18 WASM crates unpublished** — Significant WASM capabilities not accessible from JavaScript
8. **No unified discovery** — Users can't discover available capabilities from the CLI

## 10. Success Criteria

- [ ] `npx ruvector` lists all available command groups
- [ ] `npx ruvector brain search "auth"` works (ADR-070)
- [ ] `npx ruvector llm chat` works (ruvllm integration)
- [ ] `npx ruvector rvf inspect file.rvf` works
- [ ] `npx ruvector graph query "MATCH (n) RETURN n"` works
- [ ] `npx ruvector edge status` works (ADR-070)
- [ ] `npx ruvector identity generate` works (ADR-070)
- [ ] All 18 missing WASM crates published to npm
- [ ] `npx ruvector install` shows all optional packages with install status
- [ ] JSON output via `--json` flag on all commands
- [ ] Version bumped to 0.2.0 with full command hierarchy

## 11. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-065 | npm Publishing Strategy — tier-based publish order, semver, TypeScript requirements |
| ADR-070 | npx ruvector Unified Integration — brain, edge, mcp, identity commands (subset of this ADR) |
| ADR-069 | Edge-Net Integration — edge-net + brain distributed compute |
| ADR-059 | Shared Brain Google Cloud — backend that brain commands call |
| ADR-066 | SSE MCP Transport — MCP protocol for mcp commands |
