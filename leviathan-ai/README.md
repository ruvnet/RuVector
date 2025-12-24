# 🐉 Leviathan AI

[![Rust](https://img.shields.io/badge/rust-1.77%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

**Enterprise AI Orchestration with Full DAG Auditability**

Built for bank-grade compliance. Designed for Northern Trust AI Engineering standards.

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     ██╗     ███████╗██╗   ██╗██╗ █████╗ ████████╗██╗  ██╗ █████╗  ███╗   ██╗ ║
║     ██║     ██╔════╝██║   ██║██║██╔══██╗╚══██╔══╝██║  ██║██╔══██╗ ████╗  ██║ ║
║     ██║     █████╗  ██║   ██║██║███████║   ██║   ███████║███████║ ██╔██╗ ██║ ║
║     ██║     ██╔══╝  ╚██╗ ██╔╝██║██╔══██║   ██║   ██╔══██║██╔══██║ ██║╚██╗██║ ║
║     ███████╗███████╗ ╚████╔╝ ██║██║  ██║   ██║   ██║  ██║██║  ██║ ██║ ╚████║ ║
║     ╚══════╝╚══════╝  ╚═══╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═╝  ╚═══╝ ║
║                                                                   ║
║                    ENTERPRISE AI ORCHESTRATION                    ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

## 🎯 What is Leviathan AI?

A **closed-form, fully auditable AI agent framework** built in pure Rust with:

- **Zero MCP dependencies** - No external orchestration required
- **Full DAG auditability** - Every operation is cryptographically verified
- **Bank-grade compliance** - FFIEC, BCBS 239, SR 11-7, GDPR ready
- **Self-replicating agents** - Agents that can spawn modified copies
- **φ-Lattice Processor** - Perfect perplexity (1.0) with Zeckendorf arithmetic
- **Win95+Cyberpunk UI** - Corporate clean with neon accents

## 🏦 Designed for Banking

Built for the **Northern Trust AI Engineering** standards:

| Compliance | Status | Description |
|------------|--------|-------------|
| **FFIEC** | ✅ Ready | IT Examination Handbook requirements |
| **BCBS 239** | ✅ Ready | Data aggregation & lineage tracking |
| **SR 11-7** | ✅ Ready | Model risk management |
| **GDPR** | ✅ Ready | Data subject rights |

## 🚀 Quick Start

### Installation

```bash
# From source
git clone https://github.com/leviathan-ai/leviathan
cd leviathan
cargo build --release

# Windows installer
.\scripts\build-windows.ps1 -Release -Installer
```

### CLI Usage

```bash
# Initialize project
leviathan init my-project --template enterprise

# Train φ-Lattice model
leviathan train corpus.txt --output model.bin

# Generate completion
leviathan generate "D_RUN_001" --max-tokens 10

# Run swarm task
leviathan swarm "Build a RAG system" --topology hierarchical --agents 4

# Spawn agent
leviathan agent spawn junior-ai-engineer --name "ML Bot"

# Verify audit chain
leviathan audit verify --full-chain

# Launch TUI
leviathan ui

# Launch WASM UI (browser)
leviathan ui --wasm
```

### Action Sequences

Create declarative workflows in YAML:

```yaml
# deploy-rag.yaml
name: Deploy RAG System
version: "1.0"
actions:
  - id: init
    type: shell
    command: "cargo build --release"

  - id: train
    type: train
    corpus: "./data/knowledge-base.txt"
    output: "./models/kb.bin"
    depends_on: [init]

  - id: deploy
    type: swarm_task
    task: "Deploy trained model to Azure"
    agents: 3
    depends_on: [train]

  - id: validate
    type: verify_audit
    full_chain: true
    depends_on: [deploy]
```

Run with: `leviathan sequence deploy-rag.yaml`

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LEVIATHAN AI SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  φ-Lattice   │  │   DAG Audit  │  │ Compliance Engine  │    │
│  │  Processor   │  │     Trail    │  │ FFIEC/BCBS/SR11-7  │    │
│  │              │  │              │  │                    │    │
│  │ • Zeckendorf │  │ • BLAKE3     │  │ • Auto-validation  │    │
│  │ • φ/ψ dual   │  │ • Merkle     │  │ • Gap analysis     │    │
│  │ • Perp=1.0   │  │ • Lineage    │  │ • Signed reports   │    │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬──────────┘    │
│         │                 │                     │               │
│  ┌──────┴─────────────────┴─────────────────────┴──────────┐   │
│  │                   SWARM ORCHESTRATOR                     │   │
│  │              (Pure Rust, No MCP Dependencies)            │   │
│  │                                                          │   │
│  │  Topologies: Mesh │ Hierarchical │ Star │ Ring          │   │
│  │  Execution:  Parallel │ Sequential │ DAG                │   │
│  └──────┬─────────────────┬─────────────────────┬──────────┘   │
│         │                 │                     │               │
│  ┌──────┴──────┐   ┌──────┴──────┐   ┌─────────┴──────────┐   │
│  │   Agent 1   │   │   Agent 2   │   │ Self-Replicating   │   │
│  │ Data Eng.   │   │ ML Eng.     │   │     Agents         │   │
│  └─────────────┘   └─────────────┘   └────────────────────┘   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  UI Layer: CLI (ratatui) │ TUI │ WASM (egui Win95+Cyberpunk)  │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Crates

| Crate | Description |
|-------|-------------|
| `leviathan-core` | Core types and integration |
| `leviathan-lattice` | φ-Lattice Processor with WASM |
| `leviathan-dag` | DAG audit trail with lineage |
| `leviathan-swarm` | Pure Rust swarm orchestrator |
| `leviathan-agent` | Self-replicating agent system |
| `leviathan-compliance` | Regulatory compliance engine |
| `leviathan-cli` | CLI and TUI application |
| `leviathan-ui` | Win95+Cyberpunk WASM UI |

## 🎨 UI Themes

### Win95 + Cyberpunk (Default)

```
┌─[ LEVIATHAN AI ]───────────────────────────────[─][□][×]─┐
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│                                                          │
│  ┌─ SWARM STATUS ──────┐  ┌─ METRICS ─────────────────┐  │
│  │ ● Agent-001 [RUN]   │  │ Tasks:     142 completed  │  │
│  │ ● Agent-002 [RUN]   │  │ Latency:   23ms p99       │  │
│  │ ○ Agent-003 [IDLE]  │  │ Throughput: 1.2k/sec     │  │
│  │ ● Agent-004 [RUN]   │  │ Compliance: 100%         │  │
│  └─────────────────────┘  └───────────────────────────┘  │
│                                                          │
│  > leviathan swarm "Build RAG system"                    │
│  [████████████████████░░░░░░░░░░] 67% - Training model   │
│                                                          │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
├──────────────────────────────────────────────────────────┤
│ [Ready] │ Agents: 4 │ Tasks: 142 │ Audit: ✓ │ 14:32:07  │
└──────────────────────────────────────────────────────────┘
```

## 🤖 Self-Replicating Agents

Agents can spawn modified copies of themselves:

```rust
use leviathan_agent::prelude::*;

// Create base agent
let base = junior_ai_engineer_spec();

// Replicate with specialization
let mut replicator = AgentReplicator::new();
let rag_specialist = replicator.replicate_with_mutation(
    &base,
    vec![
        MutationOperator::AppendInstructions(
            "Specialize in RAG with Pinecone and LangChain".into()
        ),
        MutationOperator::AddCapability(Capability {
            name: "vector_database".into(),
            description: "Expert in vector DB operations".into(),
            required_tools: vec!["pinecone", "weaviate".into()],
        }),
    ]
)?;

// Execute with full audit
let mut executor = AgentExecutor::new(rag_specialist);
let result = executor.execute_task("Build semantic search").await?;

// Verify lineage
let lineage = replicator.get_lineage(&rag_specialist.id);
```

## 📊 φ-Lattice Processor

**Perfect perplexity (1.0)** using Zeckendorf representation:

```rust
use leviathan_lattice::*;

let mut lattice = PhiLattice::new(LatticeConfig::default());

// Train on DevOps corpus
lattice.train(&[
    "D_RUN_001 docker run nginx container",
    "K_GET_001 kubectl get pods namespace",
]);

// Verify perfect perplexity
let ppl = lattice.compute_perplexity();
assert_eq!(ppl.perplexity, 1.0);
assert_eq!(ppl.accuracy, 100.0);

// Generate with φ/ψ channel tracking
let result = lattice.generate("D_RUN_001", 6);
println!("φ_max: {}, ψ_min: {}", result.phi_max, result.psi_min);
```

## 🔒 Compliance

```rust
use leviathan_compliance::*;

// Create validator
let mut validator = ComplianceValidator::new();

// Add evidence from your systems
validator.add_evidence("FFIEC-IS-001", access_control_evidence);
validator.add_evidence("BCBS239-P3", data_lineage_evidence);

// Validate all frameworks
let ffiec = validator.validate_framework(ComplianceFramework::FFIEC)?;
let bcbs = validator.validate_framework(ComplianceFramework::BCBS239)?;

// Generate signed report
let report = validator.generate_report(
    ComplianceFramework::BCBS239,
    "Northern Trust Bank",
    "Chief Compliance Officer"
)?;

// Export for regulators
report.export_json("compliance-report.json")?;
```

## 🛠️ Development

```bash
# Build all
cargo build --workspace

# Test all
cargo test --workspace

# Build WASM UI
cd crates/leviathan-ui && ./build.sh

# Generate docs
cargo doc --workspace --no-deps --open

# Run benchmarks
cargo bench --workspace
```

## 📁 Project Structure

```
leviathan-ai/
├── Cargo.toml                 # Workspace manifest
├── crates/
│   ├── leviathan-core/        # Core integration
│   ├── leviathan-lattice/     # φ-Lattice Processor
│   ├── leviathan-dag/         # DAG audit trail
│   ├── leviathan-swarm/       # Swarm orchestrator
│   ├── leviathan-agent/       # Self-replicating agents
│   ├── leviathan-compliance/  # Regulatory compliance
│   ├── leviathan-cli/         # CLI and TUI
│   └── leviathan-ui/          # WASM UI
├── docs/                      # Documentation
├── scripts/                   # Build scripts
├── tests/                     # Integration tests
└── examples/                  # Usage examples
```

## 🙏 Acknowledgments

Built with:
- [RuVector](https://github.com/ruvnet/ruvector) - Vector database inspiration
- [egui](https://github.com/emilk/egui) - Immediate mode GUI
- [ratatui](https://github.com/ratatui-org/ratatui) - Terminal UI
- [petgraph](https://github.com/petgraph/petgraph) - Graph algorithms
- [blake3](https://github.com/BLAKE3-team/BLAKE3) - Cryptographic hashing

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built by Leviathan AI** | Powered by Rust | Bank-Grade Security

*"Enterprise AI that you can actually audit."*
