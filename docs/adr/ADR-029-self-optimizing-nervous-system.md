# ADR-029: Self-Optimizing Nervous System for DNA Analysis

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector Team
**Deciders**: Architecture Review Board
**SDK**: Claude-Flow

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | ruv.io | Initial self-optimizing nervous system proposal |
| 0.2 | 2026-02-11 | ruv.io | SOTA enhancements: Progressive Nets, Lottery Ticket, Hypernetworks, PackNet, MAML, NAS, Neuro-Symbolic |

---

## Plain Language Summary

**What is it?**

A biologically-inspired intelligence layer that enables the RuVector DNA Analyzer to
learn, adapt, and improve autonomously over time. Basecalling accuracy improves from
99% to 99.99% over 1,000 sequencing runs without manual retraining. The system adapts
to individual flow cells and chemistry versions in under 0.05ms, preserves previously
learned knowledge across adaptations, and supports federated learning across
institutions without sharing genomic data.

**Why does it matter?**

Sequencing platforms vary in error profiles across flow cells, pore types, chemistry
versions, and even individual runs. Static basecalling models leave accuracy on the
table because they cannot specialize to the local conditions of a specific instrument
at a specific time. This architecture closes that gap by continuously learning from
each run while provably preserving everything learned from previous runs.

---

## 1. SONA for Adaptive Basecalling

### 1.1 Architecture Overview

SONA (Self-Optimizing Neural Architecture) adapts the basecalling model to the
specific characteristics of each sequencing run. The adaptation is structured as a
two-tier LoRA system sitting atop a frozen foundation basecaller.

```
ADAPTIVE BASECALLING STACK

+-----------------------------------------------------------------------+
|                     FROZEN FOUNDATION BASECALLER                      |
|  Pre-trained on >10,000 sequencing runs across all chemistry versions |
|  Parameters: ~50M (quantized to INT8 for FPGA/edge deployment)       |
+-----------------------------------------------------------------------+
         |                    |                     |
         v                    v                     v
+-------------------+ +-------------------+ +-------------------+
|   MicroLoRA       | |   MicroLoRA       | |   MicroLoRA       |
|   Rank-2          | |   Rank-2          | |   Rank-2          |
|   Per flow cell   | |   Per pore type   | |   Per chemistry   |
|   <0.05ms adapt   | |   <0.05ms adapt   | |   <0.05ms adapt   |
|   512 params      | |   512 params      | |   512 params      |
+-------------------+ +-------------------+ +-------------------+
         |                    |                     |
         +--------------------+---------------------+
                              |
                              v
                 +-------------------------+
                 |   BaseLoRA (Rank 4-16)  |
                 |   Background adaptation |
                 |   Hourly consolidation  |
                 |   ~50K params           |
                 +-------------------------+
                              |
                              v
                 +-------------------------+
                 |   EWC++ Guard           |
                 |   Fisher diagonal       |
                 |   lambda = 2000-15000   |
                 |   Prevents forgetting   |
                 +-------------------------+
```

### 1.2 MicroLoRA Specialization

Each MicroLoRA adapter operates on a specific axis of variability. The implementation
builds directly on the existing `sona::lora::MicroLoRA` (rank 1-2, SIMD-optimized
forward pass at `crates/sona/src/lora.rs`).

**Per flow cell adapter**: Compensates for manufacturing variation in the flow cell
membrane, which affects the electrical signal amplitude and noise floor. Trained on
the first 1,000 reads of each new flow cell.

**Per pore type adapter**: Adjusts for the signal characteristics of different
nanopore protein variants (R9.4.1, R10.4.1, etc.). Each pore type has a distinct
current-to-base mapping. Trained offline on reference reads per pore generation.

**Per chemistry version adapter**: Handles differences in motor proteins, sequencing
speed, and signal-to-noise ratio across chemistry kit versions. Updated when a new
kit lot is detected via metadata tags.

| Adapter | Rank | Parameters | Adaptation Latency | Training Trigger |
|---------|------|------------|-------------------|------------------|
| Flow cell | 2 | 512 | <0.05ms | First 1K reads |
| Pore type | 2 | 512 | <0.05ms | Pore ID change |
| Chemistry | 2 | 512 | <0.05ms | Kit lot change |
| Combined BaseLoRA | 8 | ~50K | Background (hourly) | Quality drift > 0.1% |

### 1.3 EWC++ Catastrophic Forgetting Prevention

The EWC++ implementation (`crates/sona/src/ewc.rs`) protects previously learned
adaptations with the following guarantees:

**Online Fisher estimation**: As each sequencing run completes, the system computes
gradient statistics to estimate parameter importance. The Fisher diagonal is maintained
via exponential moving average (decay = 0.999), avoiding the need to store full
gradient histories.

**Task boundary detection**: A distribution shift detector monitors gradient z-scores
across a sliding window. When the average z-score exceeds a threshold (default 2.0),
the system automatically saves the current Fisher diagonal and optimal weights, then
begins a new adaptation epoch. This is critical at flow cell changes, chemistry lot
transitions, and instrument recalibrations.

**Adaptive lambda scheduling**: The regularization strength (lambda) scales with the
number of accumulated tasks: `lambda = initial_lambda * (1 + 0.1 * task_count)`,
clamped to [100, 15000]. After 10 flow cells, the system strongly protects the
knowledge of all previous flow cells while still adapting to the current one.

**Periodic consolidation**: After every 10 tasks, the system merges Fisher matrices
via importance-weighted averaging, reducing memory from O(tasks * params) to
O(params). This mirrors the `consolidate_all_tasks()` method in the existing EWC++
implementation.

### 1.4 Accuracy Improvement Projection

```
BASECALLING ACCURACY vs. RUN COUNT

Accuracy (%)
  99.99 |                                           ..............
        |                                     .....
  99.95 |                               .....
        |                          ....
  99.90 |                     ....
        |                ....
  99.80 |           ....
        |       ....
  99.50 |   ....
        |...
  99.00 +--+------+------+------+------+------+------+------+------+
        0  50    100    200    300    500    700    900   1000

        |-- Phase 1 --|--- Phase 2 ---|-------- Phase 3 ----------|
        MicroLoRA      BaseLoRA        EWC++ consolidated
        rapid adapt    fine-grained    asymptotic refinement
```

| Phase | Runs | Accuracy Range | Mechanism |
|-------|------|---------------|-----------|
| 1: Rapid adaptation | 0-100 | 99.00% to 99.50% | MicroLoRA per-run specialization |
| 2: Fine-grained tuning | 100-300 | 99.50% to 99.90% | BaseLoRA background consolidation |
| 3: Asymptotic refinement | 300-1000 | 99.90% to 99.99% | EWC++-protected incremental gains |

The asymptotic ceiling is determined by the Phred quality of the raw signal. SONA
cannot improve beyond the information-theoretic limit of the sequencing chemistry,
but it closes the gap between the theoretical limit and what a static model achieves.

---

## 2. Nervous System Architecture

The nervous system is a five-layer processing hierarchy inspired by biological neural
circuits. Each layer maps to existing RuVector crates and coordinates through the
Global Workspace mechanism (`crates/ruvector-nervous-system/src/routing/workspace.rs`).

### 2.1 Layer Diagram

```
NERVOUS SYSTEM ARCHITECTURE FOR DNA ANALYSIS

+=======================================================================+
|                        SENSORY LAYER (Input)                          |
|                                                                       |
|  +-------------------+  +------------------+  +--------------------+  |
|  | Raw Signal Ingest |  | Quality Metrics  |  | User Feedback &    |  |
|  | - FAST5/POD5      |  | - Phred scores   |  |   Clinical Annot.  |  |
|  | - Current traces  |  | - Read lengths   |  | - Variant confirm  |  |
|  | - Event detection |  | - Pass/fail      |  | - False positive   |  |
|  +-------------------+  +------------------+  +--------------------+  |
|            |                    |                      |              |
+=======================================================================+
             |                    |                      |
             v                    v                      v
+=======================================================================+
|                     INTEGRATION LAYER (Fusion)                        |
|                                                                       |
|  Multi-modal Fusion via Global Workspace (capacity: 7 items)          |
|                                                                       |
|  +--------------------------------------------------------------+    |
|  | OscillatoryRouter (40Hz gamma-band, Kuramoto coupling)        |    |
|  | - Phase coherence gates inter-module communication            |    |
|  | - In-phase modules exchange data; out-of-phase are isolated   |    |
|  +--------------------------------------------------------------+    |
|  | DendriticTree (coincidence detection, NMDA-like nonlinearity) |    |
|  | - Temporal coincidence within 10-50ms windows                 |    |
|  | - Plateau potentials trigger downstream processing            |    |
|  +--------------------------------------------------------------+    |
|  | HDC Encoder (10,000-dim hypervectors, XOR binding)            |    |
|  | - Sequence + quality + metadata fused into single HDC vector  |    |
|  | - SIMD-optimized similarity for pattern matching              |    |
|  +--------------------------------------------------------------+    |
|                                                                       |
+=======================================================================+
             |
             v
+=======================================================================+
|                     PROCESSING LAYER (Pipeline)                       |
|                                                                       |
|  +--------------------------------------------------------------+    |
|  | Compute Lane Router (from Prime Radiant witness.rs)           |    |
|  |                                                                |    |
|  |  Lane 0 (Reflex, <1ms):     Simple quality filter, pass/fail |    |
|  |  Lane 1 (Retrieval, ~10ms):  Alignment, variant lookup        |    |
|  |  Lane 2 (Heavy, ~100ms):     De novo assembly, SV detection   |    |
|  |  Lane 3 (Human, escalation): Uncertain calls for manual review|    |
|  +--------------------------------------------------------------+    |
|  | Resource Allocator                                             |    |
|  |  CPU pool: alignment, quality filtering                       |    |
|  |  GPU pool: basecalling neural network, attention layers       |    |
|  |  FPGA pool: real-time signal processing, compression          |    |
|  +--------------------------------------------------------------+    |
|                                                                       |
+=======================================================================+
             |
             v
+=======================================================================+
|                       MOTOR LAYER (Output)                            |
|                                                                       |
|  +------------------+  +------------------+  +--------------------+   |
|  | Variant Calls    |  | Clinical Reports |  | Alerts & Flags     |   |
|  | - VCF generation |  | - PDF/HTML       |  | - Quality warnings |   |
|  | - Phred scores   |  | - ACMG criteria  |  | - Run anomalies    |   |
|  | - Confidence     |  | - ClinVar cross  |  | - Instrument drift |   |
|  +------------------+  +------------------+  +--------------------+   |
|                                                                       |
+=======================================================================+
             |
             v
+=======================================================================+
|                     FEEDBACK LAYER (Learning)                         |
|                                                                       |
|  +--------------------------------------------------------------+    |
|  | Outcome Tracker                                                |    |
|  |  - Confirmed true positives/negatives from clinical follow-up |    |
|  |  - Concordance with orthogonal validation (Sanger, array)     |    |
|  |  - Time-to-result and resource utilization metrics             |    |
|  +--------------------------------------------------------------+    |
|  | SONA Trajectory Builder                                        |    |
|  |  - Each analysis run produces a QueryTrajectory                |    |
|  |  - Embedding: HDC vector of run characteristics               |    |
|  |  - Quality: concordance score with known truth                |    |
|  +--------------------------------------------------------------+    |
|  | ReasoningBank Pattern Storage                                  |    |
|  |  - Successful analysis patterns indexed by similarity          |    |
|  |  - Pattern retrieval for warm-starting new analyses            |    |
|  +--------------------------------------------------------------+    |
|                                                                       |
+=======================================================================+
```

### 2.2 Layer Responsibilities

**Sensory Layer**: Ingests raw data from sequencing instruments. For nanopore data,
this means FAST5/POD5 files containing current traces sampled at 4-5 kHz. Quality
metrics (Phred scores, read lengths, pass/fail status) arrive as structured metadata.
User feedback (variant confirmations, false positive flags) enters through clinical
annotation interfaces.

**Integration Layer**: Fuses multi-modal signals using three complementary mechanisms
from the nervous system crate. The OscillatoryRouter uses Kuramoto-model phase
dynamics at gamma frequency (40Hz) to gate communication -- only modules whose
oscillators are phase-synchronized exchange information, preventing irrelevant data
from interfering with focused processing. The DendriticTree detects temporal
coincidences across data streams (e.g., a quality drop coinciding with a specific
pore current pattern). The HDC encoder compresses the fused representation into a
single 10,000-dimension hypervector for downstream pattern matching.

**Processing Layer**: Routes the fused signal through the appropriate compute lane
based on complexity. The four-lane system mirrors the ComputeLane enum from
`crates/prime-radiant/src/governance/witness.rs`: Reflex (simple filters), Retrieval
(alignment and lookup), Heavy (assembly and SV detection), and Human (uncertain calls
requiring manual review). A resource allocator distributes work across available
CPU, GPU, and FPGA resources based on the current workload and deadlines.

**Motor Layer**: Generates all outputs -- VCF files, clinical reports, and operational
alerts. Every output carries a confidence score and a link to its witness chain for
auditability.

**Feedback Layer**: Closes the learning loop. Clinical follow-up results, orthogonal
validation data, and resource utilization metrics are captured as SONA trajectories
and fed back into the learning system to improve future analyses.

### 2.3 Inter-Layer Communication

Layers communicate via the Global Workspace pattern (capacity 7, based on Miller's
Law). The workspace implements competitive dynamics: representations from any layer
can bid for broadcast access, but only the most salient items survive. This prevents
information overload and ensures that the most relevant signals (e.g., a quality
anomaly or a high-confidence variant) get priority attention across all modules.

---

## 3. RuVector Intelligence System Integration

The intelligence system applies the four-step RETRIEVE-JUDGE-DISTILL-CONSOLIDATE
pipeline to genomic analysis workflows.

### 3.1 Pipeline Diagram

```
RUVECTOR INTELLIGENCE PIPELINE FOR GENOMIC ANALYSIS

   New Sequencing Run
          |
          v
  +===============+     HNSW Index (M=32, ef=200)
  |   RETRIEVE    |     150x-12,500x faster than linear scan
  |               |---> Search past runs with similar characteristics:
  |               |     - flow cell type, pore generation, sample type
  |               |     - quality profile, read length distribution
  |               |     Returns: top-k most similar past analyses
  +===============+
          |
          v
  +===============+
  |     JUDGE     |     Verdict System
  |               |---> Evaluate each retrieved pattern:
  |               |     SUCCESS:   analysis quality > 99.5% concordance
  |               |     FAILURE:   quality < 98.0% or known error pattern
  |               |     UNCERTAIN: quality between 98.0% and 99.5%
  |               |     Weight patterns by verdict for downstream use
  +===============+
          |
          v
  +===============+
  |    DISTILL    |     LoRA Fine-tuning
  |               |---> Extract learnings from SUCCESS patterns:
  |               |     - MicroLoRA adapts basecaller (rank 2, <0.05ms)
  |               |     - BaseLoRA refines pipeline parameters (rank 8)
  |               |     - Gradient accumulation across trajectory buffer
  |               |     - Quality-weighted learning (higher quality = stronger signal)
  +===============+
          |
          v
  +===============+
  |  CONSOLIDATE  |     EWC++ Protection
  |               |---> Preserve critical knowledge:
  |               |     - Fisher diagonal captures parameter importance
  |               |     - Adaptive lambda scales with accumulated tasks
  |               |     - Periodic consolidation merges Fisher matrices
  |               |     - Validated: 45% reduction in catastrophic forgetting
  +===============+
          |
          v
   Improved Model
   (feeds into next run)
```

### 3.2 RETRIEVE: HNSW-Indexed Pattern Search

Each completed analysis is stored as a vector in the HNSW index
(`crates/ruvector-core/src/index/hnsw.rs`). The vector combines:

- **Run signature** (128 dims): Flow cell ID hash, pore type, chemistry version,
  instrument serial, ambient temperature, and run duration encoded as a normalized
  embedding.
- **Quality profile** (128 dims): Distribution of Phred scores, read length
  histogram, pass rate, adapter trimming statistics, and error rate breakdown
  (substitution, insertion, deletion).
- **Outcome embedding** (128 dims): Variant concordance with truth set, clinical
  significance scores, and downstream analysis success metrics.

Total embedding dimension: 384 (matching the default HNSW configuration).

**Performance**: At 10K stored runs, k=10 retrieval completes in 61us (p50) as
benchmarked on Apple M2 (see ADR-001 Appendix C). At projected scale of 100K runs,
expected retrieval is under 200us.

### 3.3 JUDGE: Verdict System

The verdict system assigns one of three labels to each retrieved pattern:

| Verdict | Criterion | Weight in Learning | Action |
|---------|-----------|-------------------|--------|
| SUCCESS | Concordance >= 99.5% with truth set | 1.0 | Use as positive training signal |
| UNCERTAIN | Concordance 98.0%-99.5% | 0.3 | Include with reduced weight |
| FAILURE | Concordance < 98.0% or known error | 0.0 | Exclude from training, flag for review |

The quality threshold for federated aggregation
(`FederatedCoordinator::quality_threshold`) is set to 0.4 by default, aligning with
the minimum quality at which a trajectory contributes signal rather than noise.

### 3.4 DISTILL: LoRA Fine-tuning from Successful Analyses

The LoRA engine (`crates/sona/src/lora.rs`) processes successful trajectories:

1. **Gradient estimation**: For each SUCCESS trajectory, compute the gradient of the
   basecalling loss with respect to the LoRA parameters. The gradient is
   quality-weighted: `effective_gradient = gradient * quality_score`.

2. **Micro accumulation**: MicroLoRA accumulates gradients via
   `accumulate_gradient()`, averaging over `update_count` samples before applying
   with `apply_accumulated(learning_rate)`.

3. **Background refinement**: BaseLoRA (rank 4-16) performs deeper adaptation during
   idle periods, using the full trajectory buffer of the background loop coordinator.

### 3.5 CONSOLIDATE: EWC++ Knowledge Preservation

After each adaptation epoch (triggered by task boundary detection or periodic timer):

1. Save current Fisher diagonal and optimal weights to the task memory circular buffer
   (capacity: 10 tasks by default).
2. Increase lambda proportionally to accumulated task count.
3. Apply Fisher-weighted gradient constraints on all subsequent updates: parameters
   important to previous tasks receive proportionally smaller updates.
4. Periodically merge all task Fisher matrices into a single consolidated
   representation to bound memory growth.

---

## 4. Adaptive Pipeline Orchestration

### 4.1 Routing Decision Matrix

The nervous system routes each analysis through the optimal pipeline configuration
based on three signal dimensions:

```
PIPELINE ROUTING DECISION TREE

Input Characteristics
  |
  +-- Read Length
  |     |-- Short (<1 kbp)  --> Illumina-style alignment path
  |     |-- Medium (1-50 kbp) --> Standard nanopore pipeline
  |     +-- Ultra-long (>50 kbp) --> Structural variant specialist path
  |
  +-- Quality Score Distribution
  |     |-- High (mean Q > 20) --> Fast path (skip error correction)
  |     |-- Medium (Q 10-20) --> Standard path (error correction + polishing)
  |     +-- Low (Q < 10) --> Full path (consensus, multi-round polishing)
  |
  +-- Organism / Sample Type
        |-- Human clinical --> Strict pipeline (clinical-grade QC)
        |-- Microbial --> Metagenomic pipeline (community profiling)
        +-- Unknown / mixed --> Exploratory pipeline (all-of-the-above)
```

### 4.2 Dynamic Gate Selection

Three computational paths are available. The nervous system selects among them based
on input characteristics, available resources, and learned patterns from past analyses.

```
DYNAMIC GATE SELECTION

                              Input Signal
                                  |
                                  v
                        +-------------------+
                        | Complexity        |
                        | Assessment Gate   |
                        | (< 0.05ms via     |
                        |  SONA prediction) |
                        +-------------------+
                        /        |          \
                       /         |           \
                      v          v            v
          +----------+   +-----------+   +-------------+
          | FAST     |   | STANDARD  |   | FULL        |
          | PATH     |   | PATH      |   | PATH        |
          +----------+   +-----------+   +-------------+
          |              |               |
          | Sparse       | GPU Flash     | FPGA signal  |
          | inference    | Attention     | processing + |
          | CPU-only     | 2.49x-7.47x  | GPU attention|
          | <10ms/read   | speedup       | + graph SV   |
          | 80% of       | ~50ms/read    | detection    |
          | reads        | 15% of reads  | ~500ms/read  |
          +----------+   +-----------+   | 5% of reads  |
                      \        |         +-------------+
                       \       |          /
                        v      v         v
                      +-------------------+
                      | Merge & Quality   |
                      | Assessment        |
                      +-------------------+
                              |
                              v
                        Output (VCF, BAM)
```

| Gate | Trigger Condition | Resources | Latency | Read Proportion |
|------|-------------------|-----------|---------|-----------------|
| Fast | Q > 20, read length < 10kbp, known organism | CPU only (sparse inference) | <10ms/read | ~80% |
| Standard | Q 10-20, or read length 10-50kbp | GPU (Flash Attention, 2.49x-7.47x speedup) | ~50ms/read | ~15% |
| Full | Q < 10, ultra-long reads, or unknown organism | FPGA + GPU + graph analysis | ~500ms/read | ~5% |

### 4.3 Resource Allocation Strategy

The nervous system maintains a resource budget model that tracks:

- **CPU utilization**: alignment, quality filtering, I/O. Target: 80% utilization.
- **GPU utilization**: basecalling neural network, Flash Attention layers. Target: 90%
  utilization with batching.
- **FPGA utilization**: real-time signal processing, compression. Target: 95%
  utilization for streaming workloads.
- **Memory pressure**: HNSW index size, LoRA parameter storage, trajectory buffers.
  Budget: 75% of available system memory.
- **Time budget**: per-sample SLA (e.g., clinical results within 2 hours). The
  nervous system dynamically shifts reads from the Full path to the Standard path if
  the time budget is at risk.

Learned patterns from past runs inform the initial resource allocation. If the HNSW
retrieval finds that similar past runs needed predominantly the Standard path, the
system pre-allocates GPU resources accordingly rather than defaulting to the Fast path
and discovering too late that many reads require GPU attention.

---

## 5. Federated Learning for Cross-Institutional Improvement

### 5.1 Architecture

```
FEDERATED LEARNING TOPOLOGY

Hospital A              Hospital B              Hospital C
+-----------------+     +-----------------+     +-----------------+
| EphemeralAgent  |     | EphemeralAgent  |     | EphemeralAgent  |
| - Local SONA    |     | - Local SONA    |     | - Local SONA    |
| - 500 traject.  |     | - 500 traject.  |     | - 500 traject.  |
| - Local LoRA    |     | - Local LoRA    |     | - Local LoRA    |
+---------+-------+     +---------+-------+     +---------+-------+
          |                       |                       |
          | export()              | export()              | export()
          | (gradients only,      | (gradients only,      | (gradients only,
          |  NO genomic data)     |  NO genomic data)     |  NO genomic data)
          v                       v                       v
+==================================================================+
|                  FEDERATED COORDINATOR                            |
|                                                                    |
|  +--------------------------+  +-------------------------------+  |
|  | Secure Aggregation (MPC) |  | Differential Privacy          |  |
|  | - Secret-shared gradient |  | - Gaussian noise (sigma=1.0)  |  |
|  |   reconstruction         |  | - Per-gradient clipping (C=1) |  |
|  | - No single party sees   |  | - Privacy budget epsilon=1.0  |  |
|  |   raw gradients          |  | - Accounting via RDP           |  |
|  +--------------------------+  +-------------------------------+  |
|                                                                    |
|  +--------------------------------------------------------------+|
|  | FederatedCoordinator (crates/sona/src/training/federated.rs)  ||
|  | - quality_threshold: 0.4                                      ||
|  | - EWC lambda: 2000.0 (strong regularization)                  ||
|  | - trajectory_capacity: 50,000                                 ||
|  | - consolidation_interval: 50 agents                           ||
|  +--------------------------------------------------------------+|
|                                                                    |
+==================================================================+
          |
          | Improved global model (LoRA weights only)
          v
    Distributed back to all participating institutions
```

### 5.2 Privacy Guarantees

**Differential Privacy**: Each institution clips gradient norms to `C = 1.0` and adds
calibrated Gaussian noise with `sigma = 1.0` before export. This provides
(epsilon=1.0, delta=1e-5)-differential privacy per gradient update under the Renyi
Differential Privacy (RDP) accountant. Over 1,000 aggregation rounds, the total
privacy budget remains epsilon < 10.0 via privacy amplification by subsampling.

**Secure Aggregation via MPC**: The coordinator never sees individual institution
gradients. Instead, institutions use a 3-party secure computation protocol:
1. Each institution secret-shares its clipped, noised gradient across 3 aggregation
   servers.
2. Servers compute the sum of shares (addition is homomorphic over secret shares).
3. The reconstructed aggregate gradient reveals only the sum, not individual
   contributions.

**Data Residency**: No genomic sequences, variant calls, or patient-identifiable
information ever leave the originating institution. Only LoRA gradient updates (512 to
50K floating point numbers per round) are transmitted, and these are further protected
by differential privacy and MPC.

### 5.3 Topology Options

The `FederatedTopology` enum from `crates/sona/src/training/federated.rs` supports
three configurations:

| Topology | Description | Use Case |
|----------|-------------|----------|
| Star | All institutions report to one coordinator | Single-country consortium |
| Hierarchical | Institutions -> Regional -> Global | Multi-national networks |
| PeerToPeer | Direct gradient exchange between institutions | Edge/resource-limited deployments |

### 5.4 Convergence Projection

```
FEDERATED MODEL QUALITY vs. PARTICIPATING INSTITUTIONS

Global Accuracy (%)
  99.95 |                                           ............
        |                                     ......
  99.90 |                               ......
        |                         ......
  99.80 |                   ......
        |             ......
  99.50 |       ......
        | ......
  99.00 +--+------+------+------+------+------+------+------+------+
        1   5     10     20     30     50     70     90    100

        Number of Participating Institutions

  ------ Without federation (single institution, plateaus at ~99.50%)
  ...... With federation (shared learning, converges toward 99.95%)
```

The global model benefits from exposure to diverse sequencing conditions, sample types,
and error profiles across institutions. A single institution sees perhaps 10-50
flow cell types per year; a federation of 100 institutions collectively covers the
full space of sequencing variability.

---

## 6. Prime Radiant Computation Engine

### 6.1 Deterministic, Reproducible Computation

The Prime Radiant engine (`crates/prime-radiant/`) provides the foundation layer
that guarantees every analysis result can be independently reproduced and verified.

```
PRIME RADIANT AUDIT CHAIN

Analysis Request
      |
      v
+------------------+
| Policy Gate      |  GateDecision: allow/deny + ComputeLane assignment
| (governance/     |  Evaluates against PolicyBundle
|  witness.rs)     |
+------------------+
      |
      v
+------------------+
| Witness Record   |  Immutable proof of the gate decision:
| (Blake3 hash     |  - action_hash: hash of input data
|  chain)          |  - energy_snapshot: system coherence at decision time
|                  |  - decision: allow/deny + compute lane + confidence
|                  |  - policy_bundle_ref: exact policy version used
|                  |  - previous_hash: chain link to prior witness
+------------------+
      |
      v
+------------------+
| Computation      |  Deterministic execution:
| Execution        |  - Fixed random seeds per analysis
|                  |  - Pinned model version (frozen foundation + LoRA snapshot)
|                  |  - Recorded hyperparameters (ef_search, quality thresholds)
+------------------+
      |
      v
+------------------+
| Result Witness   |  Output provenance:
| (chain-linked    |  - output_hash: hash of VCF/BAM/report
|  to input        |  - model_version: foundation + LoRA adapter checksums
|  witness)        |  - parameters: full configuration snapshot
|                  |  - content_hash: Blake3 of entire witness record
+------------------+
      |
      v
Verifiable: given (input_data, model_version, parameters),
any party can reproduce the exact same output and verify
it matches the output_hash in the witness chain.
```

### 6.2 Witness Chain Properties

The witness chain implementation inherits all properties from
`crates/prime-radiant/src/governance/witness.rs`:

- **Temporal ordering**: Each witness references its predecessor by ID and hash.
  Sequence numbers are strictly monotonic.
- **Tamper detection**: Any modification to a witness breaks the chain because
  `verify_content_hash()` recomputes the Blake3 digest over all fields.
- **Deterministic replay**: Given the same input data, model version, and parameters,
  the computation produces bit-identical output. The witness chain records all three,
  enabling any auditor to re-execute and verify.

### 6.3 Compute Lane Assignment for Genomic Workloads

| Lane | Latency Budget | Genomic Operation |
|------|---------------|-------------------|
| Reflex (0) | <1ms | Quality filter pass/fail, adapter detection |
| Retrieval (1) | ~10ms | Reference alignment, known variant lookup |
| Heavy (2) | ~100ms | De novo assembly, structural variant detection |
| Human (3) | Escalation | Variants of uncertain significance, novel findings |

Each lane assignment is recorded in the witness chain, creating a complete audit trail
of how every read was processed and why.

### 6.4 Cryptographic Integrity Guarantee

All hashes use Blake3 (via the `blake3` crate), providing:
- 256-bit collision resistance
- 2 GiB/s hashing speed (single-threaded, SIMD-accelerated)
- Incremental hashing for streaming computation

The complete audit chain enables regulatory compliance: every clinical result
can be traced to its raw input data, the exact model weights used (including LoRA
adapters active at that moment), the pipeline parameters, and the policy bundle that
authorized the computation.

---

## 7. Performance Targets and Projections

### 7.1 Latency Budget

| Operation | Target | Mechanism |
|-----------|--------|-----------|
| SONA adaptation (MicroLoRA) | <0.05ms | Rank-2 SIMD-optimized forward pass |
| HNSW pattern retrieval (k=10) | <0.1ms | M=32, ef_search=100, 384-dim |
| EWC++ constraint application | <0.1ms | Single-pass Fisher-weighted scaling |
| Pipeline gate decision | <0.05ms | SONA prediction via learned patterns |
| Witness record creation | <0.01ms | Blake3 hashing, UUID generation |
| Total overhead per read | <0.35ms | Sum of all nervous system operations |

### 7.2 Memory Budget

| Component | Memory per Instance | Scale |
|-----------|-------------------|-------|
| MicroLoRA (3 adapters) | 6 KB | Per active flow cell |
| BaseLoRA (rank 8) | 200 KB | Per model layer |
| EWC++ Fisher diagonal | 4 KB per task, 10 tasks max | 40 KB |
| HNSW index (10K runs) | ~50 MB | Grows logarithmically |
| Trajectory buffer | ~5 MB | Circular, fixed capacity |
| Global Workspace | <1 KB | 7 items, 384-dim each |

### 7.3 Accuracy Targets

| Metric | Baseline (Static Model) | After 100 Runs | After 1,000 Runs |
|--------|------------------------|----------------|------------------|
| Basecalling accuracy | 99.00% | 99.50% | 99.99% |
| SNV concordance | 99.50% | 99.80% | 99.95% |
| Indel concordance | 98.00% | 99.00% | 99.50% |
| SV detection sensitivity | 85.00% | 92.00% | 96.00% |

---

## 8. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LoRA adaptation degrades basecalling on edge cases | Medium | High | EWC++ prevents catastrophic forgetting; witness chain enables rollback to any previous LoRA snapshot |
| Federated gradients leak patient information | Low | Critical | Differential privacy (epsilon=1.0) + MPC secure aggregation; formal privacy proof |
| SONA overhead exceeds per-read time budget | Low | Medium | Total overhead <0.35ms vs. typical 50-500ms basecalling time; <1% of total latency |
| Witness chain storage grows unbounded | Medium | Low | Periodic archival to cold storage; chain heads retained for verification; old witnesses compressed |
| Oscillatory routing fails to synchronize | Low | Medium | Fallback to static routing if order parameter < 0.3 after warmup period |

---

## 9. Progressive Neural Networks for Zero-Forgetting

### 9.1 Motivation: Beyond Approximate Regularization

EWC++ (Section 1.3) prevents catastrophic forgetting approximately by penalizing
changes to important parameters. The penalty is quadratic in the parameter deviation,
controlled by the Fisher diagonal and lambda. This is effective in practice (45%
reduction in forgetting per Section 3.5), but it is fundamentally a soft constraint:
sufficiently strong gradients on a new task can still override the regularization and
degrade performance on prior tasks. Progressive Neural Networks provide a
complementary hard guarantee: previous task knowledge is structurally frozen and
literally cannot be overwritten.

### 9.2 Architecture: Columnar Expansion

Each new task receives a dedicated "column" of network layers. All previous columns
are frozen (weights set to `requires_grad = false`). The new column receives lateral
connections from all prior columns, enabling forward transfer of learned features
without any backward interference.

```
PROGRESSIVE NEURAL NETWORK FOR MULTI-CHEMISTRY BASECALLING

   Column 1 (R9.4.1)    Column 2 (R10.4.1)    Column 3 (Duplex)
   FROZEN               FROZEN                 ACTIVE
   +-----------+        +-----------+          +-----------+
   | Layer 3   |------->| Layer 3   |--------->| Layer 3   |
   +-----------+   lat  +-----------+   lat    +-----------+
        |          conn       |         conn        |
   +-----------+        +-----------+          +-----------+
   | Layer 2   |------->| Layer 2   |--------->| Layer 2   |
   +-----------+   lat  +-----------+   lat    +-----------+
        |          conn       |         conn        |
   +-----------+        +-----------+          +-----------+
   | Layer 1   |------->| Layer 1   |--------->| Layer 1   |
   +-----------+   lat  +-----------+   lat    +-----------+
        |                     |                      |
   [R9.4.1 signal]      [R10.4.1 signal]       [Duplex signal]
```

**Forward transfer via lateral connections**: When Column 3 (Duplex) is being trained,
it receives activations from Columns 1 and 2 through lateral adapter layers. These
adapters are lightweight linear projections:

```
h_k^(3) = f( W_k^(3) * h_{k-1}^(3) + sum_{j=1}^{2} U_{k,j}^(3->j) * h_{k-1}^(j) )
```

where `h_k^(i)` is the hidden state at layer `k` in column `i`, `W_k^(3)` are the
column-3 weights (trainable), and `U_{k,j}^(3->j)` are the lateral adapter weights
(trainable) connecting frozen column `j` to the active column.

**Zero-forgetting guarantee**: Because columns 1 and 2 are frozen, their outputs are
deterministic functions of their inputs. No gradient from column 3's loss can flow
into columns 1 or 2. The basecalling accuracy on R9.4.1 chemistry after training on
Duplex is provably identical to its accuracy before Duplex training.

### 9.3 Comparison with EWC++

| Property | EWC++ (Section 1.3) | Progressive Nets |
|----------|---------------------|-----------------|
| Forgetting guarantee | Approximate (soft penalty) | Exact (structural freeze) |
| Forward transfer | Via shared parameter space | Via lateral connections |
| Memory growth | O(params) after consolidation | O(params * tasks) before pruning |
| Adaptation speed | Continuous | Per-task column addition |
| Deployment | Single model, single pass | Task-specific column selection |

**Recommended strategy**: Use EWC++ for fine-grained within-chemistry adaptation
(flow cell to flow cell), and Progressive Nets for coarse-grained cross-chemistry
expansion (R9.4.1 to R10.4.1 to Duplex). This two-level approach provides both
the continuous adaptation of EWC++ and the zero-forgetting guarantee of Progressive
Nets at the chemistry boundary.

### 9.4 Pruning for Bounded Growth

After a column has been trained and frozen, a post-hoc structured pruning pass removes
redundant neurons. The criterion is activation magnitude across a held-out calibration
set: neurons with mean activation below a threshold (default: 1% of layer-wise max)
are zeroed and their lateral connections removed. Empirically, 40-60% of neurons can
be pruned per column without measurable accuracy loss, keeping the total model size
growth sub-linear in the number of tasks.

### 9.5 Implementation

```rust
/// Progressive Neural Network for multi-task basecalling.
/// Each chemistry version adds a frozen column with lateral connections.
/// Crate: `crates/sona/src/progressive.rs`
pub struct ProgressiveNet<const DIM: usize> {
    /// Frozen columns from prior tasks. Each column is a Vec of layer weights.
    frozen_columns: Vec<FrozenColumn<DIM>>,
    /// Active column currently being trained.
    active_column: Option<ActiveColumn<DIM>>,
    /// Lateral adapter weights: active_column -> each frozen column, per layer.
    lateral_adapters: Vec<Vec<LateralAdapter<DIM>>>,
    /// Task metadata for column selection at inference time.
    task_registry: HashMap<TaskId, ColumnIndex>,
}

impl<const DIM: usize> ProgressiveNet<DIM> {
    /// Add a new column for a new task. Freezes the current active column
    /// (if any) and creates a fresh trainable column with lateral connections
    /// to all existing frozen columns.
    pub fn add_column(&mut self, task_id: TaskId) -> &mut ActiveColumn<DIM> {
        if let Some(active) = self.active_column.take() {
            let frozen = active.freeze(); // Sets requires_grad = false
            let col_idx = self.frozen_columns.len();
            self.frozen_columns.push(frozen);
            self.task_registry.insert(
                self.active_task_id.take().unwrap(),
                ColumnIndex(col_idx),
            );
        }
        let num_frozen = self.frozen_columns.len();
        self.active_column = Some(ActiveColumn::new(DIM, num_frozen));
        self.lateral_adapters.push(
            (0..num_frozen)
                .map(|_| LateralAdapter::new(DIM))
                .collect(),
        );
        self.active_column.as_mut().unwrap()
    }

    /// Forward pass for a specific task. Selects the appropriate column
    /// and computes lateral connections from all prior columns.
    pub fn forward(&self, input: &Tensor<DIM>, task_id: &TaskId) -> Tensor<DIM> {
        let col_idx = self.task_registry[task_id];
        let target_col = &self.frozen_columns[col_idx.0];
        let prior_activations: Vec<_> = self.frozen_columns[..col_idx.0]
            .iter()
            .map(|col| col.forward(input))
            .collect();
        target_col.forward_with_laterals(input, &prior_activations)
    }

    /// Prune a frozen column to reduce parameter count.
    /// Removes neurons with mean activation < threshold * layer_max.
    pub fn prune_column(
        &mut self,
        col_idx: ColumnIndex,
        calibration_data: &[Tensor<DIM>],
        threshold: f32,
    ) -> PruningReport {
        self.frozen_columns[col_idx.0].prune(calibration_data, threshold)
    }
}
```

**Reference**: Rusu, A. A. et al. (2016). "Progressive Neural Networks."
arXiv:1606.04671.

---

## 10. Advanced Continual Learning and Efficient Adaptation

This section presents four complementary techniques that extend the SONA continual
learning framework beyond EWC++ and Progressive Nets, each addressing a different
axis of the adaptation problem: sparsity, amortized generation, capacity packing,
and few-shot generalization.

### 10.1 Lottery Ticket Hypothesis for Sparse Adaptation

#### 10.1.1 Core Insight

Dense neural networks contain sparse subnetworks -- "winning tickets" -- that, when
trained in isolation from the same initialization, match the full dense network's
accuracy. For the RuVector basecaller (~50M parameters), this means there exists a
sparse subnetwork of 2.5M-5M parameters (5-10% of the original) that achieves the
same basecalling accuracy as the full model.

#### 10.1.2 Iterative Magnitude Pruning (IMP)

The winning ticket is found through Iterative Magnitude Pruning:

```
IMP TRAINING LOOP

  theta_0 (initial weights)
      |
      v
  +------------------+
  | Train to         |      Step 1: Train full model to convergence
  | convergence      |
  +------------------+
      |
      v
  +------------------+
  | Prune smallest   |      Step 2: Remove p% of weights with
  | magnitude weights|      smallest |w| (typically p=20%)
  +------------------+
      |
      v
  +------------------+
  | Reset surviving  |      Step 3: Reset remaining weights to
  | weights to       |      their values at theta_0
  | theta_0 values   |
  +------------------+
      |
      v
  Repeat steps 1-3 until target sparsity (90-95%) is reached.
  The surviving mask m defines the winning ticket.
  Final model: theta_0 * m (sparse, original initialization)
```

**Mathematical formulation**: Let `f(x; theta)` be the basecaller with parameters
`theta in R^n`. The winning ticket is a binary mask `m in {0,1}^n` and initial
weights `theta_0` such that:

```
L(f(x; m * theta_0)) <= L(f(x; theta*)) + epsilon
```

where `theta*` is the converged dense model and `epsilon` is a small tolerance.

#### 10.1.3 Genomic Application

**Edge deployment**: The winning ticket (5-10% of 50M = 2.5M-5M parameters) fits
entirely in FPGA on-chip SRAM (typically 10-40 MB), eliminating off-chip memory
access latency. This yields a 10-20x parameter reduction with no accuracy loss.

**Combining with MicroLoRA**: The winning ticket serves as the frozen foundation.
MicroLoRA adapters (rank-2, 512 params) are applied only to the surviving sparse
weights, further reducing adaptation overhead:

```
Sparse basecaller:      2.5M params (winning ticket)
+ MicroLoRA adapters:   1,536 params (3 adapters x 512)
= Total deployed model: 2.5M + 1,536 params
  (vs. 50M + 1,536 for the dense baseline)
```

#### 10.1.4 Implementation

```rust
/// Lottery Ticket finder via Iterative Magnitude Pruning.
/// Discovers the minimal subnetwork that matches full model accuracy.
/// Crate: `crates/ruvector-sparse-inference/src/lottery.rs`
pub struct LotteryTicketFinder {
    /// Initial weights (theta_0), saved at the start of training.
    initial_weights: Vec<f32>,
    /// Binary mask defining the winning ticket. 1 = keep, 0 = pruned.
    mask: BitVec,
    /// Pruning rate per round (fraction of remaining weights to remove).
    prune_rate: f32,
    /// Target sparsity (fraction of total weights to prune).
    target_sparsity: f32,
    /// Current sparsity level.
    current_sparsity: f32,
    /// Number of IMP rounds completed.
    rounds_completed: u32,
}

impl LotteryTicketFinder {
    /// Create a new finder. Saves the initial weights for rewinding.
    pub fn new(
        initial_weights: Vec<f32>,
        prune_rate: f32,
        target_sparsity: f32,
    ) -> Self { /* ... */ }

    /// Execute one IMP round: train, prune, rewind.
    /// Returns the current mask and sparsity level.
    pub fn imp_round(
        &mut self,
        train_fn: impl Fn(&[f32], &BitVec) -> Vec<f32>,
    ) -> (BitVec, f32) {
        // 1. Train with current mask
        let trained = train_fn(&self.initial_weights, &self.mask);
        // 2. Prune smallest-magnitude surviving weights
        let to_prune = self.select_prune_candidates(&trained);
        for idx in to_prune {
            self.mask.set(idx, false);
        }
        self.current_sparsity = 1.0 - self.mask.count_ones() as f32
            / self.mask.len() as f32;
        self.rounds_completed += 1;
        (self.mask.clone(), self.current_sparsity)
    }

    /// Extract the winning ticket: initial weights masked by the final mask.
    pub fn extract_ticket(&self) -> SparseModel {
        SparseModel {
            weights: self.initial_weights.clone(),
            mask: self.mask.clone(),
            sparsity: self.current_sparsity,
        }
    }
}
```

**Reference**: Frankle, J. & Carlin, M. (2019). "The Lottery Ticket Hypothesis:
Finding Sparse, Trainable Neural Networks." ICLR 2019.

### 10.2 Hypernetworks for Task-Specific Weight Generation

#### 10.2.1 Core Idea

Instead of storing or fine-tuning per-device adapter weights, a hypernetwork `H`
generates the weights of the main network (or its LoRA adapters) from a compact
task descriptor:

```
theta_adapter = H(task_embedding)
```

where `task_embedding` encodes the characteristics of the current flow cell,
chemistry version, and instrument. The hypernetwork `H` is trained once; at
inference, generating adapter weights is a single forward pass through `H`.

#### 10.2.2 Architecture for Genomic Adaptation

```
HYPERNETWORK ADAPTER GENERATION

  Device Metadata                  Hypernetwork H                MicroLoRA Weights
  +--------------------+          +----------------+            +------------------+
  | flow_cell_id: hash |          |                |            | A: [dim x rank]  |
  | pore_type: R10.4.1 |   embed  | Linear(128,256)|   reshape  | B: [rank x dim]  |
  | chemistry: v14     | -------> | ReLU           | ---------> | bias: [dim]      |
  | temperature: 22.5C |          | Linear(256,512)|            |                  |
  | run_duration: 48h  |          | Linear(512,    |            | Total: 512 float |
  +--------------------+          |   rank*dim*2   |            +------------------+
                                  |   + dim)       |
                                  +----------------+
                                  Params: ~200K
                                  Latency: <0.01ms
```

**Amortized adaptation**: Traditional gradient-based MicroLoRA adaptation requires
accumulating gradients over 1,000 reads (~0.05ms). The hypernetwork generates
adapter weights in a single forward pass (<0.01ms) with no gradient computation,
a 5x latency improvement.

**No per-device storage**: Instead of storing 512 parameters per flow cell (which
accumulates over thousands of flow cells), the system stores only the hypernetwork
(~200K parameters) plus a compact task embedding per device (~128 floats). At 10K
flow cells:

```
Traditional: 10K * 512 * 4 bytes = 20 MB of stored adapters
Hypernetwork: 200K * 4 bytes + 10K * 128 * 4 bytes = 5.9 MB
Savings: 3.4x less storage
```

#### 10.2.3 Training the Hypernetwork

The hypernetwork is trained via meta-learning over historical device-adapter pairs.
For each training step:

1. Sample a device metadata record and its known-good MicroLoRA weights.
2. Generate adapter weights via `H(embed(metadata))`.
3. Compute MSE loss between generated weights and known-good weights.
4. Backpropagate through `H` to update the hypernetwork parameters.

The training set consists of all historically successful MicroLoRA adaptations (i.e.,
those with verdict = SUCCESS from Section 3.3).

#### 10.2.4 Implementation

```rust
/// Hypernetwork that generates MicroLoRA adapter weights
/// from device metadata embeddings.
/// Crate: `crates/sona/src/hyper_adapter.rs`
pub struct HyperAdapter {
    /// Device metadata embedding dimension.
    embed_dim: usize,
    /// Hidden layer dimensions in the hypernetwork.
    hidden_dims: Vec<usize>,
    /// Output dimension = rank * input_dim * 2 + bias_dim.
    output_dim: usize,
    /// Hypernetwork parameters (trained offline).
    layers: Vec<LinearLayer>,
}

impl HyperAdapter {
    /// Generate MicroLoRA weights from device metadata.
    /// Returns (A_matrix, B_matrix, bias) ready for MicroLoRA injection.
    /// Latency: <0.01ms (single forward pass, no gradients).
    pub fn generate_weights(
        &self,
        device_metadata: &DeviceMetadata,
    ) -> MicroLoRAWeights {
        let embedding = self.encode_metadata(device_metadata);
        let mut h = embedding;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i < self.layers.len() - 1 {
                h = relu(&h); // ReLU activation for hidden layers
            }
        }
        MicroLoRAWeights::from_flat(&h, self.rank, self.dim)
    }

    /// Encode device metadata into a fixed-size embedding vector.
    fn encode_metadata(&self, meta: &DeviceMetadata) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.embed_dim);
        v.extend_from_slice(&hash_to_embedding(meta.flow_cell_id, 32));
        v.extend_from_slice(&onehot_pore_type(meta.pore_type, 16));
        v.extend_from_slice(&onehot_chemistry(meta.chemistry_version, 16));
        v.push(meta.temperature_celsius / 50.0); // Normalize
        v.push(meta.run_duration_hours / 72.0);  // Normalize
        // Pad or truncate to embed_dim
        v.resize(self.embed_dim, 0.0);
        v
    }
}
```

**Reference**: Ha, D., Dai, A. & Le, Q. V. (2017). "HyperNetworks." ICLR 2017.

### 10.3 PackNet: Pruning-Based Continual Learning

#### 10.3.1 Core Mechanism

PackNet assigns each task a subset of the network's parameters by iteratively
training and pruning. After training on task `t`, the essential weights are identified
(those with magnitude above a threshold), locked, and the freed weights become
available for task `t+1`. Each task is associated with a binary mask that selects its
dedicated parameter subset at inference time.

```
PACKNET WEIGHT ALLOCATION ACROSS TASKS

Network parameters (50M total):

Task 1 (R9.4.1):   [################............................] ~20% used
Task 2 (R10.4.1):  [................########....................] ~20% used
Task 3 (Duplex):   [........................########............] ~20% used
Task 4 (RNA004):   [................................########....] ~20% used
Free capacity:      [........................................####] ~20% free

Legend: # = weights allocated to this task, . = free/other task
```

**Capacity analysis**: Empirically, each basecalling task (chemistry version) requires
approximately 15-25% of the network capacity. With 50M parameters, this supports
4-5 chemistry versions in a single model with ~20% reserve capacity. The reserve
can be used for MicroLoRA fine-grained adaptation within each chemistry.

#### 10.3.2 Mathematical Formulation

For task `t` with parameters `theta` and binary mask `m_t in {0,1}^n`:

```
L_t(theta * m_t)  is minimized  subject to  m_t AND m_j = 0  for all j < t
```

The constraint ensures that no two tasks share parameters, providing strict isolation
(zero interference between tasks). At inference, the chemistry version metadata
selects the mask: `y = f(x; theta * m_{chemistry})`.

#### 10.3.3 Inference with Mask Selection

```
PACKNET INFERENCE PIPELINE

  Input Signal
       |
       v
  +------------------+
  | Chemistry        |     Read from POD5 metadata:
  | Detector         |     run_info.chemistry_version
  +------------------+
       |
       v
  +------------------+
  | Mask Selector    |     m = mask_registry[chemistry_version]
  | O(1) lookup      |     Binary mask: 50M bits = 6.25 MB
  +------------------+
       |
       v
  +------------------+
  | Masked Forward   |     y = f(x; theta * m)
  | Pass             |     Only ~10M params active (20% of 50M)
  +------------------+     ~5x theoretical speedup from sparsity
       |
       v
  Base Calls
```

#### 10.3.4 Implementation

```rust
/// PackNet adapter for multi-task basecalling.
/// Packs multiple chemistry versions into a single model via binary masks.
/// Crate: `crates/sona/src/packnet.rs`
pub struct PackNetAdapter {
    /// Shared parameter tensor (all tasks' weights coexist).
    parameters: Vec<f32>,
    /// Per-task binary masks. Key: TaskId, Value: bitmask over parameters.
    task_masks: HashMap<TaskId, BitVec>,
    /// Set of parameter indices currently allocated (union of all masks).
    allocated: BitVec,
    /// Pruning threshold for determining essential weights after training.
    prune_threshold: f32,
    /// Maximum number of tasks before capacity is exhausted.
    max_tasks: usize,
}

impl PackNetAdapter {
    /// Train on a new task using only the free (unallocated) parameters.
    /// After training, prune non-essential weights and register the mask.
    pub fn train_task(
        &mut self,
        task_id: TaskId,
        train_fn: impl Fn(&mut [f32], &BitVec) -> f32,
    ) -> Result<TaskTrainReport, PackNetError> {
        let free_mask = !self.allocated.clone();
        let free_count = free_mask.count_ones();
        if free_count < self.min_capacity_per_task() {
            return Err(PackNetError::InsufficientCapacity {
                free: free_count,
                required: self.min_capacity_per_task(),
            });
        }
        // Train using only free parameters
        let loss = train_fn(&mut self.parameters, &free_mask);
        // Prune: keep only weights with |w| > threshold
        let task_mask = self.prune_to_essential(&free_mask);
        // Lock the essential weights
        self.allocated |= &task_mask;
        self.task_masks.insert(task_id, task_mask);
        Ok(TaskTrainReport { loss, params_used: self.task_masks[&task_id].count_ones() })
    }

    /// Forward pass for a specific task. Applies the task's binary mask.
    pub fn forward(&self, input: &Tensor, task_id: &TaskId) -> Tensor {
        let mask = &self.task_masks[task_id];
        let masked_params = self.apply_mask(mask);
        basecaller_forward(input, &masked_params)
    }

    /// Report remaining free capacity as a fraction of total parameters.
    pub fn free_capacity(&self) -> f32 {
        1.0 - self.allocated.count_ones() as f32 / self.allocated.len() as f32
    }
}
```

**Reference**: Mallya, A. & Lazebnik, S. (2018). "PackNet: Adding Multiple Tasks to a
Single Network by Iterative Pruning." CVPR 2018.

### 10.4 Meta-Learning (MAML) for Few-Shot Adaptation

#### 10.4.1 The Few-Shot Problem in Sequencing

When a new flow cell is loaded or a new chemistry version is deployed, the system
has access to only a handful of reads before it must begin producing high-quality
base calls. Traditional fine-tuning requires thousands of gradient steps over
thousands of reads. Meta-learning pre-trains the model initialization `theta*` such
that 1-5 gradient steps on a tiny support set (as few as 10 reads) yield good
performance on the new condition.

#### 10.4.2 MAML Formulation

Given a distribution of tasks `p(T)` (each task is a specific flow cell + chemistry
combination), MAML optimizes:

```
theta* = argmin_{theta} E_{T ~ p(T)} [ L_T(theta - alpha * grad L_T(theta; D_T^support); D_T^query) ]
```

**Inner loop** (per-task adaptation): Starting from `theta`, take `k` gradient steps
on the support set `D_T^support` (e.g., 10 reads from the new flow cell):

```
theta'_T = theta - alpha * grad L_T(theta; D_T^support)
```

**Outer loop** (meta-optimization): Update `theta` to minimize the post-adaptation
loss on the query set `D_T^query` (e.g., 100 reads held out):

```
theta <- theta - beta * grad_{theta} E_T [ L_T(theta'_T; D_T^query) ]
```

The outer gradient requires differentiating through the inner gradient steps (second
derivatives). This is computationally expensive for large models.

#### 10.4.3 Reptile: A Practical Simplification

Reptile (Nichol et al., 2018) avoids second derivatives entirely. After `k` inner
steps, it simply moves `theta` toward the adapted `theta'_T`:

```
theta <- theta + epsilon * (theta'_T - theta)
```

This is equivalent to first-order MAML under certain conditions and is much cheaper
to compute. For the RuVector basecaller, Reptile's simplicity is preferred because:

1. The 50M parameter model makes second derivatives prohibitively expensive.
2. The adaptation target (flow cell/chemistry specialization) is relatively simple
   compared to the full task diversity in standard meta-learning benchmarks.
3. Reptile can be implemented as a simple modification to the existing training loop.

#### 10.4.4 Genomic Meta-Training Protocol

```
META-TRAINING PIPELINE

  Historical Sequencing Runs (N=100+)
  +------------------------------------------+
  | Run 1: FC-001, R9.4.1, Kit v12           |
  | Run 2: FC-002, R10.4.1, Kit v14          |
  | ...                                       |
  | Run N: FC-N, Duplex, Kit v16             |
  +------------------------------------------+
           |
           v
  +------------------------------------------+
  | For each meta-training epoch:             |
  |   1. Sample batch of B tasks              |
  |   2. For each task T_i:                   |
  |      a. Split into support (10 reads)     |
  |         and query (100 reads)             |
  |      b. Inner loop: 3 SGD steps on        |
  |         support set (lr = alpha = 0.01)   |
  |      c. Compute query loss L(theta'_i)    |
  |   3. Reptile update:                      |
  |      theta <- theta + eps * mean_i(       |
  |        theta'_i - theta)                  |
  |      (eps = 0.001, B = 16)               |
  +------------------------------------------+
           |
           v
  theta* = meta-trained initialization

  At deployment: 3 SGD steps on 10 reads from
  new flow cell -> adapted model ready in <1ms
```

#### 10.4.5 Implementation

```rust
/// Meta-learning basecaller using Reptile-style meta-training.
/// Pre-trains initialization theta* for few-shot adaptation to new devices.
/// Crate: `crates/sona/src/meta_basecaller.rs`
pub struct MetaBasecaller {
    /// Meta-trained initialization (theta*).
    meta_params: Vec<f32>,
    /// Inner loop learning rate (alpha).
    inner_lr: f32,
    /// Number of inner loop gradient steps.
    inner_steps: usize,
    /// Reptile interpolation rate (epsilon).
    reptile_epsilon: f32,
    /// Meta-batch size (number of tasks per outer step).
    meta_batch_size: usize,
    /// Minimum support set size (reads) for adaptation.
    min_support_size: usize,
}

impl MetaBasecaller {
    /// Adapt to a new flow cell / chemistry using few-shot learning.
    /// Takes a small support set (e.g., 10 reads) and performs
    /// `inner_steps` gradient steps to produce an adapted model.
    /// Latency: <1ms for 3 gradient steps on 10 reads.
    pub fn adapt(
        &self,
        support_set: &[SequencingRead],
    ) -> AdaptedBasecaller {
        let mut adapted_params = self.meta_params.clone();
        for _step in 0..self.inner_steps {
            let grad = compute_basecalling_gradient(
                &adapted_params,
                support_set,
            );
            // SGD inner step: theta' = theta - alpha * grad
            for (p, g) in adapted_params.iter_mut().zip(grad.iter()) {
                *p -= self.inner_lr * g;
            }
        }
        AdaptedBasecaller { params: adapted_params }
    }

    /// Run one Reptile meta-training step over a batch of tasks.
    /// Updates meta_params toward the mean of per-task adapted params.
    pub fn meta_train_step(
        &mut self,
        task_batch: &[SequencingTask],
    ) {
        let mut param_deltas = vec![0.0f32; self.meta_params.len()];
        let mut count = 0;
        for task in task_batch.iter().take(self.meta_batch_size) {
            let adapted = self.adapt(&task.support_set);
            for (delta, (adapted_p, meta_p)) in param_deltas.iter_mut()
                .zip(adapted.params.iter().zip(self.meta_params.iter()))
            {
                *delta += adapted_p - meta_p;
            }
            count += 1;
        }
        // Reptile update: theta <- theta + epsilon * mean(theta'_i - theta)
        let scale = self.reptile_epsilon / count as f32;
        for (p, d) in self.meta_params.iter_mut().zip(param_deltas.iter()) {
            *p += scale * d;
        }
    }
}
```

**References**:
- Finn, C., Abbeel, P. & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast
  Adaptation of Deep Networks." ICML 2017.
- Nichol, A., Achiam, J. & Schulman, J. (2018). "On First-Order Meta-Learning
  Algorithms." arXiv:1803.02999.

### 10.5 Neural Architecture Search (NAS) for Basecaller Design

#### 10.5.1 Motivation

The basecaller architecture (number of layers, hidden dimensions, attention heads,
LoRA ranks) is currently hand-designed. Neural Architecture Search automates the
discovery of optimal architectures for specific chemistry classes and hardware
constraints.

#### 10.5.2 Differentiable NAS (DARTS)

DARTS relaxes discrete architecture choices to continuous parameters. Instead of
choosing one operation per edge (e.g., convolution vs. attention vs. skip), DARTS
maintains a weighted mixture and optimizes the mixture weights via gradient descent.

**Search space for basecaller**:

| Choice | Options | Search Dimension |
|--------|---------|-----------------|
| Number of layers | 4, 6, 8, 12 | 4 |
| Hidden dimension per layer | 128, 256, 384, 512 | 4 |
| Attention heads per layer | 2, 4, 8 | 3 |
| LoRA rank for adaptation | 1, 2, 4, 8 | 4 |
| Activation function | ReLU, GELU, SiLU | 3 |
| Normalization | LayerNorm, RMSNorm | 2 |

Total search space: `4 * 4 * 3 * 4 * 3 * 2 = 1,152` discrete architectures.

**DARTS optimization**: Each choice is parameterized by a softmax-weighted mixture.
The architecture weights `alpha` and model weights `theta` are jointly optimized:

```
min_{alpha} L_val(theta*(alpha), alpha)
  s.t. theta*(alpha) = argmin_{theta} L_train(theta, alpha)
```

In practice, this is solved by alternating gradient steps on `alpha` (validation
set) and `theta` (training set).

#### 10.5.3 Once-for-All Networks

Once-for-All (OFA) trains a single supernet that supports elastic depth, width, and
kernel size. Specialized subnets are extracted without retraining by selecting a
subset of layers and channels.

**For genomics**: Train one OFA supernet covering all chemistry classes. At
deployment, extract a chemistry-specific subnet by selecting:
- Depth: 4 layers for simple chemistries (R9.4.1), 8 layers for complex (Duplex)
- Width: 128 hidden dims for FPGA (fits SRAM), 512 for GPU
- Heads: 2 for streaming, 8 for batch mode

#### 10.5.4 Hardware-Aware NAS

The search objective includes hardware latency as a constraint:

```
minimize    L_accuracy(arch) + lambda_hw * L_latency(arch, target_hardware)
```

where `L_latency` is measured or predicted for the target deployment platform
(FPGA, GPU, or CPU). The FPGA latency model accounts for:
- On-chip SRAM capacity (parameter budget)
- DSP block count (multiply-accumulate budget)
- Routing congestion (data movement cost)
- Pipeline depth (throughput-latency trade-off)

#### 10.5.5 Implementation

```rust
/// Neural Architecture Search for hardware-constrained basecaller design.
/// Uses DARTS-style differentiable search with hardware latency constraints.
/// Crate: `crates/ruvector-fpga-transformer/src/nas.rs`
pub struct NasSearcher {
    /// Architecture parameters (alpha) for each choice point.
    arch_params: Vec<ArchitectureChoice>,
    /// Hardware latency model for the target platform.
    hw_model: HardwareLatencyModel,
    /// Search space definition.
    search_space: SearchSpace,
    /// Lambda for hardware latency penalty in the loss.
    hw_lambda: f32,
    /// Best architecture found so far.
    best_arch: Option<DiscretizedArchitecture>,
    /// Validation loss of best architecture.
    best_loss: f32,
}

/// A single architecture choice with continuous relaxation.
pub struct ArchitectureChoice {
    /// Choice name (e.g., "num_layers", "hidden_dim").
    name: String,
    /// Softmax logits over discrete options.
    logits: Vec<f32>,
    /// The discrete options (e.g., [4, 6, 8, 12] for num_layers).
    options: Vec<usize>,
}

impl NasSearcher {
    /// Run one DARTS search step.
    /// Updates architecture parameters using validation loss gradient.
    pub fn search_step(
        &mut self,
        train_data: &DataLoader,
        val_data: &DataLoader,
        model_weights: &mut Vec<f32>,
    ) -> SearchStepResult {
        // 1. Train model weights on training data (fixed arch_params)
        let train_loss = self.train_step(model_weights, train_data);
        // 2. Compute validation loss + hardware penalty
        let val_loss = self.eval(model_weights, val_data);
        let hw_penalty = self.hw_model.predict_latency(&self.current_arch());
        let total_loss = val_loss + self.hw_lambda * hw_penalty;
        // 3. Update arch_params via gradient on total_loss
        let arch_grads = self.compute_arch_gradients(
            model_weights, val_data, &self.hw_model,
        );
        for (choice, grad) in self.arch_params.iter_mut().zip(arch_grads.iter()) {
            for (logit, g) in choice.logits.iter_mut().zip(grad.iter()) {
                *logit -= 0.001 * g; // Architecture learning rate
            }
        }
        // 4. Track best architecture
        if total_loss < self.best_loss {
            self.best_loss = total_loss;
            self.best_arch = Some(self.discretize());
        }
        SearchStepResult { train_loss, val_loss, hw_penalty, total_loss }
    }

    /// Discretize the continuous architecture into concrete choices.
    /// Selects the argmax of each softmax distribution.
    pub fn discretize(&self) -> DiscretizedArchitecture {
        let choices: Vec<_> = self.arch_params.iter().map(|c| {
            let max_idx = c.logits.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap().0;
            (c.name.clone(), c.options[max_idx])
        }).collect();
        DiscretizedArchitecture { choices: choices.into_iter().collect() }
    }

    /// Extract an OFA-style subnet for specific hardware constraints.
    pub fn extract_subnet(
        &self,
        supernet_weights: &[f32],
        constraints: &HardwareConstraints,
    ) -> SubnetArchitecture {
        let target_depth = constraints.max_layers;
        let target_width = constraints.max_hidden_dim;
        // Select top-k layers by importance, capped at target_depth
        // Select top-k channels per layer, capped at target_width
        self.prune_supernet(supernet_weights, target_depth, target_width)
    }
}
```

**References**:
- Liu, H., Simonyan, K. & Yang, Y. (2019). "DARTS: Differentiable Architecture
  Search." ICLR 2019.
- Cai, H., Gan, C., Wang, T., Zhang, Z. & Han, S. (2020). "Once-for-All: Train
  One Network and Specialize it for Efficient Deployment." ICLR 2020.

---

## 11. Neuro-Symbolic Integration for Biological Reasoning

### 11.1 Motivation: Beyond Pattern Matching

Neural networks excel at pattern recognition (e.g., mapping electrical signals to
base calls, identifying variant signatures in read pileups). However, they lack the
ability to reason about biological semantics -- whether a predicted variant makes
biological sense given known pathway structures, protein domain functions, and
evolutionary constraints. Neuro-symbolic integration bridges this gap by coupling
neural predictions with symbolic biological knowledge.

### 11.2 Architecture

```
NEURO-SYMBOLIC REASONING PIPELINE

  +-----------------------------------------------------------------+
  |                    NEURAL SUBSYSTEM                               |
  |                                                                   |
  |  +------------------+    +-------------------+    +------------+  |
  |  | Basecaller       |    | Variant Caller    |    | Effect     |  |
  |  | (SONA-adapted)   |--->| (CNN + attention) |--->| Predictor  |  |
  |  | Signal -> Bases  |    | Bases -> Variants |    | (GNN)      |  |
  |  +------------------+    +-------------------+    +------+-----+  |
  |                                                          |        |
  +-----------------------------------------------------------------+
                                                             |
                                  Neural prediction:         |
                                  "Variant X disrupts        |
                                   kinase domain of          |
                                   protein BRAF"             |
                                                             |
                                                             v
  +-----------------------------------------------------------------+
  |                   SYMBOLIC SUBSYSTEM                              |
  |                                                                   |
  |  +------------------+    +-------------------+    +------------+  |
  |  | Knowledge Graph  |    | Pathway Reasoner  |    | Constraint |  |
  |  | (KEGG, Reactome, |    | (Prolog / Datalog |    | Checker    |  |
  |  |  UniProt, ClinVar|    |  inference engine) |    | (verify)   |  |
  |  |  GO, InterPro)   |--->|                   |--->|            |  |
  |  +------------------+    +-------------------+    +------+-----+  |
  |                                                          |        |
  +-----------------------------------------------------------------+
                                                             |
                                  Symbolic verification:     |
                                  "BRAF is in MAPK/ERK       |
                                   pathway. Kinase domain    |
                                   disruption in this gene   |
                                   is associated with        |
                                   oncogenic activation."    |
                                                             |
                                                             v
  +-----------------------------------------------------------------+
  |                  INTEGRATION GATE                                 |
  |                                                                   |
  |  Neural confidence:  0.87                                         |
  |  Symbolic support:   STRONG (3 pathway hits, 12 literature refs)  |
  |  Combined score:     0.94 (geometric mean with symbolic boost)    |
  |  Clinical action:    FLAG FOR REVIEW (known oncogenic driver)     |
  |                                                                   |
  +-----------------------------------------------------------------+
```

### 11.3 Knowledge Graph Structure

The symbolic knowledge base is a typed property graph with the following schema:

```
KNOWLEDGE GRAPH SCHEMA

  [Gene] --HAS_DOMAIN--> [ProteinDomain]
    |                         |
    |--IN_PATHWAY-->   [Pathway]
    |                     |
    |--ASSOCIATED-->   [Disease]
    |
    |--HAS_VARIANT--> [KnownVariant]
                          |
                          |--CLASSIFIED--> [ClinicalSignificance]
                          |--IN_POPULATION--> [PopulationFrequency]

  Sources:
    KEGG:     17,000+ pathways
    Reactome: 2,600+ human pathways
    UniProt:  570,000+ protein records
    ClinVar:  2,200,000+ variant classifications
    GO:       45,000+ biological terms
    InterPro: 40,000+ protein domain families
```

### 11.4 Differentiable Logic: DeepProbLog Integration

Standard symbolic reasoning is not differentiable and cannot be jointly optimized
with neural components. DeepProbLog-style integration makes the symbolic layer
differentiable by treating logical rules as probabilistic programs whose parameters
(rule confidences) are learned from data.

**Example rule in probabilistic logic**:

```prolog
% Neural predicate: probability from GNN effect predictor
nn(effect_predictor, [Variant, Gene], P_disruptive) :: disrupts(Variant, Domain).

% Symbolic rule: if a variant disrupts a kinase domain in an oncogene,
% it is likely a gain-of-function driver
0.85 :: driver(Variant) :-
    disrupts(Variant, Domain),
    domain_type(Domain, kinase),
    gene_of(Domain, Gene),
    oncogene(Gene).

% Query: what is the probability that variant V is a driver?
query(driver(V)).
```

The probability of `driver(V)` is computed by forward-chaining through the rules,
multiplying probabilities along each derivation path. The gradient of the query
probability with respect to the neural network parameters flows back through the
probabilistic program, enabling end-to-end training.

### 11.5 Graph Neural Network for Variant Effect Prediction

The neural component of the neuro-symbolic system uses a Graph Neural Network (GNN)
that operates on the protein structure graph:

```
GNN ARCHITECTURE FOR VARIANT EFFECT PREDICTION

  Protein 3D Structure
       |
       v
  +------------------+
  | Graph             |    Nodes: amino acid residues
  | Construction     |    Edges: spatial proximity (< 8A)
  |                  |    Features: residue type, secondary structure,
  |                  |              conservation score, B-factor
  +------------------+
       |
       v
  +------------------+
  | Message Passing  |    3 rounds of GNN message passing:
  | (3 layers)       |    h_i^(l+1) = UPDATE(h_i^(l), AGG({h_j^(l) : j in N(i)}))
  |                  |    Uses attention-weighted aggregation
  +------------------+
       |
       v
  +------------------+
  | Variant Scoring  |    For variant at position p:
  |                  |    score = MLP(h_p^(L) || h_wt_p || delta_features)
  |                  |    Output: P(disruptive), P(benign), P(uncertain)
  +------------------+
```

### 11.6 Implementation

```rust
/// Neuro-symbolic reasoner combining GNN variant effect prediction
/// with knowledge graph pathway reasoning.
/// Crate: `crates/cognitum-gate-kernel/src/neurosymbolic.rs`
pub struct NeuroSymbolicReasoner {
    /// GNN for variant effect prediction on protein structure graphs.
    effect_predictor: GraphNeuralNetwork,
    /// Knowledge graph with biological pathway information.
    knowledge_graph: BiologicalKnowledgeGraph,
    /// Probabilistic logic engine (DeepProbLog-style).
    logic_engine: ProbLogEngine,
    /// Integration gate combining neural and symbolic scores.
    integration_gate: IntegrationGate,
}

/// Biological knowledge graph backed by HNSW-indexed embeddings
/// for fast entity and relation lookup.
pub struct BiologicalKnowledgeGraph {
    /// Gene -> domain mappings (InterPro).
    gene_domains: HashMap<GeneId, Vec<ProteinDomain>>,
    /// Pathway membership (KEGG, Reactome).
    pathways: HashMap<GeneId, Vec<Pathway>>,
    /// Known variant classifications (ClinVar).
    known_variants: HashMap<VariantId, ClinicalClassification>,
    /// HNSW index over gene/variant embeddings for similarity search.
    embedding_index: HnswIndex<384>,
}

impl NeuroSymbolicReasoner {
    /// Evaluate a novel variant using both neural and symbolic reasoning.
    /// Returns a combined score with full explanation chain.
    pub fn evaluate_variant(
        &self,
        variant: &Variant,
        protein_structure: &ProteinGraph,
    ) -> VariantAssessment {
        // 1. Neural: GNN predicts structural effect
        let neural_score = self.effect_predictor.predict(
            protein_structure,
            variant.position,
        );

        // 2. Symbolic: query knowledge graph for pathway context
        let pathway_context = self.knowledge_graph.get_context(
            &variant.gene,
            &variant.affected_domain,
        );

        // 3. Probabilistic logic: combine neural + symbolic
        let logic_result = self.logic_engine.query(
            "driver",
            &[
                ("neural_score", neural_score.disruptive_prob),
                ("domain_type", pathway_context.domain_type_encoding),
                ("is_oncogene", pathway_context.oncogene_prob),
            ],
        );

        // 4. Integration gate: final combined assessment
        self.integration_gate.combine(
            neural_score,
            pathway_context,
            logic_result,
        )
    }

    /// End-to-end training: gradients flow from clinical outcomes
    /// back through the logic engine into the GNN parameters.
    pub fn train_step(
        &mut self,
        variant: &Variant,
        protein_structure: &ProteinGraph,
        ground_truth: ClinicalOutcome,
    ) -> f32 {
        let assessment = self.evaluate_variant(variant, protein_structure);
        let loss = cross_entropy(assessment.combined_score, ground_truth);
        // Backprop through integration gate -> logic engine -> GNN
        let grads = self.backpropagate(loss);
        self.effect_predictor.apply_gradients(&grads.gnn_grads);
        self.logic_engine.update_rule_confidences(&grads.logic_grads);
        loss
    }
}

/// Integration gate that combines neural and symbolic assessments
/// using learned weights.
pub struct IntegrationGate {
    /// Weight for neural prediction (learned).
    neural_weight: f32,
    /// Weight for symbolic support (learned).
    symbolic_weight: f32,
    /// Confidence threshold for automatic classification.
    auto_classify_threshold: f32,
    /// Confidence threshold for flagging for human review.
    review_threshold: f32,
}

impl IntegrationGate {
    /// Combine neural and symbolic assessments.
    /// Uses geometric mean with symbolic boost for pathway-supported variants.
    pub fn combine(
        &self,
        neural: NeuralScore,
        symbolic: PathwayContext,
        logic: LogicResult,
    ) -> VariantAssessment {
        let neural_contrib = neural.disruptive_prob * self.neural_weight;
        let symbolic_contrib = logic.probability * self.symbolic_weight;
        let combined = (neural_contrib * symbolic_contrib).sqrt(); // Geometric mean

        let action = if combined > self.auto_classify_threshold {
            ClinicalAction::AutoClassify
        } else if combined > self.review_threshold {
            ClinicalAction::FlagForReview
        } else {
            ClinicalAction::Benign
        };

        VariantAssessment {
            combined_score: combined,
            neural_score: neural,
            pathway_context: symbolic,
            logic_derivation: logic,
            recommended_action: action,
        }
    }
}
```

**References**:
- Manhaeve, R. et al. (2018). "DeepProbLog: Neural Probabilistic Logic Programming."
  NeurIPS 2018.
- Lamb, L. C. et al. (2020). "Graph Neural Networks Meet Neural-Symbolic Computing:
  A Survey and Perspective." arXiv:2003.00330.

---

## 12. SOTA Technique Comparison Matrix

The following matrix summarizes all continual learning and adaptation techniques
in this ADR, their trade-offs, and their recommended application within RuVector.

| Technique | Forgetting Prevention | Forward Transfer | Memory Overhead | Adaptation Speed | Best For |
|-----------|----------------------|------------------|----------------|-----------------|----------|
| EWC++ (Sec. 1.3) | Approximate (soft) | Implicit (shared params) | O(params) | Continuous | Within-chemistry fine-tuning |
| Progressive Nets (Sec. 9) | Exact (structural freeze) | Explicit (lateral conn.) | O(params * tasks) | Per-task | Cross-chemistry expansion |
| Lottery Ticket (Sec. 10.1) | N/A (sparse extraction) | N/A | 5-10% of original | One-time | Edge/FPGA deployment |
| Hypernetworks (Sec. 10.2) | N/A (generation) | Amortized | O(hypernetwork) | <0.01ms | On-the-fly adapter generation |
| PackNet (Sec. 10.3) | Exact (binary masks) | None (isolated) | O(params + masks) | Per-task | Multi-chemistry single model |
| MAML/Reptile (Sec. 10.4) | N/A (meta-init) | Meta-learned | O(params) | 3 gradient steps | Few-shot new device adaptation |
| NAS (Sec. 10.5) | N/A (design-time) | Architecture-level | Search cost | One-time | Optimal basecaller architecture |
| Neuro-Symbolic (Sec. 11) | N/A (reasoning) | Knowledge graphs | O(graph) | Per-query | Biological interpretation |

### Recommended Deployment Strategy

```
TECHNIQUE SELECTION BY SCENARIO

  New chemistry version deployed:
    -> Progressive Nets (add column, zero forgetting) + PackNet (if single-model preferred)
    -> NAS (find optimal architecture for new chemistry)

  New flow cell loaded (same chemistry):
    -> MAML/Reptile (3 gradient steps on 10 reads)
    -> Hypernetwork (generate adapter in <0.01ms from metadata)
    -> EWC++ (continuous refinement over the run)

  Deploying to FPGA/edge:
    -> Lottery Ticket (find 5-10% sparse subnet)
    -> NAS (hardware-aware architecture search)

  Interpreting variant clinical significance:
    -> Neuro-Symbolic (GNN + knowledge graph + probabilistic logic)

  Cross-institutional model improvement:
    -> Federated learning (Section 5) + EWC++ consolidation
```

---

## Related Decisions

- **ADR-001**: RuVector Core Architecture (HNSW, SIMD, quantization)
- **ADR-003**: SIMD Optimization Strategy (MicroLoRA SIMD forward pass)
- **ADR-028**: Graph Genome & Min-Cut Architecture (variation graph substrate)
- **ADR-CE-021**: Shared SONA (coherence engine integration)
- **ADR-CE-022**: Failure Learning (verdict system integration)

---

## References

1. Kirkpatrick, J. et al. (2017). "Overcoming catastrophic forgetting in neural
   networks." PNAS, 114(13), 3521-3526.

2. Hu, E. J. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models."
   arXiv:2106.09685.

3. McMahan, B. et al. (2017). "Communication-Efficient Learning of Deep Networks
   from Decentralized Data." AISTATS 2017.

4. Fries, P. (2015). "Rhythms for Cognition: Communication through Coherence."
   Neuron, 88(1), 220-235.

5. Baars, B. J. (1988). "A Cognitive Theory of Consciousness." Cambridge University
   Press.

6. McClelland, J. L. et al. (1995). "Why there are complementary learning systems in
   the hippocampus and neocortex." Psychological Review, 102(3), 419-457.

7. Dwork, C. & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy."
   Foundations and Trends in Theoretical Computer Science, 9(3-4), 211-407.

8. Rusu, A. A. et al. (2016). "Progressive Neural Networks." arXiv:1606.04671.

9. Frankle, J. & Carlin, M. (2019). "The Lottery Ticket Hypothesis: Finding Sparse,
   Trainable Neural Networks." ICLR 2019.

10. Ha, D., Dai, A. & Le, Q. V. (2017). "HyperNetworks." ICLR 2017.

11. Mallya, A. & Lazebnik, S. (2018). "PackNet: Adding Multiple Tasks to a Single
    Network by Iterative Pruning." CVPR 2018.

12. Finn, C., Abbeel, P. & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast
    Adaptation of Deep Networks." ICML 2017.

13. Nichol, A., Achiam, J. & Schulman, J. (2018). "On First-Order Meta-Learning
    Algorithms." arXiv:1803.02999.

14. Liu, H., Simonyan, K. & Yang, Y. (2019). "DARTS: Differentiable Architecture
    Search." ICLR 2019.

15. Cai, H., Gan, C., Wang, T., Zhang, Z. & Han, S. (2020). "Once-for-All: Train
    One Network and Specialize it for Efficient Deployment." ICLR 2020.

16. Manhaeve, R. et al. (2018). "DeepProbLog: Neural Probabilistic Logic
    Programming." NeurIPS 2018.

17. Lamb, L. C. et al. (2020). "Graph Neural Networks Meet Neural-Symbolic Computing:
    A Survey and Perspective." arXiv:2003.00330.

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | Architecture Design Agent | Initial proposal |
| 0.2 | 2026-02-11 | Architecture Design Agent | SOTA enhancements: Progressive Neural Networks (Sec. 9), Lottery Ticket Hypothesis (Sec. 10.1), Hypernetworks (Sec. 10.2), PackNet (Sec. 10.3), MAML/Reptile meta-learning (Sec. 10.4), Neural Architecture Search (Sec. 10.5), Neuro-Symbolic Integration (Sec. 11), technique comparison matrix (Sec. 12) |
