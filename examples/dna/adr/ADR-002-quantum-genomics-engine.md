# ADR-002: Quantum Genomics Engine

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector Team
**Deciders**: Architecture Review Board
**SDK**: Claude-Flow

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | ruv.io | Initial quantum genomics engine proposal |

---

## Context

### The Genomics Computational Bottleneck

Modern genomics confronts a data explosion that outpaces Moore's Law. A single
human genome contains approximately 3.2 billion base pairs. Whole-genome sequencing
(WGS) at 30x coverage produces ~100 GB of raw read data per sample. The critical
computational tasks -- sequence alignment, variant calling, haplotype phasing,
de novo assembly, phylogenetic inference, and protein structure prediction -- each
pose optimization problems whose classical complexity ranges from O(N log N) to
NP-hard.

| Genomic Operation | Classical Complexity | Bottleneck |
|-------------------|---------------------|------------|
| k-mer exact search | O(N) per query | Linear scan over 3.2B base pairs |
| Sequence alignment (BWA-MEM2) | O(N log N) with FM-index | Index construction and seed extension |
| Variant calling (GATK HaplotypeCaller) | O(R * H * L) per active region | Local assembly of haplotype candidates |
| Haplotype assembly | NP-hard (MEC formulation) | Minimum error correction on read fragments |
| De novo genome assembly | O(N) edge traversal on de Bruijn graph | Graph construction and Eulerian path finding |
| Phylogenetic tree inference (ML) | NP-hard (Felsenstein, 1978) | Tree topology search over super-exponential space |
| Protein folding energy minimization | NP-hard (Crescenzi & Pode, 1998) | Conformational search in continuous space |
| Variant pathogenicity prediction | O(features * samples) | Feature engineering and model training |

Where:
- N = reference genome length (~3.2 x 10^9 for human)
- R = number of reads (~10^9 for 30x WGS)
- H = number of candidate haplotypes per active region
- L = active region length

### How Quantum Computing Transforms Genomics

Quantum computation offers three distinct advantages for genomics workloads:

1. **Quadratic speedup via amplitude amplification**: Grover-family algorithms
   reduce unstructured search from O(N) to O(sqrt(N)). For k-mer lookup across
   3.2 billion base pairs, this transforms a 3.2 x 10^9 search into a
   ~56,568-query problem.

2. **Combinatorial optimization via variational quantum methods**: QAOA and
   quantum annealing encode NP-hard problems (haplotype assembly, phylogenetic
   tree inference) as Hamiltonians whose ground states correspond to optimal
   solutions. While provable exponential advantage remains unproven for NP-hard
   problems, empirical evidence suggests polynomial-factor improvements in
   convergence for structured combinatorial landscapes.

3. **Quantum simulation of molecular systems**: VQE and quantum phase estimation
   compute molecular ground-state energies with polynomial resources in the number
   of orbitals, enabling first-principles modeling of protein-ligand interactions,
   DNA-protein binding energetics, and drug-target affinity without classical
   force-field approximations.

### Why Now: The NISQ-to-Simulation Bridge

Current quantum hardware (50-1000+ physical qubits, error rates of 10^-3 to 10^-4)
cannot yet process genome-scale data directly. However, ruQu's state-vector simulator
(ADR-QE-001) provides a platform to:

- Develop and validate quantum genomics algorithms at tractable scales (8-25 qubits)
- Benchmark quantum approaches against classical baselines with exact arithmetic
- Design hybrid classical-quantum pipelines that are hardware-ready
- Establish circuit templates and oracle constructions that transfer directly to
  future fault-tolerant quantum processors

The ruqu-wasm compilation target (ADR-QE-003) additionally enables browser-based
quantum genomics experimentation, making these algorithms accessible to
bioinformaticians without quantum hardware or local installation.

---

## Decision

### Architecture Overview

Introduce a `quantum-genomics` module within `ruqu-algorithms` that implements
quantum algorithms specialized for genomic data processing. The module leverages
all four ruQu crates:

```
                    ┌─────────────────────────────────────────────┐
                    │          Quantum Genomics Engine             │
                    │       (ruqu-algorithms::genomics)            │
                    ├─────────────────────────────────────────────┤
                    │                                              │
                    │  ┌─────────┐  ┌──────────┐  ┌───────────┐ │
                    │  │ k-mer   │  │ Haplotype│  │ Phylo-    │ │
                    │  │ Search  │  │ Assembly │  │ genetics  │ │
                    │  │(Grover) │  │ (QAOA)   │  │(Anneal)   │ │
                    │  └────┬────┘  └────┬─────┘  └─────┬─────┘ │
                    │       │            │              │        │
                    │  ┌────┴────┐  ┌────┴─────┐  ┌────┴─────┐ │
                    │  │ VQE     │  │ Quantum  │  │ QML      │ │
                    │  │Molecular│  │ Walk     │  │ Variant  │ │
                    │  │Interact.│  │ Assembly │  │ Classify │ │
                    │  └────┬────┘  └────┬─────┘  └────┬─────┘ │
                    │       │            │              │        │
                    │  ┌────┴────────────┴──────────────┴─────┐ │
                    │  │        DNA Error Correction           │ │
                    │  │   (QEC codes mapped to sequencing)    │ │
                    │  └──────────────────────────────────────┘ │
                    └────────────────┬────────────────────────┬──┘
                                     │                        │
                    ┌────────────────┴────┐    ┌─────────────┴───────┐
                    │     ruqu-core       │    │    ruqu-exotic      │
                    │  (state vector sim) │    │ (hybrid discoveries)│
                    ├─────────────────────┤    ├─────────────────────┤
                    │  ruqu-wasm          │    │  Decoherence        │
                    │  (browser target)   │    │  fingerprinting     │
                    └─────────────────────┘    │  for sequence       │
                                               │  similarity         │
                                               └─────────────────────┘
```

### Module Structure

```
ruqu-algorithms/
  src/
    genomics/
      mod.rs                 # Public API and genomic type definitions
      kmer_search.rs         # Grover-based k-mer search
      haplotype_assembly.rs  # QAOA for haplotype phasing
      vqe_molecular.rs       # VQE for DNA-protein interactions
      quantum_walk.rs        # Quantum walks on de Bruijn graphs
      dna_error_correction.rs # QEC-to-sequencing error mapping
      phylogenetics.rs       # Quantum annealing for tree optimization
      variant_classifier.rs  # Quantum ML for pathogenicity prediction
      encoding.rs            # DNA base-pair to qubit encoding schemes
      hybrid_pipeline.rs     # Classical-quantum decision boundary logic
```

---

## 1. Grover's Algorithm for k-mer Search

### Problem Statement

A k-mer is a subsequence of length k extracted from a DNA sequence. k-mer
search is fundamental to sequence alignment, genome assembly, and metagenomic
classification. Given a reference genome of length N and a query k-mer of length
k, find all positions where the k-mer occurs.

Classical approaches use hash tables (O(1) lookup after O(N) preprocessing) or
FM-indices (O(k) lookup after O(N) construction). Grover's algorithm offers a
different trade-off: no preprocessing index, with O(sqrt(N)) query complexity.

### Quantum Formulation

**Encoding**: Represent each position in the reference genome as a quantum basis
state. For a genome of length N, we require n = ceil(log2(N)) qubits.

For the human genome: n = ceil(log2(3.2 x 10^9)) = 32 qubits.

**Oracle construction**: The oracle O_kmer marks basis state |i> if the k-mer
starting at position i in the reference matches the query:

```
O_kmer |i> = (-1)^{f(i)} |i>

where f(i) = 1 if reference[i..i+k] == query[0..k]
             0 otherwise
```

In simulation, we implement this via the index-based oracle from ADR-QE-006,
pre-computing all matching positions:

```rust
/// Construct a Grover oracle for k-mer search.
///
/// Pre-scans the reference to identify matching positions,
/// then encodes them as target indices for Grover's algorithm.
///
/// Pre-scan: O(N * k) classical
/// Quantum search: O(sqrt(N/m)) iterations where m = match count
pub struct KmerOracle {
    /// Reference genome encoded as 2-bit per base (A=00, C=01, G=10, T=11)
    reference: Vec<u8>,
    /// Query k-mer (2-bit encoded)
    query: Vec<u8>,
    /// k-mer length
    k: usize,
    /// Pre-computed matching positions
    matches: Vec<usize>,
}

impl GroverOracle for KmerOracle {
    fn is_marked(&self, index: usize, _n_qubits: usize) -> bool {
        if index + self.k > self.reference.len() {
            return false;
        }
        self.reference[index..index + self.k] == self.query[..]
    }
}
```

### Complexity Analysis

| Approach | Preprocessing | Per-Query | Space |
|----------|--------------|-----------|-------|
| Linear scan | None | O(N * k) | O(1) |
| Hash table | O(N) | O(k) average | O(N) |
| FM-index (BWT) | O(N) | O(k) | O(N) |
| Suffix array | O(N) | O(k log N) | O(N) |
| **Grover (quantum)** | **None** | **O(sqrt(N) * k)** | **O(n) qubits** |
| **Grover (sim, index oracle)** | **O(N * k) pre-scan** | **O(sqrt(N/m))** | **O(2^n) amplitudes** |

For the human genome with k = 31 (standard for short-read alignment):

```
Classical linear scan:    3.2 x 10^9 comparisons per query
Grover (quantum):         sqrt(3.2 x 10^9) = 56,568 oracle queries
Speedup factor:           56,568x fewer queries
```

**Simulation ceiling**: A 32-qubit simulation requires 2^32 * 16 bytes = 64 GB
of state vector memory, which is feasible on native hardware but exceeds the WASM
limit. For browser-based demonstration, a reduced reference (up to 2^25 = 33M
bases, covering a single chromosome) is practical.

### Optimal Iteration Count for Genomic Search

When multiple k-mer matches exist (common for repetitive regions), the iteration
count adjusts:

```
iterations = floor(pi/4 * sqrt(N/m))
```

where m is the number of matching positions. For a k-mer with m = 100 matches
in a genome of N = 3.2 x 10^9:

```
iterations = floor(pi/4 * sqrt(3.2e9 / 100)) = floor(4,443) = 4,443
```

The algorithm amplifies all m matches simultaneously, and any measurement yields
a matching position with probability approaching 1.

---

## 2. QAOA for Haplotype Assembly

### Problem Statement

Haplotype assembly determines which alleles at heterozygous sites co-occur on the
same chromosome. Given a set of sequencing reads spanning multiple heterozygous
sites, the goal is to partition reads into two groups (one per haplotype) that
minimize the total number of read-allele conflicts.

This is the Minimum Error Correction (MEC) problem, proven NP-hard by reduction
from MAX-CUT (Lancia et al., 2001).

### Quantum Formulation

**Encoding**: Assign one qubit per sequencing read fragment. Qubit |0> assigns
the read to haplotype H1; qubit |1> assigns it to haplotype H2.

For F fragments covering S heterozygous SNP sites, the cost Hamiltonian encodes
the MEC objective:

```
H_MEC = sum_{j=1}^{S} sum_{i: fragment i covers site j}
        w_{ij} * (1 - z_i * a_{ij}) / 2
```

where:
- z_i in {+1, -1} is the haplotype assignment for fragment i (qubit measurement)
- a_{ij} in {+1, -1} is the allele observed by fragment i at site j
- w_{ij} is the base quality weight (Phred score converted to probability)

This maps directly to a weighted MAX-CUT instance on a fragment-conflict graph,
which is the canonical problem for QAOA (ADR-QE-007).

### QAOA Circuit for Haplotype Assembly

```
Fragment-Conflict Graph Construction:
=====================================

For each pair of fragments (i, j) covering overlapping SNP sites:
  - Compute conflict score: number of sites where fragments disagree
  - Edge weight = sum of quality-weighted conflicts

  Fragment 1: --A--C--*--T--G--    (* = no coverage at this site)
  Fragment 2: --A--T--G--T--*--
                   ^  ^
                   conflicts at sites 2 and 3

  Edge weight(1,2) = Q(site2) + Q(site3)

QAOA encodes this as:
  Phase separator: exp(-i * gamma * w_{ij} * Z_i Z_j / 2) per edge
  Mixer: exp(-i * beta * X_i) per qubit
```

```rust
/// Construct a QAOA instance for haplotype assembly.
///
/// Converts the MEC problem into a weighted MAX-CUT graph
/// and delegates to the QAOA engine from ADR-QE-007.
pub struct HaplotypeAssemblyQaoa {
    /// Fragment-SNP matrix (rows = fragments, columns = SNP sites)
    /// Values: 0 = ref allele, 1 = alt allele, -1 = no coverage
    fragment_matrix: Vec<Vec<i8>>,
    /// Base quality scores (Phred-scaled) per fragment per site
    quality_matrix: Vec<Vec<f64>>,
    /// QAOA depth parameter
    p: usize,
}

impl HaplotypeAssemblyQaoa {
    /// Build the fragment-conflict graph.
    ///
    /// Two fragments are connected by an edge if they cover
    /// at least one common SNP site with conflicting alleles.
    /// Edge weight = sum of quality-weighted conflicts.
    pub fn build_conflict_graph(&self) -> Graph {
        let n_fragments = self.fragment_matrix.len();
        let mut edges = Vec::new();

        for i in 0..n_fragments {
            for j in (i + 1)..n_fragments {
                let mut weight = 0.0;
                for s in 0..self.fragment_matrix[i].len() {
                    let a_i = self.fragment_matrix[i][s];
                    let a_j = self.fragment_matrix[j][s];
                    // Both fragments cover this site and disagree
                    if a_i >= 0 && a_j >= 0 && a_i != a_j {
                        let q = (self.quality_matrix[i][s]
                                + self.quality_matrix[j][s]) / 2.0;
                        weight += q;
                    }
                }
                if weight > 0.0 {
                    edges.push((i, j, weight));
                }
            }
        }

        Graph {
            n_vertices: n_fragments,
            edges,
        }
    }

    /// Solve haplotype assembly via QAOA.
    pub fn solve(
        &self,
        optimizer: &mut dyn ClassicalOptimizer,
    ) -> HaplotypeResult {
        let graph = self.build_conflict_graph();
        let qaoa_result = qaoa_maxcut(&graph, self.p, optimizer, &default_config());

        // Decode partition into haplotype assignment
        let assignment: Vec<u8> = (0..graph.n_vertices)
            .map(|v| ((qaoa_result.best_cut.partition >> v) & 1) as u8)
            .collect();

        HaplotypeResult {
            haplotype_assignment: assignment,
            mec_score: qaoa_result.best_cost,
            approximation_ratio: qaoa_result.approximation_ratio,
        }
    }
}
```

### Qubit Requirements and Scaling

| Active Region | Fragments | Qubits Needed | Simulatable? | QAOA Depth |
|---------------|-----------|---------------|-------------|------------|
| Small (200bp) | 10-20 | 10-20 | Yes (native + WASM) | p = 3-5 |
| Medium (500bp) | 30-50 | 30-50 | Marginal (native only, 30 qubits) | p = 3 |
| Large (1kbp) | 50-100 | 50-100 | No (requires hardware) | p = 1-3 |
| Exome region | 200+ | 200+ | No (requires hardware) | p = 1 |

The hybrid approach: use classical haplotype assembly (HapCUT2, WhatsHap) for
most regions, and invoke QAOA for difficult regions where classical methods fail
to converge (high heterozygosity, structural variants, repetitive regions).

---

## 3. VQE for Molecular Interaction Modeling

### Problem Statement

Understanding DNA-protein binding, drug-nucleic acid intercalation, and
epigenetic modification chemistry requires computing molecular ground-state
energies. Classical force fields (AMBER, CHARMM) use parameterized potentials
that approximate quantum mechanical interactions. VQE computes these energies
from first principles using the electronic structure Hamiltonian.

### Quantum Formulation

The molecular Hamiltonian in second quantization:

```
H = sum_{pq} h_{pq} a_p^dagger a_q
  + (1/2) sum_{pqrs} h_{pqrs} a_p^dagger a_q^dagger a_s a_r
  + E_nuc
```

where h_{pq} and h_{pqrs} are one-electron and two-electron integrals computed
classically, and E_nuc is the nuclear repulsion energy.

**Qubit mapping**: The Jordan-Wigner transformation maps fermionic operators
to qubit operators:

```
a_p^dagger -> (1/2)(X_p - iY_p) * prod_{j<p} Z_j
a_p        -> (1/2)(X_p + iY_p) * prod_{j<p} Z_j
```

This yields a Hamiltonian expressed as a sum of Pauli strings, compatible with
the PauliSum representation from ADR-QE-005.

### Genomics-Specific Molecular Targets

| Target System | Orbitals | Qubits (JW) | Pauli Terms | VQE Feasibility |
|---------------|----------|-------------|-------------|-----------------|
| Hydrogen bond (N-H...O) | 4 | 8 | ~15 | Fully simulatable |
| Base pair stacking (2 bases) | 8-12 | 16-24 | ~200-1000 | Simulatable (native) |
| Methyltransferase active site | 12-16 | 24-32 | ~2000-5000 | Marginal (native) |
| Intercalator-DNA complex | 16-20 | 32-40 | ~5000-20000 | Requires hardware |
| CRISPR-Cas9 guide RNA binding | 20+ | 40+ | >20000 | Future hardware |

```rust
/// Construct a molecular Hamiltonian for a DNA-relevant system.
///
/// Uses the Jordan-Wigner transformation to convert the fermionic
/// Hamiltonian into a qubit Hamiltonian (PauliSum).
pub struct GenomicMolecularHamiltonian {
    /// One-electron integrals h_{pq}
    pub one_electron: Vec<Vec<f64>>,
    /// Two-electron integrals h_{pqrs} (chemist notation)
    pub two_electron: Vec<Vec<Vec<Vec<f64>>>>,
    /// Nuclear repulsion energy
    pub nuclear_repulsion: f64,
    /// Number of spatial orbitals
    pub n_orbitals: usize,
    /// Number of electrons
    pub n_electrons: usize,
}

impl GenomicMolecularHamiltonian {
    /// Convert to qubit Hamiltonian via Jordan-Wigner.
    ///
    /// Number of qubits = 2 * n_orbitals (spin-up + spin-down).
    /// Number of Pauli terms = O(n_orbitals^4).
    pub fn to_pauli_sum(&self) -> PauliSum {
        let n_qubits = 2 * self.n_orbitals;
        let mut terms = Vec::new();

        // One-body terms
        for p in 0..n_qubits {
            for q in 0..n_qubits {
                let h_pq = self.one_electron[p / 2][q / 2];
                if h_pq.abs() > 1e-12 {
                    // Jordan-Wigner: a_p^dag a_q -> Pauli string
                    let pauli_terms = jordan_wigner_one_body(p, q, h_pq);
                    terms.extend(pauli_terms);
                }
            }
        }

        // Two-body terms
        // ... (analogous, yielding O(n^4) Pauli strings)

        // Nuclear repulsion as identity term
        terms.push((
            Complex64::new(self.nuclear_repulsion, 0.0),
            PauliString::identity(n_qubits),
        ));

        PauliSum { terms, n_qubits }
    }

    /// Construct VQE solver for this molecular system.
    pub fn vqe_solver(&self) -> VqeSolver {
        let hamiltonian = self.to_pauli_sum();
        let ansatz = ansatz::uccsd(self.n_electrons, self.n_orbitals);
        VqeSolver::new(hamiltonian, ansatz)
    }
}
```

### Accuracy Targets

| Property | Chemical Accuracy | VQE Target |
|----------|------------------|------------|
| Binding energy (hydrogen bond) | 1 kcal/mol = 0.0016 Hartree | < 0.001 Hartree |
| Base stacking energy | ~0.5 kcal/mol | < 0.001 Hartree |
| Proton transfer barrier | 1-2 kcal/mol | < 0.002 Hartree |
| Drug binding affinity | ~1 kcal/mol | < 0.001 Hartree |

The exact expectation value computation in ruqu-core (no shot noise) is critical
for achieving chemical accuracy, as sampling-based estimation requires 10^6+ shots
to reach the 0.001 Hartree threshold.

---

## 4. Quantum Walks on de Bruijn Graphs for Genome Assembly

### Problem Statement

De novo genome assembly constructs a genome sequence from sequencing reads without
a reference. The dominant paradigm represents k-mers as nodes in a de Bruijn graph,
where an edge connects k-mers that overlap by k-1 bases. The assembled genome
corresponds to an Eulerian path through this graph.

For a genome of length N with k-mer size k, the de Bruijn graph has up to N nodes
and N edges. Finding an Eulerian path is O(N) classically, but the real challenge
is resolving ambiguities from repeats, sequencing errors, and heterozygosity.

### Quantum Walk Formulation

A quantum walk on the de Bruijn graph explores multiple paths simultaneously
through superposition. The walk operator alternates between:

1. **Coin operator C**: Local superposition at each node over outgoing edges
2. **Shift operator S**: Move the walker along edges based on coin state

```
U_walk = S * (C tensor I_position)
```

After t steps, measurement of the walker's position yields a node on a
high-probability path. The quantum walk exhibits quadratic speedup over classical
random walks for finding marked vertices (Szegedy, 2004).

**Hitting time advantage**: For a de Bruijn graph with N nodes, the quantum walk
hitting time to a marked node scales as:

```
Classical random walk:  O(N) expected steps
Quantum walk:           O(sqrt(N)) steps (Szegedy's theorem)
```

### Implementation

```rust
/// Quantum walk on a de Bruijn graph for genome assembly.
///
/// The walker's Hilbert space has dimension |V| * d_max, where
/// |V| is the number of vertices and d_max is the maximum degree.
/// Each basis state |v, c> represents being at vertex v with
/// coin state c (selecting an outgoing edge).
pub struct DeBruijnQuantumWalk {
    /// Adjacency structure of the de Bruijn graph
    adjacency: Vec<Vec<usize>>,
    /// k-mer sequences labeling each node
    node_labels: Vec<Vec<u8>>,
    /// Maximum vertex degree
    max_degree: usize,
    /// Number of vertices
    n_vertices: usize,
}

impl DeBruijnQuantumWalk {
    /// Construct from k-mer set.
    ///
    /// Builds the de Bruijn graph where nodes are k-mers and
    /// edges connect (k-1)-overlapping k-mers.
    pub fn from_kmers(kmers: &[Vec<u8>], k: usize) -> Self {
        // Build adjacency from suffix-prefix overlaps
        // Each k-mer's (k-1)-suffix links to k-mers
        // whose (k-1)-prefix matches
        todo!()
    }

    /// Compute the number of qubits required.
    ///
    /// Position register: ceil(log2(n_vertices)) qubits
    /// Coin register: ceil(log2(max_degree)) qubits
    /// Total: position_qubits + coin_qubits
    pub fn qubit_count(&self) -> usize {
        let pos_qubits = (self.n_vertices as f64).log2().ceil() as usize;
        let coin_qubits = (self.max_degree as f64).log2().ceil() as usize;
        pos_qubits + coin_qubits
    }

    /// Execute quantum walk for t steps.
    ///
    /// Returns probability distribution over vertices after t steps.
    pub fn walk(
        &self,
        steps: usize,
        start_vertex: usize,
    ) -> Vec<f64> {
        let n_qubits = self.qubit_count();
        let mut state = QuantumState::new(n_qubits);

        // Initialize at start vertex
        let pos_qubits = (self.n_vertices as f64).log2().ceil() as usize;
        initialize_position(&mut state, start_vertex, pos_qubits);

        for _step in 0..steps {
            // Coin operation: Grover diffusion on coin register
            // at each vertex (conditional on position register)
            apply_coin_operator(&mut state, &self.adjacency, pos_qubits);

            // Shift operation: move walker based on coin state
            apply_shift_operator(
                &mut state,
                &self.adjacency,
                pos_qubits,
                self.qubit_count() - pos_qubits,
            );
        }

        // Measure position register to get vertex probabilities
        extract_position_probabilities(&state, pos_qubits, self.n_vertices)
    }
}
```

### Scalability

| Graph Size | Vertices | Position Qubits | Coin Qubits | Total Qubits | Simulatable? |
|------------|----------|----------------|-------------|-------------|-------------|
| Bacterial gene | 1K | 10 | 2-3 | 12-13 | Yes (WASM) |
| Bacterial genome | 10K | 14 | 3-4 | 17-18 | Yes (native) |
| Viral genome | 30K | 15 | 3 | 18 | Yes (native) |
| Human chromosome | 100M | 27 | 4 | 31 | Marginal |
| Human genome | 3.2B | 32 | 4 | 36 | No (hardware) |

The practical simulation envelope covers bacterial and viral genomes.
For human-scale assembly, the quantum walk serves as a subroutine for
resolving local repeat structures within classically pre-processed subgraphs.

---

## 5. Quantum Error Correction Mapped to DNA Sequencing Errors

### The Analogy

DNA sequencing introduces errors (substitutions, insertions, deletions) at rates
of 0.1-15% depending on the technology:

| Technology | Error Rate | Dominant Error Type |
|------------|-----------|-------------------|
| Illumina (short-read) | 0.1-1% | Substitutions |
| PacBio HiFi | 0.1-0.5% | Insertions/deletions |
| Oxford Nanopore | 5-15% | Insertions/deletions |
| MGI/DNBSEQ | 0.1-0.5% | Substitutions |

Quantum error correction (QEC) protects quantum information against decoherence
by encoding logical qubits into entangled physical qubits with redundant syndrome
measurements. The mathematical structure maps onto DNA sequencing error correction:

| QEC Concept | Genomic Analogue |
|-------------|-----------------|
| Physical qubit | Individual base read |
| Logical qubit | True genomic base |
| Pauli X error (bit flip) | Base substitution (A -> T) |
| Pauli Z error (phase flip) | Strand-specific error (quality score degradation) |
| Syndrome measurement | Multi-read consensus at each position |
| Decoder (MWPM) | Variant caller / consensus algorithm |
| Code distance d | Coverage depth (number of overlapping reads) |
| Threshold theorem | Minimum coverage for reliable calling |

### Formal Mapping

Define a "genomic stabilizer code" as follows:

- **Data qubits**: Each read base at a given genomic position, encoded as
  |A>=|00>, |C>=|01>, |G>=|10>, |T>=|11> (2 qubits per base)
- **Stabilizer generators**: Pairwise agreement checks between reads covering
  the same position
- **Syndrome**: The pattern of agreements and disagreements across reads
- **Decoder**: Determines the consensus base that minimizes total disagreement
  (equivalent to MEC)

```rust
/// Map DNA sequencing coverage to a stabilizer code.
///
/// Each genomic position with coverage c generates a code with:
/// - c "data qubits" (one per read base at that position)
/// - c*(c-1)/2 pairwise stabilizer checks
/// - Code distance proportional to c
///
/// This provides a principled framework for consensus calling
/// with error correction guarantees analogous to QEC thresholds.
pub struct GenomicStabilizerCode {
    /// Coverage at this genomic position
    coverage: usize,
    /// Read bases at this position (2-bit encoded)
    read_bases: Vec<u8>,
    /// Quality scores (Phred-scaled)
    quality_scores: Vec<f64>,
    /// Computed syndrome (pairwise disagreement pattern)
    syndrome: Vec<bool>,
}

impl GenomicStabilizerCode {
    /// Extract syndrome from read pileup.
    ///
    /// For c reads, generates c*(c-1)/2 pairwise agreement bits.
    /// A syndrome bit is 1 if the corresponding pair disagrees.
    pub fn extract_syndrome(&mut self) {
        let c = self.coverage;
        self.syndrome = Vec::with_capacity(c * (c - 1) / 2);

        for i in 0..c {
            for j in (i + 1)..c {
                self.syndrome.push(self.read_bases[i] != self.read_bases[j]);
            }
        }
    }

    /// Compute effective code distance.
    ///
    /// For a repetition-like code with c reads, the effective
    /// distance is approximately c (can correct floor((c-1)/2) errors).
    /// Quality-weighted distance accounts for varying error rates.
    pub fn effective_distance(&self) -> f64 {
        let total_quality: f64 = self.quality_scores.iter()
            .map(|&q| 1.0 - 10.0_f64.powf(-q / 10.0))
            .sum();
        total_quality
    }

    /// Decode using MWPM via ruQu's existing decoder.
    ///
    /// Maps the genomic syndrome to a format compatible with
    /// ruQu's MWPM decoder for optimal error correction.
    pub fn decode(&self, decoder: &mut MWPMDecoder) -> ConsensusBase {
        let bitmap = DetectorBitmap::from_bools(&self.syndrome);
        let correction = decoder.decode(&bitmap);
        apply_genomic_correction(&self.read_bases, &correction)
    }
}
```

### Threshold Analysis

The QEC threshold theorem states that if the physical error rate p is below
a threshold p*, the logical error rate decreases exponentially with code
distance d:

```
p_logical ~ (p / p*)^{floor((d+1)/2)}
```

Applied to sequencing:

| Coverage (d) | Illumina (p=0.01) | ONT (p=0.10) | p* = 0.15 |
|--------------|-------------------|--------------|-----------|
| 5x | 10^-5 | 0.008 | -- |
| 10x | 10^-10 | 6 x 10^-5 | -- |
| 20x | 10^-20 | 4 x 10^-9 | -- |
| 30x | 10^-30 | 2 x 10^-13 | -- |
| 50x | 10^-50 | 10^-21 | -- |

This provides a rigorous framework for determining minimum coverage requirements
based on sequencing platform error rates, using the same mathematical machinery
as quantum error correction threshold estimation.

---

## 6. Quantum Annealing for Phylogenetic Tree Optimization

### Problem Statement

Phylogenetic tree inference determines evolutionary relationships among species
or sequences. Maximum likelihood (ML) phylogenetics evaluates:

```
L(T, theta) = P(D | T, theta)
```

where T is the tree topology, theta are branch lengths, and D is the sequence
alignment data. The number of possible unrooted binary tree topologies for n taxa
is:

```
(2n - 5)!! = (2n-5) * (2n-7) * ... * 3 * 1
```

For n = 20 taxa: (2*20 - 5)!! = 35!! = 2.2 x 10^20 topologies.

This super-exponential search space makes exhaustive evaluation impossible.
Classical heuristics (RAxML, IQ-TREE) use hill-climbing with random restarts,
but may become trapped in local optima.

### Quantum Annealing Formulation

Encode tree topologies as binary strings representing the Prufer sequence of the
tree. A Prufer sequence uniquely encodes a labeled tree on n vertices as a
sequence of n-2 labels, each in {1, ..., n}.

**Qubit encoding**: Each label in the Prufer sequence requires ceil(log2(n))
qubits. The full encoding requires (n-2) * ceil(log2(n)) qubits.

**Cost Hamiltonian**: The negative log-likelihood is encoded as:

```
H_cost = -sum_{sites} log P(site_pattern | T(z), theta)
```

where T(z) is the tree decoded from the Prufer sequence z.

For the quantum annealing schedule:

```
H(s) = (1 - s) * H_mixer + s * H_cost

where s goes from 0 to 1 over the annealing schedule
H_mixer = -sum_i X_i (transverse field)
```

```rust
/// Quantum annealing for phylogenetic tree optimization.
///
/// Encodes tree topologies as Prufer sequences and uses
/// simulated quantum annealing (SQA) to search the topology space.
pub struct PhylogeneticAnnealer {
    /// Sequence alignment (taxa x sites)
    alignment: Vec<Vec<u8>>,
    /// Number of taxa
    n_taxa: usize,
    /// Substitution model (JC69, K2P, GTR, etc.)
    substitution_model: SubstitutionModel,
    /// Annealing schedule parameters
    schedule: AnnealingSchedule,
}

/// Annealing schedule: controls the transverse field strength.
pub struct AnnealingSchedule {
    /// Number of Trotter slices (imaginary time steps)
    pub trotter_slices: usize,
    /// Initial transverse field strength
    pub gamma_initial: f64,
    /// Final transverse field strength
    pub gamma_final: f64,
    /// Inverse temperature
    pub beta: f64,
    /// Number of Monte Carlo sweeps per temperature step
    pub sweeps_per_step: usize,
}

impl PhylogeneticAnnealer {
    /// Run simulated quantum annealing.
    ///
    /// Uses path-integral Monte Carlo to simulate the quantum
    /// annealing process on a classical computer.
    /// For small n_taxa (<= 12), also runs exact QAOA via ruqu-core
    /// for comparison.
    pub fn anneal(&self) -> PhylogeneticResult {
        let qubits_per_label = (self.n_taxa as f64).log2().ceil() as usize;
        let total_qubits = (self.n_taxa - 2) * qubits_per_label;

        if total_qubits <= 25 {
            // Exact quantum simulation via QAOA
            self.anneal_exact_qaoa(total_qubits)
        } else {
            // Simulated quantum annealing (path-integral MC)
            self.anneal_simulated(total_qubits)
        }
    }

    /// Compute log-likelihood for a given tree topology.
    fn log_likelihood(&self, prufer_sequence: &[usize]) -> f64 {
        let tree = decode_prufer_tree(prufer_sequence, self.n_taxa);
        felsenstein_pruning(&tree, &self.alignment, &self.substitution_model)
    }
}
```

### Qubit and Scaling Requirements

| Taxa | Prufer Length | Qubits/Label | Total Qubits | Topology Space |
|------|-------------|-------------|-------------|----------------|
| 6 | 4 | 3 | 12 | 945 |
| 8 | 6 | 3 | 18 | 135,135 |
| 10 | 8 | 4 | 32 | 3.4 x 10^7 |
| 12 | 10 | 4 | 40 | 1.4 x 10^10 |
| 15 | 13 | 4 | 52 | 7.9 x 10^14 |
| 20 | 18 | 5 | 90 | 2.2 x 10^20 |

Exact quantum simulation covers up to 8 taxa (18 qubits). For larger instances,
the simulated quantum annealing (SQA) approach provides quantum-inspired
optimization on classical hardware, with the circuit templates ready for
execution on future quantum annealers.

---

## 7. Quantum Machine Learning for Variant Pathogenicity Prediction

### Problem Statement

Determining whether a genetic variant is pathogenic (disease-causing) or benign
is central to clinical genomics. Current classifiers (CADD, REVEL, AlphaMissense)
use classical machine learning on features extracted from conservation scores,
protein structure, and functional annotations. Quantum machine learning (QML)
offers potential advantages for:

- High-dimensional feature spaces via quantum kernel methods
- Entanglement-based feature correlations not captured by classical kernels
- Exponential-dimension Hilbert space for implicit feature expansion

### Quantum Kernel Classification

The quantum kernel approach encodes variant features into quantum states and
computes the kernel matrix K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2 in the
exponential-dimensional Hilbert space.

**Feature encoding**: Map variant features to qubit rotations:

```
|phi(x)> = U(x) |0...0>

where U(x) = prod_{l=1}^{L} [W_l * S(x)]

S(x) = tensor_{j=1}^{n} Rz(x_j) Ry(x_j)    (feature map)
W_l = entangling layer (CZ ladder)              (entanglement)
```

```rust
/// Quantum kernel for variant pathogenicity classification.
///
/// Features per variant (standard clinical genomics features):
/// - Conservation scores (PhyloP, phastCons, GERP++)
/// - Protein impact (SIFT, PolyPhen, Grantham distance)
/// - Structural features (solvent accessibility, secondary structure)
/// - Population frequency (gnomAD AF)
/// - Functional annotations (regulatory, splice-site proximity)
pub struct VariantQuantumKernel {
    /// Number of features (typically 10-20 for clinical genomics)
    n_features: usize,
    /// Number of qubits = n_features (one qubit per feature)
    n_qubits: usize,
    /// Number of feature map layers
    n_layers: usize,
}

impl VariantQuantumKernel {
    /// Encode a variant's features into a quantum state.
    ///
    /// Each feature x_j is mapped to rotations Rz(x_j) * Ry(x_j)
    /// on qubit j, followed by entangling CZ gates between
    /// adjacent qubits. The layer is repeated n_layers times.
    pub fn encode(&self, features: &[f64]) -> QuantumState {
        assert_eq!(features.len(), self.n_features);
        let mut state = QuantumState::new(self.n_qubits);
        state.hadamard_all();

        for _layer in 0..self.n_layers {
            // Feature rotation
            for (j, &x_j) in features.iter().enumerate() {
                state.rz(x_j, j);
                state.ry(x_j, j);
            }
            // Entangling layer
            for j in 0..(self.n_qubits - 1) {
                state.cz(j, j + 1);
            }
        }

        state
    }

    /// Compute quantum kernel entry K(x_i, x_j).
    ///
    /// K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
    ///             = |sum_k conj(amp_i[k]) * amp_j[k]|^2
    pub fn kernel_entry(&self, x_i: &[f64], x_j: &[f64]) -> f64 {
        let state_i = self.encode(x_i);
        let state_j = self.encode(x_j);

        let inner_product: Complex64 = state_i.amplitudes.iter()
            .zip(state_j.amplitudes.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        inner_product.norm_sqr()
    }

    /// Build full kernel matrix for a training set.
    ///
    /// Returns a symmetric n x n matrix where K[i][j] = kernel(x_i, x_j).
    /// Complexity: O(n^2 * 2^{n_qubits} * n_layers)
    pub fn kernel_matrix(&self, dataset: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = dataset.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            matrix[i][i] = 1.0; // Self-kernel is always 1
            for j in (i + 1)..n {
                let k = self.kernel_entry(&dataset[i], &dataset[j]);
                matrix[i][j] = k;
                matrix[j][i] = k;
            }
        }

        matrix
    }
}
```

### Classification Pipeline

```
Variant Pathogenicity Prediction Pipeline
==========================================

1. Feature extraction (classical):
   - Conservation: PhyloP, phastCons, GERP++ scores
   - Protein: SIFT, PolyPhen, Grantham distance
   - Structure: AlphaFold pLDDT, solvent accessibility
   - Population: gnomAD allele frequency
   - Regulatory: ENCODE annotations, splice-site distance

2. Quantum kernel computation:
   - Encode each variant's features into quantum state
   - Compute kernel matrix K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
   - Time: O(n^2 * 2^{n_features} * layers)

3. Classical SVM with quantum kernel:
   - Solve dual SVM: max sum_i alpha_i - (1/2) sum_{ij} alpha_i alpha_j y_i y_j K_{ij}
   - Classify new variants via kernel evaluation against support vectors

4. Output: Pathogenicity score in [0, 1]
```

### Qubit Requirements

| Feature Set | Features | Qubits | State Size | Kernel Time (per pair) |
|-------------|----------|--------|-----------|----------------------|
| Minimal (conservation only) | 4 | 4 | 256 B | <0.01ms |
| Standard (clinical) | 10 | 10 | 16 KB | ~0.1ms |
| Extended (all features) | 16 | 16 | 1 MB | ~10ms |
| Full (with structural) | 20 | 20 | 16 MB | ~500ms |
| Comprehensive | 25 | 25 | 512 MB | ~60s |

For clinical use, the standard 10-feature set at 10 qubits is fully practical
in both native and WASM environments.

---

## 8. Hybrid Classical-Quantum Pipeline

### Decision Boundary Framework

Not every genomic computation benefits from quantum processing. The hybrid
pipeline routes operations to classical or quantum backends based on problem
characteristics:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 HYBRID ROUTING DECISION ENGINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: Genomic operation + problem parameters                       │
│                                                                      │
│  Decision criteria:                                                  │
│  1. Problem size (N)         ─┐                                      │
│  2. NP-hardness              ─┼── Quantum advantage estimate         │
│  3. Available qubits         ─┤                                      │
│  4. Required accuracy        ─┘                                      │
│                                                                      │
│  Route:                                                              │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────┐    │
│  │  CLASSICAL    │  │  HYBRID        │  │  QUANTUM             │    │
│  │              │  │                │  │                      │    │
│  │  BWA-MEM2    │  │  Classical     │  │  Full quantum        │    │
│  │  GATK        │  │  preprocessing │  │  algorithm           │    │
│  │  HapCUT2     │  │  + quantum     │  │  (simulator or       │    │
│  │  RAxML       │  │  subroutine    │  │   hardware)          │    │
│  └──────────────┘  └────────────────┘  └──────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Routing Rules

```rust
/// Determine whether to use classical, hybrid, or quantum processing.
pub enum ProcessingMode {
    /// Pure classical (no quantum advantage at this scale)
    Classical,
    /// Classical preprocessing + quantum subroutine
    Hybrid {
        classical_fraction: f64,
        quantum_subroutine: QuantumSubroutine,
    },
    /// Full quantum processing
    Quantum,
}

pub fn route_genomic_operation(
    operation: &GenomicOperation,
    available_qubits: usize,
) -> ProcessingMode {
    match operation {
        GenomicOperation::KmerSearch { genome_size, k } => {
            let required_qubits = (*genome_size as f64).log2().ceil() as usize;
            if required_qubits <= available_qubits {
                ProcessingMode::Quantum
            } else if *genome_size > 1_000_000 {
                // Classical FM-index is faster for indexed search
                ProcessingMode::Classical
            } else {
                ProcessingMode::Hybrid {
                    classical_fraction: 0.9,
                    quantum_subroutine: QuantumSubroutine::Grover,
                }
            }
        }

        GenomicOperation::HaplotypeAssembly { n_fragments, .. } => {
            if *n_fragments <= available_qubits && *n_fragments <= 25 {
                ProcessingMode::Quantum
            } else if *n_fragments <= 50 {
                // Partition into blocks, quantum solve each block
                ProcessingMode::Hybrid {
                    classical_fraction: 0.5,
                    quantum_subroutine: QuantumSubroutine::QAOA,
                }
            } else {
                ProcessingMode::Classical // HapCUT2 / WhatsHap
            }
        }

        GenomicOperation::MolecularInteraction { n_orbitals, .. } => {
            let n_qubits = 2 * n_orbitals;
            if n_qubits <= available_qubits {
                ProcessingMode::Quantum
            } else {
                ProcessingMode::Classical // DFT / force fields
            }
        }

        GenomicOperation::PhylogeneticTree { n_taxa, .. } => {
            let qubits_needed = (*n_taxa - 2) *
                (*n_taxa as f64).log2().ceil() as usize;
            if qubits_needed <= available_qubits && *n_taxa <= 8 {
                ProcessingMode::Quantum
            } else {
                ProcessingMode::Hybrid {
                    classical_fraction: 0.8,
                    quantum_subroutine: QuantumSubroutine::QuantumAnnealing,
                }
            }
        }

        GenomicOperation::VariantClassification { n_features, .. } => {
            if *n_features <= available_qubits {
                ProcessingMode::Hybrid {
                    classical_fraction: 0.3, // SVM is classical, kernel is quantum
                    quantum_subroutine: QuantumSubroutine::QuantumKernel,
                }
            } else {
                ProcessingMode::Classical
            }
        }

        GenomicOperation::GenomeAssembly { graph_size, .. } => {
            let walk_qubits = (*graph_size as f64).log2().ceil() as usize + 4;
            if walk_qubits <= available_qubits {
                ProcessingMode::Hybrid {
                    classical_fraction: 0.7,
                    quantum_subroutine: QuantumSubroutine::QuantumWalk,
                }
            } else {
                ProcessingMode::Classical // SPAdes / Canu
            }
        }
    }
}
```

### Decision Boundary Summary

| Operation | Classical Preferred When | Quantum Preferred When |
|-----------|------------------------|----------------------|
| k-mer search | Pre-built FM-index available | No index; searching unstructured data |
| Haplotype assembly | < 20 fragments or > 50 fragments | 20-50 fragments (QAOA sweet spot) |
| Molecular interaction | > 32 orbitals or force-field sufficient | <= 16 orbitals, chemical accuracy needed |
| Phylogenetics | > 8 taxa or fast heuristic sufficient | <= 8 taxa, global optimum required |
| Variant classification | < 8 or > 25 features | 10-20 features (quantum kernel advantage zone) |
| Genome assembly | Reference-guided assembly available | De novo, repeat-rich regions |

---

## 9. Qubit Requirements Summary

### Per-Operation Qubit Budget

| Genomic Operation | Qubits Required | Formula | 25-Qubit Sim Covers |
|-------------------|----------------|---------|---------------------|
| k-mer search (Grover) | ceil(log2(N)) | N = genome length | Genomes up to 33M bp |
| Haplotype assembly (QAOA) | F | F = fragment count | Up to 25 fragments |
| VQE molecular | 2 * M | M = spatial orbitals | Up to 12 orbitals |
| Quantum walk (assembly) | ceil(log2(V)) + ceil(log2(d)) | V = vertices, d = degree | Graphs up to 4M nodes |
| Phylogenetics (annealing) | (n-2) * ceil(log2(n)) | n = taxa count | Up to 8 taxa |
| Variant QML kernel | F | F = features | Up to 25 features |
| DNA error correction | 2 * C | C = coverage depth | Up to 12x coverage |

### Hardware Roadmap Projection

| Timeframe | Hardware Qubits | Error Rate | Genomic Operations Enabled |
|-----------|----------------|-----------|--------------------------|
| **2026 (simulation)** | 25 (simulated) | 0 (exact) | All at reduced scale |
| **2027-2028 (NISQ)** | 100-500 | 10^-3 | Haplotype assembly (50 fragments), variant QML |
| **2029-2030 (early FT)** | 1000-5000 logical | 10^-6 | Phylogenetics (15 taxa), bacterial assembly |
| **2032+ (mature FT)** | 10^4-10^5 logical | 10^-10 | Full human genome k-mer search, protein folding |

---

## 10. Error Mitigation for Biological Data

### Quantum Noise in Genomic Context

Quantum computations on NISQ hardware introduce errors that compound with circuit
depth. For genomic algorithms, which typically require deep circuits (Grover
iterations, QAOA layers, VQE optimization loops), error mitigation is essential.

### Mitigation Strategies

| Strategy | Technique | Applicable Algorithms | Overhead |
|----------|-----------|----------------------|----------|
| **Zero-noise extrapolation (ZNE)** | Run at multiple noise levels, extrapolate to zero | VQE molecular, QAOA haplotype | 3-5x circuit executions |
| **Probabilistic error cancellation (PEC)** | Quasi-probability decomposition of ideal operations | All algorithms | Exponential sampling overhead |
| **Symmetry verification** | Post-select on states satisfying known symmetries | VQE (electron number), haplotype (read consistency) | Discard fraction of shots |
| **Dynamical decoupling** | Insert identity-equivalent pulse sequences during idle | Grover, quantum walks | Minimal overhead |
| **Quantum subspace expansion (QSE)** | Expand variational ansatz in error-corrected subspace | VQE molecular | O(n^2) additional measurements |
| **Readout error mitigation** | Calibrate and invert measurement confusion matrix | All algorithms | O(2^n) calibration (for n measured qubits) |

### Genomics-Specific Error Handling

Genomic data has inherent redundancy (coverage depth, biological constraints)
that can be exploited for error mitigation:

```rust
/// Genomics-aware error mitigation.
///
/// Exploits biological constraints to detect and correct
/// quantum computation errors.
pub struct GenomicErrorMitigation {
    /// Symmetry constraints from biology
    pub symmetries: Vec<BiologicalSymmetry>,
    /// Noise model for the target hardware
    pub noise_model: NoiseModel,
}

pub enum BiologicalSymmetry {
    /// Base pair complementarity (A-T, C-G)
    /// Total A count must approximately equal T count in double-stranded DNA
    ComplementarityConservation,

    /// Codon reading frame preservation
    /// Protein-coding variants must maintain triplet structure
    ReadingFramePreservation,

    /// Allele frequency bounds
    /// Variant allele frequency must be in [0, 1]
    AlleleFrequencyBounds,

    /// Phylogenetic ultrametricity
    /// For a molecular clock, leaf-to-root distances must be approximately equal
    Ultrametricity { tolerance: f64 },

    /// Electron number conservation (for VQE molecular)
    /// The encoded molecular system must preserve total electron count
    ElectronNumberConservation { n_electrons: usize },
}

impl GenomicErrorMitigation {
    /// Post-select measurement results that satisfy biological constraints.
    ///
    /// Discard quantum computation results that violate known
    /// biological rules, reducing the effective error rate at the
    /// cost of reduced sampling efficiency.
    pub fn post_select(
        &self,
        measurements: &[MeasurementOutcome],
    ) -> Vec<MeasurementOutcome> {
        measurements.iter()
            .filter(|m| self.symmetries.iter().all(|s| s.is_satisfied(m)))
            .cloned()
            .collect()
    }

    /// Zero-noise extrapolation with genomic-aware scaling.
    ///
    /// Runs the circuit at noise levels [1x, 1.5x, 2x] and
    /// extrapolates to zero noise using Richardson extrapolation.
    pub fn zero_noise_extrapolate(
        &self,
        circuit: &QuantumCircuit,
        observable: &PauliSum,
        noise_factors: &[f64],
    ) -> f64 {
        let expectations: Vec<f64> = noise_factors.iter()
            .map(|&factor| {
                let noisy = self.noise_model.scale(factor);
                execute_with_noise(circuit, observable, &noisy)
            })
            .collect();

        richardson_extrapolation(&noise_factors, &expectations)
    }
}
```

---

## 11. WASM Quantum Genomics: Browser-Based Simulation

### Architecture

The ruqu-wasm crate (ADR-QE-003) enables quantum genomics algorithms to run
entirely in the browser. This is significant for bioinformatics education,
clinical decision support tools, and privacy-sensitive genomic analysis where
data must not leave the client.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Browser Environment                            │
│                                                                   │
│  ┌──────────────┐     ┌──────────────────────────────────────┐  │
│  │  JavaScript  │     │         WebAssembly Module            │  │
│  │  Frontend    │     │                                      │  │
│  │              │     │  ruqu-wasm                            │  │
│  │  - Upload    │────>│    │                                  │  │
│  │    VCF/BAM   │     │    ├── ruqu-core (state vector sim)  │  │
│  │  - Configure │     │    │                                  │  │
│  │    algorithm │     │    ├── ruqu-algorithms::genomics      │  │
│  │  - Display   │<────│    │     ├── kmer_search              │  │
│  │    results   │     │    │     ├── haplotype_assembly       │  │
│  │              │     │    │     ├── variant_classifier       │  │
│  │  WebWorker   │     │    │     └── ...                      │  │
│  │  (optional)  │     │    │                                  │  │
│  └──────────────┘     │    └── WASM SIMD128 acceleration     │  │
│                        │                                      │  │
│                        │  Constraints:                         │  │
│                        │  - Max 25 qubits (1 GB state)        │  │
│                        │  - ~2x slower than native             │  │
│                        │  - Single-threaded (unless SAB)       │  │
│                        └──────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### WASM-Feasible Genomic Operations

| Operation | Max Problem Size (WASM) | Typical Time | Use Case |
|-----------|------------------------|-------------|----------|
| k-mer search (Grover) | 33M bp genome (25 qubits) | 30-120s | Bacterial genome search |
| Haplotype assembly (QAOA) | 25 fragments, p=3 | 5-30s | Single active region |
| VQE molecular | 12 orbitals (24 qubits) | 10-60s per iteration | Base pair energetics |
| Variant QML kernel | 20 features | ~500ms per pair | Clinical variant classification |
| Quantum walk | 4M node graph (22 qubits) | 10-60s | Viral genome assembly |
| Phylogenetics | 6 taxa (12 qubits) | 1-10s | Small clade analysis |

### JavaScript API Extension

```javascript
import { QuantumGenomics } from 'ruqu-wasm';

// k-mer search
const searcher = new QuantumGenomics.KmerSearcher(referenceSequence);
const results = searcher.groverSearch(queryKmer, { k: 31 });
console.log(`Found ${results.matches.length} matches in ${results.elapsed_ms}ms`);
console.log(`Grover iterations: ${results.iterations}`);

// Variant classification
const classifier = new QuantumGenomics.VariantClassifier({
    nFeatures: 10,
    nLayers: 3,
    trainingData: labeledVariants,
});
const prediction = classifier.predict(variantFeatures);
console.log(`Pathogenicity score: ${prediction.score}`);

// Haplotype assembly
const assembler = new QuantumGenomics.HaplotypeAssembler({
    fragments: fragmentMatrix,
    qualities: qualityMatrix,
    qaoaDepth: 3,
});
const haplotypes = assembler.solve();
console.log(`MEC score: ${haplotypes.mecScore}`);
```

---

## 12. Performance Projections

### Speedup Estimates: Quantum vs. Classical

| Operation | Classical Baseline | Quantum Algorithm | Theoretical Speedup | Practical Speedup (Simulation) | Hardware Projection |
|-----------|-------------------|-------------------|--------------------|-----------------------------|-------------------|
| k-mer search (3.2B bp) | O(N) = 3.2 x 10^9 | Grover O(sqrt(N)) | 56,568x fewer queries | 1x (simulation overhead) | 56,568x (32-qubit FT) |
| Haplotype assembly (50 fragments) | O(2^F) exact, O(F^3) heuristic | QAOA p=5 | ~10-100x over exact | 2-5x over heuristic (25 qubits) | 10-50x (50-qubit NISQ) |
| VQE molecular (12 orbitals) | O(N^7) CCSD(T) | VQE O(poly(N) * iterations) | 10-100x | 1-5x (exact expectations) | 50-200x (24-qubit FT) |
| Quantum walk (10K nodes) | O(N) classical random walk | O(sqrt(N)) quantum walk | 100x | 10-50x (14-qubit sim) | 100x (20-qubit) |
| Phylogenetics (8 taxa) | O(N! * k) heuristic ML | Quantum annealing | 10-1000x over exhaustive | 5-20x (18-qubit sim) | 100-1000x (50-qubit) |
| Variant classification (10 features) | O(F * N) classical SVM | Quantum kernel SVM | Unknown (problem-dependent) | 1-3x (kernel advantage) | 2-10x (10-qubit NISQ) |

### Key Caveats

1. **Simulation does not achieve quantum speedup**: The state-vector simulator
   runs on classical hardware and does not provide the O(sqrt(N)) query complexity
   advantage of real quantum hardware. The simulator's value lies in algorithm
   development, validation, and hardware-readiness.

2. **QAOA advantage is depth-dependent**: QAOA with depth p=1 often does not
   outperform classical greedy algorithms. Meaningful advantage requires p >= 3-5,
   which increases circuit depth and noise sensitivity.

3. **VQE advantage is problem-specific**: For small molecules (< 16 orbitals),
   classical methods (FCI, CCSD(T)) may be faster. VQE's advantage emerges for
   strongly correlated systems where classical methods fail to converge.

4. **Quantum walk advantage requires coherent traversal**: Decoherence during the
   walk erases the quadratic speedup. Error correction or short walk times are
   essential.

---

## Alternatives Considered

### Alternative 1: Classical-Only Genomics Pipeline

Use established tools (BWA-MEM2, GATK, SPAdes, RAxML) without quantum components.

**Rejected because**: While classical tools are mature and performant, they face
fundamental computational barriers for NP-hard problems (haplotype assembly,
phylogenetics) and cannot model quantum molecular interactions from first
principles. The quantum genomics engine provides a research platform for
developing algorithms that will yield practical advantage on future hardware.

### Alternative 2: Cloud Quantum Hardware Only

Delegate all quantum computation to IBM Quantum, Google Quantum AI, or IonQ
cloud services.

**Rejected because**: (1) Genomic data is privacy-sensitive (HIPAA, GDPR) and
should not be transmitted to third-party quantum cloud services without careful
compliance evaluation. (2) Cloud quantum hardware introduces network latency
incompatible with interactive analysis. (3) Current NISQ hardware error rates
limit practical utility for genomic-scale problems. The simulation approach
enables development without cloud dependency.

### Alternative 3: Quantum-Inspired Classical Algorithms Only

Implement tensor network, simulated annealing, and other quantum-inspired
classical algorithms without true quantum simulation.

**Rejected because**: Quantum-inspired algorithms discard the circuit-level
structure that makes algorithms transferable to quantum hardware. By maintaining
quantum circuit representations, the genomics engine produces hardware-ready
circuits that can be executed on real quantum processors when they become
available, without redesign.

---

## Consequences

### Benefits

1. **Hardware-ready algorithm library**: All quantum genomics algorithms produce
   standard quantum circuits that can be compiled to any gate-model quantum
   processor via OpenQASM export (future work).

2. **Exact validation environment**: The state-vector simulator provides exact
   arithmetic for algorithm validation, free from the statistical noise of
   hardware-based evaluation.

3. **Browser accessibility**: Via ruqu-wasm, bioinformaticians can experiment
   with quantum genomics algorithms without installing quantum computing
   software or accessing specialized hardware.

4. **Unified framework**: By building on the existing ruQu crate architecture
   (Grover from ADR-QE-006, QAOA from ADR-QE-007, VQE from ADR-QE-005, QEC from
   ADR-QE-008), the genomics module reuses battle-tested quantum primitives.

5. **Principled hybrid routing**: The classical-quantum decision boundary
   framework prevents the anti-pattern of forcing quantum solutions where
   classical methods are superior.

6. **QEC-genomics bridge**: The formal mapping between quantum error correction
   and sequencing error correction provides a novel theoretical framework for
   coverage analysis and consensus calling.

### Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Quantum advantage does not materialize for genomics | Medium | High | Framework is useful for education and algorithm development regardless |
| Simulation scale (25 qubits) too small for real problems | High | Medium | Design for hardware portability; use simulation for validation |
| Genomic encoding overhead reduces effective qubit count | Medium | Medium | Develop compact encodings; use ancilla-free constructions |
| Competition from specialized classical algorithms | High | Low | Hybrid pipeline gracefully falls back to classical |
| Privacy concerns with quantum cloud hardware | Medium | High | WASM-based local simulation eliminates data transmission |
| QML kernel advantage unclear for variant classification | High | Low | Compare against classical kernels; publish null results |

---

## References

### Quantum Computing

- Grover, L.K. "A fast quantum mechanical algorithm for database search." STOC 1996.
- Farhi, E., Goldstone, J., Gutmann, S. "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028, 2014.
- Peruzzo, A. et al. "A variational eigenvalue solver on a photonic quantum processor." Nature Communications 5, 4213, 2014.
- Szegedy, M. "Quantum speed-up of Markov chain based algorithms." FOCS 2004.
- Schuld, M., Killoran, N. "Quantum machine learning in feature Hilbert spaces." Physical Review Letters 122, 040504, 2019.
- Kadowaki, T., Nishimori, H. "Quantum annealing in the transverse Ising model." Physical Review E 58, 5355, 1998.

### Genomics

- Li, H. "Aligning sequence reads, clone sequences and assembly contigs with BWA-MEM." arXiv:1303.3997, 2013.
- McKenna, A. et al. "The Genome Analysis Toolkit: a MapReduce framework for analyzing next-generation DNA sequencing data." Genome Research 20, 1297-1303, 2010.
- Lancia, G. et al. "SNP problems, complexity and algorithms." ESA 2001.
- Patterson, M. et al. "WhatsHap: Weighted Haplotype Assembly for Future-Generation Sequencing Reads." Journal of Computational Biology 22, 498-509, 2015.
- Felsenstein, J. "Evolutionary trees from DNA sequences: a maximum likelihood approach." Journal of Molecular Evolution 17, 368-376, 1981.
- Pevzner, P.A. et al. "An Eulerian path approach to DNA fragment assembly." PNAS 98, 9748-9753, 2001.
- Cheng, J. et al. "Accurate proteome-wide missense variant effect prediction with AlphaMissense." Science 381, eadg7492, 2023.

### RuVector ADRs

- [ADR-QE-001: Quantum Engine Core Architecture](./ADR-QE-001-quantum-engine-core-architecture.md)
- [ADR-QE-002: Crate Structure & Integration](./ADR-QE-002-crate-structure-integration.md)
- [ADR-QE-003: WASM Compilation Strategy](./ADR-QE-003-wasm-compilation-strategy.md)
- [ADR-QE-005: VQE Algorithm Support](./ADR-QE-005-vqe-algorithm-support.md)
- [ADR-QE-006: Grover's Search Implementation](./ADR-QE-006-grover-search-implementation.md)
- [ADR-QE-007: QAOA MaxCut Implementation](./ADR-QE-007-qaoa-maxcut-implementation.md)
- [ADR-QE-008: Surface Code Error Correction](./ADR-QE-008-surface-code-error-correction.md)
- [ADR-QE-014: Exotic Discoveries](./ADR-QE-014-exotic-discoveries.md)
- [ADR-001: ruQu Architecture](../../crates/ruQu/docs/adr/ADR-001-ruqu-architecture.md)
