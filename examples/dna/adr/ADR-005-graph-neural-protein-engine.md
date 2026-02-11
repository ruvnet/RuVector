# ADR-005: Graph Neural Network Protein Structure & Interaction Engine

**Status**: Proposed
**Date**: 2026-02-11
**Authors**: ruv.io, RuVector Team
**Deciders**: Architecture Review Board
**Target Crates**: `ruvector-gnn`, `ruvector-graph`, `ruvector-mincut`, `ruvector-mincut-gated-transformer`

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-11 | ruv.io | Initial proposal for protein graph neural engine |

---

## Context

### Proteins as Graphs: A Natural Representation

Proteins are inherently graph-structured objects at every scale of biological organization. A protein's three-dimensional fold is determined by the spatial arrangement of amino acid residues connected through covalent peptide bonds, hydrogen bonds, disulfide bridges, and non-covalent contacts. This structure maps directly onto graph formalisms:

- **Nodes** represent biological entities (amino acid residues, atoms, whole proteins, enzymes, genes)
- **Edges** encode relationships (covalent bonds, spatial proximity, physical interactions, regulatory influence, catalytic activity)

Traditional sequence-based deep learning (CNNs, RNNs, transformers) treats proteins as 1D strings of amino acids, discarding the rich 3D contact topology. Structure prediction methods like AlphaFold2 recover this topology through expensive iterative attention over multiple sequence alignments (MSAs) and template search. Graph Neural Networks operate directly on the native graph topology, avoiding the information bottleneck of flattening 3D structure into sequential representations.

### The Multi-Scale Biological Graph Hierarchy

Biological systems exhibit graph structure at six distinct scales, each requiring different GNN architectures and graph formulations:

| Scale | Nodes | Edges | Typical Size | Key Application |
|-------|-------|-------|-------------|-----------------|
| **Atomic** | Atoms (C, N, O, S, ...) | Covalent/non-covalent bonds | 500-50,000 atoms | Drug binding, molecular docking |
| **Residue** | Amino acid residues | Spatial contacts (< 8A) | 50-5,000 residues | Protein folding, stability prediction |
| **Domain** | Protein domains | Domain-domain interfaces | 2-20 domains | Multi-domain architecture |
| **Protein-Protein** | Whole proteins | Physical interactions | ~20,000 proteins (human) | PPI network analysis |
| **Metabolic** | Enzymes, metabolites | Catalytic reactions | ~2,500 reactions (human) | Pathway flux analysis |
| **Regulatory** | Genes, transcription factors | Regulatory influence | ~25,000 genes (human) | Gene network inference |

### Why RuVector's Existing Graph Stack Is Uniquely Suited

RuVector already provides the foundational components required for a protein graph engine:

| Requirement | RuVector Component | Capability |
|-------------|-------------------|------------|
| Message passing GNN layers | `ruvector-gnn::RuvectorLayer` | Multi-head attention, GRU update, weighted aggregation |
| Graph storage and query | `ruvector-graph::GraphDB` | Property graphs, Cypher queries, ACID transactions, hyperedges |
| Community detection | `ruvector-mincut::cluster::hierarchy` | 3-level hierarchical decomposition with expander detection |
| Graph partitioning | `ruvector-mincut::MinCutBuilder` | Exact and approximate min-cut with subpolynomial update time |
| Gated inference | `ruvector-mincut-gated-transformer` | Coherence-gated transformer with spectral position encoding |
| Spectral methods | `ruvector-mincut-gated-transformer::spectral` | Sparse CSR Laplacian, eigendecomposition, spectral encoding |
| Continual learning | `ruvector-gnn::ewc` | Elastic Weight Consolidation for preventing catastrophic forgetting |
| Differentiable search | `ruvector-gnn::search` | Soft attention-based search over embedding spaces |
| WASM compilation | `ruvector-mincut::wasm` | Browser-deployable graph algorithms via WebAssembly |

The missing piece is a unifying architecture that maps these capabilities onto the specific graph representations and biological constraints of protein science.

---

## Decision

### Implement a Multi-Scale GNN Architecture for Protein and Molecular Analysis

We will build a `ProteinGraphEngine` that provides:

1. Six graph representation layers spanning atom-level to genome-scale
2. Four specialized GNN architectures matched to biological graph properties
3. Graph partitioning-based drug target identification via `ruvector-mincut`
4. Coherence-gated pathway inference via `ruvector-mincut-gated-transformer`
5. WASM-deployable interactive 3D protein visualization
6. Integration with the genomic attention layers from ADR-001 through ADR-004

---

## Graph Representations

### 1. Residue Contact Graphs

**Biological context.** In a folded protein, amino acid residues that are distant in the primary sequence may be spatially adjacent. The set of such spatial proximities -- the contact map -- largely determines the protein's three-dimensional fold and mechanical properties.

**Formal definition.** Given a protein with `N` residues, the residue contact graph is:

```
G_contact = (V, E, X_v, X_e)

V = {r_1, r_2, ..., r_N}           -- one node per amino acid residue
E = {(r_i, r_j) : d(C_alpha_i, C_alpha_j) < tau}  -- edges from spatial proximity

X_v in R^{N x d_v}                 -- node feature matrix
X_e in R^{|E| x d_e}              -- edge feature matrix
```

**Distance threshold.** The standard C-alpha to C-alpha distance threshold is tau = 8.0 Angstroms, which captures the dominant non-covalent interactions while maintaining sparsity. The average node degree under this threshold is approximately 10-14 for globular proteins, yielding |E| ~ 6N.

**Node features (d_v = 41):**

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Amino acid type | 20 | One-hot encoding of the 20 standard amino acids |
| Secondary structure | 3 | One-hot: helix, strand, coil |
| Solvent accessibility | 1 | Relative solvent-accessible surface area [0, 1] |
| Phi/psi angles | 4 | sin(phi), cos(phi), sin(psi), cos(psi) |
| Sequence position | 1 | Normalized position i/N in [0, 1] |
| Conservation score | 1 | From multiple sequence alignment (Shannon entropy) |
| B-factor | 1 | Crystallographic temperature factor (flexibility) |
| Charge | 1 | Residue formal charge at pH 7.4 |
| Hydrophobicity | 1 | Kyte-Doolittle hydropathy index |
| Evolutionary coupling | 8 | Top-8 coevolution features from direct coupling analysis |

**Edge features (d_e = 7):**

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Euclidean distance | 1 | C-alpha to C-alpha distance in Angstroms |
| Sequence separation | 1 | |i - j| / N normalized sequence distance |
| Contact type | 3 | One-hot: backbone-backbone, backbone-sidechain, sidechain-sidechain |
| Direction cosines | 2 | Orientation of the C-alpha vector in local coordinate frame |

**RuVector mapping:**

```rust
// Residue contact graph construction using ruvector-graph
use ruvector_graph::{GraphDB, NodeBuilder, EdgeBuilder};

fn build_contact_graph(
    residues: &[Residue],
    coords: &[[f32; 3]],  // C-alpha coordinates
    threshold: f32,        // 8.0 Angstroms
) -> GraphDB {
    let mut db = GraphDB::new();

    // Add residue nodes with feature vectors
    for (i, res) in residues.iter().enumerate() {
        let features = encode_residue_features(res, i, residues.len());
        db.add_node(NodeBuilder::new()
            .with_label("Residue")
            .with_property("index", i)
            .with_property("amino_acid", res.aa_type)
            .with_property("embedding", features)
            .build());
    }

    // Add contact edges based on spatial proximity
    for i in 0..residues.len() {
        for j in (i + 1)..residues.len() {
            let dist = euclidean_distance(&coords[i], &coords[j]);
            if dist < threshold {
                let edge_features = encode_edge_features(
                    &residues[i], &residues[j], dist, i, j, residues.len()
                );
                db.add_edge(EdgeBuilder::new()
                    .from(i).to(j)
                    .with_label("SpatialContact")
                    .with_property("distance", dist)
                    .with_property("features", edge_features)
                    .build());
            }
        }
    }
    db
}
```

### 2. Molecular Graphs (Atom-Level)

**Biological context.** Drug-protein interactions, enzymatic catalysis, and ligand binding all depend on atom-level geometry and electronic structure. Modeling at atomic resolution is essential for predicting binding affinities, identifying pharmacophores, and virtual screening of drug candidates.

**Formal definition:**

```
G_mol = (V_atom, E_bond, X_atom, X_bond)

V_atom = {a_1, a_2, ..., a_M}      -- one node per atom
E_bond = E_cov U E_noncov           -- covalent and non-covalent bonds

E_cov  = {(a_i, a_j) : covalent bond between atoms i, j}
E_noncov = {(a_i, a_j) : d(a_i, a_j) < 5.0A and non-bonded interaction}
```

**Node features (d_atom = 16):**

| Feature | Dim | Description |
|---------|-----|-------------|
| Atomic number | 1 | Z (or one-hot over {C, N, O, S, P, H, halogen, metal, other}) |
| Element type | 9 | One-hot encoding of common biological elements |
| Hybridization | 3 | sp, sp2, sp3 |
| Formal charge | 1 | Integer charge |
| Aromaticity | 1 | Boolean: is aromatic ring member |
| Num hydrogens | 1 | Implicit hydrogen count |

**Edge features (d_bond = 6):**

| Feature | Dim | Description |
|---------|-----|-------------|
| Bond type | 4 | One-hot: single, double, triple, aromatic |
| Is conjugated | 1 | Boolean conjugation flag |
| Distance | 1 | Interatomic distance in Angstroms |

**Drug-protein complex graph.** For modeling drug binding, we construct a heterogeneous graph combining the protein residue graph and the drug molecular graph with cross-edges representing binding contacts:

```
G_complex = G_protein U G_drug U E_binding

E_binding = {(r_i, a_j) : min_atom_distance(r_i, a_j) < 4.5A}
```

This heterogeneous formulation is critical because protein nodes and drug-atom nodes have different feature dimensions and different edge semantics.

### 3. Protein-Protein Interaction (PPI) Networks

**Biological context.** Proteins rarely act in isolation. The human interactome comprises approximately 20,000 proteins connected by an estimated 300,000-650,000 binary physical interactions. PPI networks govern signal transduction, immune response, transcriptional regulation, and virtually every cellular process. Disruptions in PPI networks underlie cancer, neurodegeneration, and infectious disease.

**Formal definition:**

```
G_PPI = (V_protein, E_interact, X_protein, X_interact)

V_protein = {p_1, p_2, ..., p_K}    -- K ~ 20,000 for human proteome
E_interact = {(p_i, p_j) : physical interaction detected}

Adjacency matrix: A in {0, 1}^{K x K}
Degree matrix: D = diag(d_1, d_2, ..., d_K) where d_i = sum_j A_{ij}
```

**Node features (d_protein = 128):**

| Feature | Dim | Description |
|---------|-----|-------------|
| Gene ontology terms | 64 | Binary vector of top-64 GO annotations |
| Domain composition | 32 | Pfam domain fingerprint |
| Subcellular localization | 12 | One-hot: nucleus, cytoplasm, membrane, etc. |
| Expression profile | 16 | Tissue expression vector from GTEx |
| Sequence length | 1 | log(sequence_length) |
| Isoelectric point | 1 | Predicted pI |
| Molecular weight | 1 | log(MW in Daltons) |
| Disorder content | 1 | Fraction of intrinsically disordered residues |

**Edge features (d_interact = 8):**

| Feature | Dim | Description |
|---------|-----|-------------|
| Experimental evidence | 3 | One-hot: Y2H, AP-MS, structural |
| Confidence score | 1 | STRING/IntAct confidence [0, 1] |
| Co-expression | 1 | Pearson correlation of expression profiles |
| Co-evolution | 1 | Mirrortree evolutionary coupling score |
| Interface area | 1 | Buried surface area at interface (Angstroms squared) |
| Stoichiometry | 1 | Interaction stoichiometry (1:1, 1:2, etc.) |

**Graph-theoretic properties of PPI networks:**

```
Scale-free degree distribution: P(k) ~ k^(-gamma), gamma in [2.1, 2.4]
Clustering coefficient: C ~ 0.1 - 0.3 (much higher than random)
Average path length: <l> ~ 4-5 (small-world property)
Modularity: Q ~ 0.4 - 0.6 (strong community structure)
```

These properties have direct implications for GNN architecture: the heavy-tailed degree distribution means neighbor sampling (as in GraphSAGE) is essential for hub proteins with degree > 100.

### 4. Metabolic Pathway Graphs

**Biological context.** Metabolism is the set of enzyme-catalyzed chemical reactions that sustain life. Metabolic pathways form directed bipartite graphs connecting metabolites (substrates and products) through enzymatic reactions. Understanding metabolic flux is essential for drug design (enzyme inhibitors), metabolic engineering (synthetic biology), and diagnosing inborn errors of metabolism.

**Formal definition.** Metabolic networks are naturally represented as directed hypergraphs, but we model them as bipartite graphs for GNN compatibility:

```
G_metab = (V_enzyme U V_metabolite, E_substrate U E_product)

V_enzyme = {e_1, ..., e_P}          -- enzyme nodes
V_metabolite = {m_1, ..., m_Q}      -- metabolite nodes

E_substrate = {(m_i, e_j) : m_i is a substrate of enzyme e_j}   -- directed: metabolite -> enzyme
E_product   = {(e_j, m_k) : e_j produces metabolite m_k}        -- directed: enzyme -> metabolite

This forms a directed bipartite graph with two edge types.
```

For the human metabolic network (from KEGG/Recon3D):
- |V_enzyme| ~ 1,500-2,500
- |V_metabolite| ~ 2,000-4,000
- |E_substrate| + |E_product| ~ 8,000-15,000

**RuVector hyperedge representation.** Since enzymatic reactions involve multiple substrates and multiple products simultaneously, `ruvector-graph::Hyperedge` provides a natural representation:

```rust
use ruvector_graph::{Hyperedge, HyperedgeBuilder};

// A single enzymatic reaction as a hyperedge
let reaction = HyperedgeBuilder::new()
    .with_label("Reaction")
    .with_property("enzyme", "hexokinase")
    .with_property("ec_number", "2.7.1.1")
    .with_source_nodes(vec![glucose_id, atp_id])     // substrates
    .with_target_nodes(vec![g6p_id, adp_id])         // products
    .with_property("delta_g", -16.7)                  // Gibbs free energy (kJ/mol)
    .with_property("km", 0.1)                         // Michaelis constant (mM)
    .with_property("vmax", 100.0)                     // Maximum velocity
    .build();
```

### 5. Gene Regulatory Networks (GRN)

**Biological context.** Gene regulatory networks encode the control logic of the cell. Transcription factors (TFs) bind to promoter and enhancer regions to activate or repress target gene expression. Understanding GRNs is central to developmental biology, cancer genomics, and cell reprogramming (e.g., Yamanaka factors for induced pluripotent stem cells).

**Formal definition:**

```
G_GRN = (V_gene, E_regulate, X_gene, sigma)

V_gene = {g_1, ..., g_N}            -- N ~ 25,000 human genes
E_regulate = {(g_i, g_j, sigma_{ij}) : g_i regulates g_j}

sigma: E -> {+1, -1}                -- activation (+1) or repression (-1)

This is a signed directed graph.
```

**The signed Laplacian for GRNs.** Standard GNN message passing does not distinguish activating from repressing edges. We use the signed graph Laplacian:

```
L_signed = D - A_signed

where A_signed[i,j] = sigma_{ij} * w_{ij}

This means:
- Activating edges (sigma = +1): standard smoothing (connected nodes become similar)
- Repressing edges (sigma = -1): anti-smoothing (connected nodes become dissimilar)
```

The signed Laplacian naturally integrates with `ruvector-mincut`'s spectral machinery by treating negative-weight edges as repulsive forces in the graph partitioning.

### 6. Signaling Cascade Graphs

**Biological context.** Cellular signaling cascades (e.g., MAPK/ERK, PI3K/AKT, Wnt, Notch) transmit information from cell surface receptors to nuclear transcription factors through chains of phosphorylation, ubiquitination, and protein-protein interaction events. These cascades are multi-layer directed acyclic graphs (DAGs) where aberrant signaling drives cancer and autoimmune disease.

**Formal definition:**

```
G_signal = (V_layer_0 U V_layer_1 U ... U V_layer_L, E_activate)

V_layer_k = {proteins active at layer k of the cascade}
E_activate = {(v_i^k, v_j^{k+1}) : protein i at layer k activates protein j at layer k+1}

Layer structure:
  Layer 0: Receptors (e.g., EGFR, FGFR)
  Layer 1: Adaptor proteins (e.g., GRB2, SOS)
  Layer 2: Small GTPases (e.g., RAS)
  Layer 3: MAP kinase kinase kinases (e.g., RAF)
  Layer 4: MAP kinase kinases (e.g., MEK)
  Layer 5: MAP kinases (e.g., ERK)
  Layer 6: Transcription factors (e.g., MYC, FOS)
```

The layered DAG structure maps naturally to the hierarchical message passing in `ruvector-gnn::RuvectorLayer`, where each GNN layer processes one signaling layer in topological order.

---

## GNN Architectures

### Architecture 1: SE(3)-Equivariant Message Passing for 3D Structure Prediction

**Purpose.** Predict protein 3D structure from sequence, modeling the spatial arrangement of residues with guaranteed equivariance under rotations and translations of the coordinate frame.

**Equivariance requirement.** A function f: R^{3N} -> R^{3N} is SE(3)-equivariant if for any rotation R in SO(3) and translation t in R^3:

```
f(Rx_1 + t, Rx_2 + t, ..., Rx_N + t) = Rf(x_1, ..., x_N) + t
```

This guarantees that the predicted structure transforms correctly when the input coordinate frame is changed -- a physical invariance that non-equivariant models must learn from data.

**Message passing formulation.** At each GNN layer l, the node features h_i^l and coordinate features x_i^l are updated simultaneously:

```
Message computation (with radial basis functions):
  m_{ij} = phi_m(h_i^l, h_j^l, ||x_i^l - x_j^l||^2, e_{ij})

  where phi_m is an MLP and e_{ij} are edge features.
  Note: we use ||x_i - x_j||^2 (squared distance) rather than
  the vector (x_i - x_j) to ensure invariance in the scalar channel.

Coordinate update (equivariant):
  x_i^{l+1} = x_i^l + (1/|N(i)|) * sum_{j in N(i)} (x_i^l - x_j^l) * phi_x(m_{ij})

  where phi_x: R^d -> R^1 maps messages to scalar coordinate weights.
  This is equivariant because (x_i - x_j) transforms as a vector
  and phi_x outputs a scalar.

Feature update (invariant):
  h_i^{l+1} = phi_h(h_i^l, sum_{j in N(i)} m_{ij})

  where phi_h is an MLP. This is invariant because m_{ij} depends
  only on distances and invariant features.
```

**Radial basis function expansion.** Distances are expanded into a basis to enable smooth learning over continuous distance values:

```
RBF_k(d) = exp(-gamma * (d - mu_k)^2)

where mu_k are K centers uniformly spaced in [0, tau]
and gamma = 1 / (2 * delta^2) with delta = tau / K
```

With K = 50 radial basis functions spanning [0, 20A], the GNN has a smooth, learnable representation of distance that avoids the discontinuity of hard distance cutoffs.

**Implementation mapping to ruvector-gnn:**

```rust
use ruvector_gnn::layer::{RuvectorLayer, Linear, LayerNorm};

/// SE(3)-equivariant layer built on RuvectorLayer
struct EquivariantProteinLayer {
    /// Invariant feature update (uses RuvectorLayer message passing)
    feature_layer: RuvectorLayer,
    /// Coordinate update MLP (scalar output for equivariance)
    coord_mlp: Linear,
    /// Radial basis function parameters
    rbf_centers: Vec<f32>,  // K = 50 centers
    rbf_gamma: f32,
    /// Distance projection
    dist_proj: Linear,
}

impl EquivariantProteinLayer {
    fn forward(
        &self,
        h: &[Vec<f32>],       // Node features [N x d]
        x: &[[f32; 3]],       // 3D coordinates [N x 3]
        edge_index: &[(usize, usize)],
        edge_features: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, Vec<[f32; 3]>) {
        let n = h.len();
        let mut h_new = Vec::with_capacity(n);
        let mut x_new = Vec::with_capacity(n);

        for i in 0..n {
            let neighbors: Vec<usize> = edge_index.iter()
                .filter(|(src, _)| *src == i)
                .map(|(_, dst)| *dst)
                .collect();

            // Compute distance-based edge weights with RBF expansion
            let edge_weights: Vec<f32> = neighbors.iter()
                .map(|&j| {
                    let dist = euclidean_distance_3d(&x[i], &x[j]);
                    self.rbf_weight(dist)
                })
                .collect();

            let neighbor_features: Vec<Vec<f32>> = neighbors.iter()
                .map(|&j| h[j].clone())
                .collect();

            // Invariant feature update via RuvectorLayer
            let h_i = self.feature_layer.forward(&h[i], &neighbor_features, &edge_weights);

            // Equivariant coordinate update
            let mut dx = [0.0f32; 3];
            for &j in &neighbors {
                let diff = [
                    x[i][0] - x[j][0],
                    x[i][1] - x[j][1],
                    x[i][2] - x[j][2],
                ];
                let msg = self.compute_message(&h[i], &h[j], &x[i], &x[j]);
                let scalar_weight = self.coord_mlp.forward(&msg)[0];
                for d in 0..3 {
                    dx[d] += diff[d] * scalar_weight;
                }
            }
            let scale = 1.0 / neighbors.len().max(1) as f32;
            let x_i = [
                x[i][0] + dx[0] * scale,
                x[i][1] + dx[1] * scale,
                x[i][2] + dx[2] * scale,
            ];

            h_new.push(h_i);
            x_new.push(x_i);
        }

        (h_new, x_new)
    }
}
```

**Comparison with AlphaFold2:**

| Aspect | AlphaFold2 | RuVector SE(3)-GNN |
|--------|-----------|-------------------|
| Input | MSA + templates | Residue contact graph |
| Attention | Evoformer (O(N^2 L)) | Graph attention (O(N * avg_degree)) |
| Equivariance | Invariant Point Attention (IPA) | SE(3)-equivariant message passing |
| MSA requirement | Yes (slow search) | Optional (direct structure) |
| Inference time | ~minutes per protein | Target: seconds per protein |
| Memory | O(N^2) for attention | O(N * avg_degree) for sparse graph |

### Architecture 2: Graph Attention Networks (GAT) for PPI Prediction

**Purpose.** Predict protein-protein interactions from node features and network topology. Given a PPI graph with known interactions, predict whether an unobserved edge exists between two proteins.

**GAT layer formulation.** The key innovation of GAT over GCN is the use of learned attention coefficients to weight neighbor contributions, rather than using fixed degree-based normalization.

**Attention coefficient computation:**

```
For node i and neighbor j in N(i):

  e_{ij} = LeakyReLU(a^T [W h_i || W h_j])

  where:
    W in R^{d' x d}   -- shared weight matrix
    a in R^{2d'}       -- attention weight vector
    ||               -- concatenation operator
    LeakyReLU with negative slope 0.2

Normalized attention:
  alpha_{ij} = softmax_j(e_{ij}) = exp(e_{ij}) / sum_{k in N(i)} exp(e_{ik})

Node update:
  h_i' = sigma(sum_{j in N(i)} alpha_{ij} W h_j)
```

**Multi-head extension.** We use K independent attention heads and concatenate (intermediate layers) or average (final layer):

```
Intermediate layers:
  h_i' = ||_{k=1}^{K} sigma(sum_{j in N(i)} alpha_{ij}^k W^k h_j)

  Output dimension: K * d'

Final layer:
  h_i' = sigma((1/K) sum_{k=1}^{K} sum_{j in N(i)} alpha_{ij}^k W^k h_j)

  Output dimension: d'
```

**Link prediction objective.** For PPI prediction, the final node embeddings are combined pairwise and scored:

```
score(p_i, p_j) = MLP([h_i' || h_j' || h_i' * h_j'])

Loss = BCE(score(p_i, p_j), y_{ij})

where y_{ij} = 1 if interaction exists, 0 otherwise.
Negative sampling: for each positive edge, sample 5-10 negative pairs.
```

**Implementation using ruvector-gnn attention:**

```rust
use ruvector_gnn::layer::MultiHeadAttention;

struct PpiGatLayer {
    attention: MultiHeadAttention,  // From ruvector-gnn
    linear_proj: Linear,
    num_heads: usize,
}

impl PpiGatLayer {
    fn forward(
        &self,
        node_features: &[Vec<f32>],
        adjacency: &HashMap<usize, Vec<usize>>,
    ) -> Vec<Vec<f32>> {
        let n = node_features.len();
        let mut outputs = Vec::with_capacity(n);

        for i in 0..n {
            let neighbors = adjacency.get(&i)
                .map(|ns| ns.as_slice())
                .unwrap_or(&[]);

            if neighbors.is_empty() {
                outputs.push(self.linear_proj.forward(&node_features[i]));
                continue;
            }

            // Gather neighbor key/value features
            let keys: Vec<Vec<f32>> = neighbors.iter()
                .map(|&j| node_features[j].clone())
                .collect();
            let values = keys.clone();

            // Multi-head attention aggregation
            let attended = self.attention.forward(
                &node_features[i],
                &keys,
                &values,
            );

            outputs.push(attended);
        }

        outputs
    }
}
```

**Scalability for the human proteome.** The full human PPI graph has ~20,000 nodes and ~300,000+ edges. Key scalability strategies:

1. **GraphSAGE-style sampling**: Sample k = 15 neighbors per node per layer (from `ruvector-postgres::graphsage`)
2. **Mini-batch training**: Sample subgraphs of ~2,000 nodes per batch
3. **Cluster-based sampling**: Use `ruvector-mincut` community detection to form balanced mini-batches that respect community structure

### Architecture 3: Gated Graph Transformers for Pathway Analysis

**Purpose.** Model metabolic and signaling pathways as dynamic, context-dependent processes where the relevance of each reaction or signaling event depends on the current cellular state.

**Integration with ruvector-mincut-gated-transformer.** The coherence-gated transformer architecture provides exactly the right control primitives for biological pathway analysis:

| Transformer Concept | Biological Interpretation |
|---------------------|--------------------------|
| Token | Pathway node (enzyme, metabolite, signaling protein) |
| Attention | Functional coupling between pathway components |
| Coherence (lambda) | Pathway flux consistency |
| Gate decision | Whether a pathway branch is active under current conditions |
| KV cache flush | Reset pathway state on perturbation (drug, mutation) |
| Tier selection | Adaptive detail level (overview vs. detailed mechanistic) |

**Gated pathway transformer layer:**

```
For each pathway node v_i at layer l:

  1. Compute coherence energy from local pathway neighborhood:
     E_i = sum_{j in N(i)} ||rho_i(h_i) - rho_j(h_j)||^2

     where rho_i, rho_j are restriction maps encoding expected
     consistency between connected pathway components.

  2. Gate decision:
     If E_i < theta_active:  Full computation (active pathway branch)
     If E_i < theta_dormant: Reduced computation (partially active)
     If E_i >= theta_dormant: Skip (dormant pathway branch)

  3. For active nodes, apply gated graph transformer:
     q_i = W_Q h_i^l
     k_j = W_K h_j^l  for j in N(i)
     v_j = W_V h_j^l  for j in N(i)

     alpha_{ij} = softmax(q_i^T k_j / sqrt(d_k) + b_{ij})

     where b_{ij} is a learned edge bias incorporating:
     - Reaction directionality (substrate -> product)
     - Thermodynamic favorability (delta_G)
     - Enzyme kinetics (Km, Vmax)

     h_i^{l+1} = LayerNorm(h_i^l + sum_j alpha_{ij} v_j)
```

**Spectral position encoding for pathway topology.** Using `ruvector-mincut-gated-transformer::spectral::SparseCSR`, we compute spectral encodings from the pathway graph Laplacian:

```
Pathway Laplacian: L = D - A_pathway

Eigendecomposition: L u_k = lambda_k u_k, k = 1, ..., K

Position encoding for node i:
  PE_i = [u_1(i), u_2(i), ..., u_K(i)] in R^K

This captures global pathway topology:
- u_1 (Fiedler vector): main pathway branching point
- u_2, u_3: secondary structural features
- Higher eigenvectors: fine-grained local structure
```

### Architecture 4: Heterogeneous GNNs for Multi-Modal Biological Graph Integration

**Purpose.** Integrate all six graph representations into a unified multi-scale model where information flows across scales -- from atomic interactions to genome-wide regulatory effects.

**Heterogeneous graph formulation.** A heterogeneous graph has multiple node types and edge types:

```
G_hetero = (V, E, tau_V, tau_E)

Node types tau_V = {atom, residue, protein, enzyme, metabolite, gene, TF}
Edge types tau_E = {covalent_bond, spatial_contact, PPI, substrate, product,
                    activates, represses, phosphorylates}

Each node type has its own feature dimension:
  d_{atom} = 16, d_{residue} = 41, d_{protein} = 128, etc.

Each edge type has its own transformation:
  W_{tau_E} in R^{d_out x d_in}  -- one weight matrix per edge type
```

**Heterogeneous message passing:**

```
For node i of type tau_i, aggregate messages from all edge types:

  m_i^{tau_e} = AGG({W_{tau_e} h_j : j in N_{tau_e}(i)})

  where N_{tau_e}(i) is the set of neighbors connected by edge type tau_e.

Combine messages across edge types:

  h_i' = UPDATE(h_i, sum_{tau_e in T_E} alpha_{tau_e} m_i^{tau_e})

  where alpha_{tau_e} are learned edge-type attention weights:

  alpha_{tau_e} = softmax_{tau_e}(a_{tau_e}^T tanh(W_{combine} m_i^{tau_e}))
```

**Cross-scale message passing.** Information flows between scales via designated cross-scale edges:

```
Atom -> Residue:     Aggregate atom features to residue representation
Residue -> Protein:  Pool residue features to whole-protein embedding
Protein -> PPI:      Protein embeddings become PPI node features
Enzyme -> Metabolic: Protein embeddings inform enzyme node features
Gene -> GRN:         Gene expression features from genomic attention layers

These cross-scale edges use pooling operations:

  h_{residue_i} = POOL({h_{atom_j} : atom_j in residue_i})

  POOL can be: mean, attention-weighted sum, or Set2Set
```

---

## Graph Partitioning Applications

### Community Detection in PPI Networks via ruvector-mincut

**Biological motivation.** Proteins that participate in the same biological process tend to cluster together in PPI networks, forming functional modules (protein complexes, signaling pathways, metabolic subsystems). Identifying these modules is equivalent to community detection in graph theory.

**Application of ruvector-mincut hierarchical decomposition.** The 3-level cluster hierarchy in `ruvector-mincut::cluster::hierarchy` maps directly onto biological organization:

```
Level 0 (Expanders):  Tightly connected protein subcomplexes
                       Example: individual subunits of the ribosome
                       phi-expansion guarantees no sparse internal cuts

Level 1 (Preclusters): Functional units with bounded boundaries
                        Example: the entire ribosomal complex
                        Bounded boundary ratio ensures clean module separation

Level 2 (Clusters):    Biological pathways or cellular compartments
                        Example: the translation machinery (ribosome + tRNA + initiation factors)
                        Mirror cuts track cross-pathway interactions
```

**Implementation:**

```rust
use ruvector_mincut::cluster::hierarchy::{HierarchyConfig, ClusterHierarchy};
use ruvector_mincut::MinCutBuilder;

fn detect_protein_modules(
    ppi_edges: &[(usize, usize, f32)],  // (protein_i, protein_j, confidence)
    num_proteins: usize,
) -> Vec<ProteinModule> {
    // Configure hierarchy for biological networks
    let config = HierarchyConfig {
        phi: 0.05,                        // Lower expansion for biological modularity
        max_expander_size: 200,           // Largest subcomplex
        min_expander_size: 3,             // Minimum module size
        target_precluster_size: 50,       // Typical pathway size
        max_boundary_ratio: 0.4,          // Allow some cross-talk
        track_mirror_cuts: true,          // Track inter-module edges
    };

    let hierarchy = ClusterHierarchy::build(ppi_edges, num_proteins, config);

    // Extract modules at each level
    hierarchy.clusters_at_level(HierarchyLevel::Precluster)
        .map(|cluster| ProteinModule {
            proteins: cluster.vertices.clone(),
            internal_edges: cluster.internal_edges.clone(),
            boundary_proteins: cluster.boundary_edges.iter()
                .flat_map(|(u, v)| vec![*u, *v])
                .collect(),
            cohesion: cluster.volume as f64 / cluster.boundary_edges.len().max(1) as f64,
        })
        .collect()
}
```

### Min-Cut for Drug Target Identification

**Biological insight.** A drug target is a protein whose inhibition maximally disrupts a disease-related pathway. This is precisely the bottleneck identification problem: find the minimum set of nodes (or edges) whose removal disconnects the disease pathway from its downstream effects.

**Formal statement.** Given a signaling or metabolic pathway graph G and a disease phenotype node set T:

```
Find the minimum vertex cut S* such that:
  S* = argmin_{S subset V} |S| subject to:
    removing S disconnects source nodes (receptor/stimulus)
    from target nodes T (disease phenotype)

This is equivalent to finding:
  min-cut(source_set, target_set) in the pathway graph
```

**Using ruvector-mincut for target identification:**

```rust
use ruvector_mincut::{MinCutBuilder, DynamicMinCut};

fn identify_drug_targets(
    pathway_edges: &[(usize, usize, f32)],
    receptor_nodes: &[usize],       // Source: pathway entry points
    phenotype_nodes: &[usize],      // Target: disease phenotype nodes
) -> Vec<DrugTarget> {
    // Build min-cut structure with pathway topology
    let mut mincut = MinCutBuilder::new()
        .exact()
        .with_edges(pathway_edges.to_vec())
        .build()
        .expect("Failed to build min-cut structure");

    let cut_value = mincut.min_cut_value();

    // The minimum cut edges identify the bottleneck proteins
    // Proteins incident to cut edges are candidate drug targets
    let cut_edges = mincut.min_cut_edges();

    let mut targets: Vec<DrugTarget> = cut_edges.iter()
        .flat_map(|(u, v)| vec![
            DrugTarget {
                protein_id: *u,
                essentiality_score: compute_essentiality(*u, pathway_edges),
                druggability: predict_druggability(*u),
                cut_contribution: 1.0 / cut_value as f64,
            },
            DrugTarget {
                protein_id: *v,
                essentiality_score: compute_essentiality(*v, pathway_edges),
                druggability: predict_druggability(*v),
                cut_contribution: 1.0 / cut_value as f64,
            },
        ])
        .collect();

    // Rank by combined essentiality and druggability
    targets.sort_by(|a, b| {
        let score_a = a.essentiality_score * a.druggability;
        let score_b = b.essentiality_score * b.druggability;
        score_b.partial_cmp(&score_a).unwrap()
    });

    targets.dedup_by_key(|t| t.protein_id);
    targets
}
```

**Dynamic drug target discovery.** When a mutation is identified in a patient's genome, the pathway graph changes (an edge is strengthened, weakened, or removed). Using `ruvector-mincut`'s dynamic update capability:

```rust
// Patient has a gain-of-function mutation in BRAF
// This strengthens the BRAF -> MEK edge in the MAPK pathway
mincut.insert_edge(braf_id, mek_id, 5.0)  // Increased weight
    .expect("Edge update failed");

// The min-cut may now have shifted, revealing new drug targets
let new_cut_value = mincut.min_cut_value();
let new_targets = mincut.min_cut_edges();
// Perhaps now the bottleneck is downstream of BRAF, suggesting
// MEK inhibitors rather than BRAF inhibitors
```

This leverages the subpolynomial O(n^{o(1)}) amortized update time of `ruvector-mincut`, enabling real-time exploration of patient-specific drug targets.

### Spectral Clustering on Gene Co-Expression Networks

**Biological context.** Genes that are co-expressed across conditions (tissues, developmental stages, disease states) are often co-regulated and functionally related. Co-expression networks are dense weighted graphs where edge weights represent Pearson correlation of expression profiles.

**Spectral clustering formulation.** Given the co-expression adjacency matrix W in R^{N x N}:

```
1. Compute the normalized graph Laplacian:
   L_norm = I - D^{-1/2} W D^{-1/2}

   where D = diag(d_1, ..., d_N), d_i = sum_j W_{ij}

2. Compute the k smallest eigenvectors of L_norm:
   L_norm u_i = lambda_i u_i, i = 1, ..., k

   lambda_1 = 0 <= lambda_2 <= ... <= lambda_k

3. Form the spectral embedding matrix:
   U = [u_1, u_2, ..., u_k] in R^{N x k}

4. Normalize rows of U to unit length:
   T_i = U_i / ||U_i||

5. Apply k-means clustering to the rows of T
```

The number of clusters k is determined by the eigengap: k = argmax_i (lambda_{i+1} - lambda_i).

**Using ruvector-mincut spectral machinery:**

```rust
use ruvector_mincut_gated_transformer::spectral::SparseCSR;

fn spectral_cluster_coexpression(
    correlation_matrix: &[Vec<f32>],  // N x N Pearson correlations
    threshold: f32,                    // Minimum correlation for edge (e.g., 0.7)
) -> Vec<usize> {  // Cluster assignment per gene
    let n = correlation_matrix.len();

    // Build sparse adjacency from thresholded correlations
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if correlation_matrix[i][j].abs() > threshold {
                edges.push((i, j, correlation_matrix[i][j].abs()));
            }
        }
    }

    // Build sparse Laplacian using ruvector-mincut-gated-transformer CSR format
    let laplacian = SparseCSR::from_laplacian_edges(&edges, n);

    // Power iteration for bottom-k eigenvectors
    let k = estimate_num_clusters(&edges, n);
    let eigenvectors = power_iteration_bottom_k(&laplacian, k, 100);

    // Normalize and cluster
    let normalized = normalize_rows(&eigenvectors);
    kmeans_cluster(&normalized, k)
}
```

---

## Performance Analysis

### Message Passing Complexity

**Per-layer complexity for each GNN architecture:**

| Architecture | Time per Layer | Space | Notes |
|-------------|---------------|-------|-------|
| SE(3)-Equivariant | O(N * d * k) | O(N * d + E * d) | k = avg degree, d = hidden dim |
| GAT | O(N * d * k + E * d) | O(N * d + E) | E attention coefficients |
| Gated Transformer | O(N * d^2 + E * d) | O(N * d + N * d_KV) | Includes KV cache |
| Heterogeneous GNN | O(sum_tau N_tau * d_tau * k_tau) | O(sum_tau N_tau * d_tau) | Per edge-type |

**Concrete estimates for biological graphs:**

| Graph Type | N | E | d | L (layers) | Forward Pass |
|-----------|---|---|---|-------------|-------------|
| Residue contact (single protein) | 300 | 2,000 | 128 | 8 | ~2 ms |
| Molecular (drug-protein complex) | 5,000 | 15,000 | 64 | 6 | ~15 ms |
| PPI (human proteome) | 20,000 | 300,000 | 128 | 3 | ~200 ms* |
| Metabolic (human) | 6,000 | 12,000 | 64 | 4 | ~8 ms |
| GRN (human) | 25,000 | 200,000 | 64 | 3 | ~150 ms* |
| Signaling cascade | 500 | 2,000 | 64 | 6 | ~3 ms |

*With GraphSAGE sampling (k=15 neighbors per node)

### Scalability to Proteome-Scale Graphs (>100K Nodes)

For graphs exceeding 100,000 nodes (e.g., pan-proteome PPI networks, metagenomics, or multi-species comparative analysis), we employ a three-tier scalability strategy:

**Tier 1: Neighbor Sampling (GraphSAGE-style)**

Using the GraphSAGE implementation from `ruvector-postgres::graphsage`:

```
Per-node computation: O(k^L * d^2) where k = sample size, L = layers

With k = 15, L = 3, d = 128:
  Per-node: 15^3 * 128^2 = ~55M FLOPs
  Total for 100K nodes: ~5.5 TFLOPs
  At 10 TFLOPS (CPU): ~550 ms per epoch
```

**Tier 2: Cluster-Based Mini-Batching**

Using `ruvector-mincut` community detection to form mini-batches that respect graph structure:

```
1. Run hierarchical clustering to identify C communities
2. Each mini-batch = one community + 1-hop boundary nodes
3. Train on communities in round-robin order

Benefits:
- Edges within mini-batch are complete (no sampling bias)
- Boundary nodes prevent information loss at community edges
- Load balancing via balanced partitioning
```

**Tier 3: Distributed Graph Partitioning**

For graphs exceeding single-machine memory, using `ruvector-graph::distributed`:

```
1. Shard graph across machines using ruvector-graph ShardStrategy
2. Each shard processes its subgraph independently (embarrassingly parallel)
3. Boundary messages synchronized via ruvector-graph RpcClient
4. Gossip protocol (ruvector-graph::distributed::gossip) for convergence
```

### WASM Deployment for Interactive 3D Visualization

**Architecture.** The WASM deployment pipeline compiles the core graph and GNN computation to WebAssembly, enabling browser-based interactive visualization of protein structures, PPI networks, and pathway analysis results.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BROWSER (WASM Runtime)                          │
│                                                                     │
│  ┌──────────────────┐  ┌────────────────────┐  ┌────────────────┐  │
│  │  3D Renderer     │  │  Graph Layout      │  │  GNN Inference │  │
│  │  (WebGL/WebGPU)  │  │  (Force-directed)  │  │  (ruvector-gnn │  │
│  │                  │  │                    │  │   via WASM)    │  │
│  │  - Protein       │  │  - PPI network     │  │                │  │
│  │    ribbon/        │  │    visualization   │  │  - Residue     │  │
│  │    surface       │  │  - Pathway maps    │  │    embedding   │  │
│  │  - Contact       │  │  - Community       │  │  - PPI link    │  │
│  │    highlights    │  │    coloring        │  │    prediction  │  │
│  │  - Drug binding  │  │  - Min-cut         │  │  - Pathway     │  │
│  │    sites         │  │    overlay         │  │    analysis    │  │
│  └──────────────────┘  └────────────────────┘  └────────────────┘  │
│           ▲                     ▲                      ▲            │
│           └─────────────────────┴──────────────────────┘            │
│                    ruvector-mincut WASM module                      │
│               (graph partitioning, spectral layout)                 │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  ruvector-mincut::wasm  (compiled to wasm32-unknown-unknown) │   │
│  │  - min_cut_value(), insert_edge(), delete_edge()             │   │
│  │  - SIMD via wasm::simd for distance computations             │   │
│  │  - Agentic batch processing via wasm::agentic                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**WASM module size targets:**

| Module | Estimated Size | Contents |
|--------|---------------|----------|
| ruvector-gnn (core) | ~200 KB | RuvectorLayer, attention, search |
| ruvector-mincut (core) | ~350 KB | Min-cut, community detection |
| Spectral position encoding | ~80 KB | CSR Laplacian, power iteration |
| Graph layout engine | ~120 KB | Force-directed, spectral layout |
| **Total** | **~750 KB** | Full interactive capability |

**Interactive performance targets:**

| Operation | Target Latency | Context |
|-----------|---------------|---------|
| Protein contact graph construction | < 50 ms | Typical 300-residue protein |
| GNN forward pass (single protein) | < 100 ms | 8-layer SE(3)-equivariant |
| PPI community detection | < 500 ms | 5,000-node subnetwork |
| Min-cut drug target highlight | < 200 ms | 500-node pathway |
| Force-directed layout iteration | < 16 ms | 60 FPS animation target |
| Node hover / selection query | < 5 ms | Interactive responsiveness |

---

## Integration with Genomic Attention Layers

### Cross-Architecture Information Flow

The protein graph engine connects to the genomic attention layers (ADR-001 through ADR-004) through three integration points:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    MULTI-SCALE BIOLOGICAL ANALYSIS PIPELINE              │
│                                                                          │
│  ┌─────────────────────┐     ┌──────────────────────┐                   │
│  │  Genomic Attention   │     │  Protein Graph Engine │                   │
│  │  (ADR-001 - ADR-004) │     │  (ADR-005, this doc)  │                   │
│  │                      │     │                       │                   │
│  │  DNA/RNA sequence    │────>│  Gene Regulatory      │                   │
│  │  attention outputs   │ (1) │  Network node features │                   │
│  │                      │     │                       │                   │
│  │  Variant effect      │────>│  PPI edge weight      │                   │
│  │  predictions         │ (2) │  modulation           │                   │
│  │                      │     │                       │                   │
│  │  Epigenetic state    │────>│  Signaling cascade    │                   │
│  │  vectors             │ (3) │  gate signals         │                   │
│  └─────────────────────┘     └──────────────────────┘                   │
│                                                                          │
│  Integration Points:                                                     │
│  (1) Genomic attention -> GRN: Sequence context embeddings become        │
│      node features for transcription factor / target gene nodes          │
│  (2) Variant attention -> PPI: Predicted variant effects modulate        │
│      PPI edge confidence (missense mutation weakens interaction)         │
│  (3) Epigenetic attention -> Signaling: Chromatin state vectors          │
│      serve as gate signals for pathway activity analysis                 │
└──────────────────────────────────────────────────────────────────────────┘
```

### Integration Point 1: Genomic Context as GRN Node Features

The genomic attention model produces per-gene context embeddings that capture regulatory potential, promoter accessibility, and evolutionary conservation. These embeddings serve as initial node features for the gene regulatory network GNN:

```
h_gene_i^{(0)} = [genomic_attention_embedding(gene_i) || expression_vector(gene_i)]

Dimension: d_genomic (e.g., 256) + d_expression (e.g., 16) = 272
```

### Integration Point 2: Variant Effects as PPI Edge Modulation

When the DNA analyzer identifies a missense variant in a protein-coding gene, the variant effect prediction from the genomic attention model modulates the corresponding PPI edges:

```
For variant V in protein P_i:
  delta_score = variant_attention_model(V)  -- in [-1, 1]

  For each PPI edge (P_i, P_j):
    w'_{ij} = w_{ij} * (1 + delta_score * interface_overlap(V, interface_{ij}))

  where interface_overlap measures whether the variant falls in
  the interaction interface with protein P_j.
```

### Integration Point 3: Epigenetic State as Pathway Gate Signals

Chromatin accessibility (ATAC-seq), histone modifications, and DNA methylation patterns determine which genes are transcriptionally active. These epigenetic state vectors become gate signals for the coherence-gated pathway transformer:

```
For pathway node v_i (enzyme or signaling protein):
  gene_i = encoding_gene(v_i)
  epigenetic_state_i = epigenetic_attention_model(gene_i)

  gate_signal_i = {
    lambda: pathway_coherence(v_i),
    activity: sigmoid(W_gate * epigenetic_state_i),
  }

  If activity < theta_active:
    Skip node (gene is silenced, protein not expressed)
  Else:
    Process normally with coherence gating
```

This integration means the protein graph engine does not operate in isolation but receives continuous context from the genomic attention layers, enabling personalized pathway analysis where a patient's specific genomic and epigenetic context shapes the graph structure and inference.

---

## Continual Learning and Forgetting Mitigation

### EWC for Protein Model Adaptation

When the model is fine-tuned on new protein data (e.g., a newly characterized protein family), Elastic Weight Consolidation from `ruvector-gnn::ewc` prevents catastrophic forgetting of previously learned protein representations:

```
L_total = L_task + (lambda_ewc / 2) * sum_i F_i * (theta_i - theta_i*)^2

where:
  L_task  = current task loss (e.g., new protein family classification)
  F_i     = Fisher information matrix diagonal for parameter theta_i
  theta_i = current parameter value
  theta_i*= parameter value after previous training
  lambda_ewc = importance weight (0.4 default in ruvector-gnn)
```

This is critical for a clinical DNA analyzer that must continuously incorporate new variant-protein associations from the literature without losing previously validated predictions.

### Replay Buffer for Rare Protein Interactions

The `ruvector-gnn::ReplayBuffer` with reservoir sampling maintains a diverse set of training examples from previously seen protein interactions, ensuring that rare but clinically important interactions (e.g., transient signaling complexes) are not forgotten during training on common interactions.

---

## Consequences

### Positive

- **Native graph representation**: Proteins modeled as graphs avoid the information loss of sequential representations
- **Multi-scale integration**: Six graph types span from atoms to genome, connected through cross-scale message passing
- **Leverages existing stack**: Builds on proven `ruvector-gnn`, `ruvector-graph`, `ruvector-mincut`, and `ruvector-mincut-gated-transformer` infrastructure
- **Clinical drug target discovery**: Min-cut bottleneck identification provides mathematically rigorous drug target ranking
- **Dynamic personalization**: Patient-specific mutations dynamically update graph structure via `ruvector-mincut` O(n^{o(1)}) updates
- **Browser deployment**: WASM compilation enables interactive protein visualization without server-side computation
- **Continual learning**: EWC and replay buffer prevent catastrophic forgetting as new protein data arrives

### Negative

- **SE(3)-equivariance overhead**: Coordinate updates add ~30% compute compared to non-equivariant GNNs
- **Heterogeneous graph complexity**: Multiple node/edge types increase implementation and testing surface
- **Data dependency**: PPI networks are incomplete (~30% estimated coverage of true human interactions), introducing false negatives
- **Feature engineering**: Node and edge features require domain-specific biological knowledge to design

### Risks

- **PPI network noise**: False positive interactions in experimental datasets may mislead community detection
- **Scale mismatch**: Cross-scale message passing (atom-to-genome) requires careful normalization to prevent gradient explosion
- **WASM memory limits**: Browser WASM has a default 4 GB memory limit; the full human PPI graph may require streaming or server-side partitioning

---

## References

1. Kipf, T. N. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
2. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS*.
3. Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph Attention Networks. *ICLR*.
4. Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) Equivariant Graph Neural Networks. *ICML*.
5. Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583-589.
6. Gainza, P. et al. (2020). Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning. *Nature Methods*, 17, 184-192.
7. Kreuzer, D., Beaini, D., Hamilton, W. L., Letourneau, V., & Tossou, P. (2021). Rethinking Graph Transformers with Spectral Attention. *NeurIPS*.
8. Raposo, D. et al. (2024). Mixture-of-Depths: Dynamically allocating compute in transformer-based language models. *arXiv:2404.02258*.
9. Szklarczyk, D. et al. (2023). The STRING database in 2023: protein-protein association networks and functional enrichment analyses for any sequenced genome of interest. *Nucleic Acids Research*, 51(D1), D483-D489.
10. Kanehisa, M. et al. (2023). KEGG for taxonomy-based analysis of pathways and genomes. *Nucleic Acids Research*, 51(D1), D587-D592.
11. Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.
12. Von Luxburg, U. (2007). A tutorial on spectral clustering. *Statistics and Computing*, 17(4), 395-416.
