//! Gene regulatory network data generation.
//!
//! Builds synthetic gene regulatory networks based on known biological motifs,
//! with 4 functional modules: cell cycle, apoptosis, growth signaling, and
//! housekeeping. Also generates an oncogenic "cancer" variant where growth
//! signaling overrides apoptosis controls.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Module assignment for a 16-gene network.
pub const MODULE_CELL_CYCLE: &[usize] = &[0, 1, 2, 3];
pub const MODULE_APOPTOSIS: &[usize] = &[4, 5, 6, 7];
pub const MODULE_GROWTH: &[usize] = &[8, 9, 10, 11];
pub const MODULE_HOUSEKEEPING: &[usize] = &[12, 13, 14, 15];

pub const MODULE_NAMES: &[&str] = &[
    "Cell Cycle",
    "Apoptosis",
    "Growth Signaling",
    "Housekeeping",
];

/// All modules with their gene indices.
pub fn all_modules() -> Vec<(&'static str, &'static [usize])> {
    vec![
        (MODULE_NAMES[0], MODULE_CELL_CYCLE),
        (MODULE_NAMES[1], MODULE_APOPTOSIS),
        (MODULE_NAMES[2], MODULE_GROWTH),
        (MODULE_NAMES[3], MODULE_HOUSEKEEPING),
    ]
}

/// A gene regulatory network represented as a weighted adjacency matrix.
pub struct GeneNetwork {
    pub n_genes: usize,
    /// Flat row-major adjacency/regulation weights. Positive = activation,
    /// negative = repression. Range roughly [-1, 1].
    pub adjacency: Vec<f64>,
    /// Human-readable gene labels.
    pub gene_labels: Vec<String>,
    /// Module index for each gene (0..3).
    pub module_ids: Vec<usize>,
    /// Network variant label.
    pub variant: String,
}

impl GeneNetwork {
    /// Count non-zero edges (absolute value > 0.001).
    pub fn n_edges(&self) -> usize {
        self.adjacency.iter().filter(|&&w| w.abs() > 0.001).count()
    }
}

/// Transition probability matrix for consciousness analysis.
pub struct TransitionMatrix {
    pub size: usize,
    pub data: Vec<f64>,
}

/// Build the normal (healthy) 16-gene regulatory network.
///
/// Module structure:
/// - Cell cycle (genes 0-3): cyclin cascade with strong internal regulation
/// - Apoptosis (genes 4-7): pro- and anti-apoptotic balance
/// - Growth signaling (genes 8-11): receptor tyrosine kinase cascade
/// - Housekeeping (genes 12-15): weakly connected to all other modules
///
/// Within-module: strong connections (0.3-0.5)
/// Between-module: weak connections (0.01-0.05)
/// Housekeeping: weakly connected to everything
pub fn build_normal_network() -> GeneNetwork {
    let n = 16;
    let mut adj = vec![0.0f64; n * n];
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let gene_labels = vec![
        // Cell cycle
        "CycD".into(),
        "CDK4".into(),
        "CycE".into(),
        "CDK2".into(),
        // Apoptosis
        "BAX".into(),
        "BCL2".into(),
        "CASP3".into(),
        "p53".into(),
        // Growth signaling
        "EGFR".into(),
        "RAS".into(),
        "RAF".into(),
        "ERK".into(),
        // Housekeeping
        "GAPDH".into(),
        "ACTB".into(),
        "RPL13A".into(),
        "HPRT".into(),
    ];

    let module_ids: Vec<usize> = (0..n).map(|i| i / 4).collect();

    // Within-module connections: strong, directed cascade-like
    let modules: &[&[usize]] = &[
        MODULE_CELL_CYCLE,
        MODULE_APOPTOSIS,
        MODULE_GROWTH,
        MODULE_HOUSEKEEPING,
    ];

    for module in modules {
        for (idx, &from) in module.iter().enumerate() {
            for (jdx, &to) in module.iter().enumerate() {
                if from == to {
                    continue;
                }
                // Sequential cascade: strong forward, moderate feedback
                let base: f64 = if jdx == idx + 1 {
                    0.45 // strong forward connection
                } else if idx == jdx + 1 {
                    0.20 // feedback
                } else {
                    0.15 // lateral
                };
                let noise: f64 = rng.gen_range(-0.05..0.05);
                adj[from * n + to] = (base + noise).clamp(0.0, 0.5);
            }
        }
    }

    // Apoptosis module: add inhibitory connections (BCL2 inhibits BAX/CASP3)
    adj[5 * n + 4] = -0.35; // BCL2 -| BAX
    adj[5 * n + 6] = -0.30; // BCL2 -| CASP3
    adj[7 * n + 4] = 0.40; // p53 -> BAX (pro-apoptotic)
    adj[7 * n + 5] = -0.25; // p53 -| BCL2

    // Between-module connections: weak
    // Growth signaling -> Cell cycle (growth promotes division)
    adj[11 * n + 0] = 0.04; // ERK -> CycD
    adj[11 * n + 1] = 0.03; // ERK -> CDK4

    // Growth signaling -> Apoptosis (growth suppresses apoptosis)
    adj[11 * n + 5] = 0.03; // ERK -> BCL2 (anti-apoptotic)

    // Apoptosis -> Cell cycle (apoptosis inhibits division)
    adj[6 * n + 2] = -0.02; // CASP3 -| CycE

    // Housekeeping: weak connections to all modules
    for &hk in MODULE_HOUSEKEEPING {
        for g in 0..12 {
            let w = rng.gen_range(0.005..0.02);
            adj[hk * n + g] = w;
            adj[g * n + hk] = w * 0.5;
        }
    }

    GeneNetwork {
        n_genes: n,
        adjacency: adj,
        gene_labels,
        module_ids,
        variant: "Normal".into(),
    }
}

/// Build the cancer (oncogenic) variant.
///
/// Key rewiring:
/// 1. Growth signaling is constitutively active (stronger internal connections)
/// 2. Growth signaling overrides apoptosis controls (strong cross-module edges)
/// 3. p53 pathway is disrupted (weakened connections)
/// 4. Cell cycle checkpoints are bypassed
///
/// Expected: higher cross-module Phi due to loss of modular boundaries.
pub fn build_cancer_network() -> GeneNetwork {
    let mut net = build_normal_network();
    net.variant = "Cancer".into();
    let n = net.n_genes;

    // 1. Constitutively active growth signaling (boost internal connections)
    for &from in MODULE_GROWTH {
        for &to in MODULE_GROWTH {
            if from != to {
                let idx = from * n + to;
                net.adjacency[idx] = (net.adjacency[idx] * 1.5).clamp(-0.5, 0.5);
            }
        }
    }

    // 2. Growth overrides apoptosis (strong cross-module edges)
    net.adjacency[11 * n + 5] = 0.30; // ERK -> BCL2 (strong anti-apoptotic)
    net.adjacency[10 * n + 5] = 0.25; // RAF -> BCL2
    net.adjacency[11 * n + 4] = -0.20; // ERK -| BAX
    net.adjacency[9 * n + 6] = -0.15; // RAS -| CASP3

    // 3. p53 pathway disruption (simulate TP53 mutation)
    net.adjacency[7 * n + 4] = 0.05; // p53 -> BAX weakened
    net.adjacency[7 * n + 5] = -0.05; // p53 -| BCL2 weakened

    // 4. Cell cycle checkpoint bypass
    net.adjacency[11 * n + 0] = 0.25; // ERK -> CycD (strong growth drive)
    net.adjacency[11 * n + 1] = 0.20; // ERK -> CDK4
    net.adjacency[6 * n + 2] = -0.005; // CASP3 -| CycE weakened

    net
}

/// Convert a gene regulatory network to a transition probability matrix.
///
/// Method:
/// 1. Take absolute values of adjacency weights (treat activation/repression
///    as "information flow" regardless of sign)
/// 2. Add self-regulation (diagonal) as baseline activity
/// 3. Row-normalize to get transition probabilities
pub fn network_to_tpm(net: &GeneNetwork) -> TransitionMatrix {
    let n = net.n_genes;
    let mut tpm = vec![0.0f64; n * n];

    for i in 0..n {
        for j in 0..n {
            // Use absolute adjacency weight as transition strength
            tpm[i * n + j] = net.adjacency[i * n + j].abs();
        }
        // Add self-regulation baseline (genes maintain their own state)
        tpm[i * n + i] += 0.1;
    }

    // Row-normalize
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| tpm[i * n + j]).sum();
        if row_sum > 1e-30 {
            for j in 0..n {
                tpm[i * n + j] /= row_sum;
            }
        }
    }

    TransitionMatrix { size: n, data: tpm }
}

/// Extract a sub-TPM for a subset of genes.
pub fn extract_sub_tpm(tpm: &TransitionMatrix, genes: &[usize]) -> TransitionMatrix {
    let n = genes.len();
    let mut sub = vec![0.0f64; n * n];
    for (si, &gi) in genes.iter().enumerate() {
        let row_sum: f64 = genes.iter().map(|&gj| tpm.data[gi * tpm.size + gj]).sum();
        for (sj, &gj) in genes.iter().enumerate() {
            sub[si * n + sj] = tpm.data[gi * tpm.size + gj] / row_sum.max(1e-30);
        }
    }
    TransitionMatrix { size: n, data: sub }
}

/// Generate a null-model network by shuffling connections while preserving
/// degree distribution (configuration model).
pub fn generate_null_tpm(net: &GeneNetwork, rng: &mut impl rand::Rng) -> TransitionMatrix {
    let n = net.n_genes;
    let mut adj = net.adjacency.clone();

    // Shuffle non-diagonal entries while preserving row sums
    for i in 0..n {
        let mut row_vals: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| adj[i * n + j]).collect();

        // Fisher-Yates shuffle
        for k in (1..row_vals.len()).rev() {
            let swap_idx = rng.gen_range(0..=k);
            row_vals.swap(k, swap_idx);
        }

        let mut idx = 0;
        for j in 0..n {
            if j != i {
                adj[i * n + j] = row_vals[idx];
                idx += 1;
            }
        }
    }

    let shuffled = GeneNetwork {
        n_genes: n,
        adjacency: adj,
        gene_labels: net.gene_labels.clone(),
        module_ids: net.module_ids.clone(),
        variant: "Null".into(),
    };

    network_to_tpm(&shuffled)
}
