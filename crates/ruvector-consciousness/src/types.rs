//! Core types for consciousness computation.
//!
//! Provides transition probability matrices, partition representations,
//! and result types for Φ and emergence metrics.

use serde::{Deserialize, Serialize};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Transition Probability Matrix (TPM)
// ---------------------------------------------------------------------------

/// A row-major transition probability matrix for a discrete system.
///
/// Entry `tpm[i][j]` = P(state j at t+1 | state i at t).
/// Rows must sum to 1.0. Stored as a flat Vec for cache locality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionMatrix {
    /// Flat row-major storage: `data[i * cols + j]`.
    pub data: Vec<f64>,
    /// Number of states (rows = cols for square TPM).
    pub n: usize,
}

impl TransitionMatrix {
    /// Create from a flat row-major vec. Panics if `data.len() != n * n`.
    #[inline]
    pub fn new(n: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), n * n, "TPM data length must be n*n");
        Self { data, n }
    }

    /// Create an identity TPM (each state maps to itself).
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self { data, n }
    }

    /// Get element at (row, col).
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.n + col]
    }

    /// Get element without bounds checking.
    ///
    /// # Safety
    /// `row < self.n && col < self.n` must hold.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> f64 {
        *self.data.get_unchecked(row * self.n + col)
    }

    /// Set element at (row, col).
    #[inline(always)]
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * self.n + col] = val;
    }

    /// Number of states.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.n
    }

    /// Raw data slice.
    #[inline(always)]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Extract a sub-TPM for the given element indices (marginalize).
    pub fn marginalize(&self, indices: &[usize]) -> TransitionMatrix {
        let k = indices.len();
        let mut sub = vec![0.0; k * k];
        for (si, &i) in indices.iter().enumerate() {
            let mut row_sum = 0.0;
            for (sj, &j) in indices.iter().enumerate() {
                let val = self.get(i, j);
                sub[si * k + sj] = val;
                row_sum += val;
            }
            // Re-normalize row.
            if row_sum > 0.0 {
                for sj in 0..k {
                    sub[si * k + sj] /= row_sum;
                }
            }
        }
        TransitionMatrix { data: sub, n: k }
    }
}

// ---------------------------------------------------------------------------
// Partition
// ---------------------------------------------------------------------------

/// A bipartition of system elements into two non-empty sets.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bipartition {
    /// Bitmask: bit i = 1 means element i is in set A, 0 means set B.
    pub mask: u64,
    /// Total number of elements.
    pub n: usize,
}

impl Bipartition {
    /// Elements in set A.
    pub fn set_a(&self) -> Vec<usize> {
        (0..self.n).filter(|&i| self.mask & (1 << i) != 0).collect()
    }

    /// Elements in set B.
    pub fn set_b(&self) -> Vec<usize> {
        (0..self.n)
            .filter(|&i| self.mask & (1 << i) == 0)
            .collect()
    }

    /// Check if this is a valid bipartition (both sets non-empty).
    #[inline]
    pub fn is_valid(&self) -> bool {
        let full = (1u64 << self.n) - 1;
        self.mask != 0 && self.mask != full
    }
}

/// Iterator over all valid bipartitions of n elements.
pub struct BipartitionIter {
    current: u64,
    max: u64,
    n: usize,
}

impl BipartitionIter {
    pub fn new(n: usize) -> Self {
        assert!(n <= 63, "bipartition iter supports at most 63 elements");
        Self {
            current: 1, // skip mask=0 (empty set A)
            max: (1u64 << n) - 1,
            n,
        }
    }
}

impl Iterator for BipartitionIter {
    type Item = Bipartition;

    fn next(&mut self) -> Option<Self::Item> {
        // Skip masks where set B is empty (mask == max).
        while self.current < self.max {
            let mask = self.current;
            self.current += 1;
            return Some(Bipartition { mask, n: self.n });
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.max - self.current) as usize;
        (remaining, Some(remaining))
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Algorithm used for Φ computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhiAlgorithm {
    /// Exact: enumerate all 2^n - 2 bipartitions.
    Exact,
    /// Greedy bisection approximation.
    GreedyBisection,
    /// Spectral approximation via Fiedler vector.
    Spectral,
    /// Stochastic sampling of partition space.
    Stochastic,
    /// Hierarchical approximation for large systems.
    Hierarchical,
    /// GeoMIP: hypercube BFS with automorphism pruning.
    GeoMIP,
    /// Quantum-inspired collapse search.
    Collapse,
}

impl std::fmt::Display for PhiAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact => write!(f, "exact"),
            Self::GreedyBisection => write!(f, "greedy-bisection"),
            Self::Spectral => write!(f, "spectral"),
            Self::Stochastic => write!(f, "stochastic"),
            Self::Hierarchical => write!(f, "hierarchical"),
            Self::GeoMIP => write!(f, "geomip"),
            Self::Collapse => write!(f, "collapse"),
        }
    }
}

/// Result of a Φ (integrated information) computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiResult {
    /// The integrated information value Φ.
    pub phi: f64,
    /// The minimum information partition (MIP) that achieves Φ.
    pub mip: Bipartition,
    /// Number of partitions evaluated.
    pub partitions_evaluated: u64,
    /// Total partitions in search space.
    pub total_partitions: u64,
    /// Algorithm used.
    pub algorithm: PhiAlgorithm,
    /// Wall-clock time for computation.
    pub elapsed: Duration,
    /// Convergence history (Φ estimate per iteration for approximate methods).
    pub convergence: Vec<f64>,
}

/// Result of a causal emergence computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceResult {
    /// Effective information at the micro level.
    pub ei_micro: f64,
    /// Effective information at the macro level.
    pub ei_macro: f64,
    /// Causal emergence = EI_macro - EI_micro.
    pub causal_emergence: f64,
    /// The coarse-graining that maximizes emergence.
    pub coarse_graining: Vec<usize>,
    /// Determinism component.
    pub determinism: f64,
    /// Degeneracy component.
    pub degeneracy: f64,
    /// Wall-clock time.
    pub elapsed: Duration,
}

// ---------------------------------------------------------------------------
// IIT 4.0 types
// ---------------------------------------------------------------------------

/// A mechanism is a subset of system elements that has causal power.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mechanism {
    /// Bitmask of mechanism elements.
    pub elements: u64,
    /// Total number of system elements.
    pub n: usize,
}

impl Mechanism {
    pub fn new(elements: u64, n: usize) -> Self {
        Self { elements, n }
    }

    /// Number of elements in the mechanism.
    pub fn size(&self) -> usize {
        self.elements.count_ones() as usize
    }

    /// Indices of mechanism elements.
    pub fn indices(&self) -> Vec<usize> {
        (0..self.n).filter(|&i| self.elements & (1 << i) != 0).collect()
    }
}

/// A purview is the set of elements a mechanism has causal power over.
pub type Purview = Mechanism;

/// A distinction (concept in IIT 3.0) specifies how a mechanism
/// constrains its cause and effect purviews.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distinction {
    /// The mechanism (subset of elements).
    pub mechanism: Mechanism,
    /// Cause repertoire: distribution over past states.
    pub cause_repertoire: Vec<f64>,
    /// Effect repertoire: distribution over future states.
    pub effect_repertoire: Vec<f64>,
    /// Cause purview (the elements the mechanism has causal power over in the past).
    pub cause_purview: Purview,
    /// Effect purview (elements causally constrained in the future).
    pub effect_purview: Purview,
    /// φ_cause: intrinsic information of the cause.
    pub phi_cause: f64,
    /// φ_effect: intrinsic information of the effect.
    pub phi_effect: f64,
    /// φ = min(φ_cause, φ_effect): the distinction's integrated information.
    pub phi: f64,
}

/// A relation specifies how multiple distinctions overlap in cause-effect space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Indices into the CES distinctions vec.
    pub distinction_indices: Vec<usize>,
    /// Relation φ (irreducibility of the overlap).
    pub phi: f64,
    /// Order of the relation (number of distinctions involved).
    pub order: usize,
}

/// The Cause-Effect Structure (CES): the full quale / experience.
///
/// In IIT 4.0, the CES is the set of all distinctions and relations
/// specified by a system in a state — the "shape" of experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CauseEffectStructure {
    /// System size (number of elements).
    pub n: usize,
    /// Current state of the system.
    pub state: usize,
    /// All distinctions (mechanisms with non-zero φ).
    pub distinctions: Vec<Distinction>,
    /// Relations between distinctions.
    pub relations: Vec<Relation>,
    /// System-level Φ (big phi — irreducibility of the whole CES).
    pub big_phi: f64,
    /// Sum of all distinction φ values (structure integrated information).
    pub sum_phi: f64,
    /// Computation time.
    pub elapsed: Duration,
}

/// Result of Integrated Information Decomposition (ΦID).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiIdResult {
    /// Redundant information: shared across all sources.
    pub redundancy: f64,
    /// Unique information per source.
    pub unique: Vec<f64>,
    /// Synergistic information: only available from the whole.
    pub synergy: f64,
    /// Total mutual information.
    pub total_mi: f64,
    /// Transfer entropy (directional information flow).
    pub transfer_entropy: f64,
    /// Computation time.
    pub elapsed: Duration,
}

/// Result of Partial Information Decomposition (PID).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidResult {
    /// Redundant information (shared by all sources about the target).
    pub redundancy: f64,
    /// Unique information per source about the target.
    pub unique: Vec<f64>,
    /// Synergistic information (only available from all sources jointly).
    pub synergy: f64,
    /// Total mutual information I(sources; target).
    pub total_mi: f64,
    /// Number of sources.
    pub num_sources: usize,
    /// Computation time.
    pub elapsed: Duration,
}

/// Result of streaming (online) Φ computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingPhiResult {
    /// Current Φ estimate.
    pub phi: f64,
    /// Number of time steps processed.
    pub time_steps: usize,
    /// Exponentially weighted moving average of Φ.
    pub phi_ewma: f64,
    /// Variance of Φ estimates.
    pub phi_variance: f64,
    /// Change-point detected in Φ trajectory.
    pub change_detected: bool,
    /// History of Φ estimates (most recent window).
    pub history: Vec<f64>,
}

/// Approximation bound for Φ estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiBound {
    /// Lower bound on Φ.
    pub lower: f64,
    /// Upper bound on Φ.
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95% confidence).
    pub confidence: f64,
    /// Number of samples used.
    pub samples: u64,
    /// Bound source (which method produced this bound).
    pub method: String,
}

/// Compute budget for consciousness computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeBudget {
    /// Maximum wall-clock time.
    pub max_time: Duration,
    /// Maximum partitions to evaluate (0 = unlimited).
    pub max_partitions: u64,
    /// Maximum memory in bytes (0 = unlimited).
    pub max_memory: usize,
    /// Target approximation ratio (1.0 = exact, <1.0 = approximate).
    pub approximation_ratio: f64,
}

impl Default for ComputeBudget {
    fn default() -> Self {
        Self {
            max_time: Duration::from_secs(30),
            max_partitions: 0,
            max_memory: 0,
            approximation_ratio: 1.0,
        }
    }
}

impl ComputeBudget {
    /// Budget for exact computation (generous limits).
    pub fn exact() -> Self {
        Self::default()
    }

    /// Budget for fast approximate computation.
    pub fn fast() -> Self {
        Self {
            max_time: Duration::from_millis(100),
            max_partitions: 1000,
            max_memory: 64 * 1024 * 1024,
            approximation_ratio: 0.9,
        }
    }
}
