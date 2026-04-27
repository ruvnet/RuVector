//! FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search
//!
//! Implements the algorithm from Chen et al., WWW 2023 (arXiv:2206.11408).
//!
//! ## Core idea
//!
//! During graph beam search, most exact distance computations O(D) are wasted
//! because the candidate falls outside the top-k and would never update the
//! result set. FINGER avoids these wasted computations by:
//!
//! 1. **Build**: for each graph node `u`, compute K orthonormal basis vectors
//!    that span the subspace of edge-residual vectors `{v - u | v ∈ N(u)}`.
//!    Precompute projections of each edge onto this basis.
//!
//! 2. **Search**: when at node `u`, pay O(K×D) once to project the query
//!    residual `(query - u)`. Then for each neighbor `v`, approximate
//!    `dist(query, v)` in O(K) using the precomputed edge projections.
//!    Skip neighbors whose approximate distance exceeds the current
//!    worst result; only compute the exact O(D) distance for survivors.
//!
//! ## Index variants
//!
//! | Type | k_basis | Description |
//! |---|---|---|
//! | [`FingerIndex::exact`] | N/A | Exact beam search, no approximation |
//! | [`FingerIndex::finger_k4`] | 4 | Sweet spot for D=64-256 |
//! | [`FingerIndex::finger_k8`] | 8 | Higher accuracy, less speedup at low D |
//!
//! ## Example
//!
//! ```no_run
//! use ruvector_finger::{FlatGraph, FingerIndex, GraphWalk};
//!
//! let data: Vec<Vec<f32>> = (0..1000)
//!     .map(|i| vec![(i % 64) as f32; 64])
//!     .collect();
//! let graph = FlatGraph::build(&data, 16).unwrap();
//! let index = FingerIndex::build_with_k(&graph, 4).unwrap();
//! let query = vec![0.0f32; 64];
//! let (results, stats) = index.search(&query, 10, 200).unwrap();
//! println!("found {} results, pruned {:.1}% of edges", results.len(), stats.prune_rate() * 100.0);
//! ```

pub mod basis;
pub mod dist;
pub mod error;
pub mod graph;
pub mod index;
pub mod search;

pub use basis::NodeBasis;
pub use error::FingerError;
pub use graph::{recall_at_k, FlatGraph, GraphWalk};
pub use index::FingerIndex;
pub use search::SearchStats;
