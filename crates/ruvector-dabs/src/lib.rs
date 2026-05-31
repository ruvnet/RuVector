//! `ruvector-dabs` — Distance Adaptive Beam Search for HNSW
//!
//! Implements the DABS algorithm from:
//! Al-Jazzazi et al., "Distance Adaptive Beam Search for Provably Accurate
//! Graph-Based Nearest Neighbor Search", NeurIPS 2025, arXiv:2505.15636.
//!
//! ## The problem
//!
//! Standard HNSW uses a fixed expansion width (ef): the beam search
//! continues until ef candidates have been considered. This wastes distance
//! computations on nodes that are clearly worse than the current k-th result,
//! and offers no theoretical recall guarantee.
//!
//! ## The DABS solution
//!
//! Replace the fixed-ef stopping criterion with a distance-ratio test:
//!
//! ```text
//! Stop when d(q, x_best_unexplored) > (1 + γ) * d(q, j_k)
//! ```
//!
//! where j_k is the current k-th nearest discovered node. With γ = 0 this
//! is pure greedy descent; with γ > 0 the search terminates once all
//! unexplored nodes are guaranteed to be at most (1+γ) times farther than
//! any result.
//!
//! **Provable bound**: on navigable graphs, the returned results satisfy
//! `d(q, result_i) ≤ (1 + γ)² * d(q, true_i)` for each position i.
//!
//! ## Variants compared
//!
//! | Mode | Termination | Recall | Speed |
//! |------|-------------|--------|-------|
//! | `Flat` | exhaustive | exact | slow |
//! | `FixedEf { ef }` | ef candidates | high | medium |
//! | `Dabs { gamma }` | distance ratio | provable | fast |
//!
//! ## Quick start
//!
//! ```rust
//! use ruvector_dabs::{DabsIndex, SearchMode};
//!
//! let vecs: Vec<Vec<f32>> = (0..1000)
//!     .map(|i| vec![i as f32, 0.0, 0.0, 0.0])
//!     .collect();
//! let idx = DabsIndex::build(vecs, 16).unwrap();
//! let query = vec![500.5_f32, 0.0, 0.0, 0.0];
//! let (ids, stats) = idx.search(&query, 5, SearchMode::Dabs { gamma: 0.5 }).unwrap();
//! println!("top-5 ids: {ids:?}  (dist_ops={})", stats.dist_computations);
//! ```

pub mod dist;
pub mod error;
pub mod graph;
pub mod index;

pub use error::DabsError;
pub use graph::DabsGraph;
pub use index::{recall_at_k, DabsIndex, SearchMode, SearchStats};
