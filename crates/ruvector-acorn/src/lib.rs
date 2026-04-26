//! ACORN: Predicate-Agnostic Filtered HNSW for ruvector
//!
//! Implements the ACORN algorithm from:
//! Patel et al., "ACORN: Performant and Predicate-Agnostic Search Over
//! Vector Embeddings and Structured Data", SIGMOD 2024, arXiv:2403.04871.
//!
//! ## The problem
//!
//! Standard filtered vector search runs the ANN graph traversal first, then
//! discards results that fail the predicate. At low selectivity (e.g., only
//! 1% of the dataset passes) the beam exhausts before finding k valid
//! candidates — recall collapses to near zero.
//!
//! ## The ACORN solution
//!
//! Two changes to standard HNSW:
//! 1. **Denser graph**: build with γ·M neighbors per node instead of M.
//!    More edges keep the graph navigable even in sparse predicate subgraphs.
//! 2. **Predicate-agnostic traversal**: during search, expand ALL neighbors
//!    regardless of whether the current node passes the predicate. Failing
//!    nodes are skipped in results but their neighborhood is still explored.
//!
//! ## Variants in this crate
//!
//! | Struct | γ | M | Edge budget | Use when |
//! |--------|---|---|-------------|----------|
//! | `FlatFilteredIndex` | N/A | N/A | 0 | Baseline, high selectivity |
//! | `AcornIndex1` | 1 | 16 | 16/node | Moderate selectivity (≥10%) |
//! | `AcornIndexGamma` | 2 | 16 | 32/node | Low selectivity (<10%) |

pub mod dist;
pub mod error;
pub mod graph;
pub mod index;
pub mod search;

pub use error::AcornError;
pub use graph::AcornGraph;
pub use index::{recall_at_k, AcornIndex1, AcornIndexGamma, FilteredIndex, FlatFilteredIndex};
