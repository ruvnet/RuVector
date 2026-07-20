//! # ruvector-spectral-ivf
//!
//! Spectral IVF: graph-Laplacian partitioned approximate nearest-neighbour search.
//!
//! Standard IVF partitions vectors using k-means (Euclidean centroid assignment).
//! Spectral IVF instead builds a kNN graph over the vector corpus, computes the
//! **Fiedler vector** (second eigenvector of the graph Laplacian) via power
//! iteration, then recursively bisects until `n_partitions` cells are formed.
//!
//! The Fiedler vector is the continuous relaxation of the minimum balanced graph
//! cut: it places graph-connected vectors on the same side of the partition
//! boundary, producing cells that respect neighbourhood topology rather than
//! Euclidean distance to a centroid.
//!
//! ## Three variants
//!
//! | Variant | Graph edges | Partition method |
//! |---------|-------------|-----------------|
//! | `KMeansIvf` | none | Lloyd's k-means |
//! | `SpectralIvf` | cosine-similarity kNN | Fiedler bisection |
//! | `CoherenceSpectralIvf` | cosine²-weighted kNN | Fiedler bisection |
//!
//! ## RuVector ecosystem fit
//!
//! - Connects to `ruvector-mincut` (Fiedler = mincut relaxation)
//! - Connects to `ruvector-coherence` (coherence-weighted edges in variant 3)
//! - Connects to `ruvector-graph` (kNN graph as first-class structure)
//! - Connects to `ruvector-diskann` (partitions map to SSD pages)
//! - Connects to `ruvector-spann` (same IVF trait, different partitioner)
//! - Exposes as MCP memory tool (partition-scoped agent memory namespaces)
//! - Embeds cleanly in WASM (zero unsafe, no OS calls in core path)

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod distance;
pub mod graph;
pub mod index;
pub mod kmeans;
pub mod spectral;

pub use index::{AnnIndex, CoherenceSpectralIvf, KMeansIvf, SearchResult, SpectralIvf};
