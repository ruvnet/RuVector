//! # ruvector-spann
//!
//! SPANN-inspired Partition Spilling for Boundary-Safe ANN in RuVector.
//!
//! ## What this implements
//!
//! SPANN (Space Partition-based Approximate Nearest Neighbor, Microsoft Research,
//! NeurIPS 2021) addresses the "boundary problem" in partition-based ANN: vectors
//! near Voronoi cell boundaries are often closer to the query than the partition
//! centroid suggests, yet they go unvisited when only the nearest partition is
//! probed. SPANN fixes this by "spilling" boundary vectors into adjacent partitions
//! at index build time, so every partition visited at query time already contains
//! the vectors that would have been missed by hard-assignment.
//!
//! ## Three variants
//!
//! | Variant | Assignment | Spill decision |
//! |---------|-----------|----------------|
//! | `SinglePartition` | 1 centroid per vector | Never spill (baseline IVF) |
//! | `SpillPartition` | up to 2 centroids per vector | Spill if d2/d1 < threshold |
//! | `CoherenceSpill` | up to 2 centroids per vector | Spill threshold driven by coherence ratio |
//!
//! ## RuVector ecosystem fit
//!
//! - Connects to `ruvector-diskann` (disk-first storage; partitions map to disk pages)
//! - Connects to `ruvector-coherence` (coherence score drives spill decision in variant 3)
//! - Connects to `ruvector-rairs` (IVF baseline complemented by spilling)
//! - Connects to MCP agent memory (partitions as namespace-isolated memory segments)
//! - Connects to ruFlo (spill threshold as a workflow-tunable parameter)

pub mod distance;
pub mod index;
pub mod kmeans;

pub use index::{
    CoherenceSpill, CoherenceSpillConfig, PartitionIndex, SearchResult, SinglePartition,
    SinglePartitionConfig, SpillPartition, SpillPartitionConfig,
};
