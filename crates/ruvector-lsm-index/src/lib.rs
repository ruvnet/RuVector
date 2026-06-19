//! # ruvector-lsm-index
//!
//! LSM-style epoch-segmented vector index for RuVector.
//!
//! Three tiers: hot (flat linear scan) → warm (NSW graph) → cold (NSW graph).
//! Inserts always land in the hot tier (O(1)). Compaction merges tiers
//! synchronously when capacity thresholds are crossed.
//!
//! ## Example
//!
//! ```rust
//! use ruvector_lsm_index::{LsmVectorIndex, LsmConfig};
//!
//! let mut index = LsmVectorIndex::new(LsmConfig::default());
//! index.insert(1, vec![1.0, 0.0, 0.0]);
//! index.insert(2, vec![0.0, 1.0, 0.0]);
//! index.insert(3, vec![0.0, 0.0, 1.0]);
//!
//! let results = index.search(&[1.0, 0.0, 0.0], 2);
//! assert_eq!(results[0].1, 1); // id=1 is nearest to [1,0,0]
//! ```

pub mod distance;
pub mod flat;
pub mod lsm;
pub mod nsw;

pub use flat::FlatSegment;
pub use lsm::{LsmConfig, LsmStats, LsmVectorIndex};
pub use nsw::NswSegment;
