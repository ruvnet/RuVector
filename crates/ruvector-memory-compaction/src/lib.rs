//! MinCut-Guided Agent Memory Compaction for RuVector
//!
//! Provides three compaction strategies that prune a time-stamped vector
//! memory store down to a target retention fraction:
//!
//! | Variant              | Strategy                              | Best for              |
//! |----------------------|---------------------------------------|-----------------------|
//! | `GreedyAgeCompactor` | FIFO – evict oldest entries           | Simple baseline       |
//! | `DecayScoreCompactor`| Exponential decay + diversity rank    | Balanced quality      |
//! | `MinCutCompactor`    | k-NN graph topology + isolation score | Max quality retention |
//!
//! ## Example
//!
//! ```rust
//! use ruvector_memory_compaction::{
//!     MemoryStore, MemoryEntry, CompactionConfig,
//!     GreedyAgeCompactor, DecayScoreCompactor, MinCutCompactor, MemoryCompactor,
//! };
//!
//! let mut store = MemoryStore::new();
//! for i in 0u64..20 {
//!     store.insert(MemoryEntry::new(i, vec![i as f32; 4], i * 1000));
//! }
//!
//! let cfg = CompactionConfig { retain_fraction: 0.5, ..Default::default() };
//! let result = GreedyAgeCompactor.compact(&store, &cfg);
//! assert_eq!(result.entries_after, 10);
//! ```

pub mod compactor;
pub mod graph;
pub mod store;

pub use compactor::{DecayScoreCompactor, GreedyAgeCompactor, MemoryCompactor, MinCutCompactor};
pub use store::{CompactionConfig, CompactionResult, MemoryEntry, MemoryStore};
