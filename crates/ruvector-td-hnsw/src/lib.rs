//! # ruvector-td-hnsw — Temporal-Decay ANN for Agent Memory
//!
//! Nearest-neighbour search where stored vectors carry insertion timestamps
//! and older entries are penalised with a higher effective distance.
//! Recency-biased retrieval lets agent memory systems naturally surface fresh
//! knowledge without separate compaction or eviction passes.
//!
//! ## Variants
//!
//! | Variant | Strategy | Use Case |
//! |---|---|---|
//! | `Baseline` | Pure L2, no time weighting | correctness reference |
//! | `TemporalDecay` | L2 × exp-decay weight | general agent memory |
//! | `CoherenceGated` | Temporal decay + age+distance gate | high-churn memory |
//!
//! ## Quick start
//!
//! ```rust
//! use ruvector_td_hnsw::{TdIndex, IndexVariant, DecayConfig};
//!
//! let config = DecayConfig::standard(2.0, 3600.0); // 1-hour half-life
//! let mut index = TdIndex::new(IndexVariant::TemporalDecay, config);
//!
//! index.insert(1, vec![0.1, 0.9, 0.3], 0);       // old entry
//! index.insert(2, vec![0.2, 0.8, 0.4], 3500);    // recent entry
//!
//! let results = index.search(&[0.15, 0.85, 0.35], 1, 4000);
//! println!("Top result: id={}", results[0].id);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod decay;
pub mod index;

pub use decay::DecayConfig;
pub use index::{l2_sq, Entry, IndexVariant, SearchResult, TdIndex};
