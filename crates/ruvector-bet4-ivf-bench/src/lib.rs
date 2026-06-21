//! BET 4 (SepRAG, ruvnet/RuVector #534): does **lower-bound-ordered branch-and-bound**
//! IVF probing beat a tuned plain `IvfFlat` `nprobe` on unfiltered ANN over real 128-d
//! embeddings, at matched recall@10?
//!
//! This closes the BET 4 caveat left open by ADR-201: the region-pruning IVF kernel was
//! only ever run against ACORN (BET 2), never head-to-head against its natural incumbent —
//! plain IVF `nprobe`. The B&B kernel is rebuilt self-contained here (BET 2's lives only on
//! the #536 branch), over the same `ruvector-rairs` k-means substrate as the incumbent.
//!
//! Frozen gate: `docs/plans/bet4-ivf-pruning/PRE-REGISTRATION.md`.

pub mod data;
pub mod kernel;
pub mod oracle;
pub mod pca;
pub mod pq;

pub use kernel::BnBIvf;
pub use pq::{AdcCost, PqIvf};
