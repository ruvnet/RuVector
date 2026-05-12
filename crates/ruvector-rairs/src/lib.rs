//! # ruvector-rairs — IVF with Redundant Assignment + Amplified Inverse Residual
//!
//! An Inverted File (IVF) index family that recovers the low-`nprobe` recall
//! classic IVF loses near Voronoi-cell boundaries, by **redundantly assigning**
//! each vector to a primary list *and* a residual-amplified secondary list, then
//! storing the shared copies in deduplicating 32-vector blocks so the second
//! assignment costs no extra memory. Design rationale and the empirical results
//! are in `docs/adr/ADR-193`.
//!
//! > **Provenance note.** The "RAIRS / SEIL" naming and the
//! > `arXiv:2601.07183 (SIGMOD 2026)` reference cited in the design docs have
//! > not been independently verified; treat this crate as an original
//! > implementation of the redundant-assignment idea (cf. spill lists / SOAR /
//! > multi-probe LSH) and judge it on the benchmarks in `src/main.rs`, not on
//! > the citation.
//!
//! ## Index family
//!
//! | Variant        | Assignment | Layout | Description                             |
//! |----------------|------------|--------|-----------------------------------------|
//! | `IvfFlat`      | single     | flat   | baseline — one list per vector          |
//! | `RairsStrict`  | dual RAIR  | flat   | secondary assignment, no dedup          |
//! | `RairsSeil`    | dual RAIR  | SEIL   | shared 32-vector blocks, query-time dedup |
//!
//! All three satisfy [`AnnIndex`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod index;
pub mod ivf;
pub mod kmeans;
pub mod rairs;
pub mod seil;

pub use error::RairsError;
pub use index::{AnnIndex, SearchResult};
pub use ivf::IvfFlat;
pub use rairs::RairsStrict;
pub use seil::RairsSeil;
