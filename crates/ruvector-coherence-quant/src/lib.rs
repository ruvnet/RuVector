//! # ruvector-coherence-quant
//!
//! Coherence-gated mixed-precision vector quantization (ADR-353): allocates
//! each vector's quantization precision from a per-vector *coherence score*
//! instead of applying one precision uniformly across a corpus.
//!
//! The coherence score comes from reusing `ruvector-attn-mincut`'s dynamic
//! min-cut gate ([`coherence::block_coherence`]) as a self-similarity graph
//! pruner: a vector whose local neighborhood mostly survives the cut is
//! redundant (safe to drop to a cheap 1-bit code from `ruvector-turboquant`);
//! a vector near the cut boundary is a structural outlier and keeps the
//! full Turbo4 4-bit code (or, for the more aggressive variant, an exact
//! f32 residual).
//!
//! This crate is the experiment harness for that hypothesis: a deterministic
//! synthetic corpus, three encoding variants (`baseline`, `candidate_A`,
//! `candidate_B`), and the metrics needed to honestly compare them. See
//! `examples/benchmark.rs` for the full run and
//! `docs/research/nightly/2026-10-24-coherence-gated-quant/README.md` for
//! the write-up and measured results.

pub mod allocator;
pub mod coherence;
pub mod dataset;
pub mod metrics;
pub mod rng;
