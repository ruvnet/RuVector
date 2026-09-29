//! In-memory vector index for the ruvector edge `VectorShard` (ADR-351 §6.1).
//!
//! * [`QuantParams`] — versioned int8 scalar quantizer (per-dimension trained
//!   scale/offset, or a fixed range such as cosine's `[-1, 1]`).
//! * [`QuantFlatIndex`] — M2a: u8 flat scan with asymmetric distance, then an
//!   exact f32 rerank of the top candidates through [`RerankFetch`], which the
//!   store backs with paged `SELECT … WHERE iid IN (…)` (≤ [`RERANK_BATCH`]).
//! * [`HnswIndex`] — M2b: HNSW whose traversal reads only u8 codes, with
//!   fixed-stride `u32` adjacency, caller-supplied randomness ([`LevelRng`],
//!   [`SplitMix64`]), in-place update, tombstone purge and dense-iid
//!   compaction, and an `iid_base` matching the store's iids.
//! * [`IndexChunk`] — both indexes stream into ≤ 1 MiB `index_chunks` rows
//!   `(epoch, part, bytes)` (`encode_into`) and back (`from_chunk_iter`),
//!   holding one row at a time. Each row carries a CRC-32C and the last one
//!   the whole-payload sha256; any corruption, truncation, reordering, gap
//!   or structural inconsistency is a typed [`DecodeError`] so the caller
//!   falls back to replay.
//! * [`memory`] — bytes-per-slot and load-time accounting so the store can
//!   enforce caps.
//!
//! # Why not `rvf-index`
//!
//! `crates/rvf/rvf-index` (ADR §1.4, gate G4a) cannot back this crate as is:
//! `HnswGraph::insert`/`search` read neighbours through
//! `&dyn VectorStore`, whose accessor returns `&[f32]`
//! (`hnsw.rs:183-189`, `traits.rs:7`), so traversal needs every vector
//! resident as f32 — exactly the memory M2 exists to avoid. Its adjacency is
//! `BTreeMap<u64, Vec<u64>>` per layer (`hnsw.rs:98`) and its segment header
//! lacks `entry_point`/`max_layer`/`m0`. A graph↔bytes adapter would not fix
//! the traversal, so this crate carries a small self-contained HNSW instead
//! and leaves `crates/rvf` untouched.
//!
//! No `getrandom`, threads or `std::time`: randomness is supplied by the
//! caller and the crate compiles unchanged for `wasm32-unknown-unknown`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod bytes;
mod crc;
mod error;
mod flat;
mod grow;
mod heap;
mod hnsw;
pub mod memory;
mod metric;
mod persist;
mod quant;
mod rerank;
mod rng;

pub use error::{DecodeError, EmitError, EncodeError, IndexError, RerankError};
pub use flat::QuantFlatIndex;
pub use heap::Hit;
pub use hnsw::{HnswIndex, HnswParams};
pub use metric::{distance, dot, l2_sq, norm, validate_vector, Metric};
pub use persist::{
    reseal, EncodedIndex, IndexChunk, IndexDigest, IndexKind, CHUNK_HEADER_LEN, MAX_CHUNK_BYTES,
};
pub use quant::{PreparedQuery, QuantKind, QuantParams, MAX_DIM};
pub use rerank::{rerank, RerankFetch, SliceFetch, RERANK_BATCH};
pub use rng::{level_from_u64, LevelRng, SplitMix64};
