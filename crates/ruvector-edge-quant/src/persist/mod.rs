//! Persist v2 (`rbqx0002`) — the rv-quant snapshot format (ADR-351 G4c).
//!
//! Unlike `ruvector-rabitq`'s `rbpx0001` (which stores f32 originals and
//! re-encodes every row on load), v2 stores the **packed 1-bit codes, the
//! norms and the row keys**, plus the rotation kind and seed. Cold load
//! copies codes into place and regenerates only the seeded rotation,
//! checked against a fingerprint recorded at save time. The f32 originals
//! stay in the store for rerank.
//!
//! The snapshot is a sequence of frames, each ≤ 1 MiB, each CRC-32
//! protected, with a footer CRC over every frame CRC (see [`format`] for
//! the byte layout). Hosts either store frames individually
//! ([`save_frames`] / [`load_frames`]) or as one stream ([`save_to`] /
//! [`load_from`]). Every validation failure is a typed
//! [`crate::QuantError::Corrupt`]; a snapshot whose header declares more
//! than the budget allows is refused with `413` before any allocation.

pub mod format;
mod read;
mod write;

pub use format::{Header, MAX_FRAME_BYTES, VERSION};
pub use read::{load_frames, load_from, Decoder};
pub use write::{save_frames, save_to};
