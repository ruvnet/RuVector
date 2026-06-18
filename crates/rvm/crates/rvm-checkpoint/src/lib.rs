//! C2SP **tlog-checkpoint** export for RVM sealed witness segments.
//!
//! This crate turns [`rvm_witness::SealedSegment`] Merkle roots into
//! checkpoints in the [C2SP tlog-checkpoint] format, signed as
//! [C2SP signed-note] Ed25519 signatures (byte-compatible with Go's
//! `golang.org/x/mod/sumdb/note` package). That makes RVM witness-log
//! heads consumable by the existing transparency-log ecosystem:
//!
//! - **Sigsum / Rekor v2 style log tooling** can parse, verify, and
//!   countersign the emitted checkpoints.
//! - **Omniwitness cosigners** (witness networks that cosign checkpoints)
//!   can add their signature lines; this crate's verifier ignores unknown
//!   signature lines exactly as the spec requires, so cosigned checkpoints
//!   still round-trip.
//! - Any `sumdb/note`-compatible verifier (Go, age-style tooling, etc.)
//!   can validate RVM checkpoints given the log's public key in the
//!   standard `name+xxxxxxxx+base64` verifier-key form
//!   ([`NoteVerifier::to_verifier_key`]).
//!
//! # What this crate does NOT implement (out of scope here)
//!
//! - The **tlog witness HTTP protocol** (checkpoint submission/cosigning
//!   transport) — planned as R3.
//! - **Consistency proofs** between successive checkpoints — planned as R5.
//! - Inclusion-proof bundling; use [`rvm_witness::verify_inclusion`]
//!   directly against the seal root.
//!
//! # Mapping from sealed segments
//!
//! A [`SealedSegment`] covers records `[first_sequence, first_sequence + count)`,
//! so the checkpoint **tree size** is `first_sequence + count` and the
//! checkpoint **root hash** is the seal's 32-byte Merkle root. Note the RVM
//! witness Merkle tree hashes keyed-BLAKE3 chain MACs (ADR-134 v2), not RFC
//! 6962 SHA-256 leaves; the checkpoint format only transports the 32-byte
//! root, but cross-ecosystem *proof* verification requires the RVM hash
//! profile (documented at the origin).
//!
//! # R1 prev-seal binding seam
//!
//! The kernel-side seal carries a previous-seal binding
//! (`SealedSegment::version` / `prev_seal_digest`, R1). The checkpoint
//! body stays at the 3 required lines; the binding can be published
//! without a format break as a checkpoint **extension line**
//! (`rvm.prev_seal <base64 digest>`) via
//! [`Checkpoint::push_prev_seal_extension`] (or any opaque line via
//! [`Checkpoint::push_extension`]); parsing preserves extension lines
//! verbatim. Cross-checkpoint *verification* of that binding is part of
//! the consistency work (R5), not this crate.
//!
//! This is **host-side** code: `std` is used freely and the crate must not
//! be linked into the `no_std` kernel.
//!
//! [C2SP tlog-checkpoint]: https://github.com/C2SP/C2SP/blob/main/tlog-checkpoint.md
//! [C2SP signed-note]: https://github.com/C2SP/C2SP/blob/main/signed-note.md
//!
//! # Example
//!
//! ```
//! use rvm_checkpoint::{latest_checkpoint, NoteSigner, open_checkpoint};
//! use rvm_witness::SealedSegment;
//!
//! let seal = SealedSegment {
//!     version: rvm_witness::seal::SEAL_VERSION_UNCHAINED,
//!     root: [7u8; 32],
//!     first_sequence: 4096,
//!     count: 1024,
//!     prev_seal_digest: [0u8; 32],
//!     signature: [0u8; 64],
//! };
//! let cp = latest_checkpoint("ruvector.dev/rvm-witness/demo", &[seal]).unwrap();
//! assert_eq!(cp.tree_size(), 5120);
//!
//! let signer = NoteSigner::from_seed("ruvector.dev/rvm-witness/demo", &[42u8; 32]).unwrap();
//! let note = cp.to_signed_note(&signer);
//! let (parsed, verified) = open_checkpoint(&note, &[signer.verifier()]).unwrap();
//! assert_eq!(parsed, cp);
//! assert_eq!(verified.verified_by, ["ruvector.dev/rvm-witness/demo"]);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

// Exposed (doc-hidden) so the out-of-crate unit tests in tests/unit.rs can
// exercise it; not part of the supported API surface.
#[doc(hidden)]
pub mod base64;
mod checkpoint;
mod note;

pub use checkpoint::{latest_checkpoint, Checkpoint};
pub use note::{open, open_checkpoint, sign, KeyId, NoteSigner, NoteVerifier, VerifiedNote};

/// Errors produced by checkpoint construction, serialization, and
/// signed-note verification.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Checkpoint body text violates the tlog-checkpoint format.
    MalformedCheckpoint(&'static str),
    /// Signed-note envelope violates the signed-note format.
    MalformedNote(&'static str),
    /// A signature line violates the signed-note format.
    MalformedSignature(&'static str),
    /// A signer/verifier key string violates the expected encoding.
    MalformedKey(&'static str),
    /// Input is not canonical RFC 4648 §4 standard base64.
    InvalidBase64,
    /// Origin line is empty or contains a newline.
    InvalidOrigin(&'static str),
    /// Extension line is empty or contains a newline.
    InvalidExtension(&'static str),
    /// Key name is empty or contains a space, `+`, or newline.
    InvalidKeyName(&'static str),
    /// `first_sequence + count` overflows `u64`.
    TreeSizeOverflow,
    /// A key string's embedded key hash does not match the computed key ID.
    KeyIdMismatch,
    /// A signature from a *known* key failed cryptographic verification.
    /// Per signed-note semantics this rejects the whole note.
    InvalidSignature(String),
    /// No signature from any known key verified successfully
    /// ("clients MUST reject the note").
    NoVerifiedSignature,
    /// No sealed segments were provided to the adapter.
    NoSeals,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MalformedCheckpoint(m) => write!(f, "malformed checkpoint: {m}"),
            Self::MalformedNote(m) => write!(f, "malformed note: {m}"),
            Self::MalformedSignature(m) => write!(f, "malformed signature line: {m}"),
            Self::MalformedKey(m) => write!(f, "malformed key: {m}"),
            Self::InvalidBase64 => write!(f, "invalid base64 (RFC 4648 std, canonical)"),
            Self::InvalidOrigin(m) => write!(f, "invalid origin: {m}"),
            Self::InvalidExtension(m) => write!(f, "invalid extension line: {m}"),
            Self::InvalidKeyName(m) => write!(f, "invalid key name: {m}"),
            Self::TreeSizeOverflow => write!(f, "tree size overflows u64"),
            Self::KeyIdMismatch => write!(f, "embedded key hash does not match computed key ID"),
            Self::InvalidSignature(name) => {
                write!(f, "signature from known key {name:?} failed verification")
            }
            Self::NoVerifiedSignature => write!(f, "no signature from a known key verified"),
            Self::NoSeals => write!(f, "no sealed segments provided"),
        }
    }
}

impl std::error::Error for Error {}
