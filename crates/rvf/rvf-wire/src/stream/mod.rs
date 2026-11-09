//! RVF Stream Descriptor (RVSD, ADR-353): verified streaming delivery.
//!
//! * [`decode_descriptor`] parses and authenticates a signed descriptor.
//! * [`StreamVerifier`] is the receiver state machine; it releases
//!   [`VerifiedChunk`]s only after their leaf hash matched and the stages named
//!   in `requires_mask` are fully verified.
//! * [`FrameReader`] reassembles frames from arbitrary byte splits.
//! * [`StreamDescriptorBuilder`] is the producer side.
//!
//! Wire details: `crates/rvf/tests/vectors/README.md`.

mod builder;
mod descriptor;
mod frame;
mod pin;
mod verifier;

pub use builder::{FrameIter, StreamBundle, StreamDescriptorBuilder, ROOT_PAGE_SEG_ID};
pub use descriptor::{decode_descriptor, VerifiedDescriptor};
pub use frame::{
    chunk_frame, descriptor_frame, table_frame, FrameReader, FRAME_CHUNK, FRAME_CHUNK_TABLE,
    FRAME_DESCRIPTOR,
};
pub use pin::{MemoryPinStore, PinConflict, PinStore};
pub use verifier::{Event, IdentityStatus, StreamState, StreamVerifier, VerifiedChunk};

use rvf_types::stream::StreamClass;

/// Epoch pin lookup by `(signer_key_id, file_id)`: a pin set for one signer
/// never constrains another signer's lineage.
pub type EpochPinFn<'a> = dyn Fn(&[u8; 8], &[u8; 16]) -> Option<EpochPin> + 'a;

/// A pinned `(epoch, descriptor hash)` for one `file_id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpochPin {
    /// Highest accepted epoch.
    pub epoch: u64,
    /// Hash of the descriptor accepted at that epoch (mandatory). At an equal
    /// epoch a descriptor with a different hash is a conflict.
    pub descriptor_hash: [u8; 32],
}

/// Receiver policy for descriptors and streams.
#[derive(Clone, Copy)]
pub struct StreamPolicy<'a> {
    /// Maximum descriptor size in bytes (default 1 MiB).
    pub max_descriptor_bytes: usize,
    /// Maximum `seg_count`.
    pub max_segments: u32,
    /// Maximum `body_len`.
    pub max_body_len: u64,
    /// Maximum `stored_len` of one segment.
    pub max_segment_bytes: u64,
    /// Maximum declared `uncompressed_len` of one segment.
    pub max_uncompressed_len: u64,
    /// Per-class totals of `stored_len`, indexed by [`StreamClass`].
    pub max_class_bytes: [u64; StreamClass::COUNT],
    /// Trusted Ed25519 public keys (matched by `signer_key_id`).
    pub trusted_keys: &'a [[u8; 32]],
    /// Epoch pin lookup by `file_id`.
    pub epoch_pin: Option<&'a EpochPinFn<'a>>,
    /// Shared pin store: read at admission, re-checked at completion, and the
    /// target of `commit_pin`. Used in addition to `epoch_pin`.
    pub pins: Option<&'a dyn PinStore>,
    /// Trusted clock (ns), enforced only when set.
    pub now_ns: Option<u64>,
    /// Expected `policy_hash`, enforced only when set.
    pub expected_policy_hash: Option<[u8; 32]>,
    /// Counted faults tolerated before the stream is poisoned.
    pub fault_budget: u32,
    /// Cap on chunk-table bytes retained (accepted tables are kept so that
    /// duplicate chunks of finished segments can be checked). A table frame
    /// that would exceed it is refused before anything is allocated.
    ///
    /// The default is derived from the default `max_body_len` (1 TiB) at the
    /// smallest chunk size, so it is never below what the body limit allows;
    /// a tighter `max_body_len` should be paired with a tighter cap.
    pub max_table_bytes: u64,
    /// Cap on idempotent duplicate frames; `None` = `4 * total chunks + 64`.
    /// Exceeding it poisons the stream (`StreamDuplicateFlood`).
    pub max_duplicate_frames: Option<u64>,
}

impl<'a> StreamPolicy<'a> {
    /// Defaults: 1 MiB descriptor, 16384 segments, 1 TiB body, 1 GiB segment.
    pub fn new(trusted_keys: &'a [[u8; 32]]) -> Self {
        Self {
            max_descriptor_bytes: 1 << 20,
            max_segments: 16384,
            max_body_len: 1 << 40,
            max_segment_bytes: 1 << 30,
            max_uncompressed_len: 1 << 32,
            max_class_bytes: [1 << 40; StreamClass::COUNT],
            trusted_keys,
            epoch_pin: None,
            pins: None,
            now_ns: None,
            expected_policy_hash: None,
            fault_budget: 4,
            // Derived from the body limit (F2-09): 32 bytes per 4 KiB chunk
            // (the smallest chunk size) of a maximal body, so the default can
            // never be below what `max_body_len` allows. Lower both together
            // for a tighter retained-table bound.
            max_table_bytes: (1u64 << 40) / 4096 * 32,
            max_duplicate_frames: None,
        }
    }

    /// Preset for internet-facing clients (F3-06): 16 GiB body, 256 MiB
    /// segments, 4096 segments, 256 KiB descriptor, and a retained-table cap
    /// derived from the body limit (128 MiB worst case; tables are freed as
    /// their segments verify, so typical use holds only open segments).
    pub fn conservative(trusted_keys: &'a [[u8; 32]]) -> Self {
        let body = 16u64 << 30;
        Self {
            max_descriptor_bytes: 256 << 10,
            max_segments: 4096,
            max_body_len: body,
            max_segment_bytes: 256 << 20,
            max_uncompressed_len: 1 << 30,
            max_class_bytes: [body; StreamClass::COUNT],
            max_table_bytes: body / 4096 * 32,
            ..Self::new(trusted_keys)
        }
    }

    /// Preset for small devices (F3-06): 256 MiB body, 16 MiB segments, 1024
    /// segments, 64 KiB descriptor, 2 MiB retained-table cap.
    pub fn embedded(trusted_keys: &'a [[u8; 32]]) -> Self {
        let body = 256u64 << 20;
        Self {
            max_descriptor_bytes: 64 << 10,
            max_segments: 1024,
            max_body_len: body,
            max_segment_bytes: 16 << 20,
            max_uncompressed_len: 64 << 20,
            max_class_bytes: [body; StreamClass::COUNT],
            max_table_bytes: body / 4096 * 32,
            ..Self::new(trusted_keys)
        }
    }

    /// Upper bound on bytes allocated while decoding a descriptor: the
    /// signed-message copy plus the entry vector plus the id scratch vector.
    pub fn descriptor_alloc_bound(&self) -> usize {
        4 * self.max_descriptor_bytes + 4096
    }
}
