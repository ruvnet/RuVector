//! Version-2 witness log: keyed-BLAKE3 chain MACs with 128-bit links
//! (ADR-134 v2).
//!
//! Fixes the two structural weaknesses of the v1 chain:
//!
//! 1. **Cost on the syscall path.** v1 computed two hashes per append
//!    (record hash + chain hash, SHA-256 when `crypto-sha256` is on).
//!    v2 computes exactly **one** keyed BLAKE3 compression: the MAC
//!    input is `content[0..44] || prev_mac[16]` = 60 bytes, which fits
//!    a single 64-byte BLAKE3 block.
//! 2. **32-bit folded links.** v1 folded its 64-bit chain values to
//!    32 bits to fit the 64-byte record; a forged link collides after
//!    ~2^16 attempts. v2 stores full 128-bit MACs (`prev_mac`,
//!    `chain_mac`), and because the MAC is keyed, forging any link
//!    requires the chain key, not just a hash collision.
//!
//! Per-record signatures are intentionally **absent** in v2: tamper
//! evidence for exported segments comes from Merkle sealing (see
//! [`crate::seal`]), which amortizes one signature over a whole
//! segment instead of paying HMAC per record.
//!
//! # Forward security (R4): per-segment chain-key ratchet
//!
//! Every [`WitnessLogV2::seal_segment`] ratchets the chain MAC key
//! (`key_{n+1} = derive_key(`[`RATCHET_CONTEXT`]`, key_n)`) and erases
//! the old key, atomically with the seal (same lock, no window where
//! the pre-ratchet key persists after the seal). Consequences:
//!
//! - **Compromise window = the current unsealed segment only.** An
//!   attacker who extracts the live key can forge records of the
//!   current epoch, but cannot recompute MACs of any earlier epoch
//!   (`derive_key` is one-way), and earlier segments are additionally
//!   pinned by their Merkle seals and the seal chain (R1).
//! - **Verifier capability is asymmetric by design**: the holder of
//!   the *initial* key re-derives every epoch key and can verify the
//!   whole log ([`crate::verify_chain_v2_ratcheted`]); the logger
//!   itself can no longer forge history older than its last ratchet.
//! - Epoch boundaries are the log's `sequence` at each seal. Under
//!   [`CoveragePolicy::Strict`] this equals
//!   `seal.first_sequence + seal.count` of each chained seal, so the
//!   boundaries are recoverable from the seals alone.
//!
//! # Coverage invariants (R6)
//!
//! [`CoveragePolicy::Strict`] turns the two silent coverage-loss modes
//! (`segment_dropped`, `total_overwritten` of unsealed records) into
//! [`CoverageError`] backpressure from [`WitnessLogV2::try_append`].
//! [`CoveragePolicy::BestEffort`] preserves the original counter
//! behavior; all pre-existing constructors keep it for stability.

use crate::seal::{
    seal_chain_genesis, SealedSegment, SegmentAccumulator, SegmentSealSigner,
    DEFAULT_SEGMENT_SIZE,
};
use rvm_types::{WitnessRecord, WitnessRecordV2};
use spin::Mutex;

/// Domain-separation context string for chain key derivation.
pub const CHAIN_KEY_CONTEXT: &str = "rvm-witness 2026 v2 chain key";

/// Derive a 32-byte chain key from arbitrary key material using
/// BLAKE3's `derive_key` mode with the [`CHAIN_KEY_CONTEXT`] domain.
#[must_use]
pub fn derive_chain_key(material: &[u8]) -> [u8; 32] {
    blake3::derive_key(CHAIN_KEY_CONTEXT, material)
}

/// The compile-time default chain key.
///
/// **Security warning:** this key is public. Production deployments
/// MUST supply a TEE- or boot-derived key via [`WitnessLogV2::with_key`].
#[must_use]
pub fn default_chain_key() -> [u8; 32] {
    derive_chain_key(b"rvm-witness-default-chain-key-v2")
}

/// Domain-separation context string for the forward-secure chain-key
/// ratchet (R4): `key_{n+1} = blake3::derive_key(RATCHET_CONTEXT, key_n)`.
pub const RATCHET_CONTEXT: &str = "rvm-witness 2026 v2 chain key ratchet";

/// Derive the next epoch's chain key from the current one (one-way).
///
/// Applied by [`WitnessLogV2::seal_segment`] once per seal; verifiers
/// holding the initial key re-derive the same sequence (see
/// [`crate::verify_chain_v2_ratcheted`]).
#[must_use]
pub fn ratchet_chain_key(key: &[u8; 32]) -> [u8; 32] {
    blake3::derive_key(RATCHET_CONTEXT, key)
}

/// Best-effort secure erasure of 32-byte key material.
///
/// `no_std` + `forbid(unsafe_code)` rules out `write_volatile`-based
/// zeroization; this overwrites with zeros and pins the writes with
/// [`core::hint::black_box`] so the compiler cannot elide them as dead
/// stores. Transient copies inside `blake3` internals are out of reach
/// and remain a documented limitation.
pub fn erase_key(key: &mut [u8; 32]) {
    for byte in key.iter_mut() {
        *byte = 0;
    }
    core::hint::black_box(key);
}

/// Coverage policy for a [`WitnessLogV2`] (R6): whether losing Merkle
/// coverage or overwriting unsealed records is an error or a counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoveragePolicy {
    /// Coverage is an invariant. [`WitnessLogV2::try_append`] returns
    /// [`CoverageError::SegmentFull`] instead of dropping a Merkle leaf
    /// and [`CoverageError::UnsealedOverwrite`] instead of letting the
    /// ring overwrite a record that was never sealed (backpressure:
    /// seal, then retry). Recommended for all new code.
    Strict,
    /// Original behavior: appends never fail; coverage loss is only
    /// counted ([`WitnessLogV2::segment_dropped`],
    /// [`WitnessLogV2::total_overwritten`]). For callers that cannot
    /// seal synchronously.
    BestEffort,
}

/// Backpressure errors from [`WitnessLogV2::try_append`] under
/// [`CoveragePolicy::Strict`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageError {
    /// The segment accumulator is full: appending would leave the
    /// record without Merkle coverage. Seal the segment
    /// ([`WitnessLogV2::seal_segment`]) and retry.
    SegmentFull,
    /// The ring slot to be reused still holds a never-sealed record
    /// (its sequence number is given). Seal and export before
    /// appending.
    UnsealedOverwrite {
        /// Sequence number of the unsealed record that would be lost.
        sequence: u64,
    },
}

impl core::fmt::Display for CoverageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SegmentFull => write!(f, "segment full: seal before appending"),
            Self::UnsealedOverwrite { sequence } => {
                write!(f, "ring overwrite would lose unsealed record {sequence}")
            }
        }
    }
}

/// Compute a v2 chain MAC: `trunc128(BLAKE3_keyed(key, content || prev_mac))`.
///
/// `content` must be the first [`WitnessRecordV2::CONTENT_LEN`] bytes of
/// the record's canonical serialization (which includes `sequence`).
/// The 60-byte input fits one BLAKE3 block: exactly one keyed
/// compression per call.
#[must_use]
pub fn compute_chain_mac_v2(
    key: &[u8; 32],
    content: &[u8],
    prev_mac: &[u8; 16],
) -> [u8; 16] {
    debug_assert_eq!(content.len(), WitnessRecordV2::CONTENT_LEN);
    let mut buf = [0u8; WitnessRecordV2::MAC_INPUT_LEN];
    buf[..WitnessRecordV2::CONTENT_LEN].copy_from_slice(content);
    buf[WitnessRecordV2::CONTENT_LEN..].copy_from_slice(prev_mac);
    let hash = blake3::keyed_hash(key, &buf);
    let mut mac = [0u8; 16];
    mac.copy_from_slice(&hash.as_bytes()[..16]);
    mac
}

/// Append-only ring buffer of v2 witness records.
///
/// `N` is the ring capacity; `SEG` is the Merkle segment size (leaves
/// accumulated between seals, default [`DEFAULT_SEGMENT_SIZE`]).
pub struct WitnessLogV2<const N: usize, const SEG: usize = DEFAULT_SEGMENT_SIZE> {
    inner: Mutex<Inner<N, SEG>>,
}

struct Inner<const N: usize, const SEG: usize> {
    records: [WitnessRecordV2; N],
    write_pos: usize,
    /// Current 128-bit chain head (the last record's `chain_mac`, or
    /// the genesis value if empty).
    head_mac: [u8; 16],
    /// Genesis value the chain started from (zero, or a v1 anchor).
    genesis: [u8; 16],
    sequence: u64,
    total_emitted: u64,
    total_overwritten: u64,
    key: [u8; 32],
    segment: SegmentAccumulator<SEG>,
    /// Records appended while the current segment was already full
    /// (their leaves were NOT accumulated; seal more often to avoid).
    segment_dropped: u64,
    /// Coverage policy (R6). Enforced by `try_append`.
    policy: CoveragePolicy,
    /// Sequence watermark at the last seal: records with a sequence
    /// below this have had seal coverage. Exact under `Strict` (no
    /// leaves are ever dropped); under `BestEffort`, dropped records
    /// below the watermark were never actually sealed.
    sealed_up_to: u64,
    /// Digest of the most recent seal, or `seal_chain_genesis()` if no
    /// segment has been sealed yet (R1 chaining state).
    last_seal_digest: [u8; 32],
    /// Number of key ratchets performed (= number of seals; R4).
    key_epoch: u64,
}

impl<const N: usize, const SEG: usize> WitnessLogV2<N, SEG> {
    const _ASSERT_N_NONZERO: () = assert!(N > 0, "witness log capacity must be > 0");
    const _ASSERT_SEG_NONZERO: () = assert!(SEG > 0, "segment size must be > 0");

    /// Create an empty v2 log using the default chain key.
    ///
    /// **Security warning:** the default key is public; use
    /// [`Self::with_key`] with a TEE/boot-derived key in production.
    #[must_use]
    pub fn new() -> Self {
        Self::with_key(default_chain_key())
    }

    /// Create an empty v2 log with the given 32-byte chain key.
    #[must_use]
    pub fn with_key(key: [u8; 32]) -> Self {
        Self::with_key_and_genesis(key, [0u8; 16])
    }

    /// Create an empty v2 log whose chain starts from `genesis` instead
    /// of zero.
    ///
    /// Used to anchor a migrated v1 log: pass
    /// [`crate::versioned::v1_head_to_genesis`] of the verified v1 chain
    /// head so the first v2 record's `prev_mac` cryptographically binds
    /// the v1 history.
    ///
    /// Coverage policy is [`CoveragePolicy::BestEffort`], matching the
    /// behavior this constructor always had; new code should prefer
    /// [`Self::with_policy`] / [`Self::with_genesis_and_policy`] with
    /// [`CoveragePolicy::Strict`].
    #[must_use]
    pub fn with_key_and_genesis(key: [u8; 32], genesis: [u8; 16]) -> Self {
        Self::with_genesis_and_policy(key, genesis, CoveragePolicy::BestEffort)
    }

    /// Create an empty v2 log with an explicit coverage policy (R6) and
    /// the zero genesis. Use [`CoveragePolicy::Strict`] unless the
    /// caller genuinely cannot seal synchronously.
    #[must_use]
    pub fn with_policy(key: [u8; 32], policy: CoveragePolicy) -> Self {
        Self::with_genesis_and_policy(key, [0u8; 16], policy)
    }

    /// Create an empty v2 log with explicit genesis and coverage policy.
    #[must_use]
    pub fn with_genesis_and_policy(
        key: [u8; 32],
        genesis: [u8; 16],
        policy: CoveragePolicy,
    ) -> Self {
        let () = Self::_ASSERT_N_NONZERO;
        let () = Self::_ASSERT_SEG_NONZERO;
        Self {
            inner: Mutex::new(Inner {
                records: [WitnessRecordV2::zeroed(); N],
                write_pos: 0,
                head_mac: genesis,
                genesis,
                sequence: 0,
                total_emitted: 0,
                total_overwritten: 0,
                key,
                segment: SegmentAccumulator::new(0),
                segment_dropped: 0,
                policy,
                sealed_up_to: 0,
                last_seal_digest: seal_chain_genesis(),
                key_epoch: 0,
            }),
        }
    }

    /// Append a v2 record built from content fields.
    ///
    /// Fills `version`, `sequence`, `prev_mac`, and `chain_mac`, then
    /// stores the record. Returns the assigned sequence number.
    ///
    /// Cost: one keyed BLAKE3 compression (60-byte input) plus
    /// bookkeeping. No per-record signature is computed; use
    /// [`Self::seal_segment`] for exportable tamper evidence.
    ///
    /// This entry point is infallible and therefore always has
    /// [`CoveragePolicy::BestEffort`] semantics (coverage loss is
    /// counted, never refused) **even on a
    /// [`CoveragePolicy::Strict`] log**. Strict callers must use
    /// [`Self::try_append`] to get backpressure instead of silent
    /// coverage loss.
    pub fn append(&self, record: WitnessRecordV2) -> u64 {
        let mut inner = self.inner.lock();
        Self::append_locked(&mut inner, record)
    }

    /// Append a v2 record, enforcing the log's [`CoveragePolicy`] (R6).
    ///
    /// Under [`CoveragePolicy::Strict`] this fails — *before* mutating
    /// any state — when the record would lose coverage:
    ///
    /// - [`CoverageError::SegmentFull`]: the segment accumulator holds
    ///   `SEG` leaves; the record's MAC could not be accumulated for
    ///   Merkle sealing. Seal ([`Self::seal_segment`]) and retry.
    /// - [`CoverageError::UnsealedOverwrite`]: the ring is full and the
    ///   slot to be reused holds a record that was never sealed.
    ///
    /// Under [`CoveragePolicy::BestEffort`] it never fails (identical
    /// to [`Self::append`]).
    ///
    /// # Errors
    ///
    /// See above; only returned for `Strict` logs.
    pub fn try_append(&self, record: WitnessRecordV2) -> Result<u64, CoverageError> {
        let mut inner = self.inner.lock();
        if inner.policy == CoveragePolicy::Strict {
            if inner.segment.is_full() {
                return Err(CoverageError::SegmentFull);
            }
            if inner.total_emitted >= N as u64 {
                // The slot being reused holds the record appended N
                // sequence numbers ago.
                let victim = inner.sequence.wrapping_sub(N as u64);
                if victim >= inner.sealed_up_to {
                    return Err(CoverageError::UnsealedOverwrite { sequence: victim });
                }
            }
        }
        Ok(Self::append_locked(&mut inner, record))
    }

    /// Shared append path (caller holds the lock).
    fn append_locked(inner: &mut Inner<N, SEG>, mut record: WitnessRecordV2) -> u64 {
        record.version = WitnessRecordV2::VERSION;
        record.sequence = inner.sequence;
        record.prev_mac = inner.head_mac;
        let bytes = record.to_bytes();
        record.chain_mac = compute_chain_mac_v2(
            &inner.key,
            &bytes[..WitnessRecordV2::CONTENT_LEN],
            &record.prev_mac,
        );

        let seq = record.sequence;
        if inner.total_emitted >= N as u64 {
            inner.total_overwritten += 1;
        }
        let pos = inner.write_pos;
        inner.records[pos] = record;
        inner.write_pos = (pos + 1) % N;
        inner.head_mac = record.chain_mac;
        inner.sequence = seq.wrapping_add(1);
        inner.total_emitted += 1;

        // Accumulate the leaf for Merkle sealing (bookkeeping only).
        if !inner.segment.push(record.chain_mac) {
            inner.segment_dropped += 1;
        }

        seq
    }

    /// Append using the content fields of a v1 [`WitnessRecord`].
    ///
    /// Convenience for callers that still build v1 structs (emitters,
    /// gates); the v1 chain fields are ignored and replaced by v2 MACs.
    pub fn append_v1_content(&self, record: &WitnessRecord) -> u64 {
        self.append(WitnessRecordV2::from_v1_content(record))
    }

    /// Seal the current Merkle segment with `signer` and start a new one.
    ///
    /// Returns the [`SealedSegment`] (root + signature + metadata) and a
    /// copy of the [`SegmentAccumulator`] so the caller can export
    /// inclusion proofs for any record in the sealed segment. Returns
    /// `None` if no records were accumulated since the last seal.
    ///
    /// This is the **only** place signature cost is paid: one signature
    /// per up-to-`SEG` records, off the per-record append path.
    ///
    /// Hardening performed atomically with the seal (same lock):
    ///
    /// - **R1**: the seal is [chained](crate::seal::SEAL_VERSION_CHAINED)
    ///   — its digest binds the previous seal's digest (genesis constant
    ///   for the first seal), so sealed history ordering is verifiable
    ///   from seals alone via [`crate::verify_seal_chain`].
    /// - **R4**: the chain MAC key is ratcheted
    ///   ([`ratchet_chain_key`]) and the old key erased before the lock
    ///   is released — there is no window in which the pre-seal key
    ///   outlives the seal. After this returns, even this log cannot
    ///   forge records of the sealed (or any earlier) epoch; verify
    ///   multi-epoch logs with [`crate::verify_chain_v2_ratcheted`]
    ///   from the *initial* key.
    pub fn seal_segment<G: SegmentSealSigner>(
        &self,
        signer: &G,
    ) -> Option<(SealedSegment, SegmentAccumulator<SEG>)> {
        let mut inner = self.inner.lock();
        if inner.segment.is_empty() {
            return None;
        }
        let acc = inner.segment;
        let sealed = acc.seal_chained(signer, &inner.last_seal_digest)?;
        inner.last_seal_digest = sealed.digest();
        inner.sealed_up_to = inner.sequence;
        inner.segment = SegmentAccumulator::new(inner.sequence);
        // R4 ratchet: derive the next epoch key and destroy the old
        // one. `inner.key = next` overwrites the old key bytes in
        // place; the stack copy of the new key is then erased.
        let mut next = ratchet_chain_key(&inner.key);
        inner.key = next;
        erase_key(&mut next);
        inner.key_epoch += 1;
        Some((sealed, acc))
    }

    /// Current 128-bit chain head (the last record's `chain_mac`).
    ///
    /// Export this to external anchoring (CT log, QMDB, remote
    /// attestation) so truncation of the tail is detectable.
    pub fn chain_head(&self) -> [u8; 16] {
        self.inner.lock().head_mac
    }

    /// The genesis value this chain started from.
    pub fn genesis(&self) -> [u8; 16] {
        self.inner.lock().genesis
    }

    /// The **current-epoch** 32-byte chain MAC key (treat as secret).
    ///
    /// After `n` seals this is `key_n` (R4 ratchet): it verifies only
    /// records appended since the last seal. Verifiers of full logs
    /// must hold the *initial* key and re-derive epochs via
    /// [`ratchet_chain_key`] / [`crate::verify_chain_v2_ratcheted`].
    pub fn chain_key(&self) -> [u8; 32] {
        self.inner.lock().key
    }

    /// Number of chain-key ratchets performed (= seals; R4). The
    /// current [`Self::chain_key`] is `key_epoch` derivations from the
    /// initial key.
    pub fn key_epoch(&self) -> u64 {
        self.inner.lock().key_epoch
    }

    /// This log's coverage policy (R6).
    pub fn coverage_policy(&self) -> CoveragePolicy {
        self.inner.lock().policy
    }

    /// Sequence watermark at the last seal: records below it have had
    /// seal coverage (exact under [`CoveragePolicy::Strict`]).
    pub fn sealed_up_to(&self) -> u64 {
        self.inner.lock().sealed_up_to
    }

    /// Digest of the most recent seal ([`SealedSegment::digest`]), or
    /// [`seal_chain_genesis`] if nothing has been sealed. The next seal
    /// will bind this value (R1); export it alongside the chain head
    /// for external anchoring.
    pub fn last_seal_digest(&self) -> [u8; 32] {
        self.inner.lock().last_seal_digest
    }

    /// Number of leaves in the current (unsealed) segment.
    pub fn segment_len(&self) -> usize {
        self.inner.lock().segment.len()
    }

    /// True when the current segment is full and should be sealed.
    pub fn segment_is_full(&self) -> bool {
        self.inner.lock().segment.is_full()
    }

    /// Number of records appended while the segment was full (their
    /// leaves were not accumulated; they remain chain-protected only).
    pub fn segment_dropped(&self) -> u64 {
        self.inner.lock().segment_dropped
    }

    /// Total number of records ever emitted.
    pub fn total_emitted(&self) -> u64 {
        self.inner.lock().total_emitted
    }

    /// Number of records silently overwritten by ring wrap-around.
    pub fn total_overwritten(&self) -> u64 {
        self.inner.lock().total_overwritten
    }

    /// True when the used slot count has reached `watermark`.
    pub fn needs_drain(&self, watermark: usize) -> bool {
        self.len() >= watermark
    }

    /// Number of records currently in the buffer.
    #[allow(clippy::cast_possible_truncation)]
    pub fn len(&self) -> usize {
        let total = self.inner.lock().total_emitted;
        if total >= N as u64 { N } else { total as usize }
    }

    /// True if no records have been emitted.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().total_emitted == 0
    }

    /// Copy of the record at the given ring index.
    pub fn get(&self, ring_index: usize) -> Option<WitnessRecordV2> {
        if ring_index >= N {
            return None;
        }
        let inner = self.inner.lock();
        if inner.total_emitted == 0 {
            return None;
        }
        Some(inner.records[ring_index])
    }

    /// Copies the most recent records into `buf`. Returns count copied.
    pub fn snapshot(&self, buf: &mut [WitnessRecordV2]) -> usize {
        let inner = self.inner.lock();
        #[allow(clippy::cast_possible_truncation)]
        let available = if inner.total_emitted >= N as u64 {
            N
        } else {
            inner.total_emitted as usize
        };
        let to_copy = buf.len().min(available);
        if to_copy == 0 {
            return 0;
        }
        let start = if inner.total_emitted >= N as u64 {
            inner.write_pos
        } else {
            0
        };
        for (i, slot) in buf.iter_mut().enumerate().take(to_copy) {
            let idx = (start + (available - to_copy) + i) % N;
            *slot = inner.records[idx];
        }
        to_copy
    }
}

impl<const N: usize, const SEG: usize> Default for WitnessLogV2<N, SEG> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvm_types::ActionKind;

    fn make_record(kind: ActionKind, actor: u32, target: u64, ts: u64) -> WitnessRecordV2 {
        let mut r = WitnessRecordV2::zeroed();
        r.action_kind = kind as u8;
        r.actor_partition_id = actor;
        r.target_object_id = target;
        r.timestamp_ns = ts;
        r
    }

    #[test]
    fn append_assigns_sequence_and_macs() {
        let log = WitnessLogV2::<16>::new();
        let s0 = log.append(make_record(ActionKind::PartitionCreate, 1, 100, 1000));
        let s1 = log.append(make_record(ActionKind::CapabilityGrant, 1, 200, 2000));
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);

        let r0 = log.get(0).unwrap();
        let r1 = log.get(1).unwrap();
        assert_eq!(r0.version, 2);
        assert_eq!(r0.prev_mac, [0u8; 16]); // genesis
        assert_ne!(r0.chain_mac, [0u8; 16]);
        assert_eq!(r1.prev_mac, r0.chain_mac); // full-width link
        assert_eq!(log.chain_head(), r1.chain_mac);
    }

    #[test]
    fn chain_mac_is_keyed() {
        let mut r = WitnessRecordV2::zeroed();
        r.sequence = 7;
        let bytes = r.to_bytes();
        let content = &bytes[..WitnessRecordV2::CONTENT_LEN];
        let m1 = compute_chain_mac_v2(&derive_chain_key(b"key-a"), content, &[0; 16]);
        let m2 = compute_chain_mac_v2(&derive_chain_key(b"key-b"), content, &[0; 16]);
        assert_ne!(m1, m2, "different keys must produce different MACs");
    }

    #[test]
    fn chain_mac_binds_prev() {
        let r = WitnessRecordV2::zeroed();
        let bytes = r.to_bytes();
        let content = &bytes[..WitnessRecordV2::CONTENT_LEN];
        let key = default_chain_key();
        let m1 = compute_chain_mac_v2(&key, content, &[0x11; 16]);
        let m2 = compute_chain_mac_v2(&key, content, &[0x22; 16]);
        assert_ne!(m1, m2);
    }

    #[test]
    fn ring_wrap_and_counters() {
        let log = WitnessLogV2::<4>::new();
        for i in 0..10u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i * 100));
        }
        assert_eq!(log.total_emitted(), 10);
        assert_eq!(log.len(), 4);
        assert_eq!(log.total_overwritten(), 6);
    }

    #[test]
    fn genesis_anchoring() {
        let key = default_chain_key();
        let genesis = [0xA5u8; 16];
        let log = WitnessLogV2::<8>::with_key_and_genesis(key, genesis);
        assert_eq!(log.genesis(), genesis);
        assert_eq!(log.chain_head(), genesis); // empty log: head == genesis
        log.append(make_record(ActionKind::BootAttestation, 0, 0, 1));
        assert_eq!(log.get(0).unwrap().prev_mac, genesis);
    }

    #[test]
    fn snapshot_returns_most_recent() {
        let log = WitnessLogV2::<16>::new();
        for i in 0..5u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i * 100));
        }
        let mut buf = [WitnessRecordV2::zeroed(); 3];
        let copied = log.snapshot(&mut buf);
        assert_eq!(copied, 3);
        assert_eq!(buf[0].sequence, 2);
        assert_eq!(buf[2].sequence, 4);
    }

    #[test]
    fn append_v1_content_carries_fields() {
        let log = WitnessLogV2::<8>::new();
        let mut v1 = WitnessRecord::zeroed();
        v1.action_kind = ActionKind::RegionMap as u8;
        v1.proof_tier = 2;
        v1.actor_partition_id = 9;
        v1.target_object_id = 77;
        v1.capability_hash = 0xBEEF;
        v1.payload = [3; 8];
        v1.timestamp_ns = 123;
        // v1 chain fields must be ignored:
        v1.sequence = 999;
        v1.prev_hash = 0xDEAD;
        v1.record_hash = 0xFEED;

        log.append_v1_content(&v1);
        let r = log.get(0).unwrap();
        assert_eq!(r.sequence, 0); // log-assigned, not 999
        assert_eq!(r.action_kind, ActionKind::RegionMap as u8);
        assert_eq!(r.actor_partition_id, 9);
        assert_eq!(r.target_object_id, 77);
        assert_eq!(r.capability_hash, 0xBEEF);
        assert_eq!(r.payload, [3; 8]);
    }

    #[test]
    fn segment_tracking() {
        let log = WitnessLogV2::<64, 4>::new();
        assert_eq!(log.segment_len(), 0);
        for i in 0..4u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i));
        }
        assert!(log.segment_is_full());
        assert_eq!(log.segment_dropped(), 0);
        // Appending past a full segment drops leaves (counted).
        log.append(make_record(ActionKind::SchedulerEpoch, 1, 4, 4));
        assert_eq!(log.segment_dropped(), 1);
    }

    // ---- R4: forward-secure chain-key ratchet ------------------------

    fn test_seal_signer() -> crate::seal::Blake3SealSigner {
        crate::seal::Blake3SealSigner::new([0x42u8; 32])
    }

    #[test]
    fn ratchet_is_deterministic_and_changes_key() {
        let k0 = derive_chain_key(b"epoch-test");
        assert_eq!(ratchet_chain_key(&k0), ratchet_chain_key(&k0));
        assert_ne!(ratchet_chain_key(&k0), k0);
        assert_ne!(ratchet_chain_key(&ratchet_chain_key(&k0)), ratchet_chain_key(&k0));
    }

    #[test]
    fn seal_ratchets_key_atomically() {
        let signer = test_seal_signer();
        let log = WitnessLogV2::<64, 8>::new();
        let key0 = log.chain_key();
        assert_eq!(log.key_epoch(), 0);

        log.append(make_record(ActionKind::SchedulerEpoch, 1, 0, 0));
        log.seal_segment(&signer).unwrap();
        assert_eq!(log.key_epoch(), 1);
        // Verifier re-derivation matches the live key exactly.
        assert_eq!(log.chain_key(), ratchet_chain_key(&key0));
        assert_ne!(log.chain_key(), key0);

        log.append(make_record(ActionKind::SchedulerEpoch, 1, 1, 1));
        log.seal_segment(&signer).unwrap();
        assert_eq!(log.key_epoch(), 2);
        assert_eq!(log.chain_key(), ratchet_chain_key(&ratchet_chain_key(&key0)));

        // Sealing an empty segment performs no ratchet (no seal, no
        // key event — atomicity in both directions).
        assert!(log.seal_segment(&signer).is_none());
        assert_eq!(log.key_epoch(), 2);
    }

    #[test]
    fn records_after_ratchet_use_new_key() {
        let signer = test_seal_signer();
        let log = WitnessLogV2::<64, 8>::new();
        let key0 = log.chain_key();
        log.append(make_record(ActionKind::SchedulerEpoch, 1, 0, 0));
        log.seal_segment(&signer).unwrap();
        log.append(make_record(ActionKind::SchedulerEpoch, 1, 1, 1));

        let r0 = log.get(0).unwrap();
        let r1 = log.get(1).unwrap();
        let key1 = ratchet_chain_key(&key0);
        let b0 = r0.to_bytes();
        let b1 = r1.to_bytes();
        assert_eq!(
            r0.chain_mac,
            compute_chain_mac_v2(&key0, &b0[..WitnessRecordV2::CONTENT_LEN], &r0.prev_mac)
        );
        assert_eq!(
            r1.chain_mac,
            compute_chain_mac_v2(&key1, &b1[..WitnessRecordV2::CONTENT_LEN], &r1.prev_mac)
        );
        // The chain itself continues across the epoch boundary.
        assert_eq!(r1.prev_mac, r0.chain_mac);
        // The pre-ratchet key cannot produce the epoch-1 MAC.
        assert_ne!(
            r1.chain_mac,
            compute_chain_mac_v2(&key0, &b1[..WitnessRecordV2::CONTENT_LEN], &r1.prev_mac)
        );
    }

    #[test]
    fn erase_key_zeroes_buffer() {
        let mut key = [0xA7u8; 32];
        erase_key(&mut key);
        assert_eq!(key, [0u8; 32]);
    }

    // ---- R1: seals from the log form a verifiable chain --------------

    #[test]
    fn log_seals_form_verifiable_chain() {
        let signer = test_seal_signer();
        let log = WitnessLogV2::<64, 4>::with_policy(
            derive_chain_key(b"chain-test"),
            CoveragePolicy::Strict,
        );
        let mut seq = 0u64;
        let mut seal = |(): ()| {
            for _ in 0..4 {
                log.try_append(make_record(ActionKind::SchedulerEpoch, 1, seq, seq))
                    .unwrap();
                seq += 1;
            }
            log.seal_segment(&signer).unwrap().0
        };
        let seals = [seal(()), seal(()), seal(())];
        assert_eq!(crate::seal::verify_seal_chain(&seals, &signer), Ok(3));
        assert_eq!(log.last_seal_digest(), seals[2].digest());
        // Under Strict, epoch boundaries are recoverable from seals:
        // first_sequence + count of each seal.
        assert_eq!(seals[0].first_sequence + u64::from(seals[0].count), 4);
        assert_eq!(seals[1].first_sequence, 4);
        assert_eq!(seals[2].first_sequence, 8);
    }

    // ---- R6: coverage policy ------------------------------------------

    #[test]
    fn strict_segment_full_is_backpressure_not_drop() {
        let signer = test_seal_signer();
        let log = WitnessLogV2::<64, 4>::with_policy(
            default_chain_key(),
            CoveragePolicy::Strict,
        );
        assert_eq!(log.coverage_policy(), CoveragePolicy::Strict);
        for i in 0..4u64 {
            assert!(log
                .try_append(make_record(ActionKind::SchedulerEpoch, 1, i, i))
                .is_ok());
        }
        let rec = make_record(ActionKind::SchedulerEpoch, 1, 4, 4);
        assert_eq!(log.try_append(rec), Err(CoverageError::SegmentFull));
        // The refused append mutated nothing.
        assert_eq!(log.total_emitted(), 4);
        assert_eq!(log.segment_dropped(), 0);
        // Seal-then-append succeeds.
        assert!(log.seal_segment(&signer).is_some());
        assert_eq!(log.try_append(rec), Ok(4));
        assert_eq!(log.segment_dropped(), 0);
    }

    #[test]
    fn strict_refuses_unsealed_ring_overwrite() {
        let signer = test_seal_signer();
        let log = WitnessLogV2::<4, 8>::with_policy(
            default_chain_key(),
            CoveragePolicy::Strict,
        );
        for i in 0..4u64 {
            log.try_append(make_record(ActionKind::SchedulerEpoch, 1, i, i))
                .unwrap();
        }
        // Ring full, nothing sealed: record 0 would be lost.
        assert_eq!(
            log.try_append(make_record(ActionKind::SchedulerEpoch, 1, 4, 4)),
            Err(CoverageError::UnsealedOverwrite { sequence: 0 })
        );
        assert_eq!(log.total_overwritten(), 0);
        assert_eq!(log.total_emitted(), 4);
        // Sealing covers records 0..4; overwriting them is now allowed.
        assert!(log.seal_segment(&signer).is_some());
        assert_eq!(
            log.try_append(make_record(ActionKind::SchedulerEpoch, 1, 4, 4)),
            Ok(4)
        );
        assert_eq!(log.total_overwritten(), 1);
    }

    #[test]
    fn strict_stress_keeps_coverage_invariant() {
        let signer = test_seal_signer();
        let log = WitnessLogV2::<8, 4>::with_policy(
            default_chain_key(),
            CoveragePolicy::Strict,
        );
        for i in 0..100u64 {
            let rec = make_record(ActionKind::SchedulerEpoch, 1, i, i);
            if log.try_append(rec).is_err() {
                log.seal_segment(&signer).unwrap();
                log.try_append(rec).unwrap();
            }
        }
        assert_eq!(log.total_emitted(), 100);
        assert_eq!(log.segment_dropped(), 0);
        // Every record is either sealed or in the live segment.
        assert_eq!(log.sealed_up_to() + log.segment_len() as u64, 100);
    }

    #[test]
    fn best_effort_try_append_never_fails() {
        // Existing-constructor logs keep the original semantics: no
        // backpressure, losses counted exactly as before.
        let log = WitnessLogV2::<4, 2>::new();
        assert_eq!(log.coverage_policy(), CoveragePolicy::BestEffort);
        for i in 0..10u64 {
            assert!(log
                .try_append(make_record(ActionKind::SchedulerEpoch, 1, i, i))
                .is_ok());
        }
        assert_eq!(log.total_emitted(), 10);
        assert_eq!(log.segment_dropped(), 8);
        assert_eq!(log.total_overwritten(), 6);
    }
}
