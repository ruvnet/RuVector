//! Merkle segment sealing for the v2 witness log (ADR-134 v2).
//!
//! Per-record appends only *accumulate* the 16-byte chain MAC as a
//! Merkle leaf (a memcpy). All expensive crypto -- leaf hashing, tree
//! construction, and the seal signature -- is paid once per segment in
//! [`SegmentAccumulator::seal`], CT/QMDB-style. A sealed root can be
//! anchored externally, and any record in the segment can be proven
//! included via a logarithmic [`MerkleProof`].
//!
//! Domain separation: leaves are hashed as `BLAKE3(0x00 || seq || mac)`
//! and internal nodes as `BLAKE3(0x01 || left || right)`, preventing
//! leaf/node confusion attacks. Odd nodes are promoted unchanged.
//!
//! Chained seals (R1): each seal produced by
//! [`crate::WitnessLogV2::seal_segment`] binds the digest of its
//! predecessor, so the append-only ordering of the entire sealed
//! history is verifiable **from seals alone** — no chain key required —
//! via [`verify_seal_chain`] / [`verify_seal_chain_binding`].

/// Default number of leaves per segment.
///
/// 256 leaves = 4 KiB of buffered MACs and an 8 KiB scratch level at
/// seal time (kernel-stack friendly), with one seal signature amortized
/// over 256 appends. Larger deployments can instantiate
/// `WitnessLogV2<N, 1024>` for cheaper amortization.
pub const DEFAULT_SEGMENT_SIZE: usize = 256;

/// Maximum supported Merkle depth (2^32 leaves; far above any segment).
pub const MAX_MERKLE_DEPTH: usize = 32;

const LEAF_DOMAIN: u8 = 0x00;
const NODE_DOMAIN: u8 = 0x01;
const SEAL_DOMAIN: u8 = 0x02;

/// Seal format with the original digest preimage
/// `BLAKE3(0x02 || root || first_seq || count)` (no cross-segment
/// binding). Produced by [`SegmentAccumulator::seal`].
pub const SEAL_VERSION_UNCHAINED: u8 = 2;

/// Seal format whose digest binds the previous segment's seal digest:
/// `BLAKE3(0x02 || root || first_seq || count || prev_seal_digest)`.
/// Produced by [`SegmentAccumulator::seal_chained`] and
/// [`crate::WitnessLogV2::seal_segment`]. Makes append-only ordering of
/// the whole sealed history verifiable from seals alone (R1).
pub const SEAL_VERSION_CHAINED: u8 = 3;

/// Domain-derived constant used as the `prev_seal_digest` of the first
/// (genesis) seal in a chained seal sequence.
///
/// Derived rather than all-zero so a genesis link can never collide
/// with a forged "previous seal" whose digest happens to be zero.
#[must_use]
pub fn seal_chain_genesis() -> [u8; 32] {
    *blake3::hash(b"rvm-witness 2026 v3 seal-chain genesis").as_bytes()
}

/// Signs and verifies sealed segment roots.
///
/// Implemented by [`Blake3SealSigner`] (symmetric, in-crate) and, via
/// the adapter in `rvm-proof`, by every proof-crate `WitnessSigner`
/// (HMAC-SHA256, dual-HMAC, Ed25519, TEE-backed).
pub trait SegmentSealSigner {
    /// Produce a 64-byte signature over a 32-byte seal digest.
    fn sign_root(&self, digest: &[u8; 32]) -> [u8; 64];

    /// Verify a 64-byte signature over a 32-byte seal digest.
    fn verify_root(&self, digest: &[u8; 32], signature: &[u8; 64]) -> bool;
}

/// Keyed-BLAKE3 segment seal signer (symmetric MAC).
///
/// Signature layout: `sig[0..32] = BLAKE3_keyed(key, digest)`,
/// `sig[32..64] = 0`. Not publicly verifiable; single trust domain only.
#[derive(Clone)]
pub struct Blake3SealSigner {
    key: [u8; 32],
}

impl Blake3SealSigner {
    /// Create a seal signer from a 32-byte key.
    #[must_use]
    pub const fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    fn mac(&self, digest: &[u8; 32]) -> [u8; 32] {
        *blake3::keyed_hash(&self.key, digest).as_bytes()
    }
}

impl SegmentSealSigner for Blake3SealSigner {
    fn sign_root(&self, digest: &[u8; 32]) -> [u8; 64] {
        let mac = self.mac(digest);
        let mut sig = [0u8; 64];
        sig[..32].copy_from_slice(&mac);
        sig
    }

    fn verify_root(&self, digest: &[u8; 32], signature: &[u8; 64]) -> bool {
        let expected = self.sign_root(digest);
        // Branchless comparison (constant time w.r.t. content).
        let mut diff = 0u8;
        for i in 0..64 {
            diff |= expected[i] ^ signature[i];
        }
        diff == 0
    }
}

/// Hash a leaf: `BLAKE3(0x00 || sequence_le || chain_mac)`.
fn leaf_hash(sequence: u64, mac: &[u8; 16]) -> [u8; 32] {
    let mut buf = [0u8; 25];
    buf[0] = LEAF_DOMAIN;
    buf[1..9].copy_from_slice(&sequence.to_le_bytes());
    buf[9..25].copy_from_slice(mac);
    *blake3::hash(&buf).as_bytes()
}

/// Hash an internal node: `BLAKE3(0x01 || left || right)`.
fn node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 65];
    buf[0] = NODE_DOMAIN;
    buf[1..33].copy_from_slice(left);
    buf[33..65].copy_from_slice(right);
    *blake3::hash(&buf).as_bytes()
}

/// Domain-separated digest that the seal signature covers:
/// `BLAKE3(0x02 || root || first_sequence_le || count_le)`.
///
/// Binding the sequence range prevents replaying a valid (root,
/// signature) pair for a different position in the log.
#[must_use]
pub fn seal_digest(root: &[u8; 32], first_sequence: u64, count: u32) -> [u8; 32] {
    let mut buf = [0u8; 45];
    buf[0] = SEAL_DOMAIN;
    buf[1..33].copy_from_slice(root);
    buf[33..41].copy_from_slice(&first_sequence.to_le_bytes());
    buf[41..45].copy_from_slice(&count.to_le_bytes());
    *blake3::hash(&buf).as_bytes()
}

/// Chained ([`SEAL_VERSION_CHAINED`]) seal digest:
/// `BLAKE3(0x02 || root || first_sequence_le || count_le || prev_seal_digest)`.
///
/// Binding the previous segment's seal digest makes the append-only
/// ordering of an entire sealed history publicly verifiable from the
/// seals alone (no chain key needed): splicing, reordering, omitting,
/// or transplanting any seal breaks the binding of its successor. The
/// 77-byte preimage cannot collide with the 45-byte unchained preimage
/// even though both use the `0x02` domain byte.
#[must_use]
pub fn seal_digest_chained(
    root: &[u8; 32],
    first_sequence: u64,
    count: u32,
    prev_seal_digest: &[u8; 32],
) -> [u8; 32] {
    let mut buf = [0u8; 77];
    buf[0] = SEAL_DOMAIN;
    buf[1..33].copy_from_slice(root);
    buf[33..41].copy_from_slice(&first_sequence.to_le_bytes());
    buf[41..45].copy_from_slice(&count.to_le_bytes());
    buf[45..77].copy_from_slice(prev_seal_digest);
    *blake3::hash(&buf).as_bytes()
}

/// A sealed Merkle segment: exportable, externally anchorable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealedSegment {
    /// Seal format version: [`SEAL_VERSION_UNCHAINED`] (`2`, legacy
    /// digest preimage) or [`SEAL_VERSION_CHAINED`] (`3`, digest binds
    /// `prev_seal_digest`). No serialized seal format predates this
    /// field; it exists so any future persistence layer can verify both
    /// shapes under the correct rules.
    pub version: u8,
    /// Merkle root over the segment's record chain MACs.
    pub root: [u8; 32],
    /// Sequence number of the first record in the segment.
    pub first_sequence: u64,
    /// Number of records (leaves) in the segment.
    pub count: u32,
    /// Digest of the previous seal in the chain (chained seals), the
    /// [`seal_chain_genesis`] constant for the first seal of a log, or
    /// all-zero for unchained seals (ignored by their digest).
    pub prev_seal_digest: [u8; 32],
    /// Signature over [`SealedSegment::digest`].
    pub signature: [u8; 64],
}

impl SealedSegment {
    /// The digest this seal's signature covers, computed under the
    /// rules selected by [`Self::version`].
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        if self.version == SEAL_VERSION_CHAINED {
            seal_digest_chained(
                &self.root,
                self.first_sequence,
                self.count,
                &self.prev_seal_digest,
            )
        } else {
            seal_digest(&self.root, self.first_sequence, self.count)
        }
    }
}

/// Verify a sealed segment's signature (version-dispatched: unchained
/// seals verify under the legacy preimage, chained seals under the
/// prev-binding preimage). Unknown versions fail.
#[must_use]
pub fn verify_seal<G: SegmentSealSigner>(segment: &SealedSegment, signer: &G) -> bool {
    if segment.version != SEAL_VERSION_UNCHAINED && segment.version != SEAL_VERSION_CHAINED {
        return false;
    }
    signer.verify_root(&segment.digest(), &segment.signature)
}

/// Errors from chained-seal sequence verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SealChainError {
    /// The seal slice is empty.
    Empty,
    /// The seal at `index` is not a chained ([`SEAL_VERSION_CHAINED`])
    /// seal; unchained seals carry no ordering evidence.
    UnsupportedVersion {
        /// Position of the offending seal in the slice.
        index: usize,
        /// The version found.
        version: u8,
    },
    /// The seal at `index` does not bind the digest of its predecessor
    /// (or the expected start value for index 0): splice, reorder,
    /// omission, or cross-log transplant.
    BrokenBinding {
        /// Position of the offending seal in the slice.
        index: usize,
    },
    /// The seal at `index` covers sequence numbers that overlap or
    /// precede its predecessor's range.
    SequenceRegression {
        /// Position of the offending seal in the slice.
        index: usize,
    },
    /// The seal at `index` failed signature verification.
    BadSignature {
        /// Position of the offending seal in the slice.
        index: usize,
    },
}

impl core::fmt::Display for SealChainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(f, "empty seal chain"),
            Self::UnsupportedVersion { index, version } => {
                write!(f, "seal {index}: unsupported version {version}")
            }
            Self::BrokenBinding { index } => write!(f, "seal {index}: broken prev binding"),
            Self::SequenceRegression { index } => write!(f, "seal {index}: sequence regression"),
            Self::BadSignature { index } => write!(f, "seal {index}: bad signature"),
        }
    }
}

/// Keyless structural verification of a chained seal sequence starting
/// from the [`seal_chain_genesis`] constant (i.e. the full history of
/// one log). See [`verify_seal_chain_binding_from`].
///
/// # Errors
///
/// See [`verify_seal_chain_binding_from`].
pub fn verify_seal_chain_binding(seals: &[SealedSegment]) -> Result<usize, SealChainError> {
    verify_seal_chain_binding_from(seals, &seal_chain_genesis())
}

/// Keyless structural verification of a chained seal sequence: checks
/// that every seal binds the recomputed digest of its predecessor
/// (`seals[0]` must bind `expected_prev`) and that sequence ranges
/// never regress. Detects splice (replacement), reorder, omission, and
/// cross-log transplant between any two seals — **without any key**,
/// because seal digests are computed from public fields only.
///
/// Signatures are *not* checked here; pair with [`verify_seal_chain`]
/// (or per-seal [`verify_seal`] under a public-key
/// [`SegmentSealSigner`]) to also authenticate each seal.
///
/// # Errors
///
/// [`SealChainError::Empty`] for an empty slice;
/// [`SealChainError::UnsupportedVersion`] for a non-chained seal;
/// [`SealChainError::BrokenBinding`] on a prev-digest mismatch;
/// [`SealChainError::SequenceRegression`] on overlapping ranges.
pub fn verify_seal_chain_binding_from(
    seals: &[SealedSegment],
    expected_prev: &[u8; 32],
) -> Result<usize, SealChainError> {
    if seals.is_empty() {
        return Err(SealChainError::Empty);
    }
    let mut prev_digest = *expected_prev;
    let mut next_min_sequence = 0u64;
    for (index, seal) in seals.iter().enumerate() {
        if seal.version != SEAL_VERSION_CHAINED {
            return Err(SealChainError::UnsupportedVersion {
                index,
                version: seal.version,
            });
        }
        if seal.prev_seal_digest != prev_digest {
            return Err(SealChainError::BrokenBinding { index });
        }
        if seal.first_sequence < next_min_sequence {
            return Err(SealChainError::SequenceRegression { index });
        }
        next_min_sequence = seal.first_sequence + u64::from(seal.count);
        prev_digest = seal.digest();
    }
    Ok(seals.len())
}

/// Verify a chained seal sequence starting from [`seal_chain_genesis`]:
/// structural binding ([`verify_seal_chain_binding`]) **and** each
/// seal's signature.
///
/// # Errors
///
/// See [`verify_seal_chain_from`].
pub fn verify_seal_chain<G: SegmentSealSigner>(
    seals: &[SealedSegment],
    signer: &G,
) -> Result<usize, SealChainError> {
    verify_seal_chain_from(seals, signer, &seal_chain_genesis())
}

/// Verify a chained seal sequence from an arbitrary start digest:
/// checks every seal's signature and the prev-binding across the
/// slice, detecting splice, replacement, reorder, and omission between
/// any two seals. `expected_prev` is [`seal_chain_genesis`] for a full
/// history, or the digest of the last already-verified seal when
/// verifying an incremental suffix.
///
/// # Errors
///
/// All of [`verify_seal_chain_binding_from`]'s errors, plus
/// [`SealChainError::BadSignature`] for a seal whose signature fails.
pub fn verify_seal_chain_from<G: SegmentSealSigner>(
    seals: &[SealedSegment],
    signer: &G,
    expected_prev: &[u8; 32],
) -> Result<usize, SealChainError> {
    verify_seal_chain_binding_from(seals, expected_prev)?;
    for (index, seal) in seals.iter().enumerate() {
        if !signer.verify_root(&seal.digest(), &seal.signature) {
            return Err(SealChainError::BadSignature { index });
        }
    }
    Ok(seals.len())
}

/// Merkle inclusion proof for a single record in a sealed segment.
///
/// `siblings[0..depth]` are the authentication path bottom-up;
/// `present` marks levels where the node had a sibling (promotion
/// levels carry the hash up unchanged).
#[derive(Debug, Clone, Copy)]
pub struct MerkleProof {
    /// Sibling hashes, bottom-up. Only `[0..depth]` are meaningful.
    pub siblings: [[u8; 32]; MAX_MERKLE_DEPTH],
    /// Whether a sibling exists at each level (false = promotion).
    pub present: [bool; MAX_MERKLE_DEPTH],
    /// Number of tree levels above the leaves.
    pub depth: u8,
    /// Leaf index within the segment (0-based).
    pub index: u32,
    /// Sequence number of the proven record.
    pub sequence: u64,
}

/// Verify a Merkle inclusion proof against a sealed root.
///
/// `chain_mac` is the 16-byte chain MAC of the record claimed included.
#[must_use]
pub fn verify_inclusion(root: &[u8; 32], chain_mac: &[u8; 16], proof: &MerkleProof) -> bool {
    if usize::from(proof.depth) > MAX_MERKLE_DEPTH {
        return false;
    }
    let mut hash = leaf_hash(proof.sequence, chain_mac);
    let mut index = proof.index;
    for level in 0..usize::from(proof.depth) {
        if proof.present[level] {
            hash = if index & 1 == 0 {
                node_hash(&hash, &proof.siblings[level])
            } else {
                node_hash(&proof.siblings[level], &hash)
            };
        }
        // Promotion: hash carries up unchanged.
        index >>= 1;
    }
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= hash[i] ^ root[i];
    }
    diff == 0
}

/// Fixed-capacity Merkle leaf accumulator for one segment.
///
/// Appends are a 16-byte memcpy; the tree is only computed at seal /
/// proof time. `Copy` so [`crate::WitnessLogV2::seal_segment`] can hand
/// the caller a snapshot for proof generation.
#[derive(Clone, Copy)]
pub struct SegmentAccumulator<const S: usize> {
    leaves: [[u8; 16]; S],
    first_sequence: u64,
    len: usize,
}

impl<const S: usize> SegmentAccumulator<S> {
    /// Create an empty accumulator whose first leaf will correspond to
    /// the record with sequence number `first_sequence`.
    #[must_use]
    pub fn new(first_sequence: u64) -> Self {
        Self {
            leaves: [[0u8; 16]; S],
            first_sequence,
            len: 0,
        }
    }

    /// Append a record's chain MAC as the next leaf.
    ///
    /// Returns `false` (leaf dropped) if the segment is already full.
    pub fn push(&mut self, chain_mac: [u8; 16]) -> bool {
        if self.len >= S {
            return false;
        }
        self.leaves[self.len] = chain_mac;
        self.len += 1;
        true
    }

    /// Number of accumulated leaves.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if no leaves have been accumulated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// True when the accumulator holds `S` leaves.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len >= S
    }

    /// Sequence number of the first record in this segment.
    #[must_use]
    pub fn first_sequence(&self) -> u64 {
        self.first_sequence
    }

    /// Compute the Merkle root over the accumulated leaves.
    ///
    /// Returns `None` if the segment is empty.
    #[must_use]
    pub fn compute_root(&self) -> Option<[u8; 32]> {
        if self.len == 0 {
            return None;
        }
        let mut level = [[0u8; 32]; S];
        for (i, (slot, leaf)) in level
            .iter_mut()
            .zip(self.leaves.iter())
            .enumerate()
            .take(self.len)
        {
            *slot = leaf_hash(self.first_sequence + i as u64, leaf);
        }
        let mut width = self.len;
        while width > 1 {
            let mut next = 0;
            let mut i = 0;
            while i < width {
                if i + 1 < width {
                    level[next] = node_hash(&level[i], &level[i + 1]);
                } else {
                    level[next] = level[i]; // promotion
                }
                next += 1;
                i += 2;
            }
            width = next;
        }
        Some(level[0])
    }

    /// Build an inclusion proof for the leaf at `offset` (0-based
    /// within the segment). Returns `None` if out of range.
    #[must_use]
    pub fn inclusion_proof(&self, offset: usize) -> Option<MerkleProof> {
        if offset >= self.len {
            return None;
        }
        let mut proof = MerkleProof {
            siblings: [[0u8; 32]; MAX_MERKLE_DEPTH],
            present: [false; MAX_MERKLE_DEPTH],
            depth: 0,
            #[allow(clippy::cast_possible_truncation)]
            index: offset as u32,
            sequence: self.first_sequence + offset as u64,
        };
        let mut level = [[0u8; 32]; S];
        for (i, (slot, leaf)) in level
            .iter_mut()
            .zip(self.leaves.iter())
            .enumerate()
            .take(self.len)
        {
            *slot = leaf_hash(self.first_sequence + i as u64, leaf);
        }
        let mut width = self.len;
        let mut idx = offset;
        let mut depth = 0usize;
        while width > 1 {
            let sibling = idx ^ 1;
            if sibling < width {
                proof.siblings[depth] = level[sibling];
                proof.present[depth] = true;
            }
            let mut next = 0;
            let mut i = 0;
            while i < width {
                if i + 1 < width {
                    level[next] = node_hash(&level[i], &level[i + 1]);
                } else {
                    level[next] = level[i];
                }
                next += 1;
                i += 2;
            }
            width = next;
            idx >>= 1;
            depth += 1;
            if depth > MAX_MERKLE_DEPTH {
                return None;
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        {
            proof.depth = depth as u8;
        }
        Some(proof)
    }

    /// Build an inclusion proof addressed by record sequence number.
    #[must_use]
    pub fn proof_for_sequence(&self, sequence: u64) -> Option<MerkleProof> {
        let offset = sequence.checked_sub(self.first_sequence)?;
        if offset >= self.len as u64 {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)]
        self.inclusion_proof(offset as usize)
    }

    /// The raw chain MAC stored for the leaf at `offset`.
    #[must_use]
    pub fn leaf(&self, offset: usize) -> Option<[u8; 16]> {
        if offset >= self.len {
            return None;
        }
        Some(self.leaves[offset])
    }

    /// Seal this segment without cross-segment binding: compute the
    /// root and sign [`seal_digest`]`(root, first_sequence, len)`,
    /// producing a [`SEAL_VERSION_UNCHAINED`] seal.
    ///
    /// Standalone use only; prefer [`Self::seal_chained`] (or
    /// [`crate::WitnessLogV2::seal_segment`], which chains
    /// automatically) so the seal participates in publicly verifiable
    /// history ordering.
    ///
    /// Returns `None` if the segment is empty.
    #[must_use]
    pub fn seal<G: SegmentSealSigner>(&self, signer: &G) -> Option<SealedSegment> {
        let root = self.compute_root()?;
        #[allow(clippy::cast_possible_truncation)]
        let count = self.len as u32;
        let digest = seal_digest(&root, self.first_sequence, count);
        Some(SealedSegment {
            version: SEAL_VERSION_UNCHAINED,
            root,
            first_sequence: self.first_sequence,
            count,
            prev_seal_digest: [0u8; 32],
            signature: signer.sign_root(&digest),
        })
    }

    /// Seal this segment bound to its predecessor: compute the root and
    /// sign [`seal_digest_chained`]`(root, first_sequence, len,
    /// prev_seal_digest)`, producing a [`SEAL_VERSION_CHAINED`] seal.
    ///
    /// `prev_seal_digest` is the [`SealedSegment::digest`] of the
    /// previous seal, or [`seal_chain_genesis`] for the first segment
    /// of a log. Verify whole sequences with [`verify_seal_chain`].
    ///
    /// Returns `None` if the segment is empty.
    #[must_use]
    pub fn seal_chained<G: SegmentSealSigner>(
        &self,
        signer: &G,
        prev_seal_digest: &[u8; 32],
    ) -> Option<SealedSegment> {
        let root = self.compute_root()?;
        #[allow(clippy::cast_possible_truncation)]
        let count = self.len as u32;
        let digest = seal_digest_chained(&root, self.first_sequence, count, prev_seal_digest);
        Some(SealedSegment {
            version: SEAL_VERSION_CHAINED,
            root,
            first_sequence: self.first_sequence,
            count,
            prev_seal_digest: *prev_seal_digest,
            signature: signer.sign_root(&digest),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled_acc<const S: usize>(n: usize, first_seq: u64) -> SegmentAccumulator<S> {
        let mut acc = SegmentAccumulator::<S>::new(first_seq);
        for i in 0..n {
            let mut mac = [0u8; 16];
            mac[0] = i as u8;
            mac[1] = 0xC3;
            assert!(acc.push(mac));
        }
        acc
    }

    fn test_signer() -> Blake3SealSigner {
        Blake3SealSigner::new([0x42u8; 32])
    }

    #[test]
    fn root_deterministic_and_content_sensitive() {
        let a = filled_acc::<8>(5, 0);
        let b = filled_acc::<8>(5, 0);
        assert_eq!(a.compute_root(), b.compute_root());

        let mut c = filled_acc::<8>(4, 0);
        let mut mac = [0u8; 16];
        mac[0] = 0xFF;
        c.push(mac);
        assert_ne!(a.compute_root(), c.compute_root());
    }

    #[test]
    fn root_binds_sequence_numbers() {
        let a = filled_acc::<8>(5, 0);
        let b = filled_acc::<8>(5, 100);
        assert_ne!(a.compute_root(), b.compute_root());
    }

    #[test]
    fn empty_segment_has_no_root_or_seal() {
        let acc = SegmentAccumulator::<8>::new(0);
        assert!(acc.compute_root().is_none());
        assert!(acc.seal(&test_signer()).is_none());
    }

    #[test]
    fn seal_and_verify_round_trip() {
        let acc = filled_acc::<8>(7, 10);
        let signer = test_signer();
        let sealed = acc.seal(&signer).unwrap();
        assert_eq!(sealed.first_sequence, 10);
        assert_eq!(sealed.count, 7);
        assert!(verify_seal(&sealed, &signer));
    }

    #[test]
    fn seal_tamper_detected() {
        let acc = filled_acc::<8>(7, 10);
        let signer = test_signer();
        let sealed = acc.seal(&signer).unwrap();

        let mut bad = sealed;
        bad.root[0] ^= 1;
        assert!(!verify_seal(&bad, &signer));

        let mut bad = sealed;
        bad.first_sequence += 1; // range replay
        assert!(!verify_seal(&bad, &signer));

        let mut bad = sealed;
        bad.count -= 1; // truncation claim
        assert!(!verify_seal(&bad, &signer));

        let mut bad = sealed;
        bad.signature[5] ^= 0x80;
        assert!(!verify_seal(&bad, &signer));

        // Wrong key fails.
        assert!(!verify_seal(&sealed, &Blake3SealSigner::new([0x43u8; 32])));
    }

    #[test]
    fn inclusion_proof_verifies_for_every_leaf() {
        // Cover power-of-two and odd (promotion) widths.
        for n in [1usize, 2, 3, 5, 7, 8] {
            let acc = filled_acc::<8>(n, 20);
            let root = acc.compute_root().unwrap();
            for i in 0..n {
                let proof = acc.inclusion_proof(i).unwrap();
                let mac = acc.leaf(i).unwrap();
                assert!(
                    verify_inclusion(&root, &mac, &proof),
                    "leaf {i} of {n} failed"
                );
            }
        }
    }

    #[test]
    fn inclusion_proof_tamper_detected() {
        let acc = filled_acc::<8>(6, 0);
        let root = acc.compute_root().unwrap();
        let proof = acc.inclusion_proof(2).unwrap();
        let mac = acc.leaf(2).unwrap();

        // Wrong MAC fails.
        let mut bad_mac = mac;
        bad_mac[3] ^= 0xFF;
        assert!(!verify_inclusion(&root, &bad_mac, &proof));

        // Wrong sequence fails (leaf hash binds sequence).
        let mut bad = proof;
        bad.sequence += 1;
        assert!(!verify_inclusion(&root, &mac, &bad));

        // Wrong index (position swap) fails.
        let mut bad = proof;
        bad.index ^= 1;
        assert!(!verify_inclusion(&root, &mac, &bad));

        // Corrupted sibling fails.
        let mut bad = proof;
        bad.siblings[0][0] ^= 1;
        assert!(!verify_inclusion(&root, &mac, &bad));

        // Wrong root fails.
        let mut bad_root = root;
        bad_root[0] ^= 1;
        assert!(!verify_inclusion(&bad_root, &mac, &proof));
    }

    #[test]
    fn proof_for_sequence_addressing() {
        let acc = filled_acc::<8>(5, 100);
        let root = acc.compute_root().unwrap();
        let proof = acc.proof_for_sequence(103).unwrap();
        assert_eq!(proof.index, 3);
        assert!(verify_inclusion(&root, &acc.leaf(3).unwrap(), &proof));
        assert!(acc.proof_for_sequence(99).is_none());
        assert!(acc.proof_for_sequence(105).is_none());
    }

    #[test]
    fn push_past_capacity_drops() {
        let mut acc = filled_acc::<4>(4, 0);
        assert!(acc.is_full());
        assert!(!acc.push([9u8; 16]));
        assert_eq!(acc.len(), 4);
    }

    // ---- R1: chained seals and seal-chain verification ----------------

    /// Accumulator with `salt`-dependent leaf content so two "logs"
    /// produce distinct roots for the same sequence ranges.
    fn salted_acc<const S: usize>(n: usize, first_seq: u64, salt: u8) -> SegmentAccumulator<S> {
        let mut acc = SegmentAccumulator::<S>::new(first_seq);
        for i in 0..n {
            let mut mac = [salt; 16];
            mac[0] = u8::try_from(i).unwrap();
            assert!(acc.push(mac));
        }
        acc
    }

    /// Three consecutive chained seals (segments 0..8, 8..16, 16..24).
    fn build_chain3(signer: &Blake3SealSigner, salt: u8) -> [SealedSegment; 3] {
        let s0 = salted_acc::<8>(8, 0, salt)
            .seal_chained(signer, &seal_chain_genesis())
            .unwrap();
        let s1 = salted_acc::<8>(8, 8, salt)
            .seal_chained(signer, &s0.digest())
            .unwrap();
        let s2 = salted_acc::<8>(8, 16, salt)
            .seal_chained(signer, &s1.digest())
            .unwrap();
        [s0, s1, s2]
    }

    #[test]
    fn chained_seal_round_trip_and_tamper() {
        let signer = test_signer();
        let sealed = filled_acc::<8>(5, 10)
            .seal_chained(&signer, &seal_chain_genesis())
            .unwrap();
        assert_eq!(sealed.version, SEAL_VERSION_CHAINED);
        assert!(verify_seal(&sealed, &signer));

        let mut bad = sealed;
        bad.root[0] ^= 1;
        assert!(!verify_seal(&bad, &signer));

        let mut bad = sealed;
        bad.prev_seal_digest[0] ^= 1; // prev binding is signed
        assert!(!verify_seal(&bad, &signer));

        let mut bad = sealed;
        bad.first_sequence += 1;
        assert!(!verify_seal(&bad, &signer));

        // Version relabeling cannot move a signature between preimages.
        let mut bad = sealed;
        bad.version = SEAL_VERSION_UNCHAINED;
        assert!(!verify_seal(&bad, &signer));
        let mut bad = filled_acc::<8>(5, 10).seal(&signer).unwrap();
        bad.version = SEAL_VERSION_CHAINED;
        assert!(!verify_seal(&bad, &signer));
        let mut bad = sealed;
        bad.version = 9; // unknown version
        assert!(!verify_seal(&bad, &signer));
    }

    #[test]
    fn unchained_seal_still_verifies_under_old_rules() {
        let signer = test_signer();
        let sealed = filled_acc::<8>(7, 10).seal(&signer).unwrap();
        assert_eq!(sealed.version, SEAL_VERSION_UNCHAINED);
        assert_eq!(sealed.prev_seal_digest, [0u8; 32]);
        assert_eq!(
            sealed.digest(),
            seal_digest(&sealed.root, sealed.first_sequence, sealed.count)
        );
        assert!(verify_seal(&sealed, &signer));
    }

    #[test]
    fn seal_chain_accepts_honest_sequence() {
        let signer = test_signer();
        let seals = build_chain3(&signer, 0xC3);
        assert_eq!(verify_seal_chain(&seals, &signer), Ok(3));
        // Structural binding alone needs no key at all.
        assert_eq!(verify_seal_chain_binding(&seals), Ok(3));
        // A single seal chains from genesis.
        assert_eq!(verify_seal_chain(&seals[..1], &signer), Ok(1));
    }

    #[test]
    fn seal_chain_detects_middle_replacement() {
        // Replace the middle seal with a *valid* seal over different
        // content, correctly bound to s0: the successor's binding
        // exposes the splice.
        let signer = test_signer();
        let mut seals = build_chain3(&signer, 0xC3);
        let forged = salted_acc::<8>(8, 8, 0xEE)
            .seal_chained(&signer, &seals[0].digest())
            .unwrap();
        assert!(verify_seal(&forged, &signer)); // individually valid
        seals[1] = forged;
        assert_eq!(
            verify_seal_chain(&seals, &signer),
            Err(SealChainError::BrokenBinding { index: 2 })
        );
    }

    #[test]
    fn seal_chain_detects_reorder() {
        let signer = test_signer();
        let mut seals = build_chain3(&signer, 0xC3);
        seals.swap(0, 1);
        assert_eq!(
            verify_seal_chain(&seals, &signer),
            Err(SealChainError::BrokenBinding { index: 0 })
        );
        let mut seals = build_chain3(&signer, 0xC3);
        seals.swap(1, 2);
        assert_eq!(
            verify_seal_chain(&seals, &signer),
            Err(SealChainError::BrokenBinding { index: 1 })
        );
    }

    #[test]
    fn seal_chain_detects_omission() {
        let signer = test_signer();
        let seals = build_chain3(&signer, 0xC3);
        let gapped = [seals[0], seals[2]];
        assert_eq!(
            verify_seal_chain(&gapped, &signer),
            Err(SealChainError::BrokenBinding { index: 1 })
        );
        // Dropping the genesis seal is equally visible.
        assert_eq!(
            verify_seal_chain(&seals[1..], &signer),
            Err(SealChainError::BrokenBinding { index: 0 })
        );
    }

    #[test]
    fn seal_chain_detects_cross_log_transplant() {
        let signer = test_signer();
        let mut seals = build_chain3(&signer, 0xC3);
        let other = build_chain3(&signer, 0x5A); // same ranges, other log
        seals[1] = other[1];
        assert_eq!(
            verify_seal_chain(&seals, &signer),
            Err(SealChainError::BrokenBinding { index: 1 })
        );
    }

    #[test]
    fn seal_chain_genesis_handling() {
        let signer = test_signer();
        let seals = build_chain3(&signer, 0xC3);

        // A first seal bound to something other than the genesis
        // constant is rejected when verifying a full history.
        let rogue = salted_acc::<8>(8, 0, 0xC3)
            .seal_chained(&signer, &[0u8; 32])
            .unwrap();
        let mut bad = seals;
        bad[0] = rogue;
        assert_eq!(
            verify_seal_chain(&bad, &signer),
            Err(SealChainError::BrokenBinding { index: 0 })
        );

        // Incremental suffix verification from a trusted prior digest.
        assert_eq!(
            verify_seal_chain_from(&seals[1..], &signer, &seals[0].digest()),
            Ok(2)
        );
    }

    #[test]
    fn seal_chain_rejects_unchained_versions() {
        let signer = test_signer();
        let mut seals = build_chain3(&signer, 0xC3);
        seals[1] = salted_acc::<8>(8, 8, 0xC3).seal(&signer).unwrap();
        assert_eq!(
            verify_seal_chain(&seals, &signer),
            Err(SealChainError::UnsupportedVersion {
                index: 1,
                version: SEAL_VERSION_UNCHAINED
            })
        );
    }

    #[test]
    fn seal_chain_detects_bad_signature() {
        let signer = test_signer();
        let mut seals = build_chain3(&signer, 0xC3);
        seals[1].signature[7] ^= 0x40;
        assert_eq!(
            verify_seal_chain(&seals, &signer),
            Err(SealChainError::BadSignature { index: 1 })
        );
        // Wrong signer key fails every seal.
        let seals = build_chain3(&signer, 0xC3);
        assert_eq!(
            verify_seal_chain(&seals, &Blake3SealSigner::new([0x43u8; 32])),
            Err(SealChainError::BadSignature { index: 0 })
        );
    }

    #[test]
    fn seal_chain_detects_sequence_regression() {
        let signer = test_signer();
        let s0 = salted_acc::<8>(8, 0, 0xC3)
            .seal_chained(&signer, &seal_chain_genesis())
            .unwrap();
        // Overlapping range (4..12 after 0..8), correctly bound.
        let s1 = salted_acc::<8>(8, 4, 0xC3)
            .seal_chained(&signer, &s0.digest())
            .unwrap();
        assert_eq!(
            verify_seal_chain(&[s0, s1], &signer),
            Err(SealChainError::SequenceRegression { index: 1 })
        );
    }

    #[test]
    fn seal_chain_empty_rejected() {
        let signer = test_signer();
        assert_eq!(
            verify_seal_chain(&[], &signer),
            Err(SealChainError::Empty)
        );
    }
}
