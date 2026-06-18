//! Append-only ring buffer witness log (ADR-134).
//!
//! Thread-safe via `spin::Mutex`. Designed for < 500 ns emission.

use crate::hash::{compute_chain_hash, compute_record_hash};
use rvm_types::WitnessRecord;
use spin::Mutex;

/// XOR-fold a 64-bit hash into 32 bits.
///
/// This preserves entropy from both halves of the hash, unlike simple
/// truncation (`as u32`) which discards the upper 32 bits entirely.
///
/// `fold(h) = (h >> 32) ^ (h & 0xFFFF_FFFF)`
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn fold_u64_to_u32(h: u64) -> u32 {
    ((h >> 32) ^ h) as u32
}

/// Append-only ring buffer of witness records.
pub struct WitnessLog<const N: usize> {
    inner: Mutex<WitnessLogInner<N>>,
}

struct WitnessLogInner<const N: usize> {
    records: [WitnessRecord; N],
    write_pos: usize,
    chain_hash: u64,
    sequence: u64,
    total_emitted: u64,
    /// Number of records silently overwritten by ring wrap-around.
    ///
    /// The ring overwrites the oldest record on overflow; this counter
    /// makes that loss observable so callers can schedule drains before
    /// audit records are dropped.
    total_overwritten: u64,
}

impl<const N: usize> WitnessLogInner<N> {
    /// Fill the chain fields of `record` from the current chain state.
    ///
    /// Sets `sequence`, `prev_hash`, and `record_hash`, and returns the
    /// new chain hash value (to be committed via [`Self::store`]). The
    /// `record_hash` is computed over the record's content bytes
    /// (`[0..WitnessRecord::CONTENT_LEN]`, i.e. everything before the
    /// chain fields), and the chain hash binds that content hash:
    /// `chain = H(prev_chain || sequence || record_hash)`. Rewriting a
    /// record's content therefore breaks both its self-integrity hash
    /// and every subsequent chain link.
    fn fill_chain_fields(&self, record: &mut WitnessRecord) -> u64 {
        let seq = self.sequence;
        let prev_hash = self.chain_hash;

        record.sequence = seq;
        record.prev_hash = fold_u64_to_u32(prev_hash);

        // Self-integrity hash over the content bytes [0..44].
        let record_hash =
            compute_record_hash(&record.to_bytes()[..WitnessRecord::CONTENT_LEN]);
        record.record_hash = fold_u64_to_u32(record_hash);

        // Chain hash binds prev link, sequence, AND record content.
        compute_chain_hash(prev_hash, seq, record_hash)
    }

    /// Store a fully-populated record and advance the chain state.
    ///
    /// Returns the sequence number assigned to the record.
    fn store(&mut self, record: WitnessRecord, chain: u64) -> u64 {
        let seq = self.sequence;
        if self.total_emitted >= N as u64 {
            // Ring is full: this write overwrites the oldest record.
            self.total_overwritten += 1;
        }
        let pos = self.write_pos;
        self.records[pos] = record;
        self.write_pos = (pos + 1) % N;
        self.chain_hash = chain;
        self.sequence = seq.wrapping_add(1);
        self.total_emitted += 1;

        seq
    }
}

impl<const N: usize> WitnessLog<N> {
    /// Compile-time assertion: N must be greater than zero.
    ///
    /// Using a const item inside the impl block causes a compilation
    /// error when `N == 0` because dividing by zero is a const-eval
    /// failure. This replaces the previous `assert!(N > 0)` runtime
    /// panic with a hard compile-time rejection.
    const _ASSERT_N_NONZERO: () = assert!(N > 0, "witness log capacity must be > 0");

    /// Creates a new empty witness log.
    ///
    /// # Compile-time invariant
    ///
    /// `N` must be greater than zero. Attempting to instantiate
    /// `WitnessLog<0>` is a compile-time error.
    #[must_use]
    pub fn new() -> Self {
        // Reference the const to ensure the compile-time check fires.
        let () = Self::_ASSERT_N_NONZERO;
        Self {
            inner: Mutex::new(WitnessLogInner {
                records: [WitnessRecord::zeroed(); N],
                write_pos: 0,
                chain_hash: 0,
                sequence: 0,
                total_emitted: 0,
                total_overwritten: 0,
            }),
        }
    }

    /// Appends a pre-built witness record to the log.
    ///
    /// Fills `sequence`, `prev_hash`, and `record_hash`, then stores the
    /// record. Returns the sequence number.
    ///
    /// `record_hash` is the self-integrity hash of the record's content
    /// bytes (`[0..WitnessRecord::CONTENT_LEN]`), and the chain hash
    /// binds it: `chain = H(prev_chain || sequence || record_hash)`.
    /// Tampering with any content field is therefore detected by
    /// [`crate::replay::verify_chain`].
    ///
    /// # Hash truncation
    ///
    /// The internal chain hash is a full 64-bit value, but the
    /// `WitnessRecord` fields `prev_hash` and `record_hash` are 32-bit
    /// (constrained by the 64-byte record layout, ADR-134). We use
    /// XOR-folding (`high32 ^ low32`) rather than simple `as u32`
    /// truncation to preserve entropy from both halves of the hash.
    ///
    /// **Future migration note:** When SHA-256 is adopted (TEE ADR),
    /// the record format should be revised to use 64-bit (or wider)
    /// hash fields, which will require a witness format version bump.
    pub fn append(&self, mut record: WitnessRecord) -> u64 {
        let mut inner = self.inner.lock();
        let chain = inner.fill_chain_fields(&mut record);
        inner.store(record, chain)
    }

    /// Appends a pre-built witness record with signing (ADR-142 Phase 4).
    ///
    /// Like [`append`], but after filling `sequence`, `prev_hash`, and
    /// `record_hash`, signs the fully-populated record using the provided
    /// [`WitnessSigner`] and stores the signature in the `aux` field.
    ///
    /// This ensures the signature covers all fields including chain-hash
    /// metadata, unlike signing before append.
    pub fn signed_append<S: crate::signer::WitnessSigner>(
        &self,
        mut record: WitnessRecord,
        signer: &S,
    ) -> u64 {
        let mut inner = self.inner.lock();
        let chain = inner.fill_chain_fields(&mut record);

        // Sign the fully-populated record (all chain-hash fields set).
        record.aux = signer.sign(&record);

        inner.store(record, chain)
    }

    /// Returns the total number of records ever emitted.
    pub fn total_emitted(&self) -> u64 {
        self.inner.lock().total_emitted
    }

    /// Returns the number of records that have been silently overwritten
    /// by ring wrap-around (i.e. audit records lost because no drain
    /// happened in time).
    pub fn total_overwritten(&self) -> u64 {
        self.inner.lock().total_overwritten
    }

    /// Returns true when the number of used slots has reached `watermark`,
    /// signaling that the log should be drained before wrap-around starts
    /// (or continues) to overwrite records.
    ///
    /// `watermark` is a slot count, typically a fraction of the capacity
    /// `N` (e.g. `(N * 3) / 4`). Once the ring has wrapped, the used
    /// count is pinned at `N`, so any watermark `<= N` keeps reporting
    /// `true` until a drain mechanism resets the log.
    pub fn needs_drain(&self, watermark: usize) -> bool {
        self.len() >= watermark
    }

    /// Returns the number of records currently in the buffer.
    #[allow(clippy::cast_possible_truncation)]
    pub fn len(&self) -> usize {
        let total = self.inner.lock().total_emitted;
        // Safe: if total < N then total fits in usize since N is usize.
        if total >= N as u64 { N } else { total as usize }
    }

    /// Returns true if no records have been emitted.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().total_emitted == 0
    }

    /// Returns a copy of the record at the given ring index.
    pub fn get(&self, ring_index: usize) -> Option<WitnessRecord> {
        if ring_index >= N {
            return None;
        }
        let inner = self.inner.lock();
        if inner.total_emitted == 0 {
            return None;
        }
        Some(inner.records[ring_index])
    }

    /// Copies the most recent records into the buffer. Returns count copied.
    pub fn snapshot(&self, buf: &mut [WitnessRecord]) -> usize {
        let inner = self.inner.lock();
        #[allow(clippy::cast_possible_truncation)]
        let available = if inner.total_emitted >= N as u64 {
            N
        } else {
            // Safe: total_emitted < N and N is usize, so it fits.
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

impl<const N: usize> Default for WitnessLog<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvm_types::ActionKind;

    fn make_record(kind: ActionKind, actor: u32, target: u64, ts: u64) -> WitnessRecord {
        let mut r = WitnessRecord::zeroed();
        r.action_kind = kind as u8;
        r.actor_partition_id = actor;
        r.target_object_id = target;
        r.timestamp_ns = ts;
        r
    }

    #[test]
    fn test_append_and_sequence() {
        let log = WitnessLog::<16>::new();
        let s0 = log.append(make_record(ActionKind::PartitionCreate, 1, 100, 1000));
        let s1 = log.append(make_record(ActionKind::CapabilityGrant, 1, 200, 2000));
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(log.total_emitted(), 2);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_ring_wrap() {
        let log = WitnessLog::<4>::new();
        for i in 0..10u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i * 100));
        }
        assert_eq!(log.total_emitted(), 10);
        assert_eq!(log.len(), 4);
    }

    #[test]
    fn test_overwrite_counter() {
        let log = WitnessLog::<4>::new();
        // Filling the ring exactly does not overwrite anything.
        for i in 0..4u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i * 100));
        }
        assert_eq!(log.total_overwritten(), 0);

        // Every append past capacity overwrites the oldest record.
        for i in 4..10u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i * 100));
        }
        assert_eq!(log.total_overwritten(), 6);
        assert_eq!(log.total_emitted(), 10);
    }

    #[test]
    fn test_overwrite_counter_signed_append() {
        use crate::signer::default_signer;

        let log = WitnessLog::<2>::new();
        let signer = default_signer();
        for i in 0..5u64 {
            log.signed_append(
                make_record(ActionKind::SchedulerEpoch, 1, i, i * 100),
                &signer,
            );
        }
        assert_eq!(log.total_overwritten(), 3);
    }

    #[test]
    fn test_needs_drain_watermark() {
        let log = WitnessLog::<8>::new();
        assert!(!log.needs_drain(6));

        for i in 0..5u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i * 100));
        }
        assert!(!log.needs_drain(6)); // 5 used < 6

        log.append(make_record(ActionKind::SchedulerEpoch, 1, 5, 500));
        assert!(log.needs_drain(6)); // 6 used >= 6

        // Once wrapped, used count is pinned at capacity.
        for i in 6..20u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i * 100));
        }
        assert!(log.needs_drain(6));
        assert!(log.needs_drain(8));
    }

    #[test]
    fn test_hash_chain() {
        let log = WitnessLog::<16>::new();
        log.append(make_record(ActionKind::PartitionCreate, 1, 10, 100));
        log.append(make_record(ActionKind::CapabilityGrant, 1, 20, 200));

        let r0 = log.get(0).unwrap();
        let r1 = log.get(1).unwrap();
        assert_eq!(r0.prev_hash, 0);
        assert_ne!(r1.prev_hash, 0);
    }

    #[test]
    fn test_snapshot() {
        let log = WitnessLog::<16>::new();
        for i in 0..5u64 {
            log.append(make_record(ActionKind::SchedulerEpoch, 1, i, i * 100));
        }
        let mut buf = [WitnessRecord::zeroed(); 3];
        let copied = log.snapshot(&mut buf);
        assert_eq!(copied, 3);
        assert_eq!(buf[0].sequence, 2);
        assert_eq!(buf[1].sequence, 3);
        assert_eq!(buf[2].sequence, 4);
    }

    #[test]
    fn test_empty_log() {
        let log = WitnessLog::<16>::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    // -- signed_append tests (ADR-142 Phase 4) -----------------------------

    #[test]
    fn test_signed_append_sets_aux() {
        use crate::signer::{WitnessSigner, default_signer};

        let log = WitnessLog::<16>::new();
        let signer = default_signer();

        let record = make_record(ActionKind::PartitionCreate, 1, 100, 1000);
        let seq = log.signed_append(record, &signer);
        assert_eq!(seq, 0);

        let stored = log.get(0).unwrap();
        // The aux field should be non-zero (signed).
        assert_ne!(stored.aux, [0u8; 8]);
    }

    #[test]
    fn test_signed_append_signature_verifiable() {
        use crate::signer::{WitnessSigner, default_signer};

        let log = WitnessLog::<16>::new();
        let signer = default_signer();

        let record = make_record(ActionKind::CapabilityGrant, 2, 200, 2000);
        log.signed_append(record, &signer);

        let stored = log.get(0).unwrap();
        // The stored record's signature should verify.
        assert!(signer.verify(&stored));
    }

    #[test]
    fn test_signed_append_chain_hashes_included() {
        use crate::signer::{WitnessSigner, default_signer};

        let log = WitnessLog::<16>::new();
        let signer = default_signer();

        // Append two signed records.
        log.signed_append(
            make_record(ActionKind::PartitionCreate, 1, 10, 100),
            &signer,
        );
        log.signed_append(
            make_record(ActionKind::CapabilityGrant, 1, 20, 200),
            &signer,
        );

        let r0 = log.get(0).unwrap();
        let r1 = log.get(1).unwrap();

        // Chain hashes should be set.
        assert_ne!(r1.prev_hash, 0);
        // Both records should verify.
        assert!(signer.verify(&r0));
        assert!(signer.verify(&r1));
    }

    #[test]
    fn test_signed_append_tampered_record_fails_verify() {
        use crate::signer::{WitnessSigner, default_signer};

        let log = WitnessLog::<16>::new();
        let signer = default_signer();

        log.signed_append(
            make_record(ActionKind::PartitionCreate, 1, 100, 1000),
            &signer,
        );

        let mut stored = log.get(0).unwrap();
        // Tamper with the record.
        stored.actor_partition_id = 999;
        // Verify should fail.
        assert!(!signer.verify(&stored));
    }
}
