//! Epoch pin storage (ADR-353 sections 2.5 and 7).
//!
//! Pins are keyed by `(signer_key_id, file_id)` and always carry the hash of the
//! descriptor accepted at that epoch. Protocol:
//!
//! 1. At admission the verifier reads the pin (`get`) and refuses older epochs and
//!    equal epochs with a different descriptor hash.
//! 2. At completion the pin is read again ([`StreamVerifier::recheck_pin`], run
//!    automatically when the policy carries a store): a session superseded while it
//!    ran ends in `StreamStaleEpoch`, never in a verified result.
//! 3. Only after `Complete` with the identity confirmed, [`StreamVerifier::commit_pin`]
//!    records the new pin with a compare-and-set against the pin seen at admission.

use super::EpochPin;
use alloc::collections::BTreeMap;
use std::sync::Mutex;

extern crate alloc;

/// A compare-and-set that lost, or that would move a pin backwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinConflict {
    /// The pin currently stored.
    pub current: Option<EpochPin>,
}

/// Durable (or in-memory) pin storage shared by sessions.
pub trait PinStore {
    /// Current pin for `(signer_key_id, file_id)`.
    fn get(&self, signer_key_id: &[u8; 8], file_id: &[u8; 16]) -> Option<EpochPin>;

    /// Atomically replace the pin if it still equals `expected_current`.
    ///
    /// Must fail with [`PinConflict`] when the stored pin differs from
    /// `expected_current`, and must never move backwards: `new.epoch` below the
    /// stored epoch, or equal with a different descriptor hash, is a conflict.
    /// Re-recording the identical pin is a no-op success.
    fn advance(
        &self,
        signer_key_id: &[u8; 8],
        file_id: &[u8; 16],
        expected_current: Option<EpochPin>,
        new: EpochPin,
    ) -> Result<(), PinConflict>;
}

type PinKey = ([u8; 8], [u8; 16]);

/// Reference in-memory store.
#[derive(Default)]
pub struct MemoryPinStore {
    pins: Mutex<BTreeMap<PinKey, EpochPin>>,
}

impl MemoryPinStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PinStore for MemoryPinStore {
    fn get(&self, signer: &[u8; 8], file_id: &[u8; 16]) -> Option<EpochPin> {
        self.pins
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(*signer, *file_id))
            .copied()
    }

    fn advance(
        &self,
        signer: &[u8; 8],
        file_id: &[u8; 16],
        expected_current: Option<EpochPin>,
        new: EpochPin,
    ) -> Result<(), PinConflict> {
        let mut m = self.pins.lock().unwrap_or_else(|e| e.into_inner());
        let cur = m.get(&(*signer, *file_id)).copied();
        // Re-recording the identical pin is a no-op success even when this
        // session was admitted against an older view (concurrent sessions of
        // one descriptor, F3-03).
        if cur == Some(new) {
            return Ok(());
        }
        if cur != expected_current {
            return Err(PinConflict { current: cur });
        }
        if let Some(c) = cur {
            if new.epoch < c.epoch
                || (new.epoch == c.epoch && new.descriptor_hash != c.descriptor_hash)
            {
                return Err(PinConflict { current: cur });
            }
        }
        m.insert((*signer, *file_id), new);
        Ok(())
    }
}
