//! RVF-style receipt store and verification (ADR-260 §15).
//!
//! [`ReceiptStore`] persists receipts as JSON strings and delegates
//! verification to [`photonlayer_core::receipt::verify_receipt`].

use photonlayer_core::prelude::{verify_receipt, ExperimentReceipt};
use serde_json;
use std::collections::HashMap;

/// Storage and verification of [`ExperimentReceipt`]s.
///
/// Each receipt is serialised to JSON on insertion so that the raw bytes
/// remain stable. Verification deserialises and recomputes the binding digest.
#[derive(Default, Debug)]
pub struct ReceiptStore {
    /// Map from `experiment_id` to serialised JSON string.
    store: HashMap<String, String>,
}

impl ReceiptStore {
    /// Create an empty receipt store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a receipt, overwriting any existing entry for the same id.
    ///
    /// # Errors
    /// Returns an error string if serialisation fails (practically infallible
    /// for well-formed receipts).
    pub fn insert(&mut self, receipt: &ExperimentReceipt) -> Result<(), String> {
        let json = serde_json::to_string(receipt).map_err(|e| format!("serialise failed: {e}"))?;
        self.store.insert(receipt.experiment_id.clone(), json);
        Ok(())
    }

    /// Number of stored receipts.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns `true` when no receipts are stored.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Verify the receipt identified by `id`.
    ///
    /// Returns `true` if the receipt passes [`verify_receipt`].
    /// Returns `false` if the id is unknown, deserialisation fails, or
    /// verification fails (tampered fields).
    pub fn verify(&self, id: &str) -> bool {
        match self.store.get(id) {
            None => false,
            Some(json) => match serde_json::from_str::<ExperimentReceipt>(json) {
                Err(_) => false,
                Ok(receipt) => verify_receipt(&receipt),
            },
        }
    }

    /// Verify every stored receipt.
    ///
    /// Returns a map from experiment id to verification outcome.
    pub fn verify_all(&self) -> HashMap<String, bool> {
        self.store
            .iter()
            .map(|(id, json)| {
                let ok = serde_json::from_str::<ExperimentReceipt>(json)
                    .map(|r| verify_receipt(&r))
                    .unwrap_or(false);
                (id.clone(), ok)
            })
            .collect()
    }

    /// Retrieve the stored JSON string for an experiment id.
    pub fn get_json(&self, id: &str) -> Option<&str> {
        self.store.get(id).map(String::as_str)
    }

    /// Retrieve and deserialise a receipt by experiment id.
    ///
    /// Returns `None` if the id is unknown or deserialisation fails.
    pub fn get(&self, id: &str) -> Option<ExperimentReceipt> {
        self.store
            .get(id)
            .and_then(|json| serde_json::from_str(json).ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photonlayer_core::prelude::{
        build_receipt, InputImage, MetricReport, OpticalConfig, OpticalSimulator, PhaseMask,
        Provenance, ScalarSimulator,
    };

    fn make_receipt(id: &str, seed: u64) -> ExperimentReceipt {
        let n = 16;
        let pixels: Vec<f32> = (0..n * n).map(|i| (i % n) as f32 / n as f32).collect();
        let img = InputImage::from_norm_f32(n, n, pixels).unwrap();
        let mask = PhaseMask::random(n, n, seed);
        let cfg = OpticalConfig::demo(n, n);
        let frame = ScalarSimulator.simulate(&img, &mask, &cfg).unwrap();
        let metrics = MetricReport::default();
        build_receipt(
            id,
            &img,
            &mask,
            &cfg,
            &frame,
            &metrics,
            &Provenance::default(),
        )
    }

    #[test]
    fn insert_and_verify_valid_receipt() {
        let mut store = ReceiptStore::new();
        let r = make_receipt("exp-ok", 1);
        store.insert(&r).unwrap();
        assert!(store.verify("exp-ok"), "valid receipt should verify");
    }

    #[test]
    fn verify_unknown_id_returns_false() {
        let store = ReceiptStore::new();
        assert!(!store.verify("nonexistent"));
    }

    #[test]
    fn tampered_receipt_fails_verification() {
        let mut store = ReceiptStore::new();
        let mut r = make_receipt("exp-tampered", 2);
        // Tamper: mutate a hash field after building.
        r.output_hash.push('X');
        store.insert(&r).unwrap();
        assert!(!store.verify("exp-tampered"), "tampered receipt must fail");
    }

    #[test]
    fn verify_all_mixes_pass_and_fail() {
        let mut store = ReceiptStore::new();
        let good = make_receipt("good", 3);
        store.insert(&good).unwrap();

        // Insert a deliberately broken receipt by mutating JSON after insertion.
        let mut bad = make_receipt("bad", 4);
        bad.mask_hash.push('!');
        store.insert(&bad).unwrap();

        let results = store.verify_all();
        assert_eq!(results.len(), 2);
        assert!(results["good"]);
        assert!(!results["bad"]);
    }

    #[test]
    fn len_and_is_empty() {
        let mut store = ReceiptStore::new();
        assert!(store.is_empty());
        store.insert(&make_receipt("e1", 5)).unwrap();
        assert_eq!(store.len(), 1);
        store.insert(&make_receipt("e2", 6)).unwrap();
        assert_eq!(store.len(), 2);
    }
}
