//! Witness chains — tamper-evident, reproducible provenance for trained models.
//!
//! A witness chain is a hash-linked ledger of training runs. Each record seals
//! the hashes of its inputs (dataset + config), the resulting model, and the
//! held-out metrics, then links to the previous record's hash. Anyone can
//! recompute the hashes from the committed model + a re-run and confirm:
//!
//! 1. **integrity** — the stored metrics/model match their hashes;
//! 2. **chain continuity** — each record links to the prior one;
//! 3. **reproducibility** — re-training with the same data + config yields the
//!    same `model_hash` (the learner is deterministic), so the sealed metrics
//!    are checkable rather than asserted.
//!
//! This is "proof" in the sense of *verifiable provenance* — it proves the
//! reported numbers correspond to the committed model and are reproducible. It
//! does **not** prove the model beats real-world SOTA; that requires real
//! labelled data (ADR-251 §4).
//!
//! Hashing is FNV-1a (64-bit) — deterministic, dependency-free, and adequate for
//! provenance/integrity (not a cryptographic commitment).

/// FNV-1a 64-bit hash.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Round to 6 decimals so float metrics serialize and re-hash exactly.
fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// Hash a slice of f64 by their canonical (rounded) bit patterns.
pub fn hash_f64s(xs: &[f64]) -> u64 {
    let mut bytes = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        bytes.extend_from_slice(&round6(x).to_bits().to_le_bytes());
    }
    fnv1a64(&bytes)
}

/// Hash a 2-D dataset plus its labels into a single content hash.
pub fn hash_dataset(x: &[Vec<f64>], y: &[f64]) -> u64 {
    let mut acc: u64 = fnv1a64(&(x.len() as u64).to_le_bytes());
    for (row, &label) in x.iter().zip(y) {
        let mut h = hash_f64s(row);
        h ^= (label.to_bits()).rotate_left(17);
        acc = acc.wrapping_mul(0x100_0000_01b3) ^ h;
    }
    acc
}

/// One sealed training-run record.
#[derive(Clone, Debug, PartialEq)]
pub struct WitnessRecord {
    pub index: u64,
    pub prev: u64,
    pub data_hash: u64,
    pub config_hash: u64,
    pub model_hash: u64,
    pub val_auc: f64,
    pub single_auc: f64,
    pub handset_auc: f64,
    /// Sealing hash over all fields above (incl. `prev`).
    pub hash: u64,
}

impl WitnessRecord {
    /// Canonical byte encoding used for the sealing hash.
    fn payload(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(8 * 8);
        b.extend_from_slice(&self.index.to_le_bytes());
        b.extend_from_slice(&self.prev.to_le_bytes());
        b.extend_from_slice(&self.data_hash.to_le_bytes());
        b.extend_from_slice(&self.config_hash.to_le_bytes());
        b.extend_from_slice(&self.model_hash.to_le_bytes());
        b.extend_from_slice(&round6(self.val_auc).to_bits().to_le_bytes());
        b.extend_from_slice(&round6(self.single_auc).to_bits().to_le_bytes());
        b.extend_from_slice(&round6(self.handset_auc).to_bits().to_le_bytes());
        b
    }

    /// Seal a record: compute its `hash` from its content and `prev`.
    #[allow(clippy::too_many_arguments)]
    pub fn seal(
        index: u64,
        prev: u64,
        data_hash: u64,
        config_hash: u64,
        model_hash: u64,
        val_auc: f64,
        single_auc: f64,
        handset_auc: f64,
    ) -> WitnessRecord {
        let mut r = WitnessRecord {
            index,
            prev,
            data_hash,
            config_hash,
            model_hash,
            val_auc,
            single_auc,
            handset_auc,
            hash: 0,
        };
        r.hash = fnv1a64(&r.payload());
        r
    }

    /// Recompute the sealing hash and compare to the stored one.
    pub fn is_intact(&self) -> bool {
        fnv1a64(&self.payload()) == self.hash
    }

    /// Serialize to a single pipe-delimited line (hex hashes, 6-dp metrics).
    pub fn to_line(&self) -> String {
        format!(
            "{}|{:016x}|{:016x}|{:016x}|{:016x}|{:.6}|{:.6}|{:.6}|{:016x}",
            self.index,
            self.prev,
            self.data_hash,
            self.config_hash,
            self.model_hash,
            round6(self.val_auc),
            round6(self.single_auc),
            round6(self.handset_auc),
            self.hash
        )
    }

    /// Parse a line produced by [`WitnessRecord::to_line`].
    pub fn from_line(line: &str) -> Option<WitnessRecord> {
        let p: Vec<&str> = line.trim().split('|').collect();
        if p.len() != 9 {
            return None;
        }
        Some(WitnessRecord {
            index: p[0].parse().ok()?,
            prev: u64::from_str_radix(p[1], 16).ok()?,
            data_hash: u64::from_str_radix(p[2], 16).ok()?,
            config_hash: u64::from_str_radix(p[3], 16).ok()?,
            model_hash: u64::from_str_radix(p[4], 16).ok()?,
            val_auc: p[5].parse().ok()?,
            single_auc: p[6].parse().ok()?,
            handset_auc: p[7].parse().ok()?,
            hash: u64::from_str_radix(p[8], 16).ok()?,
        })
    }
}

/// A hash-linked chain of witness records.
#[derive(Clone, Debug, Default)]
pub struct WitnessChain {
    pub records: Vec<WitnessRecord>,
}

impl WitnessChain {
    pub fn new() -> Self {
        WitnessChain::default()
    }

    /// Tip hash (0 for an empty chain).
    pub fn tip(&self) -> u64 {
        self.records.last().map(|r| r.hash).unwrap_or(0)
    }

    /// Append a pre-sealed record whose `prev` must equal the current tip.
    pub fn append(&mut self, record: WitnessRecord) -> Result<(), String> {
        if record.prev != self.tip() {
            return Err(format!(
                "record {} prev {:016x} does not link to tip {:016x}",
                record.index,
                record.prev,
                self.tip()
            ));
        }
        if !record.is_intact() {
            return Err(format!("record {} fails its own seal", record.index));
        }
        self.records.push(record);
        Ok(())
    }

    /// Seal a fresh record from raw inputs and append it.
    #[allow(clippy::too_many_arguments)]
    pub fn seal_and_append(
        &mut self,
        data_hash: u64,
        config_hash: u64,
        model_hash: u64,
        val_auc: f64,
        single_auc: f64,
        handset_auc: f64,
    ) -> Result<(), String> {
        let rec = WitnessRecord::seal(
            self.records.len() as u64,
            self.tip(),
            data_hash,
            config_hash,
            model_hash,
            val_auc,
            single_auc,
            handset_auc,
        );
        self.append(rec)
    }

    /// Verify every record's seal and the chain links. Returns the verified
    /// length or the first inconsistency.
    pub fn verify(&self) -> Result<usize, String> {
        let mut prev = 0u64;
        for (i, r) in self.records.iter().enumerate() {
            if r.index as usize != i {
                return Err(format!("record {i} has out-of-order index {}", r.index));
            }
            if r.prev != prev {
                return Err(format!("record {i} broken link"));
            }
            if !r.is_intact() {
                return Err(format!("record {i} tampered (seal mismatch)"));
            }
            prev = r.hash;
        }
        Ok(self.records.len())
    }

    pub fn to_text(&self) -> String {
        let mut s = String::from(
            "# emergent-time witness chain\n\
             # index|prev|data_hash|config_hash|model_hash|val_auc|single_auc|handset_auc|hash\n",
        );
        for r in &self.records {
            s.push_str(&r.to_line());
            s.push('\n');
        }
        s
    }

    pub fn from_text(text: &str) -> WitnessChain {
        let mut chain = WitnessChain::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(r) = WitnessRecord::from_line(line) {
                chain.records.push(r);
            }
        }
        chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_is_deterministic_and_intact() {
        let a = WitnessRecord::seal(0, 0, 11, 22, 33, 0.9, 0.7, 0.8);
        let b = WitnessRecord::seal(0, 0, 11, 22, 33, 0.9, 0.7, 0.8);
        assert_eq!(a, b);
        assert!(a.is_intact());
    }

    #[test]
    fn chain_links_and_verifies() {
        let mut c = WitnessChain::new();
        c.seal_and_append(1, 2, 3, 0.90, 0.70, 0.80).unwrap();
        c.seal_and_append(4, 5, 6, 0.92, 0.71, 0.81).unwrap();
        assert_eq!(c.verify().unwrap(), 2);
        // Each record links to the previous.
        assert_eq!(c.records[1].prev, c.records[0].hash);
    }

    #[test]
    fn tamper_is_detected() {
        let mut c = WitnessChain::new();
        c.seal_and_append(1, 2, 3, 0.90, 0.70, 0.80).unwrap();
        c.seal_and_append(4, 5, 6, 0.92, 0.71, 0.81).unwrap();
        // Flip a metric without resealing → seal mismatch.
        c.records[0].val_auc = 0.99;
        assert!(c.verify().is_err());
    }

    #[test]
    fn text_round_trip_preserves_verification() {
        let mut c = WitnessChain::new();
        c.seal_and_append(1, 2, 3, 0.901234, 0.706789, 0.812345)
            .unwrap();
        c.seal_and_append(4, 5, 6, 0.923456, 0.711111, 0.815555)
            .unwrap();
        let text = c.to_text();
        let parsed = WitnessChain::from_text(&text);
        assert_eq!(parsed.records, c.records);
        assert_eq!(parsed.verify().unwrap(), 2);
    }

    #[test]
    fn dataset_hash_is_order_sensitive() {
        let x1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let x2 = vec![vec![3.0, 4.0], vec![1.0, 2.0]];
        let y = vec![1.0, 0.0];
        assert_ne!(hash_dataset(&x1, &y), hash_dataset(&x2, &y));
    }
}
