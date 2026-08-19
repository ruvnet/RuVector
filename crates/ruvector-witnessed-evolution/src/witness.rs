//! Hash-chain witnessing of an evolutionary parameter search lineage.
//!
//! Each generation's genome, fitness, and accept/reject decision is admitted
//! through a `ruvector_proof_gate::HashChainGate` as a `WritePayload` — the
//! genome doubles as the payload's vector field, and fitness/decision are
//! packed into `metadata`. This reuses `ruvector-proof-gate`'s existing
//! SHA-256 chain primitive rather than inventing a parallel one.
//!
//! # What replay actually proves
//!
//! A `WitnessedLineage` keeps the plaintext evidence (genome, fitness,
//! decision) *and* the chain of commitments to it. [`WitnessedLineage::replay_verify`]
//! is the independent auditor: it (1) recomputes each entry's payload hash
//! from the plaintext evidence and checks it against what was committed at
//! admission time, (2) re-derives the full hash chain from genesis, (3)
//! independently re-evaluates fitness for every genome against the raw
//! workload, and (4) re-derives every accept/reject decision under the
//! fixed promotion policy. Any of the four disagreeing is treated as
//! evidence of tampering — matching the threat model `ruvector-retrieval-receipt`
//! already documents for its own read-path receipts: this detects
//! post-issuance mutation of the evidence, it does not (and cannot) prove
//! the *search itself* was run honestly in the first place.

use ruvector_proof_gate::{HashChainGate, WriteGate, WritePayload, WriteReceipt};

use crate::fitness::{Fitness, Workload};
use crate::genome::Genome;

/// Fixed-width metadata layout committed alongside each genome:
/// `[accepted:u8][recall_mean:f32][avg_expansions:f32][composite:f32]`.
const METADATA_LEN: usize = 1 + 4 + 4 + 4;

fn encode_metadata(fitness: Fitness, accepted: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(METADATA_LEN);
    out.push(u8::from(accepted));
    out.extend_from_slice(&fitness.recall_mean.to_le_bytes());
    out.extend_from_slice(&fitness.avg_expansions.to_le_bytes());
    out.extend_from_slice(&fitness.composite.to_le_bytes());
    out
}

fn payload_for(generation: u64, genome: Genome, fitness: Fitness, accepted: bool) -> WritePayload {
    WritePayload::new(generation, genome.to_vec()).with_metadata(encode_metadata(fitness, accepted))
}

/// One committed generation: the plaintext evidence plus the receipt the
/// gate returned when it was admitted.
#[derive(Clone, Debug)]
pub struct GenerationRecord {
    pub generation: u64,
    pub genome: Genome,
    pub fitness: Fitness,
    pub accepted: bool,
    pub receipt: WriteReceipt,
}

/// A witnessed evolutionary lineage: plaintext evidence + hash-chain
/// commitments, both required for replay.
pub struct WitnessedLineage {
    gate: HashChainGate,
    records: Vec<GenerationRecord>,
}

impl WitnessedLineage {
    pub fn new() -> Self {
        WitnessedLineage {
            gate: HashChainGate::new(),
            records: Vec::new(),
        }
    }

    /// Commit one generation. Returns the receipt (also retained internally
    /// for replay).
    pub fn record(
        &mut self,
        generation: u64,
        genome: Genome,
        fitness: Fitness,
        accepted: bool,
    ) -> WriteReceipt {
        let payload = payload_for(generation, genome, fitness, accepted);
        let receipt = self
            .gate
            .admit(&payload)
            .expect("HashChainGate never rejects an admission");
        self.records.push(GenerationRecord {
            generation,
            genome,
            fitness,
            accepted,
            receipt: receipt.clone(),
        });
        receipt
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn chain_root(&self) -> [u8; 32] {
        self.gate.chain_root()
    }

    pub fn records(&self) -> &[GenerationRecord] {
        &self.records
    }

    /// Adversarial test hook: mutate a committed generation's stored
    /// fitness *without* recomputing its receipt — simulating an attacker
    /// who edits the evidence log after issuance but cannot forge a
    /// SHA-256 preimage. Returns `false` if `generation_idx` is out of
    /// range.
    pub fn tamper_composite(&mut self, generation_idx: usize, forged_composite: f32) -> bool {
        match self.records.get_mut(generation_idx) {
            Some(rec) => {
                rec.fitness.composite = forged_composite;
                true
            }
            None => false,
        }
    }

    /// Independently replay and verify the entire lineage against the raw
    /// workload. See module docs for exactly what this does and does not
    /// prove.
    pub fn replay_verify(&self, workload: &Workload) -> ReplayReport {
        let chain_integrity_ok = self.gate.verify_integrity();
        let mut first_divergence = None;
        let mut incumbent: Option<Fitness> = None;

        for (i, rec) in self.records.iter().enumerate() {
            let recomputed_payload =
                payload_for(rec.generation, rec.genome, rec.fitness, rec.accepted);
            let recomputed_hash = recomputed_payload.payload_hash();
            if recomputed_hash != rec.receipt.payload_hash {
                first_divergence = Some(i);
                break;
            }
            if !self.gate.verify_receipt(&rec.receipt) {
                first_divergence = Some(i);
                break;
            }

            let recomputed_fitness = workload.evaluate(rec.genome);
            if recomputed_fitness != rec.fitness {
                first_divergence = Some(i);
                break;
            }

            let expected_accept = match incumbent {
                None => true,
                Some(inc) => recomputed_fitness.composite > inc.composite,
            };
            if expected_accept != rec.accepted {
                first_divergence = Some(i);
                break;
            }
            if expected_accept {
                incumbent = Some(recomputed_fitness);
            }
        }

        ReplayReport {
            generations_checked: self.records.len(),
            chain_integrity_ok,
            first_divergence,
            verified: chain_integrity_ok && first_divergence.is_none(),
        }
    }
}

impl Default for WitnessedLineage {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of [`WitnessedLineage::replay_verify`].
#[derive(Debug, Clone, Copy)]
pub struct ReplayReport {
    pub generations_checked: usize,
    pub chain_integrity_ok: bool,
    /// Index of the first generation whose recomputation disagreed with
    /// what was committed, if any.
    pub first_divergence: Option<usize>,
    pub verified: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::WorkloadConfig;
    use crate::genome::DEFAULT_GENOME;

    fn tiny_workload() -> Workload {
        Workload::build(&WorkloadConfig {
            n_clusters: 4,
            n_per_cluster: 30,
            dims: 12,
            cluster_std: 0.15,
            m: 10,
            m_longjump: 4,
            n_queries: 20,
            k: 8,
            entry_id: 0,
            data_seed: 0x1111,
            query_seed: 0x2222,
        })
    }

    #[test]
    fn honest_lineage_replays_clean() {
        let w = tiny_workload();
        let mut lineage = WitnessedLineage::new();
        let f0 = w.evaluate(DEFAULT_GENOME);
        lineage.record(0, DEFAULT_GENOME, f0, true);
        let report = lineage.replay_verify(&w);
        assert!(report.verified, "{report:?}");
        assert!(report.chain_integrity_ok);
        assert_eq!(report.first_divergence, None);
    }

    #[test]
    fn tampered_fitness_is_detected() {
        let w = tiny_workload();
        let mut lineage = WitnessedLineage::new();
        let f0 = w.evaluate(DEFAULT_GENOME);
        lineage.record(0, DEFAULT_GENOME, f0, true);
        assert!(lineage.tamper_composite(0, f0.composite + 10.0));
        let report = lineage.replay_verify(&w);
        assert!(!report.verified);
        assert_eq!(report.first_divergence, Some(0));
    }

    #[test]
    fn out_of_range_tamper_is_a_noop() {
        let mut lineage = WitnessedLineage::new();
        assert!(!lineage.tamper_composite(5, 0.0));
    }
}
