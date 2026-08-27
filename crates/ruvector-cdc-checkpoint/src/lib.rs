//! Content-defined chunking for incremental, witness-chained index
//! checkpoints.
//!
//! # Hypothesis under test
//!
//! Given an HNSW-style vector+graph index that is checkpointed periodically
//! for durability and portability (the role `ruvector-snapshot` plays for
//! agent-memory collections), replacing "re-serialize and re-write the
//! whole snapshot" with content-defined chunking (CDC) plus a
//! content-addressed store should make the *incremental* bytes written per
//! checkpoint scale with the size of the edit, not the size of the index —
//! while every checkpoint remains exactly reconstructible and is
//! witness-chained so a consumer (an edge replica, an RVF-portable package
//! reader) can detect tampering or a missing chunk before trusting it.
//!
//! Three variants are compared, all going through the same
//! [`Checkpointer`] so the only difference is chunking strategy:
//!
//! - [`Variant::FullSnapshot`] — one chunk covering the whole blob (the
//!   "full re-write every round" baseline every naive snapshot system
//!   defaults to).
//! - [`Variant::FixedBlock`] — fixed-size block chunking plus dedup (a
//!   real, non-strawman incremental-backup baseline; its weakness is that
//!   an edit shifts every downstream block boundary).
//! - [`Variant::Cdc`] — FastCDC-style content-defined chunking plus dedup
//!   plus a sequential witness hash chain over the manifest.

pub mod chunker;
pub mod store;
pub mod witness;
pub mod workload;

use chunker::{cdc_boundaries, fixed_boundaries, CdcParams};
use std::time::{Duration, Instant};
use store::{store_ranges, ChunkStore};
use witness::{CheckpointManifest, WitnessChain};

#[derive(Debug, Clone, Copy)]
pub enum Variant {
    FullSnapshot,
    FixedBlock(usize),
    Cdc(CdcParams),
}

impl Variant {
    pub fn name(&self) -> &'static str {
        match self {
            Variant::FullSnapshot => "full_snapshot",
            Variant::FixedBlock(_) => "fixed_block",
            Variant::Cdc(_) => "cdc",
        }
    }

    fn boundaries(&self, data: &[u8]) -> Vec<(usize, usize)> {
        match self {
            Variant::FullSnapshot => {
                if data.is_empty() {
                    Vec::new()
                } else {
                    vec![(0, data.len())]
                }
            }
            Variant::FixedBlock(block_size) => fixed_boundaries(data, *block_size),
            Variant::Cdc(params) => cdc_boundaries(data, params),
        }
    }
}

/// Per-round measurement. `new_bytes` is the honest "bytes this checkpoint
/// actually had to persist" figure the acceptance test compares across
/// variants; `chunk_time` isolates chunking+hashing cost from the rest of
/// the benchmark harness.
#[derive(Debug, Clone)]
pub struct RoundStats {
    pub round: u64,
    pub blob_len: usize,
    pub chunk_count: usize,
    pub new_bytes: usize,
    pub resident_bytes_after: usize,
    pub chunk_time: Duration,
}

pub struct Checkpointer {
    variant: Variant,
    store: ChunkStore,
    chain: WitnessChain,
}

impl Checkpointer {
    pub fn new(variant: Variant) -> Self {
        Self {
            variant,
            store: ChunkStore::new(),
            chain: WitnessChain::new(),
        }
    }

    /// Persist one checkpoint of `blob` and return its stats plus the
    /// witness manifest a consumer would need to verify and reconstruct it.
    pub fn checkpoint(&mut self, round: u64, blob: &[u8]) -> (RoundStats, CheckpointManifest) {
        let t0 = Instant::now();
        let ranges = self.variant.boundaries(blob);
        let chunk_count = ranges.len();
        let (hashes, new_bytes) = store_ranges(&mut self.store, blob, &ranges);
        let chunk_time = t0.elapsed();

        let manifest = self.chain.append(round, hashes, blob);

        let stats = RoundStats {
            round,
            blob_len: blob.len(),
            chunk_count,
            new_bytes,
            resident_bytes_after: self.store.resident_bytes(),
            chunk_time,
        };
        (stats, manifest)
    }

    /// The chain root immediately *before* `manifest.round` was appended —
    /// what a verifier must have already trusted (e.g. from the previous
    /// checkpoint it verified) to check `manifest` via [`witness::verify`].
    pub fn root_before(&self, manifest: &CheckpointManifest) -> [u8; 32] {
        if manifest.round == 0 {
            witness::GENESIS_ROOT
        } else {
            self.chain.manifests[(manifest.round - 1) as usize].chain_root
        }
    }

    pub fn store(&self) -> &ChunkStore {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use workload::IndexState;

    #[test]
    fn all_three_variants_reconstruct_every_round_exactly() {
        for variant in [
            Variant::FullSnapshot,
            Variant::FixedBlock(4096),
            Variant::Cdc(CdcParams::new(512, 2048, 8192)),
        ] {
            let mut replay_state = IndexState::build(2_000, 32, 8, 11);
            let mut checkpointer = Checkpointer::new(variant);
            for round in 0..5u64 {
                if round > 0 {
                    replay_state.churn(20, 10, 20);
                }
                let blob = replay_state.serialize();
                let (_, manifest) = checkpointer.checkpoint(round, &blob);
                let root_before = checkpointer.root_before(&manifest);
                let reconstructed = witness::verify(&root_before, &manifest, checkpointer.store())
                    .unwrap_or_else(|e| {
                        panic!("{}: round {round} failed to verify: {e:?}", variant.name())
                    });
                assert_eq!(
                    reconstructed,
                    blob,
                    "{}: round {round} mismatch",
                    variant.name()
                );
            }
        }
    }

    #[test]
    fn cdc_writes_fewer_new_bytes_than_full_snapshot_after_small_churn() {
        let mut full_state = IndexState::build(5_000, 64, 12, 5);
        let mut cdc_state = IndexState::build(5_000, 64, 12, 5);
        let mut full = Checkpointer::new(Variant::FullSnapshot);
        let mut cdc = Checkpointer::new(Variant::Cdc(CdcParams::new(512, 2048, 8192)));

        let (s0, _) = full.checkpoint(0, &full_state.serialize());
        let (c0, _) = cdc.checkpoint(0, &cdc_state.serialize());
        assert_eq!(
            s0.new_bytes, c0.new_bytes,
            "first round has no history to dedup against"
        );

        full_state.churn(5, 5, 10); // ~0.4% of 5000 rows touched
        cdc_state.churn(5, 5, 10);
        let (s1, _) = full.checkpoint(1, &full_state.serialize());
        let (c1, _) = cdc.checkpoint(1, &cdc_state.serialize());

        assert!(
            c1.new_bytes < s1.new_bytes / 2,
            "expected CDC round-1 new_bytes ({}) to be well under half of full-snapshot new_bytes ({})",
            c1.new_bytes,
            s1.new_bytes
        );
    }
}
