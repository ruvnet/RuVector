//! Witness-chained checkpoint manifests.
//!
//! Mirrors the domain-separated sequential hash chain used by
//! `ruvector-proof-gate` / `ruvector-retrieval-receipt`, applied here to
//! checkpoint manifests instead of write/read receipts. A manifest commits
//! to: the ordered list of chunk hashes that reconstitute a checkpoint, the
//! full-content hash of the reconstructed bytes, and a running chain root
//! linking it to the previous checkpoint's manifest. Flipping any chunk's
//! stored bytes, reordering chunks, or splicing in a chunk from a different
//! checkpoint all change the chain root — this is the tamper-evidence
//! property under test, not confidentiality or write-path authenticity.

use crate::store::{reconstruct, ChunkHash, ChunkStore};
use sha2::{Digest, Sha256};

const GENESIS: [u8; 32] = *b"ruvector-cdc-checkpoint-genesis0";

fn chunk_leaf(seq: u64, hash: &ChunkHash) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"ruvector:cdc:leaf:");
    h.update(seq.to_le_bytes());
    h.update(hash);
    h.finalize().into()
}

fn chain_step(prev: &[u8; 32], leaf: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"ruvector:cdc:chain:");
    h.update(prev);
    h.update(leaf);
    h.finalize().into()
}

/// A single checkpoint's witness manifest.
#[derive(Debug, Clone)]
pub struct CheckpointManifest {
    pub round: u64,
    pub chunk_hashes: Vec<ChunkHash>,
    pub content_hash: [u8; 32],
    pub chain_root: [u8; 32],
}

/// Builds successive manifests, chaining each checkpoint's root into the
/// next so the sequence of checkpoints itself is tamper-evident (deleting
/// or reordering an earlier checkpoint breaks the chain, not just the
/// individual manifest).
pub struct WitnessChain {
    prev_root: [u8; 32],
    pub manifests: Vec<CheckpointManifest>,
}

impl Default for WitnessChain {
    fn default() -> Self {
        Self::new()
    }
}

impl WitnessChain {
    pub fn new() -> Self {
        Self {
            prev_root: GENESIS,
            manifests: Vec::new(),
        }
    }

    pub fn append(
        &mut self,
        round: u64,
        chunk_hashes: Vec<ChunkHash>,
        original_bytes: &[u8],
    ) -> CheckpointManifest {
        let content_hash: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(b"ruvector:cdc:content:");
            h.update(original_bytes);
            h.finalize().into()
        };

        let mut root = self.prev_root;
        for h in &chunk_hashes {
            let leaf = chunk_leaf(round, h);
            root = chain_step(&root, &leaf);
        }
        // Bind the content hash into the root too, so two different chunk
        // orderings that happen to reconstruct different bytes cannot share
        // a root by coincidence.
        root = chain_step(&root, &content_hash);

        let manifest = CheckpointManifest {
            round,
            chunk_hashes,
            content_hash,
            chain_root: root,
        };
        self.prev_root = root;
        self.manifests.push(manifest.clone());
        manifest
    }
}

/// Verification failure modes, kept explicit rather than a single boolean
/// so the benchmark and tests can assert *which* invariant broke.
#[derive(Debug, PartialEq, Eq)]
pub enum VerifyError {
    MissingChunk,
    ContentHashMismatch,
    ChainRootMismatch,
}

/// Recompute a manifest's chain root from the store contents and compare
/// against the recorded root. Also checks that reconstructing the chunks
/// yields exactly `content_hash`. This is the check a checkpoint reader
/// must run before trusting an incremental checkpoint it did not itself
/// produce.
pub fn verify(
    prev_root: &[u8; 32],
    manifest: &CheckpointManifest,
    store: &ChunkStore,
) -> Result<Vec<u8>, VerifyError> {
    let bytes = reconstruct(store, &manifest.chunk_hashes).ok_or(VerifyError::MissingChunk)?;

    let recomputed_content_hash: [u8; 32] = {
        let mut h = Sha256::new();
        h.update(b"ruvector:cdc:content:");
        h.update(&bytes);
        h.finalize().into()
    };
    if recomputed_content_hash != manifest.content_hash {
        return Err(VerifyError::ContentHashMismatch);
    }

    let mut root = *prev_root;
    for h in &manifest.chunk_hashes {
        let leaf = chunk_leaf(manifest.round, h);
        root = chain_step(&root, &leaf);
    }
    root = chain_step(&root, &manifest.content_hash);
    if root != manifest.chain_root {
        return Err(VerifyError::ChainRootMismatch);
    }

    Ok(bytes)
}

pub const GENESIS_ROOT: [u8; 32] = GENESIS;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::{cdc_boundaries, CdcParams};
    use crate::store::store_ranges;

    fn build_manifest(
        store: &mut ChunkStore,
        chain: &mut WitnessChain,
        round: u64,
        data: &[u8],
    ) -> [u8; 32] {
        let params = CdcParams::new(64, 256, 1024);
        let ranges = cdc_boundaries(data, &params);
        let (hashes, _) = store_ranges(store, data, &ranges);
        let prev = chain.prev_root;
        chain.append(round, hashes, data);
        prev
    }

    #[test]
    fn honest_manifest_verifies() {
        let mut store = ChunkStore::new();
        let mut chain = WitnessChain::new();
        let data = vec![7u8; 5000];
        let prev = build_manifest(&mut store, &mut chain, 0, &data);
        let manifest = chain.manifests[0].clone();
        let result = verify(&prev, &manifest, &store).expect("honest manifest must verify");
        assert_eq!(result, data);
    }

    #[test]
    fn manifest_lying_about_content_hash_is_rejected() {
        // An attacker (or a corrupted transport) hands over a manifest
        // whose declared `content_hash` does not match what the honest
        // chunk store actually reconstructs to. This must be caught before
        // the chain root is even considered.
        let mut store = ChunkStore::new();
        let mut chain = WitnessChain::new();
        let data = vec![9u8; 5000];
        let prev = build_manifest(&mut store, &mut chain, 0, &data);
        let manifest = chain.manifests[0].clone();

        let mut forged = manifest.clone();
        forged.content_hash[0] ^= 0xFF;
        let err = verify(&prev, &forged, &store).unwrap_err();
        assert_eq!(err, VerifyError::ContentHashMismatch);
    }

    #[test]
    fn manifest_with_forged_chain_root_is_rejected() {
        // The content hash and chunk list are honest, but the recorded
        // `chain_root` has been forged (e.g. to hide that an earlier
        // checkpoint in the chain was tampered with, or to splice in a
        // chunk list from a different round). Verification recomputes the
        // root from `prev_root` + the chunk list and must reject a root
        // that does not match, independent of the content check passing.
        let mut store = ChunkStore::new();
        let mut chain = WitnessChain::new();
        let data = vec![9u8; 5000];
        let prev = build_manifest(&mut store, &mut chain, 0, &data);
        let mut forged = chain.manifests[0].clone();
        forged.chain_root[0] ^= 0xFF;
        let err = verify(&prev, &forged, &store).unwrap_err();
        assert_eq!(err, VerifyError::ChainRootMismatch);
    }

    #[test]
    fn missing_chunk_is_detected() {
        let mut store = ChunkStore::new();
        let mut chain = WitnessChain::new();
        let data = vec![3u8; 5000];
        let prev = build_manifest(&mut store, &mut chain, 0, &data);
        let manifest = chain.manifests[0].clone();
        let empty_store = ChunkStore::new();
        assert_eq!(
            verify(&prev, &manifest, &empty_store).unwrap_err(),
            VerifyError::MissingChunk
        );
    }
}
