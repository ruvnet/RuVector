use thiserror::Error;

/// A vector write payload: the unit of admission through a WriteGate.
#[derive(Debug, Clone)]
pub struct WritePayload {
    /// Monotonic identifier assigned by the caller.
    pub id: u64,
    /// The embedding vector (f32 components).
    pub vector: Vec<f32>,
    /// Arbitrary metadata bytes (e.g. JSON-encoded tags, namespace, etc.).
    pub metadata: Vec<u8>,
    /// Originating agent identifier (16 bytes, e.g. UUID without hyphens).
    pub agent_id: [u8; 16],
    /// UNIX timestamp in nanoseconds at the write site.
    pub timestamp_ns: u64,
}

impl WritePayload {
    pub fn new(id: u64, vector: Vec<f32>) -> Self {
        Self {
            id,
            vector,
            metadata: Vec::new(),
            agent_id: [0u8; 16],
            timestamp_ns: 0,
        }
    }

    pub fn with_metadata(mut self, metadata: Vec<u8>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_agent(mut self, agent_id: [u8; 16]) -> Self {
        self.agent_id = agent_id;
        self
    }

    pub fn with_timestamp(mut self, timestamp_ns: u64) -> Self {
        self.timestamp_ns = timestamp_ns;
        self
    }

    /// SHA-256 of the canonical encoding, computed **without** allocating the
    /// intermediate byte buffer — streams each field straight into the hasher.
    /// Produces a digest identical to `Sha256::digest(self.canonical_bytes())`,
    /// so it is a drop-in fast path (no on-disk/format change).
    pub fn payload_hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(self.id.to_le_bytes());
        h.update((self.vector.len() as u32).to_le_bytes());
        for f in &self.vector {
            h.update(f.to_le_bytes());
        }
        h.update((self.metadata.len() as u32).to_le_bytes());
        h.update(&self.metadata);
        h.update(self.agent_id);
        h.update(self.timestamp_ns.to_le_bytes());
        h.finalize().into()
    }

    /// Canonical byte representation used as hash preimage.
    ///
    /// Fixed-structure concatenation: prevents length-extension issues
    /// because each field has a deterministic length or delimiter.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let vec_len = self.vector.len() * 4;
        let mut out = Vec::with_capacity(8 + vec_len + 4 + self.metadata.len() + 16 + 8);
        out.extend_from_slice(&self.id.to_le_bytes());
        // Dimension prefix prevents aliasing between different-length vectors
        out.extend_from_slice(&(self.vector.len() as u32).to_le_bytes());
        for f in &self.vector {
            out.extend_from_slice(&f.to_le_bytes());
        }
        // Metadata length prefix
        out.extend_from_slice(&(self.metadata.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.metadata);
        out.extend_from_slice(&self.agent_id);
        out.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        out
    }
}

/// Which gate variant produced a receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GateVariant {
    /// No integrity check. Used only to establish baseline throughput.
    Null = 0,
    /// Sequential SHA-256 hash chain: each entry chains the previous hash.
    HashChain = 1,
    /// Merkle Mountain Range: append-only tree with O(log n) per insert.
    Merkle = 2,
}

/// Cryptographic receipt returned by a WriteGate for each admitted payload.
#[derive(Debug, Clone)]
pub struct WriteReceipt {
    /// Monotonically increasing write sequence number within this gate.
    pub sequence: u64,
    /// SHA-256 of the canonical payload bytes.
    pub payload_hash: [u8; 32],
    /// The gate's current chain commitment after this write.
    /// - NullGate:      all zeros
    /// - HashChainGate: SHA-256(prev_commitment || payload_hash || sequence)
    /// - MerkleGate:    MMR root after this leaf was appended
    pub chain_commitment: [u8; 32],
    /// Gate variant that produced this receipt.
    pub gate_variant: GateVariant,
}

/// Errors returned when a gate rejects an admission.
#[derive(Debug, Error)]
pub enum GateError {
    #[error("admission denied: {0}")]
    Denied(String),
}
